"""Shared components for HSPN weakly supervised lesion segmentation.

The model is an SSHR-inspired HFRM-CAM network: image-level labels supervise
multi-level classification heads, while the fused CAM is used as the weak
lesion evidence map. Pixel masks are optional and are only used for validation
when available.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class TargetSpec:
    slug: str
    class_id: int
    prob_field: str
    area_field: str


TARGET_SPECS: Dict[str, Tuple[TargetSpec, ...]] = {
    "proliferation": (
        TargetSpec("mesangial", 1, "mesangial_prob", "mesangial_weakseg_area"),
        TargetSpec("endocapillary", 2, "endocapillary_prob", "endocapillary_weakseg_area"),
    ),
    "mesangial": (
        TargetSpec("mesangial", 1, "mesangial_prob", "mesangial_weakseg_area"),
    ),
    "endocapillary": (
        TargetSpec("endocapillary", 2, "endocapillary_prob", "endocapillary_weakseg_area"),
    ),
    "crescent": (
        TargetSpec("cellular_crescent", 1, "cellular_crescent_prob", "cellular_crescent_weakseg_area"),
        TargetSpec("fibrocellular_crescent", 2, "fibrocellular_crescent_prob", "fibrocellular_crescent_weakseg_area"),
        TargetSpec("fibrous_crescent", 3, "fibrous_crescent_prob", "fibrous_crescent_weakseg_area"),
    ),
    "crescent_binary": (
        TargetSpec("crescent", 1, "crescent_prob", "crescent_weakseg_area"),
    ),
}


def target_specs_for_task(task: str) -> Tuple[TargetSpec, ...]:
    task = task.strip().lower()
    if task not in TARGET_SPECS:
        raise ValueError("Unknown weakseg task '{}'. Valid: {}".format(task, ", ".join(sorted(TARGET_SPECS))))
    return TARGET_SPECS[task]


def parse_class_pixels(raw: object) -> Dict[str, int]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return {str(key): int(value) for key, value in raw.items()}
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): int(value) for key, value in data.items()}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: object, path: Path) -> None:
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=convert)


def first_value(row: Dict[str, str], names: Iterable[str]) -> str:
    lower = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value not in (None, ""):
            return str(value)
    return ""


def resolve_path(data_root: Path, raw_path: str) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path if path.exists() else None
    candidate = data_root / path
    return candidate if candidate.exists() else None


def find_split_file(data_root: Path, split: str, folder: str, stem: str, exts: Iterable[str]) -> Optional[Path]:
    for ext in exts:
        candidate = data_root / split / folder / "{}{}".format(stem, ext)
        if candidate.exists():
            return candidate
    return None


def safe_crop_id(row: Dict[str, str], image_path: Optional[Path], index: int) -> str:
    for key in ("crop_id", "case_id", "image_name", "name", "filename", "file_name"):
        value = first_value(row, (key,))
        if value:
            return Path(value.replace("\\", "/")).stem
    if image_path is not None:
        return image_path.stem
    return "crop_{:06d}".format(index)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(image.device, image.dtype)
    std = IMAGENET_STD.to(image.device, image.dtype)
    return (image * std + mean).clamp(0.0, 1.0)


def normalize_raw_tensor(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(image.device, image.dtype)
    std = IMAGENET_STD.to(image.device, image.dtype)
    return (image - mean) / std


def image_tissue_prior(raw_image: torch.Tensor, white_threshold: float = 0.92, saturation_threshold: float = 0.045) -> torch.Tensor:
    """Estimate a foreground prior from RGB appearance.

    This is intentionally weak: it only suppresses near-white background and
    lets the classifier decide which foreground pixels support each lesion.
    """

    brightness = raw_image.mean(dim=0, keepdim=True)
    saturation = raw_image.max(dim=0, keepdim=True).values - raw_image.min(dim=0, keepdim=True).values
    prior = ((brightness < white_threshold) | (saturation > saturation_threshold)).float()
    prior = F.avg_pool2d(prior.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
    return (prior > 0.25).float()


def pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    return TVF.to_tensor(image)


def mask_to_tensor(path: Optional[Path], image_size: int) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.full((image_size, image_size), -1, dtype=torch.long)
    mask = Image.open(path)
    if mask.size != (image_size, image_size):
        mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
    return torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()


def maybe_augment(image: Image.Image, mask: Optional[Image.Image], strength: str) -> Tuple[Image.Image, Optional[Image.Image]]:
    if strength in ("off", "none", "false", "0"):
        return image, mask
    if random.random() < 0.5:
        image = TVF.hflip(image)
        mask = TVF.hflip(mask) if mask is not None else None
    if random.random() < 0.5:
        image = TVF.vflip(image)
        mask = TVF.vflip(mask) if mask is not None else None
    max_angle = 15 if strength == "basic" else 30
    if random.random() < 0.75:
        angle = random.uniform(-max_angle, max_angle)
        image = TVF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=255)
        mask = TVF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0) if mask is not None else None
    if strength == "strong":
        factor = random.uniform(0.85, 1.15)
        image = TVF.adjust_contrast(image, factor)
        image = TVF.adjust_brightness(image, random.uniform(0.90, 1.10))
        image = TVF.adjust_saturation(image, random.uniform(0.90, 1.10))
    return image, mask


class HspnWeakSegDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        task: str,
        split: str,
        image_size: int,
        min_positive_pixels: int = 8,
        manifest: Optional[Path] = None,
        augmentation: str = "off",
        limit: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.task = task
        self.target_specs = target_specs_for_task(task)
        self.split = split
        self.image_size = int(image_size)
        self.min_positive_pixels = int(min_positive_pixels)
        self.augmentation = augmentation
        self.manifest = Path(manifest) if manifest else self.data_root / "manifests" / "crops.csv"
        if not self.manifest.exists():
            raise FileNotFoundError("Missing weakseg manifest: {}".format(self.manifest))

        rows: List[Dict[str, object]] = []
        for index, row in enumerate(read_csv_rows(self.manifest)):
            row_split = first_value(row, ("split",)) or split
            if row_split != split:
                continue
            image_path = resolve_path(self.data_root, first_value(row, ("image_path", "image", "source_image")))
            crop_id = safe_crop_id(row, image_path, index)
            if image_path is None:
                image_path = find_split_file(self.data_root, row_split, "images", crop_id, IMAGE_EXTS)
            if image_path is None or not image_path.exists():
                continue
            mask_path = resolve_path(self.data_root, first_value(row, ("mask_path", "mask", "source_mask")))
            if mask_path is None:
                mask_path = find_split_file(self.data_root, row_split, "masks", crop_id, (".png", ".tif", ".tiff"))
            class_pixels = parse_class_pixels(first_value(row, ("class_pixels",)))
            target = np.asarray(
                [
                    float(class_pixels.get(str(spec.class_id), 0) >= self.min_positive_pixels)
                    for spec in self.target_specs
                ],
                dtype=np.float32,
            )
            rows.append(
                {
                    **row,
                    "crop_id": crop_id,
                    "split": row_split,
                    "image_abs_path": str(image_path),
                    "mask_abs_path": str(mask_path) if mask_path is not None and mask_path.exists() else "",
                    "target": target,
                    "class_pixels_parsed": class_pixels,
                }
            )
            if limit and len(rows) >= limit:
                break
        if not rows:
            raise RuntimeError("No {} weakseg rows found under {}".format(split, self.data_root))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        image_path = Path(str(row["image_abs_path"]))
        image = Image.open(image_path).convert("RGB")
        mask_path_text = str(row.get("mask_abs_path") or "")
        mask_pil = Image.open(mask_path_text) if mask_path_text else None
        if self.split == "train":
            image, mask_pil = maybe_augment(image, mask_pil, self.augmentation)
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        if mask_pil is not None and mask_pil.size != (self.image_size, self.image_size):
            mask_pil = mask_pil.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        raw = TVF.to_tensor(image)
        prior = image_tissue_prior(raw)
        mask = torch.from_numpy(np.asarray(mask_pil, dtype=np.int64)).long() if mask_pil is not None else torch.full(
            (self.image_size, self.image_size), -1, dtype=torch.long
        )
        return {
            "image": normalize_raw_tensor(raw),
            "raw_image": raw,
            "prior": prior,
            "mask": mask,
            "target": torch.tensor(row["target"], dtype=torch.float32),
            "crop_id": str(row["crop_id"]),
            "slide_id": first_value(row, ("slide_id", "wsi_id", "case_id")) or "unknown_slide",
            "glomerulus_id": first_value(row, ("glomerulus_id", "glomerulus_index")) or str(index + 1),
            "image_path": str(image_path),
        }

    @property
    def targets(self) -> np.ndarray:
        return np.stack([row["target"] for row in self.rows]).astype(np.float32)

    def label_summary(self) -> Dict[str, object]:
        targets = self.targets
        summary: Dict[str, object] = {
            "task": self.task,
            "split": self.split,
            "samples": int(targets.shape[0]),
            "any_positive": int((targets.sum(axis=1) > 0).sum()),
            "negative": int((targets.sum(axis=1) == 0).sum()),
        }
        for index, spec in enumerate(self.target_specs):
            summary["{}_positive".format(spec.slug)] = int(targets[:, index].sum())
        return summary


class HFRM(nn.Module):
    def __init__(self, in_channels: int, deep_channels: int, context_kernel: int = 15) -> None:
        super().__init__()
        hidden = max(deep_channels // 8, 16)
        self.veto_mlp = nn.Sequential(
            nn.Linear(deep_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
            nn.Sigmoid(),
        )
        padding = context_kernel // 2
        self.context_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=context_kernel,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        nn.init.constant_(self.context_conv.weight, 1.0 / float(context_kernel * context_kernel))
        self.gamma_veto = nn.Parameter(torch.zeros(1))
        self.gamma_context = nn.Parameter(torch.zeros(1))

    def forward(self, shallow: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(deep, 1).flatten(1)
        weights = self.veto_mlp(pooled).view(shallow.shape[0], shallow.shape[1], 1, 1)
        vetoed = shallow * weights
        smoothed = self.context_conv(shallow)
        return shallow + self.gamma_veto * vetoed + self.gamma_context * smoothed


def build_resnet(backbone: str, pretrained: bool) -> nn.Module:
    factories = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if backbone not in factories:
        raise ValueError("Unsupported backbone '{}'. Valid: {}".format(backbone, ", ".join(sorted(factories))))
    factory = factories[backbone]
    if not pretrained:
        return factory(weights=None)
    try:
        weights_name = {
            "resnet18": "ResNet18_Weights",
            "resnet34": "ResNet34_Weights",
            "resnet50": "ResNet50_Weights",
        }[backbone]
        weights = getattr(models, weights_name).DEFAULT
        return factory(weights=weights)
    except Exception:
        return factory(pretrained=True)


def load_classifier_init_state(init_model: object) -> Optional[Dict[str, torch.Tensor]]:
    if init_model in (None, ""):
        return None
    checkpoint_path = Path(str(init_model))
    if not checkpoint_path.exists():
        raise FileNotFoundError("Missing classifier init model: {}".format(checkpoint_path))
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        raw_state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        raw_state = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
    else:
        raise RuntimeError("Unsupported classifier init checkpoint type: {}".format(type(checkpoint).__name__))
    return {str(key): value for key, value in raw_state.items() if torch.is_tensor(value)}


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_compatible_state(model: nn.Module, init_state: Dict[str, torch.Tensor], init_model: object) -> None:
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in strip_module_prefix(init_state).items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    if not compatible:
        raise RuntimeError("Classifier init model has no compatible weights for this backbone: {}".format(init_model))
    model.load_state_dict(compatible, strict=False)


def resnet_feature_extractor(model: nn.Module) -> nn.Module:
    return nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )


def build_aux_classifier(backbone: str, pretrained: bool, init_model: object = None) -> Tuple[nn.Module, int, Optional[Dict[str, torch.Tensor]]]:
    efficientnet_factories = {
        "efficientnet_v2_s": (models.efficientnet_v2_s, "EfficientNet_V2_S_Weights"),
        "efficientnet_v2_m": (models.efficientnet_v2_m, "EfficientNet_V2_M_Weights"),
    }
    resnet_factories = {
        "resnet18": (models.resnet18, "ResNet18_Weights"),
        "resnet34": (models.resnet34, "ResNet34_Weights"),
    }
    valid = sorted([*efficientnet_factories.keys(), *resnet_factories.keys()])
    if backbone in efficientnet_factories:
        factory, weights_name = efficientnet_factories[backbone]
        if pretrained:
            try:
                model = factory(weights=getattr(models, weights_name).DEFAULT)
            except Exception:
                model = factory(pretrained=True)
        else:
            model = factory(weights=None)
        out_channels = int(model.classifier[1].in_features)
        init_state = load_classifier_init_state(init_model)
        if init_state is not None:
            load_compatible_state(model, init_state, init_model)
        return model.features, out_channels, init_state

    if backbone in resnet_factories:
        factory, weights_name = resnet_factories[backbone]
        if pretrained:
            try:
                model = factory(weights=getattr(models, weights_name).DEFAULT)
            except Exception:
                model = factory(pretrained=True)
        else:
            model = factory(weights=None)
        out_channels = int(model.fc.in_features)
        init_state = load_classifier_init_state(init_model)
        if init_state is not None:
            load_compatible_state(model, init_state, init_model)
        return resnet_feature_extractor(model), out_channels, init_state

    raise ValueError("Unsupported classifier_backbone '{}'. Valid: shared, {}".format(backbone, ", ".join(valid)))


def classifier_head_tensors(
    init_state: Optional[Dict[str, torch.Tensor]],
    classifier_init_class_index: int,
    num_classes: int,
    expected_weight_shape: Tuple[int, int],
    expected_bias_shape: Tuple[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if init_state is None:
        return None, None
    state = strip_module_prefix(init_state)
    for prefix in ("classifier.1", "classifier", "fc"):
        head_weight = state.get("{}.weight".format(prefix))
        head_bias = state.get("{}.bias".format(prefix))
        if head_weight is None:
            continue
        if (
            int(classifier_init_class_index) >= 0
            and head_weight.ndim == 2
            and head_weight.shape[0] >= int(classifier_init_class_index) + num_classes
        ):
            start = int(classifier_init_class_index)
            head_weight = head_weight[start:start + num_classes]
            if head_bias is not None and head_bias.ndim == 1:
                head_bias = head_bias[start:start + num_classes]
        if tuple(head_weight.shape) != expected_weight_shape:
            continue
        if head_bias is not None and tuple(head_bias.shape) != expected_bias_shape:
            head_bias = None
        return head_weight, head_bias
    return None, None


class HfrmCamModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = False,
        context_kernel: int = 15,
        cam_weights: Sequence[float] = (0.2, 0.3, 0.5),
        classifier_weight: float = 0.0,
        classifier_dropout: float = 0.2,
        classifier_backbone: str = "shared",
        classifier_init_model: object = None,
        classifier_init_class_index: int = -1,
        classifier_input: str = "normalized",
        cam_pool: str = "avg",
        cam_pool_topk: float = 0.05,
    ) -> None:
        super().__init__()
        base = build_resnet(backbone, pretrained=pretrained)
        self.backbone_name = backbone
        self.num_classes = int(num_classes)
        self.cam_weights = tuple(float(v) for v in cam_weights)
        self.classifier_weight = float(classifier_weight)
        self.classifier_backbone = str(classifier_backbone)
        self.classifier_input = str(classifier_input)
        if self.classifier_input not in {"normalized", "raw"}:
            raise ValueError("Unsupported classifier_input '{}'; use normalized or raw".format(self.classifier_input))
        self.cam_pool = str(cam_pool)
        self.cam_pool_topk = float(cam_pool_topk)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        if backbone in {"resnet18", "resnet34"}:
            channels = (128, 256, 512)
        else:
            channels = (512, 1024, 2048)
        self.hfrm_l2 = HFRM(channels[0], channels[2], context_kernel=context_kernel)
        self.hfrm_l3 = HFRM(channels[1], channels[2], context_kernel=context_kernel)
        self.head_l2 = nn.Conv2d(channels[0], self.num_classes, kernel_size=1)
        self.head_l3 = nn.Conv2d(channels[1], self.num_classes, kernel_size=1)
        self.head_l4 = nn.Conv2d(channels[2], self.num_classes, kernel_size=1)
        if self.classifier_backbone == "shared":
            self.aux_classifier_features = None
            classifier_channels = channels[2]
            classifier_init_state = None
        else:
            self.aux_classifier_features, classifier_channels, classifier_init_state = build_aux_classifier(
                self.classifier_backbone,
                pretrained=pretrained,
                init_model=classifier_init_model,
            )
        self.image_head = nn.Sequential(
            nn.Dropout(p=float(classifier_dropout)),
            nn.Linear(classifier_channels, self.num_classes),
        )
        for head in (self.head_l2, self.head_l3, self.head_l4):
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        nn.init.xavier_uniform_(self.image_head[1].weight)
        nn.init.zeros_(self.image_head[1].bias)
        if classifier_init_state is not None:
            head_weight, head_bias = classifier_head_tensors(
                classifier_init_state,
                classifier_init_class_index=classifier_init_class_index,
                num_classes=self.num_classes,
                expected_weight_shape=tuple(self.image_head[1].weight.shape),
                expected_bias_shape=tuple(self.image_head[1].bias.shape),
            )
            if head_weight is not None:
                self.image_head[1].weight.data.copy_(head_weight)
            if head_bias is not None:
                self.image_head[1].bias.data.copy_(head_bias)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        feat_l2 = self.layer2(x)
        feat_l3 = self.layer3(feat_l2)
        feat_l4 = self.layer4(feat_l3)
        return feat_l2, feat_l3, feat_l4

    def pool_cam_logits(self, cam: torch.Tensor) -> torch.Tensor:
        flat = cam.flatten(2)
        if self.cam_pool in {"avg", "gap"}:
            return flat.mean(dim=-1)
        if self.cam_pool == "topk":
            k = max(1, int(round(flat.shape[-1] * self.cam_pool_topk)))
            return torch.topk(flat, k=k, dim=-1).values.mean(dim=-1)
        if self.cam_pool == "avg_topk":
            k = max(1, int(round(flat.shape[-1] * self.cam_pool_topk)))
            topk_logits = torch.topk(flat, k=k, dim=-1).values.mean(dim=-1)
            return 0.5 * flat.mean(dim=-1) + 0.5 * topk_logits
        raise ValueError("Unsupported cam_pool '{}'; use avg, topk, or avg_topk".format(self.cam_pool))

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        feat_l2, feat_l3, feat_l4 = self.forward_features(x)
        rect_l2 = self.hfrm_l2(feat_l2, feat_l4)
        rect_l3 = self.hfrm_l3(feat_l3, feat_l4)
        cams = [self.head_l2(rect_l2), self.head_l3(rect_l3), self.head_l4(feat_l4)]
        logits = [self.pool_cam_logits(cam) for cam in cams]
        total = sum(self.cam_weights)
        weights = [item / total for item in self.cam_weights]
        cam_fused_logits = sum(weight * logit for weight, logit in zip(weights, logits))
        if self.aux_classifier_features is None:
            classifier_features = feat_l4
        else:
            classifier_input = denormalize(x) if self.classifier_input == "raw" else x
            classifier_features = self.aux_classifier_features(classifier_input)
        image_logits = self.image_head(F.adaptive_avg_pool2d(classifier_features, 1).flatten(1))
        classifier_weight = max(0.0, min(1.0, self.classifier_weight))
        fused_logits = (1.0 - classifier_weight) * cam_fused_logits + classifier_weight * image_logits
        return {
            "cams": cams,
            "logits": logits,
            "cam_fused_logits": cam_fused_logits,
            "image_logits": image_logits,
            "fused_logits": fused_logits,
        }

    def fused_cam(self, x: torch.Tensor, output_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.forward(x)
        cams = output["cams"]
        total = sum(self.cam_weights)
        weights = [item / total for item in self.cam_weights]
        resized = [
            F.interpolate(F.relu(cam), size=output_size, mode="bilinear", align_corners=False)
            for cam in cams
        ]
        fused = sum(weight * cam for weight, cam in zip(weights, resized))
        fused = normalize_cams(fused)
        probs = torch.sigmoid(output["fused_logits"])
        return fused, probs


def normalize_cams(cams: torch.Tensor) -> torch.Tensor:
    flat = cams.flatten(2)
    min_v = flat.min(dim=-1).values[:, :, None, None]
    max_v = flat.max(dim=-1).values[:, :, None, None]
    return (cams - min_v) / (max_v - min_v).clamp_min(1e-6)


def fused_cam_with_tta(
    model: HfrmCamModel,
    x: torch.Tensor,
    output_size: Tuple[int, int],
    eval_tta: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cams, probs = model.fused_cam(x, output_size=output_size)
    if not eval_tta:
        return cams, probs

    cam_list = [cams]
    prob_list = [probs]
    for dims in ((-1,), (-2,)):
        aug = torch.flip(x, dims=dims)
        aug_cams, aug_probs = model.fused_cam(aug, output_size=output_size)
        cam_list.append(torch.flip(aug_cams, dims=dims))
        prob_list.append(aug_probs)
    return normalize_cams(torch.stack(cam_list, dim=0).mean(dim=0)), torch.stack(prob_list, dim=0).mean(dim=0)


def make_model(
    num_classes: int,
    backbone: str,
    pretrained: bool,
    context_kernel: int,
    cam_weights: Sequence[float],
    classifier_weight: float = 0.0,
    classifier_dropout: float = 0.2,
    classifier_backbone: str = "shared",
    classifier_init_model: object = None,
    classifier_init_class_index: int = -1,
    classifier_input: str = "normalized",
    cam_pool: str = "avg",
    cam_pool_topk: float = 0.05,
) -> HfrmCamModel:
    return HfrmCamModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        context_kernel=context_kernel,
        cam_weights=cam_weights,
        classifier_weight=classifier_weight,
        classifier_dropout=classifier_dropout,
        classifier_backbone=classifier_backbone,
        classifier_init_model=classifier_init_model,
        classifier_init_class_index=classifier_init_class_index,
        classifier_input=classifier_input,
        cam_pool=cam_pool,
        cam_pool_topk=cam_pool_topk,
    )


def positive_weight(targets: np.ndarray, device: torch.device, max_weight: float = 8.0) -> torch.Tensor:
    pos = targets.sum(axis=0)
    neg = targets.shape[0] - pos
    weights = neg / np.maximum(pos, 1.0)
    weights = np.clip(weights, 0.25, max_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        pt = torch.exp(-bce)
        return ((1.0 - pt).pow(self.gamma) * bce).mean()


def topk_cam_scores(cam_prob: torch.Tensor, ratio: float = 0.05) -> torch.Tensor:
    flat = cam_prob.flatten(2)
    k = max(1, int(round(flat.shape[-1] * ratio)))
    return torch.topk(flat, k=k, dim=-1).values.mean(dim=-1).clamp(1e-5, 1.0 - 1e-5)


@dataclass
class WeakSegLossConfig:
    cls_weights: Tuple[float, float, float] = (0.2, 0.3, 0.5)
    cam_cls_weight: float = 1.0
    image_cls_weight: float = 0.8
    fused_cls_weight: float = 0.4
    prior_weight: float = 0.08
    negative_cam_weight: float = 0.05
    area_weight: float = 0.03
    score_consistency_weight: float = 0.20
    positive_area_target: float = 0.08
    topk_ratio: float = 0.05


def weakseg_loss(
    output: Dict[str, object],
    targets: torch.Tensor,
    priors: torch.Tensor,
    criterion: nn.Module,
    config: WeakSegLossConfig,
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = output["logits"]
    cams = output["cams"]
    cls_weights = config.cls_weights
    cls_total = sum(cls_weights)
    cls_loss = sum((weight / cls_total) * criterion(logit, targets) for weight, logit in zip(cls_weights, logits))
    image_logits = output.get("image_logits")
    if image_logits is not None and config.image_cls_weight > 0:
        image_cls_loss = criterion(image_logits, targets)
    else:
        image_cls_loss = torch.zeros((), device=targets.device, dtype=targets.dtype)
    fused_logits = output.get("fused_logits")
    if fused_logits is not None and config.fused_cls_weight > 0:
        fused_cls_loss = criterion(fused_logits, targets)
    else:
        fused_cls_loss = torch.zeros((), device=targets.device, dtype=targets.dtype)

    resized = [
        F.interpolate(F.relu(cam), size=image_size, mode="bilinear", align_corners=False)
        for cam in cams
    ]
    cam_prob = normalize_cams(sum((weight / cls_total) * cam for weight, cam in zip(cls_weights, resized)))
    positive = targets[:, :, None, None]
    negative = 1.0 - positive

    negative_cam_loss = (cam_prob * negative).mean()
    positive_pixels = positive.sum().clamp_min(1.0)
    outside_prior_loss = (cam_prob * positive * (1.0 - priors)).sum() / positive_pixels / max(image_size[0] * image_size[1], 1)
    area = (cam_prob * positive).flatten(2).mean(dim=-1)
    positive_mask = targets > 0.5
    if positive_mask.any():
        area_loss = F.l1_loss(area[positive_mask], torch.full_like(area[positive_mask], config.positive_area_target))
    else:
        area_loss = torch.zeros((), device=targets.device, dtype=targets.dtype)
    cam_scores = topk_cam_scores(cam_prob.float(), ratio=config.topk_ratio)
    target_scores = targets.float()
    score_consistency = -(
        target_scores * torch.log(cam_scores)
        + (1.0 - target_scores) * torch.log(1.0 - cam_scores)
    ).mean()

    loss = (
        config.cam_cls_weight * cls_loss
        + config.image_cls_weight * image_cls_loss
        + config.fused_cls_weight * fused_cls_loss
        + config.negative_cam_weight * negative_cam_loss
        + config.prior_weight * outside_prior_loss
        + config.area_weight * area_loss
        + config.score_consistency_weight * score_consistency
    )
    parts = {
        "loss_cls": float(cls_loss.detach().cpu()),
        "loss_image_cls": float(image_cls_loss.detach().cpu()),
        "loss_fused_cls": float(fused_cls_loss.detach().cpu()),
        "loss_prior": float(outside_prior_loss.detach().cpu()),
        "loss_negative_cam": float(negative_cam_loss.detach().cpu()),
        "loss_area": float(area_loss.detach().cpu()),
        "loss_score_consistency": float(score_consistency.detach().cpu()),
    }
    return loss, parts


def metric_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    return {
        "tp": int(np.logical_and(y_true == 1, y_pred == 1).sum()),
        "fp": int(np.logical_and(y_true == 0, y_pred == 1).sum()),
        "fn": int(np.logical_and(y_true == 1, y_pred == 0).sum()),
        "tn": int(np.logical_and(y_true == 0, y_pred == 0).sum()),
    }


def counts_to_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        **counts,
    }


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = y_true.astype(np.int32)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_grid(step: float) -> List[float]:
    step = max(0.01, min(float(step), 0.50))
    return [round(float(v), 4) for v in np.arange(step, 1.0, step)]


def best_thresholds_by_f1(y_true: np.ndarray, y_prob: np.ndarray, specs: Sequence[TargetSpec], values: Sequence[float]) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for index, spec in enumerate(specs):
        best_f1 = -1.0
        best_thr = 0.5
        for threshold in values:
            pred = (y_prob[:, index] >= threshold).astype(np.int32)
            f1 = counts_to_metrics(metric_counts(y_true[:, index], pred))["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(threshold)
        thresholds[spec.slug] = best_thr
    return thresholds


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    specs: Sequence[TargetSpec],
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    per_class: Dict[str, object] = {}
    preds = np.zeros_like(y_true, dtype=np.int32)
    for index, spec in enumerate(specs):
        preds[:, index] = (y_prob[:, index] >= thresholds[spec.slug]).astype(np.int32)
        metrics = counts_to_metrics(metric_counts(y_true[:, index], preds[:, index]))
        auc = binary_auc(y_true[:, index], y_prob[:, index])
        if auc is not None:
            metrics["auc"] = auc
        metrics["threshold"] = float(thresholds[spec.slug])
        per_class[spec.slug] = metrics
    macro_f1 = float(np.mean([per_class[spec.slug]["f1"] for spec in specs]))
    exact_match = float((preds == y_true).all(axis=1).mean())
    auc_values = [per_class[spec.slug].get("auc") for spec in specs if per_class[spec.slug].get("auc") is not None]
    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "exact_match": exact_match,
        "mean_auc": float(np.mean(auc_values)) if auc_values else 0.0,
    }


def mask_metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    dice = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "dice": float(dice),
        "iou": float(iou),
        "mask_precision": float(precision),
        "mask_recall": float(recall),
        **counts,
    }


def update_mask_counts(counts: Dict[str, int], pred: np.ndarray, true: np.ndarray) -> None:
    counts["tp"] += int(np.logical_and(pred, true).sum())
    counts["fp"] += int(np.logical_and(pred, ~true).sum())
    counts["fn"] += int(np.logical_and(~pred, true).sum())
    counts["tn"] += int(np.logical_and(~pred, ~true).sum())


def postprocess_binary_mask(
    binary_mask: np.ndarray,
    min_area: int = 8,
    open_radius: int = 0,
    close_radius: int = 0,
    fill_holes: bool = False,
    keep_components: int = 0,
) -> np.ndarray:
    mask = binary_mask.astype(bool)
    if not mask.any():
        return mask
    try:
        from scipy import ndimage
    except Exception:
        return mask

    if open_radius > 0:
        structure = np.ones((open_radius * 2 + 1, open_radius * 2 + 1), dtype=bool)
        mask = ndimage.binary_opening(mask, structure=structure)
    if close_radius > 0:
        structure = np.ones((close_radius * 2 + 1, close_radius * 2 + 1), dtype=bool)
        mask = ndimage.binary_closing(mask, structure=structure)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if not mask.any():
        return mask.astype(bool)

    labels, count = ndimage.label(mask)
    components: List[Tuple[int, int]] = []
    for label in range(1, count + 1):
        area = int((labels == label).sum())
        if area >= min_area:
            components.append((label, area))
    if not components:
        return np.zeros_like(mask, dtype=bool)
    components.sort(key=lambda item: item[1], reverse=True)
    if keep_components > 0:
        components = components[:keep_components]
    keep = {label for label, _area in components}
    return np.isin(labels, list(keep))


def cam_binary_mask(
    cam: np.ndarray,
    mask_threshold: float,
    active: bool = True,
    prior: Optional[np.ndarray] = None,
    use_prior: bool = False,
    min_area: int = 8,
    open_radius: int = 0,
    close_radius: int = 0,
    fill_holes: bool = False,
    keep_components: int = 0,
    topk_area: float = 0.0,
) -> np.ndarray:
    if not active:
        return np.zeros_like(cam, dtype=bool)
    cam_np = np.asarray(cam, dtype=np.float32)
    threshold = float(mask_threshold)
    topk_area = max(0.0, min(float(topk_area), 1.0))
    if topk_area > 0:
        raw_mask = cam_np >= threshold
        if float(raw_mask.mean()) > topk_area:
            threshold = max(threshold, float(np.quantile(cam_np, 1.0 - topk_area)))
    mask = cam_np >= threshold
    if use_prior and prior is not None:
        prior_mask = np.asarray(prior).astype(bool)
        if prior_mask.shape == mask.shape:
            mask = mask & prior_mask
    return postprocess_binary_mask(
        mask,
        min_area=min_area,
        open_radius=open_radius,
        close_radius=close_radius,
        fill_holes=fill_holes,
        keep_components=keep_components,
    )


def connected_boxes(binary_mask: np.ndarray, min_area: int = 8) -> List[Tuple[int, int, int, int, int]]:
    try:
        from scipy import ndimage
    except Exception:
        ndimage = None
    mask = binary_mask.astype(bool)
    if not mask.any():
        return []
    if ndimage is None:
        ys, xs = np.where(mask)
        return [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, int(mask.sum()))]
    labels, count = ndimage.label(mask)
    boxes: List[Tuple[int, int, int, int, int]] = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        area = int(len(xs))
        if area < min_area:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, area))
    boxes.sort(key=lambda item: item[-1], reverse=True)
    return boxes


def cam_to_overlay(image: Image.Image, cam: np.ndarray, color: Tuple[int, int, int] = (220, 30, 30), alpha: float = 0.42) -> Image.Image:
    image_np = np.asarray(image.convert("RGB")).astype(np.float32)
    cam_np = np.clip(cam.astype(np.float32), 0.0, 1.0)
    color_np = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    heat = image_np * (1.0 - cam_np[:, :, None]) + color_np * cam_np[:, :, None]
    overlay = image_np * (1.0 - alpha) + heat * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def save_evidence_overlay(
    image_path: Path,
    cams: np.ndarray,
    probs: np.ndarray,
    specs: Sequence[TargetSpec],
    thresholds: Dict[str, float],
    output_path: Path,
    mask_threshold: float = 0.35,
    min_area: int = 8,
    open_radius: int = 0,
    close_radius: int = 0,
    fill_holes: bool = False,
    keep_components: int = 0,
    topk_area: float = 0.0,
    use_prior: bool = False,
) -> None:
    image = Image.open(image_path).convert("RGB")
    size = 224
    thumbs: List[Image.Image] = []
    colors = [(220, 30, 30), (30, 150, 80), (40, 90, 210), (180, 80, 180)]
    for index, spec in enumerate(specs):
        cam = cams[index]
        resized_image = image.resize((cam.shape[1], cam.shape[0]))
        overlay = cam_to_overlay(resized_image, cam, color=colors[index % len(colors)])
        overlay = overlay.resize((size, size), resample=Image.BICUBIC)
        draw = ImageDraw.Draw(overlay)
        prior = None
        if use_prior:
            prior = image_tissue_prior(TVF.to_tensor(resized_image))[0].numpy()
        pred_mask = cam_binary_mask(
            cam,
            mask_threshold=mask_threshold,
            active=float(probs[index]) >= float(thresholds[spec.slug]),
            prior=prior,
            use_prior=use_prior,
            min_area=min_area,
            open_radius=open_radius,
            close_radius=close_radius,
            fill_holes=fill_holes,
            keep_components=keep_components,
            topk_area=topk_area,
        )
        scale_x = size / cam.shape[1]
        scale_y = size / cam.shape[0]
        for x1, y1, x2, y2, _area in connected_boxes(pred_mask, min_area=min_area)[:3]:
            draw.rectangle(
                [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                outline=colors[index % len(colors)],
                width=3,
            )
        label = "{} p={:.2f} area={:.1f}%".format(spec.slug, float(probs[index]), float(pred_mask.mean() * 100.0))
        draw.rectangle([0, size - 28, size, size], fill=(255, 255, 255))
        draw.text((5, size - 23), label, fill=(20, 20, 20))
        thumbs.append(overlay)
    if not thumbs:
        return
    canvas = Image.new("RGB", (size * len(thumbs), size), color=(255, 255, 255))
    for index, thumb in enumerate(thumbs):
        canvas.paste(thumb, (index * size, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> Tuple[HfrmCamModel, Dict[str, object]]:
    try:
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint_path), map_location=device)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    target_names = ckpt.get("target_names") if isinstance(ckpt, dict) else None
    num_classes = int(args.get("num_classes") or (len(target_names) if target_names else args.get("n_class", 1)))
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    has_image_head = isinstance(state, dict) and any(str(key).startswith("image_head.") for key in state)
    model = make_model(
        num_classes=num_classes,
        backbone=args.get("backbone", "resnet18"),
        pretrained=False,
        context_kernel=int(args.get("context_kernel", 15)),
        cam_weights=args.get("cam_weights", (0.2, 0.3, 0.5)),
        classifier_weight=float(args.get("classifier_weight", 0.0)) if has_image_head else 0.0,
        classifier_dropout=float(args.get("classifier_dropout", 0.2)),
        classifier_backbone=str(args.get("classifier_backbone", "shared")) if has_image_head else "shared",
        classifier_init_model=None,
        classifier_init_class_index=int(args.get("classifier_init_class_index", -1)),
        classifier_input=str(args.get("classifier_input", "normalized")),
        cam_pool=str(args.get("cam_pool", "avg")),
        cam_pool_topk=float(args.get("cam_pool_topk", 0.05)),
    )
    model.load_state_dict(state, strict=has_image_head)
    model.to(device)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {"args": args}
