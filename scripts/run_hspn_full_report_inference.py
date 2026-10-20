"""Run the current HSPN WSI scoring pipeline from real glomerulus crops.

This script stitches together the project-final inference route:

* mesangial proliferation: MESCNN binary classifier
* endocapillary proliferation: MESCNN multi-label classifier, E output only
* crescents: Faster R-CNN detection on real glomerulus crops
* crescent subtype: ResNet ROI classifier on detected crescent boxes
* reporting: separate ISKDC and Oxford score files plus real crop evidence PNGs

The evidence images are montages of the actual crop images used for inference,
with predicted crescent boxes and per-glomerulus labels overlaid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tvf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_hspn_pathology import (  # noqa: E402
    Thresholds,
    group_and_score,
    load_glomeruli,
    write_glomerulus_csv,
    write_iskdc_csv,
    write_iskdc_json,
    write_json,
    write_oxford_csv,
    write_oxford_json,
    write_summary_csv,
)
from train_hzy_crescent_subtype_classifier import (  # noqa: E402
    CLASS_SLUGS,
    build_model as build_subtype_model,
    build_transforms as build_subtype_transforms,
)
from train_hzy_detection_boxes import make_model as build_detection_model  # noqa: E402
from train_hzy_mescnn_mesangial_binary import (  # noqa: E402
    build_transforms as build_mesangial_transforms,
    load_model as load_mesangial_init_model,
    model_probs as mesangial_model_probs,
)
from train_hzy_mescnn_proliferation_multilabel import (  # noqa: E402
    build_transforms as build_proliferation_transforms,
    load_mescnn_model as load_proliferation_init_model,
)


DEFAULT_DATA_ROOT = Path(
    "/home/duyanhong/Dataspace/HZY/"
    "HZY_HSPN_glomerulus_lesion_tasks_proliferation_768_20260822/proliferation"
)
DEFAULT_OUTPUT_DIR = Path("/home/duyanhong/Dataspace/HZY/eval/hspn_full_oxford_report_real_20260822")
DEFAULT_M_CHECKPOINT = Path(
    "/home/duyanhong/Dataspace/HZY/checkpoints/"
    "hzy_hspn_mescnn_mesangial_binary_initM_384_focal_tta_20260822/best.pth"
)
DEFAULT_E_CHECKPOINT = Path(
    "/home/duyanhong/Dataspace/HZY/checkpoints/"
    "hzy_hspn_mescnn_proliferation_multilabel_initE_384_focal_sampler_noposw_20260822/best.pth"
)
DEFAULT_DETECTION_CHECKPOINT = Path(
    "/home/duyanhong/Dataspace/HZY/checkpoints/"
    "hzy_hspn_detection_boxes/crescent_binary_fasterrcnn_coco_aug_presence_balanced_20260619/best.pth"
)
DEFAULT_SUBTYPE_CHECKPOINT = Path(
    "/home/duyanhong/Dataspace/HZY/checkpoints/"
    "hzy_hspn_crescent_subtype_classifier_resnet18_20260619/best.pth"
)
DEFAULT_M_INIT = Path(
    "/home/duyanhong/Workspace/MESCnn/mescnn_data/classification/logs/cnn/holdout/efficientnetv2-m_M_V3.pth"
)
DEFAULT_E_INIT = Path(
    "/home/duyanhong/Workspace/MESCnn/mescnn_data/classification/logs/cnn/holdout/efficientnetv2-m_E_V3.pth"
)


@dataclass
class CropRecord:
    index: int
    slide_id: str
    glomerulus_id: str
    crop_id: str
    split: str
    image_path: Path
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    mesangial_prob: float = 0.0
    endocapillary_prob: float = 0.0
    crescent_prob: float = 0.0
    crescent_boxes: List[Dict[str, float]] = field(default_factory=list)
    subtype_probs: Dict[str, float] = field(
        default_factory=lambda: {slug: 0.0 for slug in CLASS_SLUGS}
    )
    crescent_subtype: str = ""
    ifta_percent: Optional[float] = None


@dataclass
class RoiRecord:
    roi_id: str
    crop_index: int
    slide_id: str
    glomerulus_id: str
    crop_id: str
    image_path: Path
    x1: float
    y1: float
    x2: float
    y2: float
    detection_score: float
    pred_class: str = ""
    pred_prob: float = 0.0
    probs: Dict[str, float] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-crop HSPN inference and generate ISKDC/Oxford reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=None, help="Crop manifest; defaults to data-root/manifests/crops.csv.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m-checkpoint", type=Path, default=DEFAULT_M_CHECKPOINT)
    parser.add_argument("--e-checkpoint", type=Path, default=DEFAULT_E_CHECKPOINT)
    parser.add_argument("--detection-checkpoint", type=Path, default=DEFAULT_DETECTION_CHECKPOINT)
    parser.add_argument("--subtype-checkpoint", type=Path, default=DEFAULT_SUBTYPE_CHECKPOINT)
    parser.add_argument("--m-init-model", type=Path, default=DEFAULT_M_INIT)
    parser.add_argument("--e-init-model", type=Path, default=DEFAULT_E_INIT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--detection-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--no-mesangial-tta", dest="mesangial_tta", action="store_false", default=True)
    parser.add_argument("--endocapillary-tta", action="store_true", default=False)
    parser.add_argument("--mesangial-threshold", type=float, default=0.35)
    parser.add_argument("--endocapillary-threshold", type=float, default=0.20)
    parser.add_argument("--crescent-threshold", type=float, default=0.30)
    parser.add_argument("--cellular-crescent-threshold", type=float, default=0.50)
    parser.add_argument("--fibrocellular-crescent-threshold", type=float, default=0.50)
    parser.add_argument("--fibrous-crescent-threshold", type=float, default=0.50)
    parser.add_argument("--generic-crescent-active", action="store_true", default=False,
                        help="Let generic crescent detections count as Oxford active crescents if subtype is absent.")
    parser.add_argument("--min-glomeruli", type=int, default=8)
    parser.add_argument("--max-glomeruli", type=int, default=0, help="Optional cap for smoke tests; 0 means all.")
    parser.add_argument("--limit-slides", default="", help="Comma-separated slide ids for smoke tests.")
    parser.add_argument("--max-evidence-items", type=int, default=24)
    parser.add_argument("--roi-size", type=int, default=224)
    parser.add_argument("--roi-margin", type=float, default=0.12)
    parser.add_argument(
        "--tubular-atrophy-slide-csv",
        type=Path,
        default=None,
        help="Scene/slide tubular-atrophy burden CSV, mapped to the T scorer field by slide_id.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def checkpoint_args(ckpt: object) -> object:
    if isinstance(ckpt, dict):
        return ckpt.get("args", {})
    return {}


def arg_value(args: object, name: str, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def state_dict_from_checkpoint(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(ckpt, dict) and all(hasattr(value, "shape") for value in ckpt.values()):
        return ckpt
    raise RuntimeError("Could not find a model state dict in checkpoint.")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def first_nonempty(row: Dict[str, str], names: Sequence[str]) -> Optional[str]:
    lower = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def safe_int(value: object, default: int) -> int:
    number = parse_float(value)
    if number is None:
        return default
    return int(round(number))


def safe_name(value: str) -> str:
    value = value.strip().replace("\\", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value) or "item"


def derive_slide_id(row: Dict[str, str], image_path: Path) -> str:
    slide_id = first_nonempty(row, ("slide_id", "wsi_id", "case_id", "image_id"))
    if slide_id:
        return slide_id
    stem = image_path.stem
    match = re.search(r"(.+?)_g\d{3,}", stem)
    return match.group(1) if match else stem


def derive_glomerulus_id(row: Dict[str, str], image_path: Path) -> str:
    glom_id = first_nonempty(row, ("glomerulus_id", "glom_id", "instance_id"))
    if glom_id:
        return glom_id
    crop_id = first_nonempty(row, ("crop_id", "roi_id", "id")) or image_path.stem
    match = re.search(r"(g\d{3,})", crop_id)
    if match:
        return match.group(1)
    index = first_nonempty(row, ("glomerulus_index", "glom_index", "index"))
    if index is not None:
        return f"g{safe_int(index, 0):04d}"
    return image_path.stem


def resolve_image_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = data_root / path
    if candidate.exists():
        return candidate
    return candidate


def manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest:
        return args.manifest
    preferred = args.data_root / "manifests" / "crops.csv"
    if preferred.exists():
        return preferred
    return args.data_root / "manifest.csv"


def load_crop_records(args: argparse.Namespace) -> List[CropRecord]:
    path = manifest_path(args)
    rows = read_csv(path)
    slide_filter = {item.strip() for item in args.limit_slides.split(",") if item.strip()}
    records: List[CropRecord] = []
    missing = 0

    for row in rows:
        split = first_nonempty(row, ("split",)) or ""
        if args.split and split != args.split:
            continue
        image_value = first_nonempty(row, ("image_path", "path", "crop_path"))
        if image_value is None:
            continue
        image_path = resolve_image_path(args.data_root, image_value)
        slide_id = derive_slide_id(row, image_path)
        if slide_filter and slide_id not in slide_filter:
            continue
        if not image_path.exists():
            missing += 1
            continue
        crop_id = first_nonempty(row, ("crop_id", "roi_id", "id")) or image_path.stem
        record = CropRecord(
            index=len(records),
            slide_id=slide_id,
            glomerulus_id=derive_glomerulus_id(row, image_path),
            crop_id=crop_id,
            split=split or args.split,
            image_path=image_path,
            x1=parse_float(first_nonempty(row, ("x1", "xmin", "x_min", "bbox_x_min", "left"))),
            y1=parse_float(first_nonempty(row, ("y1", "ymin", "y_min", "bbox_y_min", "top"))),
            x2=parse_float(first_nonempty(row, ("x2", "xmax", "x_max", "bbox_x_max", "right"))),
            y2=parse_float(first_nonempty(row, ("y2", "ymax", "y_max", "bbox_y_max", "bottom"))),
        )
        records.append(record)
        if args.max_glomeruli and len(records) >= args.max_glomeruli:
            break

    if missing:
        print(f"Skipped {missing} manifest rows with missing crop images.")
    if not records:
        raise RuntimeError(f"No crop records were loaded from {path}")
    return records


class CropClassificationDataset(Dataset):
    def __init__(self, records: Sequence[CropRecord], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        with Image.open(self.records[index].image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, index


class CropDetectionDataset(Dataset):
    def __init__(self, records: Sequence[CropRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        with Image.open(self.records[index].image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            tensor = tvf.to_tensor(image)
        return tensor, index, (width, height)


class RoiClassificationDataset(Dataset):
    def __init__(self, rois: Sequence[RoiRecord], transform):
        self.rois = rois
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rois)

    def __getitem__(self, index: int):
        with Image.open(self.rois[index].image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, index


def detection_collate(batch):
    images, indices, sizes = zip(*batch)
    return list(images), list(indices), list(sizes)


def load_mesangial_model(checkpoint_path: Path, init_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch_load(checkpoint_path, device)
    init_model = Path(arg_value(checkpoint_args(ckpt), "init_model", init_path))
    if not init_model.exists():
        init_model = init_path
    model = load_mesangial_init_model(init_model, device)
    model.load_state_dict(state_dict_from_checkpoint(ckpt))
    model.eval()
    return model


def load_endocapillary_model(checkpoint_path: Path, init_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch_load(checkpoint_path, device)
    ckpt_args = checkpoint_args(ckpt)
    init_model = Path(arg_value(ckpt_args, "init_model", init_path))
    if not init_model.exists():
        init_model = init_path
    model = load_proliferation_init_model(init_model, device, num_outputs=2)
    model.load_state_dict(state_dict_from_checkpoint(ckpt))
    model.eval()
    return model


def load_detection_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch_load(checkpoint_path, device)
    ckpt_args = checkpoint_args(ckpt)
    detections_per_img = int(arg_value(ckpt_args, "detections_per_img", 20) or 20)
    nms_thresh = float(arg_value(ckpt_args, "nms_thresh", 0.4) or 0.4)
    model = build_detection_model(
        num_classes=2,
        pretrained="none",
        detections_per_img=detections_per_img,
        nms_thresh=nms_thresh,
    )
    model.load_state_dict(state_dict_from_checkpoint(ckpt))
    model.to(device)
    model.eval()
    return model


def load_subtype_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, List[str]]:
    ckpt = torch_load(checkpoint_path, device)
    ckpt_args = checkpoint_args(ckpt)
    class_slugs = list(ckpt.get("class_slugs", CLASS_SLUGS)) if isinstance(ckpt, dict) else list(CLASS_SLUGS)
    model_name = str(arg_value(ckpt_args, "model", "resnet18") or "resnet18")
    model = build_subtype_model(model_name, pretrained="none", num_classes=len(class_slugs), allow_random_init=True)
    model.load_state_dict(state_dict_from_checkpoint(ckpt))
    model.to(device)
    model.eval()
    return model, class_slugs


def run_mesangial_inference(records: Sequence[CropRecord], args: argparse.Namespace, device: torch.device) -> None:
    _, transform = build_mesangial_transforms("off", args.image_size)
    loader = DataLoader(
        CropClassificationDataset(records, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_mesangial_model(args.m_checkpoint, args.m_init_model, device)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            probs = mesangial_model_probs(model, images, eval_tta=args.mesangial_tta).detach().cpu().view(-1)
            for index, prob in zip(indices.tolist(), probs.tolist()):
                records[index].mesangial_prob = float(prob)


def multilabel_probs(model: nn.Module, images: torch.Tensor, eval_tta: bool) -> torch.Tensor:
    probs = torch.sigmoid(model(images))
    if not eval_tta:
        return probs
    probs_h = torch.sigmoid(model(torch.flip(images, dims=[3])))
    probs_v = torch.sigmoid(model(torch.flip(images, dims=[2])))
    return (probs + probs_h + probs_v) / 3.0


def run_endocapillary_inference(records: Sequence[CropRecord], args: argparse.Namespace, device: torch.device) -> None:
    _, transform = build_proliferation_transforms("off", args.image_size)
    loader = DataLoader(
        CropClassificationDataset(records, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_endocapillary_model(args.e_checkpoint, args.e_init_model, device)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            probs = multilabel_probs(model, images, eval_tta=args.endocapillary_tta).detach().cpu()
            for index, prob in zip(indices.tolist(), probs.tolist()):
                records[index].endocapillary_prob = float(prob[1])


def clamp_box(box: Sequence[float], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_square_box(
    box: Sequence[float], width: int, height: int, margin: float
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [float(value) for value in box]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    side = max(box_w, box_h) * (1.0 + 2.0 * max(0.0, margin))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = int(math.floor(cx - side / 2.0))
    top = int(math.floor(cy - side / 2.0))
    right = int(math.ceil(cx + side / 2.0))
    bottom = int(math.ceil(cy + side / 2.0))
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def save_roi(
    record: CropRecord,
    box: Sequence[float],
    roi_id: str,
    output_dir: Path,
    roi_size: int,
    margin: float,
) -> Optional[Path]:
    with Image.open(record.image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        expanded = expand_square_box(box, width, height, margin)
        if expanded is None:
            return None
        roi = image.crop(expanded).resize((roi_size, roi_size), Image.BILINEAR)
    roi_dir = output_dir / "crescent_predicted_rois" / "images"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_path = roi_dir / f"{safe_name(roi_id)}.jpg"
    roi.save(roi_path, quality=92)
    return roi_path


def run_crescent_detection(records: Sequence[CropRecord], args: argparse.Namespace, device: torch.device) -> List[RoiRecord]:
    loader = DataLoader(
        CropDetectionDataset(records),
        batch_size=args.detection_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
    )
    model = load_detection_model(args.detection_checkpoint, device)
    rois: List[RoiRecord] = []
    with torch.no_grad():
        for images, indices, sizes in loader:
            images = [image.to(device=device, dtype=torch.float32, non_blocking=True) for image in images]
            outputs = model(images)
            for output, index, size in zip(outputs, indices, sizes):
                record = records[int(index)]
                width, height = int(size[0]), int(size[1])
                boxes = output["boxes"].detach().cpu()
                labels = output["labels"].detach().cpu()
                scores = output["scores"].detach().cpu()
                positive_scores = [
                    float(score) for label, score in zip(labels.tolist(), scores.tolist()) if int(label) == 1
                ]
                record.crescent_prob = max(positive_scores) if positive_scores else 0.0
                for box_index, (box, label, score) in enumerate(zip(boxes.tolist(), labels.tolist(), scores.tolist())):
                    if int(label) != 1 or float(score) < args.crescent_threshold:
                        continue
                    clipped = clamp_box(box, width, height)
                    if clipped is None:
                        continue
                    roi_id = f"{record.crop_id}_crescent_roi{box_index:02d}"
                    roi_path = save_roi(
                        record,
                        clipped,
                        roi_id,
                        args.output_dir,
                        roi_size=args.roi_size,
                        margin=args.roi_margin,
                    )
                    if roi_path is None:
                        continue
                    box_payload = {
                        "x1": float(clipped[0]),
                        "y1": float(clipped[1]),
                        "x2": float(clipped[2]),
                        "y2": float(clipped[3]),
                        "score": float(score),
                    }
                    record.crescent_boxes.append(box_payload)
                    rois.append(
                        RoiRecord(
                            roi_id=roi_id,
                            crop_index=record.index,
                            slide_id=record.slide_id,
                            glomerulus_id=record.glomerulus_id,
                            crop_id=record.crop_id,
                            image_path=roi_path,
                            x1=box_payload["x1"],
                            y1=box_payload["y1"],
                            x2=box_payload["x2"],
                            y2=box_payload["y2"],
                            detection_score=float(score),
                        )
                    )
    return rois


def run_subtype_inference(
    records: Sequence[CropRecord],
    rois: Sequence[RoiRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not rois:
        return
    _, transform = build_subtype_transforms("off")
    loader = DataLoader(
        RoiClassificationDataset(rois, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model, class_slugs = load_subtype_model(args.subtype_checkpoint, device)
    best_by_crop: Dict[int, Tuple[float, str]] = {}

    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            probs = F.softmax(model(images), dim=1).detach().cpu()
            for index, row_probs in zip(indices.tolist(), probs.tolist()):
                roi = rois[int(index)]
                roi.probs = {slug: float(value) for slug, value in zip(class_slugs, row_probs)}
                pred_idx = int(torch.tensor(row_probs).argmax().item())
                roi.pred_class = class_slugs[pred_idx]
                roi.pred_prob = float(row_probs[pred_idx])
                crop = records[roi.crop_index]
                for slug, value in roi.probs.items():
                    crop.subtype_probs[slug] = max(float(crop.subtype_probs.get(slug, 0.0)), float(value))
                priority = float(roi.detection_score) * float(roi.pred_prob)
                if roi.crop_index not in best_by_crop or priority > best_by_crop[roi.crop_index][0]:
                    best_by_crop[roi.crop_index] = (priority, roi.pred_class)

    for crop_index, (_, subtype) in best_by_crop.items():
        records[crop_index].crescent_subtype = subtype


def format_optional(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def parse_fraction(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100.0 if 1.0 < number <= 100.0 else number


def apply_tubular_atrophy_by_slide(records: Sequence[CropRecord], table_csv: Optional[Path]) -> int:
    if table_csv is None:
        return 0
    values: Dict[str, float] = {}
    with table_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            slide_id = row.get("slide_id") or row.get("wsi_id") or row.get("scene_id") or row.get("case_id")
            raw_value = (
                row.get("ifta_percent")
                or row.get("tubular_atrophy_target_percent")
                or row.get("tubular_atrophy_percent")
                or row.get("t_percent")
            )
            value = parse_fraction(raw_value)
            if slide_id and value is not None:
                values[str(slide_id)] = value
    for record in records:
        record.ifta_percent = values.get(record.slide_id)
    return sum(record.ifta_percent is not None for record in records)


def write_per_glomerulus_predictions(records: Sequence[CropRecord], path: Path, args: argparse.Namespace) -> None:
    fields = [
        "slide_id",
        "glomerulus_id",
        "crop_id",
        "split",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_path",
        "mesangial_prob",
        "mesangial_pred",
        "endocapillary_prob",
        "endocapillary_pred",
        "crescent_prob",
        "crescent_pred",
        "crescent_box_count",
        "crescent_subtype",
        "cellular_crescent_prob",
        "fibrocellular_crescent_prob",
        "fibrous_crescent_prob",
        "cellular_crescent_pred",
        "fibrocellular_crescent_pred",
        "fibrous_crescent_pred",
        "segmental_sclerosis_prob",
        "adhesion_prob",
        "ifta_percent",
        "crescent_boxes_json",
    ]
    rows = []
    for record in records:
        cell_prob = float(record.subtype_probs.get("cellular_crescent", 0.0))
        fibrocell_prob = float(record.subtype_probs.get("fibrocellular_crescent", 0.0))
        fibrous_prob = float(record.subtype_probs.get("fibrous_crescent", 0.0))
        rows.append(
            {
                "slide_id": record.slide_id,
                "glomerulus_id": record.glomerulus_id,
                "crop_id": record.crop_id,
                "split": record.split,
                "x1": format_optional(record.x1),
                "y1": format_optional(record.y1),
                "x2": format_optional(record.x2),
                "y2": format_optional(record.y2),
                "image_path": str(record.image_path),
                "mesangial_prob": f"{record.mesangial_prob:.6f}",
                "mesangial_pred": int(record.mesangial_prob >= args.mesangial_threshold),
                "endocapillary_prob": f"{record.endocapillary_prob:.6f}",
                "endocapillary_pred": int(record.endocapillary_prob >= args.endocapillary_threshold),
                "crescent_prob": f"{record.crescent_prob:.6f}",
                "crescent_pred": int(record.crescent_prob >= args.crescent_threshold),
                "crescent_box_count": len(record.crescent_boxes),
                "crescent_subtype": record.crescent_subtype,
                "cellular_crescent_prob": f"{cell_prob:.6f}",
                "fibrocellular_crescent_prob": f"{fibrocell_prob:.6f}",
                "fibrous_crescent_prob": f"{fibrous_prob:.6f}",
                "cellular_crescent_pred": int(cell_prob >= args.cellular_crescent_threshold),
                "fibrocellular_crescent_pred": int(fibrocell_prob >= args.fibrocellular_crescent_threshold),
                "fibrous_crescent_pred": int(fibrous_prob >= args.fibrous_crescent_threshold),
                "segmental_sclerosis_prob": "",
                "adhesion_prob": "",
                "ifta_percent": "" if record.ifta_percent is None else f"{record.ifta_percent:.6f}",
                "crescent_boxes_json": json.dumps(record.crescent_boxes, ensure_ascii=False),
            }
        )
    write_csv(path, rows, fields)


def write_roi_predictions(rois: Sequence[RoiRecord], path: Path) -> None:
    fields = [
        "roi_id",
        "slide_id",
        "glomerulus_id",
        "crop_id",
        "roi_image_path",
        "x1",
        "y1",
        "x2",
        "y2",
        "detection_score",
        "pred_class",
        "pred_prob",
        "prob_cellular_crescent",
        "prob_fibrocellular_crescent",
        "prob_fibrous_crescent",
    ]
    rows = []
    for roi in rois:
        rows.append(
            {
                "roi_id": roi.roi_id,
                "slide_id": roi.slide_id,
                "glomerulus_id": roi.glomerulus_id,
                "crop_id": roi.crop_id,
                "roi_image_path": str(roi.image_path),
                "x1": f"{roi.x1:.3f}",
                "y1": f"{roi.y1:.3f}",
                "x2": f"{roi.x2:.3f}",
                "y2": f"{roi.y2:.3f}",
                "detection_score": f"{roi.detection_score:.6f}",
                "pred_class": roi.pred_class,
                "pred_prob": f"{roi.pred_prob:.6f}",
                "prob_cellular_crescent": f"{float(roi.probs.get('cellular_crescent', 0.0)):.6f}",
                "prob_fibrocellular_crescent": f"{float(roi.probs.get('fibrocellular_crescent', 0.0)):.6f}",
                "prob_fibrous_crescent": f"{float(roi.probs.get('fibrous_crescent', 0.0)):.6f}",
            }
        )
    write_csv(path, rows, fields)


def score_predictions(prediction_csv: Path, output_dir: Path, args: argparse.Namespace):
    thresholds = Thresholds(
        mesangial=args.mesangial_threshold,
        endocapillary=args.endocapillary_threshold,
        crescent=args.crescent_threshold,
        cellular_crescent=args.cellular_crescent_threshold,
        fibrocellular_crescent=args.fibrocellular_crescent_threshold,
        fibrous_crescent=args.fibrous_crescent_threshold,
    )
    glomeruli = load_glomeruli(
        prediction_csv,
        thresholds,
        generic_crescent_active=args.generic_crescent_active,
    )
    scores = group_and_score(glomeruli, min_glomeruli=args.min_glomeruli, iskdc_crescent_mode="all")

    score_dir = output_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    write_iskdc_csv(scores, score_dir / "iskdc_scores.csv")
    write_oxford_csv(scores, score_dir / "oxford_scores.csv")
    write_iskdc_json(scores, score_dir / "iskdc_scores.json")
    write_oxford_json(scores, score_dir / "oxford_scores.json")
    write_glomerulus_csv(scores, score_dir / "glomerulus_evidence.csv")
    write_summary_csv(scores, score_dir / "slide_scores.csv")
    write_json(scores, score_dir / "slide_scores.json")
    return scores


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill=(35, 35, 35)) -> int:
    draw.text(xy, text, font=font, fill=fill)
    try:
        box = draw.textbbox(xy, text, font=font)
        return int(box[3] - box[1])
    except Exception:
        return 18


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0])
    except Exception:
        return len(text) * 8


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font,
    max_width: int,
    fill=(35, 35, 35),
    line_gap: int = 4,
) -> int:
    x, y = xy
    start_y = y
    for line in wrap_text(draw, text, font, max_width):
        height = draw_text(draw, (x, y), line, font, fill=fill)
        y += height + line_gap
    return y - start_y


def draw_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    value: float,
    width: int,
    color: Tuple[int, int, int],
    font,
) -> None:
    value = max(0.0, min(1.0, float(value)))
    draw.text((x, y), label, fill=(45, 45, 45), font=font)
    bx = x + 170
    by = y + 5
    draw.rectangle((bx, by, bx + width, by + 14), outline=(205, 205, 205), width=1)
    draw.rectangle((bx, by, bx + int(width * value), by + 14), fill=color)
    draw.text((bx + width + 10, y), f"{value * 100:.1f}%", fill=(45, 45, 45), font=font)


def label_for_record(record: CropRecord, args: argparse.Namespace) -> str:
    labels = []
    if record.mesangial_prob >= args.mesangial_threshold:
        labels.append("M")
    if record.endocapillary_prob >= args.endocapillary_threshold:
        labels.append("E")
    if record.crescent_prob >= args.crescent_threshold:
        if record.crescent_subtype:
            short = {
                "cellular_crescent": "C-cell",
                "fibrocellular_crescent": "C-fibcell",
                "fibrous_crescent": "C-fibrous",
            }.get(record.crescent_subtype, "C")
            labels.append(short)
        else:
            labels.append("C")
    return "+".join(labels) if labels else "negative/top"


def border_color(record: CropRecord, args: argparse.Namespace) -> Tuple[int, int, int]:
    if record.crescent_subtype in {"cellular_crescent", "fibrocellular_crescent"}:
        return (210, 35, 35)
    if record.crescent_subtype == "fibrous_crescent":
        return (235, 135, 25)
    if record.crescent_prob >= args.crescent_threshold:
        return (145, 55, 190)
    if record.mesangial_prob >= args.mesangial_threshold or record.endocapillary_prob >= args.endocapillary_threshold:
        return (30, 145, 75)
    return (130, 130, 130)


def record_priority(record: CropRecord, args: argparse.Namespace) -> Tuple[int, float, float, float]:
    positive = int(
        record.mesangial_prob >= args.mesangial_threshold
        or record.endocapillary_prob >= args.endocapillary_threshold
        or record.crescent_prob >= args.crescent_threshold
    )
    return positive, record.crescent_prob, record.mesangial_prob, record.endocapillary_prob


def select_evidence_records(records: Sequence[CropRecord], args: argparse.Namespace) -> List[CropRecord]:
    ordered = sorted(records, key=lambda item: record_priority(item, args), reverse=True)
    return ordered[: max(1, args.max_evidence_items)]


def score_lookup(scores) -> Dict[str, object]:
    return {score.slide_id: score for score in scores}


def render_real_evidence_for_slide(
    slide_id: str,
    records: Sequence[CropRecord],
    score,
    output_path: Path,
    mode: str,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    selected = select_evidence_records(records, args)
    cols = 4
    tile = 210
    label_h = 55
    gap = 16
    header_h = 125
    sidebar_w = 540
    rows = max(1, math.ceil(len(selected) / cols))
    canvas_w = 40 + cols * tile + (cols - 1) * gap + 30 + sidebar_w + 35
    canvas_h = max(820, header_h + rows * (tile + label_h + gap) + 30)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    header_font = load_font(20)
    body_font = load_font(15)
    small_font = load_font(12)

    if mode == "oxford":
        title = f"Oxford MEST-C real-crop evidence: {slide_id}"
        subtitle = f"{score.oxford['MEST_C']} | glomeruli={score.total_glomeruli}"
    else:
        title = f"ISKDC real-crop evidence: {slide_id}"
        subtitle = f"ISKDC {score.iskdc['grade']} | glomeruli={score.total_glomeruli}"
    draw.text((30, 22), title, fill=(20, 35, 55), font=title_font)
    draw.text((30, 64), subtitle, fill=(55, 55, 55), font=header_font)
    draw.text((30, 94), "Real glomerulus crops with predicted crescent boxes.", fill=(80, 80, 80), font=small_font)

    index_rows = []
    grid_x = 30
    grid_y = header_h
    for rank, record in enumerate(selected, start=1):
        row = (rank - 1) // cols
        col = (rank - 1) % cols
        x = grid_x + col * (tile + gap)
        y = grid_y + row * (tile + label_h + gap)
        draw.rectangle((x, y, x + tile, y + tile), fill=(248, 248, 248), outline=(210, 210, 210), width=1)
        try:
            with Image.open(record.image_path) as image:
                image = image.convert("RGB")
                original_w, original_h = image.size
                scale = min(tile / original_w, tile / original_h)
                disp_w = max(1, int(original_w * scale))
                disp_h = max(1, int(original_h * scale))
                resized = image.resize((disp_w, disp_h), Image.BILINEAR)
                ox = x + (tile - disp_w) // 2
                oy = y + (tile - disp_h) // 2
                canvas.paste(resized, (ox, oy))
                for box in record.crescent_boxes:
                    color = border_color(record, args)
                    bx1 = ox + int(float(box["x1"]) * scale)
                    by1 = oy + int(float(box["y1"]) * scale)
                    bx2 = ox + int(float(box["x2"]) * scale)
                    by2 = oy + int(float(box["y2"]) * scale)
                    draw.rectangle((bx1, by1, bx2, by2), outline=color, width=3)
                    draw.text((bx1 + 3, max(oy, by1 - 16)), f"{float(box['score']):.2f}", fill=color, font=small_font)
        except Exception as exc:
            draw.text((x + 8, y + 8), f"image error: {exc}", fill=(150, 45, 45), font=small_font)

        color = border_color(record, args)
        draw.rectangle((x, y, x + tile, y + tile), outline=color, width=4)
        label_y = y + tile + 6
        draw.text((x, label_y), f"{record.glomerulus_id} | {label_for_record(record, args)}", fill=(30, 30, 30), font=body_font)
        draw.text(
            (x, label_y + 22),
            f"M {record.mesangial_prob:.2f}  E {record.endocapillary_prob:.2f}  Cr {record.crescent_prob:.2f}",
            fill=(60, 60, 60),
            font=small_font,
        )
        index_rows.append(
            {
                "mode": mode,
                "slide_id": slide_id,
                "rank": rank,
                "crop_id": record.crop_id,
                "glomerulus_id": record.glomerulus_id,
                "image_path": str(record.image_path),
                "label": label_for_record(record, args),
                "mesangial_prob": f"{record.mesangial_prob:.6f}",
                "endocapillary_prob": f"{record.endocapillary_prob:.6f}",
                "crescent_prob": f"{record.crescent_prob:.6f}",
                "crescent_subtype": record.crescent_subtype,
            }
        )

    side_x = grid_x + cols * tile + (cols - 1) * gap + 35
    side_width = sidebar_w - 25
    y = header_h
    draw.text((side_x, y), "Slide-level result", fill=(20, 35, 55), font=header_font)
    y += 34
    if mode == "oxford":
        lines = [
            f"Oxford: {score.oxford['MEST_C']}",
            f"M={score.oxford['M']} E={score.oxford['E']} S={score.oxford['S']} T={score.oxford['T']} C={score.oxford['C']}",
            f"M: {score.counts['mesangial']}/{score.total_glomeruli}",
            f"E: {score.counts['endocapillary']}/{score.total_glomeruli}",
            f"Active crescents: {score.counts['active_crescent']}/{score.total_glomeruli}",
            f"Fibrous review: {score.counts['fibrous_crescent']}/{score.total_glomeruli}",
            "S/T not model-supported in this run.",
        ]
        bars = [
            ("Mesangial", score.percentages["mesangial"], (30, 145, 75)),
            ("Endocapillary", score.percentages["endocapillary"], (30, 145, 75)),
            ("Active Cr", score.percentages["active_crescent"], (210, 35, 35)),
        ]
    else:
        lines = [
            f"ISKDC: {score.iskdc['grade']}",
            score.iskdc["description"],
            f"M: {score.counts['mesangial']}/{score.total_glomeruli}",
            f"E: {score.counts['endocapillary']}/{score.total_glomeruli}",
            f"Any crescent: {score.counts['any_crescent']}/{score.total_glomeruli}",
            f"Cellular: {score.counts['cellular_crescent']}/{score.total_glomeruli}",
            f"Fibrocellular: {score.counts['fibrocellular_crescent']}/{score.total_glomeruli}",
            f"Fibrous: {score.counts['fibrous_crescent']}/{score.total_glomeruli}",
        ]
        bars = [
            ("Mesangial", score.percentages["mesangial"], (30, 145, 75)),
            ("Endocapillary", score.percentages["endocapillary"], (30, 145, 75)),
            ("Any Cr", score.percentages["any_crescent"], (235, 135, 25)),
        ]
    for line in lines:
        height = draw_wrapped_text(draw, (side_x, y), line, body_font, side_width)
        y += height + 8
    y += 18
    draw.text((side_x, y), "Evidence ratios", fill=(20, 35, 55), font=header_font)
    y += 36
    for label, value, color in bars:
        draw_bar(draw, side_x, y, label, value, 180, color, body_font)
        y += 34
    if score.warnings:
        y += 18
        draw.text((side_x, y), "Warnings", fill=(165, 75, 0), font=header_font)
        y += 32
        for warning in score.warnings[:5]:
            y += draw_wrapped_text(draw, (side_x, y), f"- {warning}", small_font, side_width, fill=(125, 70, 0))
            y += 6

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return index_rows


def render_all_real_evidence(records: Sequence[CropRecord], scores, output_dir: Path, args: argparse.Namespace) -> None:
    by_slide: Dict[str, List[CropRecord]] = {}
    for record in records:
        by_slide.setdefault(record.slide_id, []).append(record)
    scores_by_slide = score_lookup(scores)
    index_rows: List[Dict[str, object]] = []
    for slide_id in sorted(by_slide):
        score = scores_by_slide.get(slide_id)
        if score is None:
            continue
        index_rows.extend(
            render_real_evidence_for_slide(
                slide_id,
                by_slide[slide_id],
                score,
                output_dir / "real_oxford_evidence" / f"{safe_name(slide_id)}_oxford_real_crops.png",
                "oxford",
                args,
            )
        )
        index_rows.extend(
            render_real_evidence_for_slide(
                slide_id,
                by_slide[slide_id],
                score,
                output_dir / "real_iskdc_evidence" / f"{safe_name(slide_id)}_iskdc_real_crops.png",
                "iskdc",
                args,
            )
        )
    write_csv(
        output_dir / "real_evidence_index.csv",
        index_rows,
        [
            "mode",
            "slide_id",
            "rank",
            "crop_id",
            "glomerulus_id",
            "image_path",
            "label",
            "mesangial_prob",
            "endocapillary_prob",
            "crescent_prob",
            "crescent_subtype",
        ],
    )


def score_table_rows(scores) -> List[Dict[str, object]]:
    rows = []
    for score in scores:
        rows.append(
            {
                "slide_id": score.slide_id,
                "glomeruli": score.total_glomeruli,
                "ISKDC": score.iskdc["grade"],
                "Oxford": score.oxford["MEST_C"],
                "M_count": score.counts["mesangial"],
                "M_percent": round(score.percentages["mesangial"], 4),
                "E_count": score.counts["endocapillary"],
                "E_percent": round(score.percentages["endocapillary"], 4),
                "active_crescent_count": score.counts["active_crescent"],
                "active_crescent_percent": round(score.percentages["active_crescent"], 4),
                "any_crescent_count": score.counts["any_crescent"],
                "any_crescent_percent": round(score.percentages["any_crescent"], 4),
            }
        )
    return rows


def write_run_summary(output_dir: Path, records: Sequence[CropRecord], rois: Sequence[RoiRecord], scores, args: argparse.Namespace) -> None:
    summary = {
        "data_root": str(args.data_root),
        "split": args.split,
        "crop_count": len(records),
        "predicted_crescent_roi_count": len(rois),
        "thresholds": {
            "mesangial": args.mesangial_threshold,
            "endocapillary": args.endocapillary_threshold,
            "crescent": args.crescent_threshold,
            "cellular_crescent": args.cellular_crescent_threshold,
            "fibrocellular_crescent": args.fibrocellular_crescent_threshold,
            "fibrous_crescent": args.fibrous_crescent_threshold,
            "generic_crescent_active_for_oxford": args.generic_crescent_active,
        },
        "tubular_atrophy_slide_csv": str(args.tubular_atrophy_slide_csv) if args.tubular_atrophy_slide_csv else "",
        "artifacts": {
            "per_glomerulus_predictions": str(output_dir / "per_glomerulus_predictions.csv"),
            "roi_predictions": str(output_dir / "crescent_roi_predictions.csv"),
            "scores": str(output_dir / "scores"),
            "real_oxford_evidence": str(output_dir / "real_oxford_evidence"),
            "real_iskdc_evidence": str(output_dir / "real_iskdc_evidence"),
        },
        "scores": score_table_rows(scores),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    md_lines = [
        "# HSPN Real-Crop Full Report Run",
        "",
        f"- Data root: `{args.data_root}`",
        f"- Split: `{args.split}`",
        f"- Glomerulus crops scored: `{len(records)}`",
        f"- Predicted crescent ROIs: `{len(rois)}`",
        f"- Scores: `{output_dir / 'scores'}`",
        f"- Real Oxford evidence: `{output_dir / 'real_oxford_evidence'}`",
        f"- Real ISKDC evidence: `{output_dir / 'real_iskdc_evidence'}`",
        "",
        "| slide_id | glomeruli | ISKDC | Oxford | M | E | active crescent | any crescent |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for score in scores:
        md_lines.append(
            "| {slide} | {total} | {iskdc} | {oxford} | {m}/{total} | {e}/{total} | {ac}/{total} | {anyc}/{total} |".format(
                slide=score.slide_id,
                total=score.total_glomeruli,
                iskdc=score.iskdc["grade"],
                oxford=score.oxford["MEST_C"],
                m=score.counts["mesangial"],
                e=score.counts["endocapillary"],
                ac=score.counts["active_crescent"],
                anyc=score.counts["any_crescent"],
            )
        )
    (output_dir / "run_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Loading crops from: {manifest_path(args)}")
    records = load_crop_records(args)
    print(f"Loaded {len(records)} real glomerulus crop(s) from split={args.split}.")
    attached_ifta = apply_tubular_atrophy_by_slide(records, args.tubular_atrophy_slide_csv)
    if args.tubular_atrophy_slide_csv:
        print(f"Attached tubular-atrophy scene burden to {attached_ifta} crop record(s).")

    print("Running mesangial MESCNN inference...")
    run_mesangial_inference(records, args, device)
    print("Running endocapillary MESCNN inference...")
    run_endocapillary_inference(records, args, device)
    print("Running crescent Faster R-CNN detection...")
    rois = run_crescent_detection(records, args, device)
    print(f"Detected {len(rois)} crescent ROI(s) above threshold.")
    print("Running crescent subtype inference...")
    run_subtype_inference(records, rois, args, device)

    prediction_csv = args.output_dir / "per_glomerulus_predictions.csv"
    write_per_glomerulus_predictions(records, prediction_csv, args)
    write_roi_predictions(rois, args.output_dir / "crescent_roi_predictions.csv")
    scores = score_predictions(prediction_csv, args.output_dir, args)
    render_all_real_evidence(records, scores, args.output_dir, args)
    write_run_summary(args.output_dir, records, rois, scores, args)
    print(f"Wrote full real-crop report to: {args.output_dir}")
    print(json.dumps(score_table_rows(scores), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
