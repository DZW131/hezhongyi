"""Fine-tune a MESCNN EfficientNetV2-M classifier for HSPN proliferation M/E labels.

This script trains a multi-label glomerulus-level classifier from the fine-grained
HZY proliferation crop manifest. Class 1 mask pixels are mapped to mesangial
hypercellularity (Oxford M), and class 2 mask pixels are mapped to endocapillary
hypercellularity (Oxford E). A crop can be positive for both labels.
"""

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torchvision is required: {}".format(exc))


TARGETS = ("mesangial", "endocapillary")
TARGET_LABELS = {
    "mesangial": "M / mesangial hypercellularity",
    "endocapillary": "E / endocapillary hypercellularity",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MESCNN on HSPN M/E proliferation labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, help="Task root containing manifests/crops.csv and train/val images")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and analysis outputs")
    parser.add_argument("--init-model", required=True, help="MESCNN .pth model used to initialize the EfficientNetV2-M backbone")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--class-weights", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--loss", choices=("bce", "focal_bce"), default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--weighted-sampler", action="store_true", default=False)
    parser.add_argument("--augmentation", choices=("off", "basic", "strong"), default="basic")
    parser.add_argument("--min-positive-pixels", type=int, default=8)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--checkpoint-metric", choices=("macro_f1", "mean_auc", "exact_match"), default="macro_f1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--limit-train", type=int, default=0, help="Debug limit; 0 means all train samples")
    parser.add_argument("--limit-val", type=int, default=0, help="Debug limit; 0 means all val samples")
    parser.add_argument("--save-montage", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_class_pixels(raw: str) -> Dict[str, int]:
    if raw is None or raw == "":
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {str(k): int(v) for k, v in data.items()}


class HspnProliferationMultiLabelDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        transform=None,
        min_positive_pixels: int = 8,
        limit: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.min_positive_pixels = int(min_positive_pixels)
        manifest = self.data_root / "manifests" / "crops.csv"
        if not manifest.exists():
            raise FileNotFoundError("Missing manifest: {}".format(manifest))

        self.rows: List[Dict[str, object]] = []
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("split") != split:
                    continue
                image_rel = row.get("image_path") or ""
                if not image_rel:
                    continue
                image_path = self.data_root / image_rel
                if image_path.suffix.lower() not in IMAGE_EXTS or not image_path.exists():
                    continue
                class_pixels = parse_class_pixels(row.get("class_pixels", ""))
                target = np.array(
                    [
                        float(class_pixels.get("1", 0) >= self.min_positive_pixels),
                        float(class_pixels.get("2", 0) >= self.min_positive_pixels),
                    ],
                    dtype=np.float32,
                )
                record = dict(row)
                record["image_abs_path"] = str(image_path)
                record["target"] = target
                record["mesangial_pixels"] = int(class_pixels.get("1", 0))
                record["endocapillary_pixels"] = int(class_pixels.get("2", 0))
                self.rows.append(record)
                if limit and len(self.rows) >= limit:
                    break
        if not self.rows:
            raise RuntimeError("No {} rows found under {}".format(split, self.data_root))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = Path(str(row["image_abs_path"]))
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(row["target"], dtype=torch.float32)
        return image, target, str(row.get("crop_id") or image_path.stem)

    @property
    def targets(self) -> np.ndarray:
        return np.stack([row["target"] for row in self.rows]).astype(np.float32)

    def label_summary(self) -> Dict[str, object]:
        y = self.targets
        both = np.logical_and(y[:, 0] == 1, y[:, 1] == 1)
        either = np.logical_or(y[:, 0] == 1, y[:, 1] == 1)
        return {
            "split": self.split,
            "samples": int(len(self.rows)),
            "mesangial_positive": int(y[:, 0].sum()),
            "endocapillary_positive": int(y[:, 1].sum()),
            "both_positive": int(both.sum()),
            "any_positive": int(either.sum()),
            "negative": int((~either).sum()),
        }


def build_transforms(augmentation: str, image_size: int):
    resize = (int(image_size), int(image_size))
    train_ops: List[object] = [transforms.Resize(resize)]
    if augmentation != "off":
        train_ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20)])
        train_ops.append(transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02))
    if augmentation == "strong":
        train_ops.extend([
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])
    train_ops.append(transforms.ToTensor())
    val_ops = [transforms.Resize(resize), transforms.ToTensor()]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        pt = torch.exp(-bce)
        focal = (1.0 - pt).pow(self.gamma) * bce
        return focal.mean()


def replace_classifier(model: nn.Module, num_outputs: int) -> None:
    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if isinstance(classifier, nn.Sequential):
            linear_idx = None
            for idx in reversed(range(len(classifier))):
                if isinstance(classifier[idx], nn.Linear):
                    linear_idx = idx
                    break
            if linear_idx is None:
                raise RuntimeError("Could not find Linear layer in model.classifier")
            in_features = classifier[linear_idx].in_features
            classifier[linear_idx] = nn.Linear(in_features, num_outputs)
            return
        if isinstance(classifier, nn.Linear):
            setattr(model, "classifier", nn.Linear(classifier.in_features, num_outputs))
            return
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        fc = getattr(model, "fc")
        setattr(model, "fc", nn.Linear(fc.in_features, num_outputs))
        return
    raise RuntimeError("Unsupported model head; expected classifier or fc")


def load_mescnn_model(path: Path, device: torch.device, num_outputs: int = 2) -> nn.Module:
    model = torch.load(str(path), map_location=device, weights_only=False)
    if not isinstance(model, nn.Module):
        raise RuntimeError("Expected a torch.nn.Module checkpoint, got {}".format(type(model)))
    replace_classifier(model, num_outputs)
    return model.to(device)


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for name, parameter in model.named_parameters():
        is_head = name.startswith("classifier") or name.startswith("fc")
        parameter.requires_grad = bool(is_head or not frozen)


def positive_weight(targets: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = targets.sum(axis=0)
    neg = targets.shape[0] - pos
    weights = neg / np.maximum(pos, 1.0)
    weights = np.clip(weights, 0.25, 8.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def sample_weights(targets: np.ndarray) -> List[float]:
    pos_counts = targets.sum(axis=0)
    neg_counts = targets.shape[0] - pos_counts
    label_weights = neg_counts / np.maximum(pos_counts, 1.0)
    weights = []
    for row in targets:
        w = 1.0
        if row.sum() > 0:
            w += float((row * label_weights).sum())
        weights.append(w)
    return weights


def metric_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def counts_to_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
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
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_grid(step: float) -> List[float]:
    step = max(min(float(step), 0.5), 0.01)
    values = np.arange(step, 1.0, step)
    return [round(float(v), 4) for v in values]


def class_metrics_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Sequence[float],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    all_metrics: Dict[str, Dict[str, float]] = {}
    best_thresholds: Dict[str, float] = {}
    for class_idx, target_name in enumerate(TARGETS):
        yt = y_true[:, class_idx]
        yp = y_prob[:, class_idx]
        best: Optional[Dict[str, float]] = None
        best_thr = 0.5
        for thr in thresholds:
            pred = (yp >= thr).astype(np.int32)
            metrics = counts_to_metrics(metric_counts(yt, pred))
            metrics["threshold"] = float(thr)
            if best is None or metrics["f1"] > best["f1"] or (
                metrics["f1"] == best["f1"] and metrics["recall"] > best["recall"]
            ):
                best = metrics
                best_thr = float(thr)
        assert best is not None
        auc = binary_auc(yt, yp)
        if auc is not None:
            best["auc"] = float(auc)
        all_metrics[target_name] = best
        best_thresholds[target_name] = best_thr
    return all_metrics, best_thresholds


def summarize_multilabel(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, object]:
    preds = np.zeros_like(y_true, dtype=np.int32)
    for i, target_name in enumerate(TARGETS):
        preds[:, i] = (y_prob[:, i] >= thresholds[target_name]).astype(np.int32)
    per_class = {}
    for i, target_name in enumerate(TARGETS):
        metrics = counts_to_metrics(metric_counts(y_true[:, i], preds[:, i]))
        auc = binary_auc(y_true[:, i], y_prob[:, i])
        if auc is not None:
            metrics["auc"] = float(auc)
        metrics["threshold"] = float(thresholds[target_name])
        per_class[target_name] = metrics
    macro_f1 = float(np.mean([per_class[t]["f1"] for t in TARGETS]))
    exact_match = float((preds == y_true).all(axis=1).mean())
    mean_auc_values = [per_class[t].get("auc") for t in TARGETS if per_class[t].get("auc") is not None]
    mean_auc = float(np.mean(mean_auc_values)) if mean_auc_values else 0.0
    return {"per_class": per_class, "macro_f1": macro_f1, "exact_match": exact_match, "mean_auc": mean_auc}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp: bool,
    threshold_values: Sequence[float],
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, float]]:
    model.eval()
    y_true_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []
    names: List[str] = []
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for images, targets, crop_ids in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
                probs = torch.sigmoid(logits)
            batch_size = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size
            y_true_list.append(targets.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())
            names.extend(list(crop_ids))
    y_true = np.concatenate(y_true_list, axis=0).astype(np.int32)
    y_prob = np.concatenate(y_prob_list, axis=0).astype(np.float32)
    best_class_metrics, thresholds = class_metrics_at_thresholds(y_true, y_prob, threshold_values)
    metrics = summarize_multilabel(y_true, y_prob, thresholds)
    metrics["loss"] = total_loss / max(total_seen, 1)
    metrics["best_thresholds"] = thresholds
    metrics["best_per_class_by_f1"] = best_class_metrics

    predictions: List[Dict[str, object]] = []
    for idx, crop_id in enumerate(names):
        row = {
            "crop_id": crop_id,
            "mesangial_true": int(y_true[idx, 0]),
            "endocapillary_true": int(y_true[idx, 1]),
            "mesangial_prob": float(y_prob[idx, 0]),
            "endocapillary_prob": float(y_prob[idx, 1]),
            "mesangial_pred": int(y_prob[idx, 0] >= thresholds["mesangial"]),
            "endocapillary_pred": int(y_prob[idx, 1] >= thresholds["endocapillary"]),
        }
        predictions.append(row)
    return metrics, predictions, thresholds


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_train_loader(dataset: HspnProliferationMultiLabelDataset, args: argparse.Namespace) -> DataLoader:
    if args.weighted_sampler:
        sampler = WeightedRandomSampler(sample_weights(dataset.targets), num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


def denormalized_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_prediction_montage(
    dataset: HspnProliferationMultiLabelDataset,
    predictions: List[Dict[str, object]],
    output_path: Path,
    max_items: int = 32,
) -> None:
    by_id = {str(row["crop_id"]): row for row in predictions}
    selected = []
    for record in dataset.rows:
        crop_id = str(record.get("crop_id"))
        pred = by_id.get(crop_id)
        if pred is None:
            continue
        wrong = (
            int(pred["mesangial_true"]) != int(pred["mesangial_pred"])
            or int(pred["endocapillary_true"]) != int(pred["endocapillary_pred"])
        )
        score = abs(float(pred["mesangial_prob"]) - 0.5) + abs(float(pred["endocapillary_prob"]) - 0.5)
        selected.append((not wrong, score, record, pred))
    selected.sort(key=lambda item: (item[0], item[1]))
    selected = selected[:max_items]
    if not selected:
        return
    tile = 180
    label_h = 44
    cols = 4
    rows = int(math.ceil(len(selected) / cols))
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (_ok_first, _score, record, pred) in enumerate(selected):
        x = (idx % cols) * tile
        y = (idx // cols) * (tile + label_h)
        image = Image.open(str(record["image_abs_path"])).convert("RGB").resize((tile, tile))
        canvas.paste(image, (x, y))
        m_true = int(pred["mesangial_true"])
        e_true = int(pred["endocapillary_true"])
        m_pred = int(pred["mesangial_pred"])
        e_pred = int(pred["endocapillary_pred"])
        color = (20, 130, 70) if (m_true == m_pred and e_true == e_pred) else (190, 45, 45)
        draw.rectangle([x, y + tile, x + tile, y + tile + label_h], fill=(248, 248, 248))
        draw.text((x + 4, y + tile + 4), "T M{} E{} | P M{} E{}".format(m_true, e_true, m_pred, e_pred), fill=color)
        draw.text((x + 4, y + tile + 22), "pM={:.2f} pE={:.2f}".format(float(pred["mesangial_prob"]), float(pred["endocapillary_prob"])), fill=(40, 40, 40))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    set_seed(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), output_dir / "analysis" / "args.json")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_tf, val_tf = build_transforms(args.augmentation, args.image_size)
    train_ds = HspnProliferationMultiLabelDataset(data_root, "train", train_tf, args.min_positive_pixels, args.limit_train)
    val_ds = HspnProliferationMultiLabelDataset(data_root, "val", val_tf, args.min_positive_pixels, args.limit_val)
    save_json({"train": train_ds.label_summary(), "val": val_ds.label_summary()}, output_dir / "analysis" / "label_summary.json")

    train_loader = make_train_loader(train_ds, args)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = load_mescnn_model(Path(args.init_model), device=device, num_outputs=len(TARGETS))
    pos_weight = positive_weight(train_ds.targets, device) if args.class_weights == "balanced" else None
    if args.loss == "focal_bce":
        criterion = FocalBCEWithLogitsLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    threshold_values = threshold_grid(args.threshold_step)

    logging.info("Device: %s", device)
    logging.info("Data root: %s", data_root)
    logging.info("Init model: %s", args.init_model)
    logging.info("Image size: %s loss=%s", args.image_size, args.loss)
    logging.info("Train summary: %s", train_ds.label_summary())
    logging.info("Val summary: %s", val_ds.label_summary())
    if pos_weight is not None:
        logging.info("BCE pos_weight: %s", [round(float(v), 4) for v in pos_weight.detach().cpu().tolist()])

    history: List[Dict[str, object]] = []
    best_score = -1.0
    best_epoch = 0
    best_metrics: Optional[Dict[str, object]] = None
    best_predictions: List[Dict[str, object]] = []
    best_thresholds: Dict[str, float] = {"mesangial": 0.5, "endocapillary": 0.5}

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_backbone_epochs
        set_backbone_frozen(model, frozen)
        model.train()
        start = time.perf_counter()
        train_loss = 0.0
        train_seen = 0

        for images, targets, _crop_ids in train_loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(targets.shape[0])
            train_loss += float(loss.item()) * batch_size
            train_seen += batch_size

        val_metrics, predictions, thresholds = evaluate_model(model, val_loader, device, criterion, args.amp, threshold_values)
        epoch_seconds = time.perf_counter() - start
        score = float(val_metrics[args.checkpoint_metric])
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_seen, 1),
            "val_loss": val_metrics["loss"],
            "macro_f1": val_metrics["macro_f1"],
            "exact_match": val_metrics["exact_match"],
            "mean_auc": val_metrics["mean_auc"],
            "mesangial_f1": val_metrics["per_class"]["mesangial"]["f1"],
            "mesangial_precision": val_metrics["per_class"]["mesangial"]["precision"],
            "mesangial_recall": val_metrics["per_class"]["mesangial"]["recall"],
            "mesangial_specificity": val_metrics["per_class"]["mesangial"]["specificity"],
            "mesangial_auc": val_metrics["per_class"]["mesangial"].get("auc", ""),
            "endocapillary_f1": val_metrics["per_class"]["endocapillary"]["f1"],
            "endocapillary_precision": val_metrics["per_class"]["endocapillary"]["precision"],
            "endocapillary_recall": val_metrics["per_class"]["endocapillary"]["recall"],
            "endocapillary_specificity": val_metrics["per_class"]["endocapillary"]["specificity"],
            "endocapillary_auc": val_metrics["per_class"]["endocapillary"].get("auc", ""),
            "mesangial_threshold": thresholds["mesangial"],
            "endocapillary_threshold": thresholds["endocapillary"],
            "backbone_frozen": int(frozen),
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        save_csv(history, output_dir / "analysis" / "history.csv")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_class": type(model).__name__,
            "target_names": TARGETS,
            "thresholds": thresholds,
            "metrics": val_metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "latest.pth")
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_predictions = predictions
            best_thresholds = thresholds
            torch.save(checkpoint, output_dir / "best.pth")
            save_json({"best_epoch": best_epoch, "metrics": best_metrics}, output_dir / "analysis" / "best_metrics.json")
            save_csv(best_predictions, output_dir / "analysis" / "best_predictions.csv")
            if args.save_montage:
                save_prediction_montage(val_ds, best_predictions, output_dir / "analysis" / "best_val_montage.jpg")

        logging.info(
            "Epoch %d/%d loss=%.4f val=%.4f macro_f1=%.4f exact=%.4f M(f1/rec/spec)=%.3f/%.3f/%.3f E(f1/rec/spec)=%.3f/%.3f/%.3f thr=%.2f/%.2f%s",
            epoch,
            args.epochs,
            row["train_loss"],
            row["val_loss"],
            row["macro_f1"],
            row["exact_match"],
            row["mesangial_f1"],
            row["mesangial_recall"],
            row["mesangial_specificity"],
            row["endocapillary_f1"],
            row["endocapillary_recall"],
            row["endocapillary_specificity"],
            row["mesangial_threshold"],
            row["endocapillary_threshold"],
            " [best]" if is_best else "",
        )

    final = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_thresholds": best_thresholds,
        "best_metrics": best_metrics,
        "label_summary": {"train": train_ds.label_summary(), "val": val_ds.label_summary()},
        "artifacts": {
            "best_checkpoint": str(output_dir / "best.pth"),
            "latest_checkpoint": str(output_dir / "latest.pth"),
            "history_csv": str(output_dir / "analysis" / "history.csv"),
            "best_predictions_csv": str(output_dir / "analysis" / "best_predictions.csv"),
            "best_metrics_json": str(output_dir / "analysis" / "best_metrics.json"),
        },
    }
    save_json(final, output_dir / "analysis" / "run_summary.json")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
