"""Fine-tune a MESCNN EfficientNetV2-M binary classifier for HSPN Oxford M.

This is a mesangial-hypercellularity-only companion experiment to
``train_hzy_mescnn_proliferation_multilabel.py``. The goal is to test whether
M performance is limited by multi-task interference from E, and to try
M-specific losses such as asymmetric loss and TTA evaluation.
"""

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MESCNN as an M/mesangial-only classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-model", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=4)
    parser.add_argument("--loss", choices=("bce", "focal_bce", "asymmetric"), default="asymmetric")
    parser.add_argument("--class-weights", choices=("balanced", "none"), default="none")
    parser.add_argument("--weighted-sampler", action="store_true", default=False)
    parser.add_argument("--augmentation", choices=("off", "basic", "strong"), default="strong")
    parser.add_argument("--min-positive-pixels", type=int, default=8)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--eval-tta", action="store_true", default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--save-montage", action="store_true", default=False)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_class_pixels(raw: str) -> Dict[str, int]:
    if not raw:
        return {}
    try:
        return {str(k): int(v) for k, v in json.loads(raw).items()}
    except json.JSONDecodeError:
        return {}


class MesangialDataset(Dataset):
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
        self.rows: List[Dict[str, object]] = []
        manifest = self.data_root / "manifests" / "crops.csv"
        if not manifest.exists():
            raise FileNotFoundError("Missing manifest: {}".format(manifest))
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("split") != split:
                    continue
                image_rel = row.get("image_path") or ""
                image_path = self.data_root / image_rel
                if image_path.suffix.lower() not in IMAGE_EXTS or not image_path.exists():
                    continue
                pixels = parse_class_pixels(row.get("class_pixels", ""))
                target = float(int(pixels.get("1", 0)) >= min_positive_pixels)
                stored = dict(row)
                stored["image_abs_path"] = str(image_path)
                stored["target"] = target
                stored["mesangial_pixels"] = int(pixels.get("1", 0))
                stored["endocapillary_pixels"] = int(pixels.get("2", 0))
                self.rows.append(stored)
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
        target = torch.tensor([float(row["target"])], dtype=torch.float32)
        crop_id = str(row.get("crop_id") or image_path.stem)
        return image, target, crop_id

    @property
    def targets(self) -> np.ndarray:
        return np.asarray([float(row["target"]) for row in self.rows], dtype=np.float32)

    def label_summary(self) -> Dict[str, int]:
        y = self.targets
        return {
            "samples": int(len(y)),
            "mesangial_positive": int(y.sum()),
            "mesangial_negative": int((y == 0).sum()),
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


def replace_classifier(model: nn.Module, num_outputs: int) -> None:
    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if isinstance(classifier, nn.Sequential):
            for idx in reversed(range(len(classifier))):
                if isinstance(classifier[idx], nn.Linear):
                    classifier[idx] = nn.Linear(classifier[idx].in_features, num_outputs)
                    return
        if isinstance(classifier, nn.Linear):
            setattr(model, "classifier", nn.Linear(classifier.in_features, num_outputs))
            return
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        fc = getattr(model, "fc")
        setattr(model, "fc", nn.Linear(fc.in_features, num_outputs))
        return
    raise RuntimeError("Unsupported model classifier head")


def load_model(path: Path, device: torch.device) -> nn.Module:
    model = torch.load(str(path), map_location=device, weights_only=False)
    if not isinstance(model, nn.Module):
        raise RuntimeError("Expected torch.nn.Module checkpoint, got {}".format(type(model)))
    replace_classifier(model, 1)
    return model.to(device)


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for name, parameter in model.named_parameters():
        is_head = name.startswith("classifier") or name.startswith("fc")
        parameter.requires_grad = bool(is_head or not frozen)


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        pt = torch.exp(-bce)
        return ((1.0 - pt).pow(self.gamma) * bce).mean()


class AsymmetricLossWithLogits(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05) -> None:
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        pos_prob = prob
        neg_prob = 1.0 - prob
        if self.clip > 0:
            neg_prob = (neg_prob + self.clip).clamp(max=1.0)
        eps = 1e-8
        pos_loss = targets * torch.log(pos_prob.clamp(min=eps))
        neg_loss = (1.0 - targets) * torch.log(neg_prob.clamp(min=eps))
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = pos_prob * targets + neg_prob * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            weight = (1.0 - pt).pow(gamma)
            pos_loss = pos_loss * weight
            neg_loss = neg_loss * weight
        return -(pos_loss + neg_loss).mean()


def positive_weight(targets: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = float(targets.sum())
    neg = float(len(targets) - pos)
    weight = min(max(neg / max(pos, 1.0), 0.25), 8.0)
    return torch.tensor([weight], dtype=torch.float32, device=device)


def build_loader(dataset: MesangialDataset, args: argparse.Namespace) -> DataLoader:
    if args.weighted_sampler:
        targets = dataset.targets
        pos = float(targets.sum())
        neg = float(len(targets) - pos)
        weights = [1.0 / max(pos if t == 1 else neg, 1.0) for t in targets]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


def metric_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def counts_to_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        **counts,
    }


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_score = y_score[order]
    i = 0
    while i < len(sorted_score):
        j = i + 1
        while j < len(sorted_score) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray, step: float) -> Dict[str, float]:
    thresholds = [round(float(x), 4) for x in np.arange(max(min(step, 0.5), 0.01), 1.0, max(min(step, 0.5), 0.01))]
    best = None
    for threshold in thresholds:
        metrics = counts_to_metrics(metric_counts(y_true, (y_prob >= threshold).astype(np.int32)))
        metrics["threshold"] = float(threshold)
        if best is None or metrics["f1"] > best["f1"] or (
            metrics["f1"] == best["f1"] and metrics["recall"] > best["recall"]
        ):
            best = metrics
    assert best is not None
    auc = binary_auc(y_true, y_prob)
    if auc is not None:
        best["auc"] = float(auc)
    return best


def model_probs(model: nn.Module, images: torch.Tensor, eval_tta: bool) -> torch.Tensor:
    logits = model(images)
    probs = torch.sigmoid(logits)
    if not eval_tta:
        return probs
    probs_h = torch.sigmoid(model(torch.flip(images, dims=[3])))
    probs_v = torch.sigmoid(model(torch.flip(images, dims=[2])))
    return (probs + probs_h + probs_v) / 3.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp: bool,
    eval_tta: bool,
    threshold_step: float,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    model.eval()
    y_true_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []
    crop_ids: List[str] = []
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for images, targets, names in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
                probs = model_probs(model, images, eval_tta)
            batch_size = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size
            y_true_list.append(targets.cpu().numpy().reshape(-1))
            y_prob_list.append(probs.cpu().numpy().reshape(-1))
            crop_ids.extend(list(names))
    y_true = np.concatenate(y_true_list).astype(np.int32)
    y_prob = np.concatenate(y_prob_list).astype(np.float32)
    metrics = best_threshold(y_true, y_prob, threshold_step)
    metrics["loss"] = total_loss / max(total_seen, 1)
    predictions = [
        {
            "crop_id": crop_id,
            "mesangial_true": int(y_true[i]),
            "mesangial_prob": float(y_prob[i]),
            "mesangial_pred": int(y_prob[i] >= metrics["threshold"]),
        }
        for i, crop_id in enumerate(crop_ids)
    ]
    return metrics, predictions


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
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_montage(dataset: MesangialDataset, predictions: List[Dict[str, object]], output_path: Path) -> None:
    by_id = {row["crop_id"]: row for row in predictions}
    selected = []
    for row in dataset.rows:
        crop_id = str(row["crop_id"])
        pred = by_id.get(crop_id)
        if not pred:
            continue
        wrong = int(pred["mesangial_true"]) != int(pred["mesangial_pred"])
        score = abs(float(pred["mesangial_prob"]) - 0.5)
        selected.append((not wrong, score, row, pred))
    selected.sort(key=lambda item: (item[0], item[1]))
    selected = selected[:32]
    if not selected:
        return
    tile = 180
    label_h = 38
    cols = 4
    rows = int(math.ceil(len(selected) / cols))
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (_ok, _score, record, pred) in enumerate(selected):
        x = (idx % cols) * tile
        y = (idx // cols) * (tile + label_h)
        image = Image.open(str(record["image_abs_path"])).convert("RGB").resize((tile, tile))
        canvas.paste(image, (x, y))
        truth = int(pred["mesangial_true"])
        guess = int(pred["mesangial_pred"])
        color = (20, 130, 70) if truth == guess else (190, 45, 45)
        draw.rectangle([x, y + tile, x + tile, y + tile + label_h], fill=(248, 248, 248))
        draw.text((x + 4, y + tile + 4), "T M{} | P M{} pM={:.2f}".format(truth, guess, float(pred["mesangial_prob"])), fill=color)
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
    train_ds = MesangialDataset(data_root, "train", train_tf, args.min_positive_pixels, args.limit_train)
    val_ds = MesangialDataset(data_root, "val", val_tf, args.min_positive_pixels, args.limit_val)
    save_json({"train": train_ds.label_summary(), "val": val_ds.label_summary()}, output_dir / "analysis" / "label_summary.json")

    train_loader = build_loader(train_ds, args)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = load_model(Path(args.init_model), device)
    pos_weight = positive_weight(train_ds.targets, device) if args.class_weights == "balanced" else None
    if args.loss == "focal_bce":
        criterion: nn.Module = FocalBCEWithLogitsLoss(args.focal_gamma, pos_weight=pos_weight)
    elif args.loss == "asymmetric":
        criterion = AsymmetricLossWithLogits(args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.freeze_backbone_epochs, 1)) if args.scheduler == "cosine" else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    logging.info("Device=%s image_size=%s loss=%s eval_tta=%s", device, args.image_size, args.loss, args.eval_tta)
    logging.info("Train summary: %s", train_ds.label_summary())
    logging.info("Val summary: %s", val_ds.label_summary())
    if pos_weight is not None:
        logging.info("pos_weight=%s", [round(float(v), 4) for v in pos_weight.detach().cpu().tolist()])

    history: List[Dict[str, object]] = []
    best_score = -1.0
    best_epoch = 0
    best_metrics: Optional[Dict[str, object]] = None
    best_predictions: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_backbone_epochs
        set_backbone_frozen(model, frozen)
        model.train()
        start = time.perf_counter()
        train_loss = 0.0
        train_seen = 0
        for images, targets, _names in train_loader:
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
        if scheduler is not None and epoch > args.freeze_backbone_epochs:
            scheduler.step()

        metrics, predictions = evaluate(model, val_loader, device, criterion, args.amp, args.eval_tta, args.threshold_step)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_seen, 1),
            "val_loss": metrics["loss"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "accuracy": metrics["accuracy"],
            "auc": metrics.get("auc", ""),
            "threshold": metrics["threshold"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "lr": optimizer.param_groups[0]["lr"],
            "backbone_frozen": int(frozen),
            "epoch_seconds": time.perf_counter() - start,
        }
        history.append(row)
        save_csv(history, output_dir / "analysis" / "history.csv")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model_class": type(model).__name__,
            "target_names": ("mesangial",),
            "thresholds": {"mesangial": metrics["threshold"]},
            "metrics": metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "latest.pth")
        is_best = float(metrics["f1"]) > best_score
        if is_best:
            best_score = float(metrics["f1"])
            best_epoch = epoch
            best_metrics = metrics
            best_predictions = predictions
            torch.save(checkpoint, output_dir / "best.pth")
            save_json({"best_epoch": best_epoch, "metrics": best_metrics}, output_dir / "analysis" / "best_metrics.json")
            save_csv(best_predictions, output_dir / "analysis" / "best_predictions.csv")
            if args.save_montage:
                save_montage(val_ds, best_predictions, output_dir / "analysis" / "best_val_montage.jpg")
        logging.info(
            "Epoch %d/%d loss=%.4f val=%.4f f1=%.4f pr=%.3f rec=%.3f spec=%.3f auc=%s thr=%.2f%s",
            epoch,
            args.epochs,
            row["train_loss"],
            row["val_loss"],
            row["f1"],
            row["precision"],
            row["recall"],
            row["specificity"],
            "{:.3f}".format(row["auc"]) if row["auc"] != "" else "NA",
            row["threshold"],
            " [best]" if is_best else "",
        )

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "label_summary": {"train": train_ds.label_summary(), "val": val_ds.label_summary()},
        "artifacts": {
            "best_checkpoint": str(output_dir / "best.pth"),
            "history_csv": str(output_dir / "analysis" / "history.csv"),
            "best_predictions_csv": str(output_dir / "analysis" / "best_predictions.csv"),
            "best_metrics_json": str(output_dir / "analysis" / "best_metrics.json"),
        },
    }
    save_json(summary, output_dir / "analysis" / "run_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
