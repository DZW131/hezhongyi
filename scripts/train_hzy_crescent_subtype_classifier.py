import argparse
import csv
import json
import logging
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import torchvision
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    raise SystemExit("torchvision is required for crescent subtype classification: {}".format(exc))


CLASS_SLUGS = ["cellular_crescent", "fibrocellular_crescent", "fibrous_crescent"]
CLASS_LABELS = {
    "cellular_crescent": "细胞性新月体",
    "fibrocellular_crescent": "纤维细胞性新月体",
    "fibrous_crescent": "纤维性新月体",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an ROI classifier for crescent subtypes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, help="Classification dataset root produced by prepare_hzy_crescent_subtype_rois.py")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and analysis outputs")
    parser.add_argument("--model", choices=("resnet18", "resnet34"), default="resnet18", help="torchvision classifier backbone")
    parser.add_argument("--pretrained", choices=("imagenet", "none"), default="imagenet", help="Backbone initialization")
    parser.add_argument("--allow-random-init", action="store_true", default=False,
                        help="If ImageNet weights cannot be loaded, continue with random initialization")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3, help="Freeze feature extractor for the first N epochs")
    parser.add_argument("--weighted-sampler", action="store_true", default=False, help="Use inverse-frequency weighted sampling")
    parser.add_argument("--class-weights", choices=("balanced", "none"), default="balanced", help="Cross-entropy class weighting")
    parser.add_argument("--augmentation", choices=("off", "basic", "strong"), default="basic", help="Training augmentation strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu")
    parser.add_argument("--amp", action="store_true", default=False, help="Use CUDA AMP")
    parser.add_argument("--save-montage", action="store_true", default=False, help="Save a validation prediction montage for the best model")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class RoiClassificationDataset(Dataset):
    def __init__(self, data_root: Path, split: str, transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        rows = read_manifest(data_root / "manifest.csv")
        self.rows = [row for row in rows if row["split"] == split and row["class_slug"] in CLASS_SLUGS]
        if not self.rows:
            raise RuntimeError("No rows found for split '{}' in {}".format(split, data_root / "manifest.csv"))
        self.class_to_idx = {class_slug: index for index, class_slug in enumerate(CLASS_SLUGS)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.data_root / row["relative_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row["class_slug"]]
        return image, label, row["roi_id"]

    @property
    def targets(self) -> List[int]:
        return [self.class_to_idx[row["class_slug"]] for row in self.rows]


def build_transforms(augmentation: str):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_ops: List[object] = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    if augmentation in {"basic", "strong"}:
        train_ops.extend(
            [
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            ]
        )
    if augmentation == "strong":
        train_ops.extend(
            [
                transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.9, 1.1)),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ]
        )
    train_ops.extend([transforms.ToTensor(), normalize])
    val_ops = [transforms.ToTensor(), normalize]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def load_weights(model_name: str, pretrained: str, allow_random_init: bool):
    if pretrained == "none":
        return None
    try:
        if model_name == "resnet18":
            return torchvision.models.ResNet18_Weights.DEFAULT
        return torchvision.models.ResNet34_Weights.DEFAULT
    except Exception as exc:
        if allow_random_init:
            logging.warning("Could not resolve ImageNet weights (%s). Falling back to random init.", exc)
            return None
        raise


def build_model(model_name: str, pretrained: str, num_classes: int, allow_random_init: bool) -> nn.Module:
    weights = load_weights(model_name, pretrained, allow_random_init)
    try:
        if model_name == "resnet18":
            model = torchvision.models.resnet18(weights=weights)
        else:
            model = torchvision.models.resnet34(weights=weights)
    except Exception as exc:
        if not allow_random_init or pretrained == "none":
            raise
        logging.warning("Could not load ImageNet weights (%s). Falling back to random init.", exc)
        model = torchvision.models.resnet18(weights=None) if model_name == "resnet18" else torchvision.models.resnet34(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = not frozen


def class_counts(targets: Iterable[int], num_classes: int) -> List[int]:
    counts = Counter(int(target) for target in targets)
    return [counts.get(index, 0) for index in range(num_classes)]


def class_weight_tensor(targets: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = class_counts(targets, num_classes)
    total = sum(counts)
    weights = [total / max(count, 1) for count in counts]
    mean_weight = sum(weights) / max(len(weights), 1)
    weights = [weight / mean_weight for weight in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_train_loader(dataset: RoiClassificationDataset, batch_size: int, num_workers: int, weighted_sampler: bool):
    if weighted_sampler:
        counts = class_counts(dataset.targets, len(CLASS_SLUGS))
        sample_weights = [1.0 / max(counts[target], 1) for target in dataset.targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> Dict[str, object]:
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    for index, slug in enumerate(CLASS_SLUGS):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - matrix[index, index])
        fn = float(matrix[index, :].sum() - matrix[index, index])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = int(matrix[index, :].sum())
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class[slug] = {
            "label": CLASS_LABELS[slug],
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    predictions: List[Dict[str, object]] = []
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for images, labels, roi_ids in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                probabilities = torch.softmax(logits, dim=1)
                pred = torch.argmax(probabilities, dim=1)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            for roi_id, truth, predicted, probs in zip(roi_ids, labels.cpu().tolist(), pred.cpu().tolist(), probabilities.cpu().tolist()):
                predictions.append(
                    {
                        "roi_id": roi_id,
                        "true_index": int(truth),
                        "true_class": CLASS_SLUGS[int(truth)],
                        "pred_index": int(predicted),
                        "pred_class": CLASS_SLUGS[int(predicted)],
                        "correct": int(truth == predicted),
                        **{"prob_{}".format(CLASS_SLUGS[idx]): float(value) for idx, value in enumerate(probs)},
                    }
                )
    matrix = confusion_matrix(y_true, y_pred, len(CLASS_SLUGS))
    metrics = metrics_from_confusion(matrix)
    metrics["loss"] = total_loss / max(total_seen, 1)
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics, predictions


def save_history(rows: List[Dict[str, object]], path: Path) -> None:
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


def write_predictions(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(matrix: List[List[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_SLUGS])
        for slug, row in zip(CLASS_SLUGS, matrix):
            writer.writerow([slug, *row])


def denormalize_image(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = (tensor.cpu() * std + mean).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def save_montage(model: nn.Module, dataset: RoiClassificationDataset, device: torch.device, output_path: Path, max_items: int = 24) -> None:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    tiles = []
    model.eval()
    with torch.no_grad():
        for index, (image_tensor, label, roi_id) in enumerate(loader):
            if index >= max_items:
                break
            image = image_tensor.to(device=device, dtype=torch.float32)
            probs = torch.softmax(model(image), dim=1)[0].cpu().tolist()
            pred_index = int(np.argmax(probs))
            pil_image = denormalize_image(image_tensor[0]).resize((160, 160))
            draw = ImageDraw.Draw(pil_image)
            truth = CLASS_SLUGS[int(label.item())]
            pred = CLASS_SLUGS[pred_index]
            color = (0, 128, 0) if truth == pred else (200, 0, 0)
            text = "T:{}\nP:{:.2f} {}".format(truth.replace("_crescent", ""), max(probs), pred.replace("_crescent", ""))
            draw.rectangle((0, 0, 159, 38), fill=(255, 255, 255))
            draw.text((4, 3), text, fill=color)
            tiles.append(pil_image)

    if not tiles:
        return
    columns = 4
    rows = int(math.ceil(len(tiles) / columns))
    canvas = Image.new("RGB", (columns * 160, rows * 160), color=(255, 255, 255))
    for idx, tile in enumerate(tiles):
        canvas.paste(tile, ((idx % columns) * 160, (idx // columns) * 160))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_seed(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (data_root / "manifest.csv").exists():
        raise FileNotFoundError("Missing manifest.csv under {}".format(data_root))

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_transform, val_transform = build_transforms(args.augmentation)
    train_dataset = RoiClassificationDataset(data_root, "train", transform=train_transform)
    val_dataset = RoiClassificationDataset(data_root, "val", transform=val_transform)

    train_loader = build_train_loader(train_dataset, args.batch_size, args.num_workers, args.weighted_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, args.pretrained, len(CLASS_SLUGS), args.allow_random_init).to(device)
    criterion_weight = None
    if args.class_weights == "balanced":
        criterion_weight = class_weight_tensor(train_dataset.targets, len(CLASS_SLUGS), device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    logging.info(
        "Training crescent subtype classifier: train=%s val=%s train_counts=%s val_counts=%s device=%s",
        len(train_dataset),
        len(val_dataset),
        class_counts(train_dataset.targets, len(CLASS_SLUGS)),
        class_counts(val_dataset.targets, len(CLASS_SLUGS)),
        device,
    )

    history: List[Dict[str, object]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_predictions: List[Dict[str, object]] = []
    best_metrics: Optional[Dict[str, object]] = None

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_backbone_epochs
        set_backbone_frozen(model, frozen=frozen)
        model.train()
        start = time.perf_counter()
        train_loss = 0.0
        train_seen = 0

        for images, labels, _roi_ids in train_loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, weight=criterion_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(labels.shape[0])
            train_loss += float(loss.item()) * batch_size
            train_seen += batch_size

        val_metrics, predictions = evaluate_model(model, val_loader, device, amp=args.amp)
        epoch_seconds = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_seen, 1),
            "val_loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "balanced_accuracy": val_metrics["balanced_accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "macro_precision": val_metrics["macro_precision"],
            "macro_recall": val_metrics["macro_recall"],
            "epoch_seconds": epoch_seconds,
            "backbone_frozen": int(frozen),
        }
        history.append(row)
        save_history(history, output_dir / "analysis" / "history.csv")

        is_best = float(val_metrics["macro_f1"]) > best_macro_f1
        torch.save(
            {
                "model_state": model.state_dict(),
                "class_slugs": CLASS_SLUGS,
                "class_labels": CLASS_LABELS,
                "args": vars(args),
                "epoch": epoch,
                "metrics": val_metrics,
            },
            output_dir / "latest.pth",
        )
        if is_best:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_predictions = predictions
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_slugs": CLASS_SLUGS,
                    "class_labels": CLASS_LABELS,
                    "args": vars(args),
                    "epoch": epoch,
                    "metrics": val_metrics,
                },
                output_dir / "best.pth",
            )
            write_predictions(predictions, output_dir / "analysis" / "best_predictions.csv")
            write_confusion_matrix(val_metrics["confusion_matrix"], output_dir / "analysis" / "best_confusion_matrix.csv")
            with (output_dir / "analysis" / "best_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump({"best_epoch": best_epoch, "metrics": val_metrics}, handle, ensure_ascii=False, indent=2)
            if args.save_montage:
                save_montage(model, val_dataset, device, output_dir / "analysis" / "best_montage.jpg")

        logging.info(
            "Epoch %s/%s train_loss=%.4f val_loss=%.4f acc=%.4f bal_acc=%.4f macro_f1=%.4f%s",
            epoch,
            args.epochs,
            row["train_loss"],
            row["val_loss"],
            row["accuracy"],
            row["balanced_accuracy"],
            row["macro_f1"],
            " [best]" if is_best else "",
        )

    if best_metrics is not None:
        print(json.dumps({"best_epoch": best_epoch, "metrics": best_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
