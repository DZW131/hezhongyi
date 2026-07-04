"""Train a ResNet18 binary classifier for proliferation detection on glomerulus crops.

The classifier learns an image-level binary label derived from the segmentation mask
produced by ``prepare_hzy_glomerulus_lesion_crops.py``: a crop is positive (class 1)
if its mask contains any non-zero pixel, otherwise negative (class 0).

This classifier is intentionally NOT a replacement for the Faster R-CNN detector.
Its purpose is to provide a Grad-CAM heatmap for hard-negative mining: we want to
see *where* the model looks when it falsely predicts "proliferation" on a negative
crop, so those crops can be shown to the doctor for confirmation.
"""

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
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import torchvision
    from torchvision import transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torchvision is required: {}".format(exc))


CLASS_NAMES = ["background", "proliferation"]
NUM_CLASSES = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 binary classifier for proliferation on glomerulus crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True,
                        help="Crop dataset root containing train/ and val/ with images/ and masks/")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and analysis")
    parser.add_argument("--model", choices=("resnet18", "resnet34"), default="resnet18")
    parser.add_argument("--pretrained", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--allow-random-init", action="store_true", default=False,
                        help="Continue with random init if ImageNet weights cannot be loaded")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--weighted-sampler", action="store_true", default=False)
    parser.add_argument("--class-weights", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--augmentation", choices=("off", "basic", "strong"), default="basic")
    parser.add_argument("--checkpoint-metric", choices=("f1", "balanced_accuracy", "accuracy"), default="f1",
                        help="Metric used to select the best checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--save-montage", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GlomerulusCropDataset(Dataset):
    """Binary classification dataset built from segmentation crops.

    Each sample's label is derived from its mask: any non-zero pixel => positive (1),
    otherwise negative (0). No external manifest is required.
    """

    def __init__(self, data_root: Path, split: str, transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.images_dir = self.data_root / split / "images"
        self.masks_dir = self.data_root / split / "masks"
        if not self.images_dir.exists():
            raise FileNotFoundError("Missing images dir: {}".format(self.images_dir))
        self.image_paths = sorted(
            p for p in self.images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.image_paths:
            raise RuntimeError("No images found in {}".format(self.images_dir))
        self.labels = [self._label_for(p) for p in self.image_paths]

    def _label_for(self, image_path: Path) -> int:
        mask_path = self.masks_dir / (image_path.stem + ".png")
        if not mask_path.exists():
            return 0
        mask = np.array(Image.open(mask_path))
        return int(mask.max() > 0)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        name = self.image_paths[index].name
        return image, label, name

    @property
    def targets(self) -> List[int]:
        return list(self.labels)


def build_transforms(augmentation: str):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_ops: List[object] = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    if augmentation in {"basic", "strong"}:
        train_ops.extend([
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        ])
    if augmentation == "strong":
        train_ops.extend([
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])
    train_ops.extend([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    val_ops = [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def build_model(model_name: str, pretrained: str, num_classes: int, allow_random_init: bool) -> nn.Module:
    weights = None
    if pretrained == "imagenet":
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT if model_name == "resnet18" \
                else torchvision.models.ResNet34_Weights.DEFAULT
        except Exception as exc:
            if not allow_random_init:
                raise
            logging.warning("Could not resolve ImageNet weights (%s). Random init.", exc)
    try:
        if model_name == "resnet18":
            model = torchvision.models.resnet18(weights=weights)
        else:
            model = torchvision.models.resnet34(weights=weights)
    except Exception as exc:
        if not allow_random_init or pretrained == "none":
            raise
        logging.warning("ImageNet load failed (%s). Random init.", exc)
        model = torchvision.models.resnet18(weights=None) if model_name == "resnet18" \
            else torchvision.models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = not frozen


def class_counts(targets: Iterable[int], num_classes: int) -> List[int]:
    counts = Counter(int(t) for t in targets)
    return [counts.get(i, 0) for i in range(num_classes)]


def class_weight_tensor(targets: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = class_counts(targets, num_classes)
    total = sum(counts)
    weights = [total / max(c, 1) for c in counts]
    mean_w = sum(weights) / max(len(weights), 1)
    weights = [w / mean_w for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_train_loader(dataset: GlomerulusCropDataset, batch_size: int, num_workers: int, weighted_sampler: bool):
    if weighted_sampler:
        counts = class_counts(dataset.targets, NUM_CLASSES)
        sample_weights = [1.0 / max(counts[t], 1) for t in dataset.targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def binary_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, object]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    names: List[str] = []
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for images, labels, image_names in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            with torch.autocast(device.type if device.type != "mps" else "cpu",
                                enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1)
            bs = int(labels.shape[0])
            total_loss += float(loss.item()) * bs
            total_seen += bs
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            y_prob.extend(probs[:, 1].cpu().tolist())
            names.extend(image_names)
    metrics = binary_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / max(total_seen, 1)
    predictions = [
        {"name": n, "true": int(t), "pred": int(p), "prob_positive": float(pr)}
        for n, t, p, pr in zip(names, y_true, y_pred, y_prob)
    ]
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
    with path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def denormalize_image(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = (tensor.cpu() * std + mean).clamp(0, 1)
    arr = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_montage(model: nn.Module, dataset: GlomerulusCropDataset, device: torch.device,
                 output_path: Path, max_items: int = 32) -> None:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    tiles = []
    model.eval()
    with torch.no_grad():
        for idx, (img_tensor, label, name) in enumerate(loader):
            if idx >= max_items:
                break
            img = img_tensor.to(device=device, dtype=torch.float32)
            probs = torch.softmax(model(img), dim=1)[0].cpu().tolist()
            pred = int(np.argmax(probs))
            pil = denormalize_image(img_tensor[0]).resize((160, 160))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(pil)
            truth = "pos" if int(label.item()) == 1 else "neg"
            pr = "pos" if pred == 1 else "neg"
            color = (0, 128, 0) if truth == pr else (200, 0, 0)
            draw.rectangle((0, 0, 159, 30), fill=(255, 255, 255))
            draw.text((4, 3), "T:{} P:{:.2f}".format(truth, probs[1]), fill=color)
            tiles.append(pil)
    if not tiles:
        return
    cols = 4
    rows = int(math.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * 160, rows * 160), color=(255, 255, 255))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % cols) * 160, (i // cols) * 160))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_seed(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_tf, val_tf = build_transforms(args.augmentation)
    train_ds = GlomerulusCropDataset(data_root, "train", transform=train_tf)
    val_ds = GlomerulusCropDataset(data_root, "val", transform=val_tf)

    train_loader = build_train_loader(train_ds, args.batch_size, args.num_workers, args.weighted_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, args.pretrained, NUM_CLASSES, args.allow_random_init).to(device)
    crit_weight = None
    if args.class_weights == "balanced":
        crit_weight = class_weight_tensor(train_ds.targets, NUM_CLASSES, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    logging.info(
        "Training proliferation classifier: train=%s val=%s train_counts=%s val_counts=%s device=%s",
        len(train_ds), len(val_ds),
        class_counts(train_ds.targets, NUM_CLASSES),
        class_counts(val_ds.targets, NUM_CLASSES),
        device,
    )

    history: List[Dict[str, object]] = []
    best_score = -1.0
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

        for images, labels, _names in train_loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type if device.type != "mps" else "cpu",
                                enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, weight=crit_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = int(labels.shape[0])
            train_loss += float(loss.item()) * bs
            train_seen += bs

        val_metrics, predictions = evaluate_model(model, val_loader, device, amp=args.amp)
        epoch_seconds = time.perf_counter() - start
        score = float(val_metrics[args.checkpoint_metric])
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_seen, 1),
            "val_loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "specificity": val_metrics["specificity"],
            "f1": val_metrics["f1"],
            "balanced_accuracy": (val_metrics["recall"] + val_metrics["specificity"]) / 2.0,
            "epoch_seconds": epoch_seconds,
            "backbone_frozen": int(frozen),
        }
        history.append(row)
        save_history(history, output_dir / "analysis" / "history.csv")

        is_best = score > best_score
        ckpt = {
            "model_state": model.state_dict(),
            "model_name": args.model,
            "class_names": CLASS_NAMES,
            "args": vars(args),
            "epoch": epoch,
            "metrics": val_metrics,
        }
        torch.save(ckpt, output_dir / "latest.pth")
        if is_best:
            best_score = score
            best_epoch = epoch
            best_predictions = predictions
            best_metrics = val_metrics
            torch.save(ckpt, output_dir / "best.pth")
            write_predictions(predictions, output_dir / "analysis" / "best_predictions.csv")
            with (output_dir / "analysis" / "best_metrics.json").open("w", encoding="utf-8") as h:
                json.dump({"best_epoch": best_epoch, "metrics": val_metrics}, h, ensure_ascii=False, indent=2)
            if args.save_montage:
                save_montage(model, val_ds, device, output_dir / "analysis" / "best_montage.jpg")

        logging.info(
            "Epoch %s/%s train_loss=%.4f val_loss=%.4f acc=%.4f prec=%.4f rec=%.4f spec=%.4f f1=%.4f%s",
            epoch, args.epochs, row["train_loss"], row["val_loss"],
            row["accuracy"], row["precision"], row["recall"], row["specificity"], row["f1"],
            " [best]" if is_best else "",
        )

    if best_metrics is not None:
        print(json.dumps({"best_epoch": best_epoch, "metrics": best_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()