import argparse
import csv
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.detection_boxes import Box, binary_confusion_metrics, match_boxes


DEFAULT_DATA_ROOT = "/home/duyanhong/Dataspace/HZY/HZY_HSPN_detection_boxes/proliferation_binary"
DEFAULT_OUTPUT_DIR = "/home/duyanhong/Dataspace/HZY/checkpoints/hzy_hspn_detection_boxes/proliferation_binary_fasterrcnn"


class YoloBoxDataset(Dataset):
    def __init__(self, data_root: Path, split: str, limit: int = 0, augmentation: str = "off", seed: int = 42):
        self.data_root = Path(data_root)
        self.split = split
        self.augmentation = augmentation
        self.seed = seed
        self.images_dir = self.data_root / split / "images"
        self.labels_dir = self.data_root / split / "labels"
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError("Missing {} images/labels under {}".format(split, data_root))
        self.image_paths = sorted(path for path in self.images_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if limit > 0:
            self.image_paths = self.image_paths[:limit]
        if not self.image_paths:
            raise RuntimeError("No images found in {}".format(self.images_dir))
        self.positive_indices, self.negative_indices = self._split_indices_by_label()

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _label_has_boxes(label_path: Path) -> bool:
        if not label_path.exists():
            return False
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                _, _, _, box_width, box_height = [float(item) for item in parts]
            except ValueError:
                continue
            if box_width > 0 and box_height > 0:
                return True
        return False

    def _split_indices_by_label(self) -> Tuple[List[int], List[int]]:
        positive_indices = []
        negative_indices = []
        for index, image_path in enumerate(self.image_paths):
            label_path = self.labels_dir / (image_path.stem + ".txt")
            if self._label_has_boxes(label_path):
                positive_indices.append(index)
            else:
                negative_indices.append(index)
        return positive_indices, negative_indices

    def _read_boxes(self, label_path: Path, width: int, height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes = []
        labels = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, box_width, box_height = [float(item) for item in parts]
                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height
                x_min = max(0.0, x_center - box_width / 2.0)
                y_min = max(0.0, y_center - box_height / 2.0)
                x_max = min(float(width), x_center + box_width / 2.0)
                y_max = min(float(height), y_center + box_height / 2.0)
                if x_max <= x_min or y_max <= y_min:
                    continue
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id) + 1)
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    @staticmethod
    def _flip_horizontal(boxes: torch.Tensor, width: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        return flipped

    @staticmethod
    def _flip_vertical(boxes: torch.Tensor, height: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        flipped = boxes.clone()
        flipped[:, 1] = height - boxes[:, 3]
        flipped[:, 3] = height - boxes[:, 1]
        return flipped

    @staticmethod
    def _jitter_image(image: Image.Image, rng: random.Random, strength: float) -> Image.Image:
        contrast = rng.uniform(1.0 - strength, 1.0 + strength)
        brightness = rng.uniform(1.0 - strength, 1.0 + strength)
        color = rng.uniform(1.0 - strength * 0.7, 1.0 + strength * 0.7)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Color(image).enhance(color)
        return image

    def _augment(self, image: Image.Image, boxes: torch.Tensor, index: int):
        if self.split != "train" or self.augmentation in ("off", "none", "false", "0"):
            return image, boxes

        rng = random.Random(self.seed + index * 1009)
        width, height = image.size
        mode = self.augmentation.lower()
        color_strength = 0.10 if mode == "basic" else 0.18

        if rng.random() < 0.5:
            image = F.hflip(image)
            boxes = self._flip_horizontal(boxes, width)
        if rng.random() < 0.5:
            image = F.vflip(image)
            boxes = self._flip_vertical(boxes, height)
        if rng.random() < (0.5 if mode == "basic" else 0.75):
            image = self._jitter_image(image, rng, color_strength)
        return image, boxes

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels = self._read_boxes(self.labels_dir / (image_path.stem + ".txt"), width, height)
        image, boxes = self._augment(image, boxes, index)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.numel() else torch.zeros((0,), dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "case_id": image_path.stem,
        }
        return F.to_tensor(image), target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class PositiveBalancedBatchSampler(Sampler[List[int]]):
    """Build batches with at least one positive image when positives exist."""

    def __init__(self, positive_indices: List[int], negative_indices: List[int], batch_size: int, seed: int = 42):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.positive_indices = list(positive_indices)
        self.negative_indices = list(negative_indices)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        if not self.positive_indices:
            self.num_batches = math.ceil(len(self.negative_indices) / max(batch_size, 1))
        elif batch_size == 1:
            self.num_batches = max(len(self.positive_indices), 1)
        else:
            self.num_batches = max(
                math.ceil(len(self.negative_indices) / max(batch_size - 1, 1)),
                math.ceil(len(self.positive_indices) / max(1, 1)),
                1,
            )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        positives = list(self.positive_indices)
        negatives = list(self.negative_indices)
        rng.shuffle(positives)
        rng.shuffle(negatives)

        if not positives:
            all_indices = negatives
            for start in range(0, len(all_indices), self.batch_size):
                yield all_indices[start:start + self.batch_size]
            return

        negative_cursor = 0
        for batch_index in range(self.num_batches):
            batch = [positives[batch_index % len(positives)]]
            while len(batch) < self.batch_size and negative_cursor < len(negatives):
                batch.append(negatives[negative_cursor])
                negative_cursor += 1
            while len(batch) < self.batch_size:
                pool = positives if not negatives else positives + negatives
                batch.append(rng.choice(pool))
            rng.shuffle(batch)
            yield batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a torchvision Faster R-CNN detector on HZY lesion bbox data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Detection dataset root produced by prepare_hzy_detection_boxes.py")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Checkpoint and metrics output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--pretrained", choices=("none", "coco"), default="none",
                        help="Detector initialization. coco replaces the classification head after loading COCO weights")
    parser.add_argument("--augmentation", choices=("off", "basic", "strong"), default="off",
                        help="Training-only detection augmentation")
    parser.add_argument("--augmentation-seed", type=int, default=42, help="Seed for deterministic per-sample augmentation")
    parser.add_argument("--ensure-positive-batches", action="store_true", default=False,
                        help="Oversample positives so every train batch contains at least one positive image")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                        help="Freeze detector backbone for the first N epochs")
    parser.add_argument("--detections-per-img", type=int, default=100,
                        help="Maximum detections kept per image during evaluation")
    parser.add_argument("--nms-thresh", type=float, default=0.5, help="ROI NMS threshold")
    parser.add_argument("--lr-step-size", type=int, default=0, help="Optional StepLR step size; 0 disables scheduler")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="StepLR gamma")
    parser.add_argument("--score-thresholds", default="0.2,0.3,0.4,0.5,0.6", help="Comma-separated detection score thresholds")
    parser.add_argument("--primary-threshold", type=float, default=0.3, help="Threshold used to rank best checkpoints")
    parser.add_argument("--checkpoint-metric", choices=("box_f1", "box_recall", "presence_f1"), default="box_f1")
    parser.add_argument(
        "--save-best-metrics",
        default="box_f1,presence_f1",
        help="Comma-separated primary-threshold metrics to save as best_<metric>.pth in addition to best.pth",
    )
    parser.add_argument("--fail-on-nonfinite-loss", action="store_true", default=False,
                        help="Raise an error instead of skipping a batch when detection loss is NaN/Inf")
    parser.add_argument("--match-iou", type=float, default=0.1, help="Relaxed IoU threshold for matching")
    parser.add_argument("--no-center-hit", action="store_true", default=False, help="Disable center-inside-GT matching")
    parser.add_argument("--limit-train", type=int, default=0, help="Optional train sample cap for smoke tests")
    parser.add_argument("--limit-val", type=int, default=0, help="Optional val sample cap for smoke tests")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    return parser.parse_args()


def parse_thresholds(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_metric_names(value: str) -> List[str]:
    names = []
    for item in value.split(","):
        metric = item.strip()
        if metric and metric not in names:
            names.append(metric)
    return names


def make_model(num_classes: int, pretrained: str = "none", detections_per_img: int = 100, nms_thresh: float = 0.5):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained == "coco" else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None, num_classes=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = detections_per_img
    model.roi_heads.nms_thresh = nms_thresh
    return model


def set_backbone_trainable(model, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def target_to_boxes(target: Dict[str, torch.Tensor]) -> List[Box]:
    boxes = []
    for box, label in zip(target["boxes"].detach().cpu().numpy(), target["labels"].detach().cpu().numpy()):
        boxes.append(Box(int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3])), class_id=int(label)))
    return boxes


def prediction_to_boxes(output: Dict[str, torch.Tensor], score_threshold: float) -> List[Box]:
    boxes = []
    raw_boxes = output["boxes"].detach().cpu().numpy()
    raw_labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    for box, label, score in zip(raw_boxes, raw_labels, scores):
        if score < score_threshold or int(label) != 1:
            continue
        boxes.append(Box(int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3])), class_id=int(label)))
    return boxes


def empty_raw_metrics():
    return {
        "presence_tp": 0,
        "presence_fp": 0,
        "presence_fn": 0,
        "presence_tn": 0,
        "gt_boxes": 0,
        "pred_boxes": 0,
        "matched_boxes": 0,
        "unmatched_gt_boxes": 0,
        "unmatched_pred_boxes": 0,
    }


def update_raw_metrics(raw: Dict[str, int], gt_boxes: List[Box], pred_boxes: List[Box], matches):
    gt_positive = bool(gt_boxes)
    pred_positive = bool(pred_boxes)
    if gt_positive and pred_positive:
        raw["presence_tp"] += 1
    elif (not gt_positive) and pred_positive:
        raw["presence_fp"] += 1
    elif gt_positive and (not pred_positive):
        raw["presence_fn"] += 1
    else:
        raw["presence_tn"] += 1

    raw["gt_boxes"] += len(gt_boxes)
    raw["pred_boxes"] += len(pred_boxes)
    raw["matched_boxes"] += len(matches)
    raw["unmatched_gt_boxes"] += max(0, len(gt_boxes) - len(matches))
    raw["unmatched_pred_boxes"] += max(0, len(pred_boxes) - len(matches))


def finalize_raw(raw: Dict[str, int]) -> Dict[str, float]:
    presence = binary_confusion_metrics(raw["presence_tp"], raw["presence_fp"], raw["presence_fn"], raw["presence_tn"])
    box_precision = raw["matched_boxes"] / raw["pred_boxes"] if raw["pred_boxes"] else 0.0
    box_recall = raw["matched_boxes"] / raw["gt_boxes"] if raw["gt_boxes"] else 0.0
    box_f1 = (2 * box_precision * box_recall) / (box_precision + box_recall) if (box_precision + box_recall) else 0.0
    return {
        **raw,
        "presence_precision": presence["precision"],
        "presence_recall": presence["recall"],
        "presence_specificity": presence["specificity"],
        "presence_f1": presence["f1"],
        "presence_accuracy": presence["accuracy"],
        "box_precision": float(box_precision),
        "box_recall": float(box_recall),
        "box_f1": float(box_f1),
    }


@torch.no_grad()
def evaluate_detector(model, dataloader, device, thresholds: List[float], match_iou: float, allow_center_hit: bool):
    model.eval()
    raw_by_threshold = {threshold: empty_raw_metrics() for threshold in thresholds}
    for images, targets in tqdm(dataloader, desc="Validate detector", unit="batch"):
        images = [image.to(device) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            gt_boxes = target_to_boxes(target)
            for threshold in thresholds:
                pred_boxes = prediction_to_boxes(output, threshold)
                matches = match_boxes(gt_boxes, pred_boxes, iou_threshold=match_iou, allow_center_hit=allow_center_hit)
                update_raw_metrics(raw_by_threshold[threshold], gt_boxes, pred_boxes, matches)
    return {str(threshold): {**finalize_raw(raw), "threshold": threshold} for threshold, raw in raw_by_threshold.items()}


def write_history(rows: List[Dict[str, float]], path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    thresholds = parse_thresholds(args.score_thresholds)
    train_dataset = YoloBoxDataset(
        Path(args.data_root),
        "train",
        limit=args.limit_train,
        augmentation=args.augmentation,
        seed=args.augmentation_seed,
    )
    val_dataset = YoloBoxDataset(Path(args.data_root), "val", limit=args.limit_val)
    if args.ensure_positive_batches and train_dataset.positive_indices:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=PositiveBalancedBatchSampler(
                train_dataset.positive_indices,
                train_dataset.negative_indices,
                batch_size=args.batch_size,
                seed=args.augmentation_seed,
            ),
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = make_model(
        num_classes=2,
        pretrained=args.pretrained,
        detections_per_img=args.detections_per_img,
        nms_thresh=args.nms_thresh,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        if args.lr_step_size > 0
        else None
    )
    best_score = -1.0
    best_scores_by_metric = {}
    best_metric_names = parse_metric_names(args.save_best_metrics)
    if args.checkpoint_metric not in best_metric_names:
        best_metric_names.insert(0, args.checkpoint_metric)
    history = []

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone_epochs > 0:
            set_backbone_trainable(model, epoch > args.freeze_backbone_epochs)
        model.train()
        epoch_loss = 0.0
        batches = 0
        nonfinite_batches = 0
        for images, targets in tqdm(train_loader, desc="Train detector epoch {}".format(epoch), unit="batch"):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
                for target in targets
            ]
            loss_dict = model(images, targets)
            loss = sum(value for value in loss_dict.values())
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                loss_parts = {
                    key: float(value.detach().cpu()) if torch.isfinite(value.detach()).item() else str(value.detach().cpu().item())
                    for key, value in loss_dict.items()
                }
                message = "Non-finite detection loss at epoch {} batch {}; parts={}".format(
                    epoch,
                    batches + nonfinite_batches,
                    loss_parts,
                )
                if args.fail_on_nonfinite_loss:
                    raise FloatingPointError(message)
                logging.warning("%s; skipping optimizer step", message)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1

        metrics = evaluate_detector(
            model,
            val_loader,
            device=device,
            thresholds=thresholds,
            match_iou=args.match_iou,
            allow_center_hit=not args.no_center_hit,
        )
        primary = metrics[str(args.primary_threshold)]
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "nonfinite_batches": nonfinite_batches,
            **{key: primary[key] for key in (
                "presence_precision",
                "presence_recall",
                "presence_specificity",
                "presence_f1",
                "box_precision",
                "box_recall",
                "box_f1",
                "gt_boxes",
                "pred_boxes",
                "matched_boxes",
            )},
        }
        history.append(row)
        write_history(history, output_dir / "history.csv")
        with (output_dir / "metrics_latest.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        score = float(primary[args.checkpoint_metric])
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "latest.pth")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output_dir / "best.pth")
        for metric_name in best_metric_names:
            metric_score = float(primary.get(metric_name, -1.0))
            if metric_score > best_scores_by_metric.get(metric_name, -1.0):
                best_scores_by_metric[metric_name] = metric_score
                torch.save(checkpoint, output_dir / "best_{}.pth".format(metric_name))

        if scheduler is not None:
            scheduler.step()

        logging.info(
            "epoch=%s loss=%.4f %s=%.4f box_f1=%.4f box_recall=%.4f presence_f1=%.4f",
            epoch,
            row["train_loss"],
            args.checkpoint_metric,
            score,
            primary["box_f1"],
            primary["box_recall"],
            primary["presence_f1"],
        )
        if nonfinite_batches:
            logging.warning("epoch=%s skipped %s non-finite-loss batches", epoch, nonfinite_batches)

    logging.info("Training complete. Best %s=%.4f at %s", args.checkpoint_metric, best_score, output_dir / "best.pth")


if __name__ == "__main__":
    main()
