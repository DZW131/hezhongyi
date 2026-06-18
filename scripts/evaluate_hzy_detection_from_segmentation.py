import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import create_dataset
from unet import UNet
from utils.checkpoint_io import load_torch_state
from utils.data_loading import BasicDataset
from utils.detection_boxes import (
    binary_confusion_metrics,
    boxes_from_binary_mask,
    match_boxes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate lesion detection errors from a segmentation checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Segmentation checkpoint path")
    parser.add_argument("--images-dir", required=True, help="Evaluation image directory")
    parser.add_argument("--masks-dir", required=True, help="Evaluation mask directory")
    parser.add_argument("--output-dir", default="evaluation_detection", help="Directory for metrics and case tables")
    parser.add_argument("--classes", type=int, default=2, help="Number of segmentation classes")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--scale", type=float, default=1.0, help="Image scale")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7",
                        help="Comma-separated foreground probability thresholds")
    parser.add_argument("--min-area", type=int, default=16, help="Minimum component area for GT/pred boxes")
    parser.add_argument("--box-margin", type=int, default=4, help="Extra pixels added around GT/pred boxes")
    parser.add_argument("--match-iou", type=float, default=0.1, help="Relaxed IoU threshold for box matching")
    parser.add_argument("--disable-center-hit", action="store_true", default=False,
                        help="Require IoU only; by default a predicted center inside GT also counts as matched")
    parser.add_argument("--save-previews", type=int, default=12,
                        help="Maximum false-negative/false-positive preview images to save at the primary threshold")
    parser.add_argument("--primary-threshold", type=float, default=0.5,
                        help="Threshold used for preview selection")
    return parser.parse_args()


class NamedDataset(BasicDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["name"] = self.ids[idx]
        return sample


def parse_thresholds(value: str) -> List[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def make_dataset(images_dir: Path, masks_dir: Path, scale: float):
    try:
        return NamedDataset(images_dir, masks_dir, scale)
    except (AssertionError, RuntimeError):
        dataset = create_dataset(images_dir, masks_dir, scale)
        if not isinstance(dataset, BasicDataset):
            raise
        dataset.__class__ = NamedDataset
        return dataset


def foreground_probability(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    if n_classes == 1:
        return torch.sigmoid(logits[:, 0])
    probabilities = torch.softmax(logits, dim=1)
    if n_classes == 2:
        return probabilities[:, 1]
    return 1.0 - probabilities[:, 0]


def update_metrics(bucket: Dict[str, int], gt_boxes, pred_boxes, matches):
    gt_positive = len(gt_boxes) > 0
    pred_positive = len(pred_boxes) > 0

    if gt_positive and pred_positive:
        bucket["presence_tp"] += 1
    elif (not gt_positive) and pred_positive:
        bucket["presence_fp"] += 1
    elif gt_positive and (not pred_positive):
        bucket["presence_fn"] += 1
    else:
        bucket["presence_tn"] += 1

    bucket["gt_boxes"] += len(gt_boxes)
    bucket["pred_boxes"] += len(pred_boxes)
    bucket["matched_boxes"] += len(matches)
    bucket["unmatched_gt_boxes"] += max(0, len(gt_boxes) - len(matches))
    bucket["unmatched_pred_boxes"] += max(0, len(pred_boxes) - len(matches))


def draw_previews(image_path: Path, mask_path: Path, preview_path: Path, gt_boxes, pred_boxes, title: str):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        draw.rectangle([box.x_min, box.y_min, box.x_max, box.y_max], outline=(30, 180, 70), width=4)
    for box in pred_boxes:
        draw.rectangle([box.x_min, box.y_min, box.x_max, box.y_max], outline=(220, 50, 50), width=4)
    draw.text((8, 8), title, fill=(0, 0, 0))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_path, quality=95)


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finalize_metrics(raw: Dict[str, int]) -> Dict[str, float]:
    presence = binary_confusion_metrics(
        tp=raw["presence_tp"],
        fp=raw["presence_fp"],
        fn=raw["presence_fn"],
        tn=raw["presence_tn"],
    )
    box_precision = raw["matched_boxes"] / raw["pred_boxes"] if raw["pred_boxes"] else 0.0
    box_recall = raw["matched_boxes"] / raw["gt_boxes"] if raw["gt_boxes"] else 0.0
    box_f1 = (2 * box_precision * box_recall) / (box_precision + box_recall) if (box_precision + box_recall) else 0.0
    return {
        **raw,
        "presence_precision": presence["precision"],
        "presence_recall": presence["recall"],
        "presence_sensitivity": presence["sensitivity"],
        "presence_specificity": presence["specificity"],
        "presence_f1": presence["f1"],
        "presence_accuracy": presence["accuracy"],
        "box_precision": float(box_precision),
        "box_recall": float(box_recall),
        "box_f1": float(box_f1),
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    thresholds = parse_thresholds(args.thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = make_dataset(Path(args.images_dir), Path(args.masks_dir), args.scale)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = UNet(n_channels=3, n_classes=args.classes)
    state_dict = load_torch_state(args.model, map_location=device)
    state_dict.pop("mask_values", None)
    model.load_state_dict(state_dict)
    model.to(device=device)
    model.eval()

    raw_metrics = {
        threshold: {
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
        for threshold in thresholds
    }
    case_rows = []
    preview_count = 0
    name_to_paths = {
        path.stem: (path, Path(args.masks_dir) / (path.stem + ".png"))
        for path in Path(args.images_dir).glob("*.jpg")
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate detection", unit="batch"):
            images = batch["image"].to(device=device, dtype=torch.float32)
            true_masks = batch["mask"].cpu().numpy()
            names = list(batch["name"])
            with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=args.amp):
                logits = model(images)
                probabilities = foreground_probability(logits, args.classes).detach().cpu().numpy()

            for item_index, name in enumerate(names):
                gt_binary = true_masks[item_index] > 0
                gt_boxes = boxes_from_binary_mask(
                    gt_binary,
                    class_id=1,
                    min_area=args.min_area,
                    margin=args.box_margin,
                )

                for threshold in thresholds:
                    pred_binary = probabilities[item_index] >= threshold
                    pred_boxes = boxes_from_binary_mask(
                        pred_binary,
                        class_id=1,
                        min_area=args.min_area,
                        margin=args.box_margin,
                    )
                    matches = match_boxes(
                        gt_boxes,
                        pred_boxes,
                        iou_threshold=args.match_iou,
                        allow_center_hit=not args.disable_center_hit,
                    )
                    update_metrics(raw_metrics[threshold], gt_boxes, pred_boxes, matches)

                    gt_positive = len(gt_boxes) > 0
                    pred_positive = len(pred_boxes) > 0
                    case_status = "tn"
                    if gt_positive and pred_positive and matches:
                        case_status = "tp_localized"
                    elif gt_positive and pred_positive and not matches:
                        case_status = "positive_wrong_location"
                    elif gt_positive and not pred_positive:
                        case_status = "fn_missed"
                    elif (not gt_positive) and pred_positive:
                        case_status = "fp_overcalled"

                    case_rows.append(
                        {
                            "threshold": threshold,
                            "case_id": name,
                            "gt_positive": int(gt_positive),
                            "pred_positive": int(pred_positive),
                            "gt_boxes": len(gt_boxes),
                            "pred_boxes": len(pred_boxes),
                            "matched_boxes": len(matches),
                            "status": case_status,
                        }
                    )

                    if (
                        abs(threshold - args.primary_threshold) < 1e-8
                        and preview_count < args.save_previews
                        and case_status in {"fn_missed", "fp_overcalled", "positive_wrong_location"}
                        and name in name_to_paths
                    ):
                        image_path, mask_path = name_to_paths[name]
                        draw_previews(
                            image_path,
                            mask_path,
                            output_dir / "previews" / "{}_{}.jpg".format(case_status, name),
                            gt_boxes,
                            pred_boxes,
                            title="{} threshold={}".format(case_status, threshold),
                        )
                        preview_count += 1

    metrics_rows = []
    metrics_by_threshold = {}
    for threshold in thresholds:
        metrics = finalize_metrics(raw_metrics[threshold])
        metrics["threshold"] = threshold
        metrics_by_threshold[str(threshold)] = metrics
        metrics_rows.append(metrics)

    write_csv(metrics_rows, output_dir / "metrics.csv")
    write_csv(case_rows, output_dir / "cases.csv")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_by_threshold, handle, indent=2)

    logging.info("Saved detection-style metrics to %s", output_dir)
    primary = metrics_by_threshold.get(str(args.primary_threshold))
    if primary:
        logging.info(
            "Primary threshold %.2f: presence_f1=%.4f, presence_recall=%.4f, presence_precision=%.4f, box_recall=%.4f, box_precision=%.4f",
            args.primary_threshold,
            primary["presence_f1"],
            primary["presence_recall"],
            primary["presence_precision"],
            primary["box_recall"],
            primary["box_precision"],
        )


if __name__ == "__main__":
    main()
