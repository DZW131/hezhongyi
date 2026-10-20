"""Evaluate the HZY tubular-atrophy tile model on a held-out case split."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unet import UNet
from utils.checkpoint_io import load_torch_state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-evidence", type=int, default=4)
    parser.add_argument(
        "--component-hit-fraction",
        type=float,
        default=0.05,
        help="Minimum fraction of a ground-truth connected component that must be predicted to count as detected.",
    )
    return parser.parse_args()


def metrics(counts):
    tp, fp, fn, tn = (int(counts[key]) for key in ("tp", "fp", "fn", "tn"))
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def detection_counts(target: np.ndarray, prediction: np.ndarray, hit_fraction: float):
    """Count target components recovered by the prediction for recall-oriented use."""
    labels, component_count = ndimage.label(target)
    hits = 0
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        if prediction[component].sum() >= max(1, int(np.ceil(component.sum() * hit_fraction))):
            hits += 1
    return {
        "target_components": int(component_count),
        "detected_components": int(hits),
        "positive_tile": int(target.any()),
        "detected_positive_tile": int(target.any() and np.logical_and(target, prediction).any()),
    }


def overlay(image, mask, color):
    base = Image.fromarray(image).convert("RGBA")
    tint = Image.new("RGBA", base.size)
    tint.paste((*color, 110), mask=Image.fromarray((mask.astype(np.uint8) * 255)))
    return Image.alpha_composite(base, tint).convert("RGB")


def save_evidence(samples, output_path):
    if not samples:
        return
    tile_size = samples[0][1].shape[0]
    header = 28
    canvas = Image.new("RGB", (tile_size * 3, (tile_size + header) * len(samples)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (slide_id, image, target, prediction) in enumerate(samples):
        top = index * (tile_size + header)
        draw.text((6, top + 6), f"{slide_id}  original", fill="black")
        draw.text((tile_size + 6, top + 6), "GT (cyan)", fill="black")
        draw.text((tile_size * 2 + 6, top + 6), "prediction (magenta)", fill="black")
        canvas.paste(Image.fromarray(image), (0, top + header))
        canvas.paste(overlay(image, target, (0, 180, 200)), (tile_size, top + header))
        canvas.paste(overlay(image, prediction, (220, 30, 140)), (tile_size * 2, top + header))
    canvas.save(output_path, quality=95)


def dataset_path(root, relative_path):
    return root / Path(relative_path.replace("\\", "/"))


def main():
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be in (0, 1)")
    if not 0 < args.component_hit_fraction <= 1:
        raise ValueError("--component-hit-fraction must be in (0, 1]")
    root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (root / "manifests" / "tiles.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"No tiles in split {args.split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=2, bilinear=False).to(device)
    state = load_torch_state(args.checkpoint, map_location=device)
    state.pop("mask_values", None)
    model.load_state_dict(state)
    model.eval()

    overall = defaultdict(int)
    per_slide = defaultdict(lambda: defaultdict(int))
    per_case = defaultdict(lambda: defaultdict(int))
    evidence_candidates = {}
    batch_size = max(1, args.batch_size)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            images = [np.asarray(Image.open(dataset_path(root, row["image_path"])).convert("RGB")) for row in batch_rows]
            targets = [np.asarray(Image.open(dataset_path(root, row["mask_path"]))) > 0 for row in batch_rows]
            tensor = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()).float().to(device) / 255.0
            predictions = model(tensor).softmax(dim=1)[:, 1].cpu().numpy() >= args.threshold
            for row, image, target, prediction in zip(batch_rows, images, targets, predictions):
                counts = {
                    "tp": int(np.logical_and(target, prediction).sum()),
                    "fp": int(np.logical_and(~target, prediction).sum()),
                    "fn": int(np.logical_and(target, ~prediction).sum()),
                    "tn": int(np.logical_and(~target, ~prediction).sum()),
                }
                slide = per_slide[row["slide_id"]]
                case_id = row["slide_id"].rsplit("_s", 1)[0]
                case = per_case[case_id]
                for key, value in counts.items():
                    overall[key] += value
                    slide[key] += value
                    case[key] += value
                found = detection_counts(target, prediction, args.component_hit_fraction)
                for key, value in found.items():
                    overall[key] += value
                    slide[key] += value
                    case[key] += value
                slide["tiles"] += 1
                case["tiles"] += 1
                if target.any() and args.max_evidence > 0:
                    target_pixels = int(target.sum())
                    current = evidence_candidates.get(case_id)
                    if current is None or target_pixels > current[0]:
                        evidence_candidates[case_id] = (
                            target_pixels, row["slide_id"], image, target, prediction
                        )

    slide_rows = []
    for slide_id, counts in sorted(per_slide.items()):
        pixel_metrics = metrics(counts)
        component_recall = counts["detected_components"] / counts["target_components"] if counts["target_components"] else 1.0
        tile_recall = counts["detected_positive_tile"] / counts["positive_tile"] if counts["positive_tile"] else 1.0
        slide_rows.append({
            "slide_id": slide_id,
            "tiles": counts["tiles"],
            "positive_tiles": counts["positive_tile"],
            "target_components": counts["target_components"],
            "detected_components": counts["detected_components"],
            "component_recall": component_recall,
            "positive_tile_recall": tile_recall,
            **pixel_metrics,
        })
    with (output_dir / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slide_rows[0]))
        writer.writeheader()
        writer.writerows(slide_rows)
    case_rows = [
        {
            "case_id": case_id,
            "tiles": counts["tiles"],
            "positive_tiles": counts["positive_tile"],
            "target_components": counts["target_components"],
            "detected_components": counts["detected_components"],
            "component_recall": counts["detected_components"] / counts["target_components"] if counts["target_components"] else 1.0,
            "positive_tile_recall": counts["detected_positive_tile"] / counts["positive_tile"] if counts["positive_tile"] else 1.0,
            "case_detected": int(counts["positive_tile"] == 0 or counts["detected_positive_tile"] > 0),
            **metrics(counts),
        }
        for case_id, counts in sorted(per_case.items())
    ]
    with (output_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    payload = {
        "split": args.split,
        "tile_count": len(rows),
        "scene_count": len(per_slide),
        "case_count": len(per_case),
        "positive_tiles": int(overall["positive_tile"]),
        "target_components": int(overall["target_components"]),
        "detected_components": int(overall["detected_components"]),
        "component_recall": overall["detected_components"] / overall["target_components"] if overall["target_components"] else 1.0,
        "positive_tile_recall": overall["detected_positive_tile"] / overall["positive_tile"] if overall["positive_tile"] else 1.0,
        "positive_case_recall": (
            sum(row["case_detected"] for row in case_rows if row["positive_tiles"] > 0)
            / sum(row["positive_tiles"] > 0 for row in case_rows)
            if any(row["positive_tiles"] > 0 for row in case_rows)
            else 1.0
        ),
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "component_hit_fraction": args.component_hit_fraction,
        "metrics": metrics(overall),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    evidence = [candidate[1:] for _, candidate in sorted(evidence_candidates.items())[:args.max_evidence]]
    save_evidence(evidence, output_dir / "evidence.jpg")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
