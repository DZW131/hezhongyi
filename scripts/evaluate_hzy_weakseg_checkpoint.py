"""Evaluate an HSPN HFRM-CAM checkpoint with optional CAM mask postprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hzy_weakseg_hfrm import (  # noqa: E402
    HspnWeakSegDataset,
    load_checkpoint_model,
    save_json,
    target_specs_for_task,
)
from train_hzy_weakseg_hfrm import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an HFRM-CAM checkpoint on one split and write compact metric summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=("proliferation", "mesangial", "endocapillary", "crescent", "crescent_binary"), default="")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-tta", action="store_true", default=False)
    parser.add_argument("--mask-thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--mask-min-area", type=int, default=8)
    parser.add_argument("--mask-open-radius", type=int, default=0)
    parser.add_argument("--mask-close-radius", type=int, default=0)
    parser.add_argument("--mask-fill-holes", action="store_true", default=False)
    parser.add_argument("--mask-keep-components", type=int, default=0)
    parser.add_argument("--mask-topk-area", type=float, default=0.0)
    parser.add_argument("--mask-use-prior", action="store_true", default=False)
    parser.add_argument(
        "--class-thresholds",
        default="",
        help="Optional comma list like cellular_crescent=0.65,fibrocellular_crescent=0.90. Defaults to checkpoint thresholds.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def task_from_checkpoint(ckpt: Dict[str, object], explicit: str) -> str:
    if explicit:
        return explicit
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    task = args.get("task")
    if not task:
        raise SystemExit("--task is required because checkpoint args do not contain task")
    return str(task)


def thresholds_from_checkpoint(ckpt: Dict[str, object], specs, override: str) -> Dict[str, float]:
    raw = ckpt.get("thresholds", {}) if isinstance(ckpt, dict) else {}
    thresholds = {spec.slug: float(raw.get(spec.slug, 0.5)) if isinstance(raw, dict) else 0.5 for spec in specs}
    if not override.strip():
        return thresholds
    for piece in override.split(","):
        if not piece.strip():
            continue
        key, value = piece.split("=", 1)
        key = key.strip()
        if key not in thresholds:
            raise ValueError("Unknown class threshold '{}'. Valid: {}".format(key, ", ".join(thresholds)))
        thresholds[key] = float(value)
    return thresholds


def parse_thresholds(text: str) -> List[float]:
    values = [float(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not values:
        raise ValueError("--mask-thresholds must include at least one value")
    return values


def flatten_result(mask_threshold: float, result: Dict[str, object], specs) -> Dict[str, object]:
    cls = result["classification"]
    seg = result.get("segmentation", {})
    row: Dict[str, object] = {
        "mask_threshold": mask_threshold,
        "macro_f1": cls["macro_f1"],
        "exact_match": cls["exact_match"],
        "mean_auc": cls["mean_auc"],
        "mean_dice": seg.get("mean_dice", ""),
        "mean_iou": seg.get("mean_iou", ""),
    }
    for spec in specs:
        cls_item = cls["per_class"][spec.slug]
        row["{}_f1".format(spec.slug)] = cls_item["f1"]
        row["{}_precision".format(spec.slug)] = cls_item["precision"]
        row["{}_recall".format(spec.slug)] = cls_item["recall"]
        row["{}_threshold".format(spec.slug)] = cls_item["threshold"]
        if spec.slug in seg:
            row["{}_dice".format(spec.slug)] = seg[spec.slug]["dice"]
            row["{}_iou".format(spec.slug)] = seg[spec.slug]["iou"]
            row["{}_pointing".format(spec.slug)] = seg[spec.slug].get("pointing_game")
    return row


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, ckpt = load_checkpoint_model(args.checkpoint, device=device)
    task = task_from_checkpoint(ckpt, args.task)
    specs = target_specs_for_task(task)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    image_size = int(args.image_size or ckpt_args.get("image_size", 384))
    thresholds = thresholds_from_checkpoint(ckpt, specs, args.class_thresholds)
    dataset = HspnWeakSegDataset(
        args.data_root,
        task=task,
        split=args.split,
        image_size=image_size,
        min_positive_pixels=int(ckpt_args.get("min_positive_pixels", 8)),
        manifest=args.manifest,
        augmentation="off",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    results = []
    flat_rows = []
    for mask_threshold in parse_thresholds(args.mask_thresholds):
        result = evaluate(
            model,
            loader,
            device,
            specs,
            thresholds,
            mask_threshold,
            eval_tta=args.eval_tta,
            mask_min_area=args.mask_min_area,
            mask_open_radius=args.mask_open_radius,
            mask_close_radius=args.mask_close_radius,
            mask_fill_holes=args.mask_fill_holes,
            mask_keep_components=args.mask_keep_components,
            mask_topk_area=args.mask_topk_area,
            mask_use_prior=args.mask_use_prior,
        )
        results.append({"mask_threshold": mask_threshold, "metrics": result})
        flat_rows.append(flatten_result(mask_threshold, result, specs))

    payload = {
        "checkpoint": str(args.checkpoint),
        "task": task,
        "split": args.split,
        "samples": len(dataset),
        "class_thresholds": thresholds,
        "postprocess": {
            "mask_min_area": args.mask_min_area,
            "mask_open_radius": args.mask_open_radius,
            "mask_close_radius": args.mask_close_radius,
            "mask_fill_holes": args.mask_fill_holes,
            "mask_keep_components": args.mask_keep_components,
            "mask_topk_area": args.mask_topk_area,
            "mask_use_prior": args.mask_use_prior,
        },
        "results": results,
        "summary_rows": flat_rows,
    }
    save_json(payload, args.output_json)
    if args.output_csv:
        write_csv(flat_rows, args.output_csv)
    print(json.dumps({"output_json": str(args.output_json), "rows": flat_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
