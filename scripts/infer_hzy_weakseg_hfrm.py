"""Infer HSPN weak segmentation CAM masks and write scorer-ready CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hzy_weakseg_hfrm import (  # noqa: E402
    HspnWeakSegDataset,
    cam_binary_mask,
    connected_boxes,
    fused_cam_with_tta,
    load_checkpoint_model,
    save_evidence_overlay,
    save_json,
    target_specs_for_task,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HFRM-CAM weak segmentation inference and export scorer-ready probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=("proliferation", "mesangial", "endocapillary", "crescent", "crescent_binary"), default="")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for masks, overlays, and summary JSON")
    parser.add_argument("--image-size", type=int, default=0, help="Override checkpoint image size; 0 uses checkpoint args")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-threshold", type=float, default=0.35)
    parser.add_argument("--mask-min-area", type=int, default=8)
    parser.add_argument("--mask-open-radius", type=int, default=0)
    parser.add_argument("--mask-close-radius", type=int, default=0)
    parser.add_argument("--mask-fill-holes", action="store_true", default=False)
    parser.add_argument("--mask-keep-components", type=int, default=0)
    parser.add_argument("--mask-topk-area", type=float, default=0.0)
    parser.add_argument("--mask-use-prior", action="store_true", default=False)
    parser.add_argument("--classification-threshold", type=float, default=0.0, help="If 0, use checkpoint per-class thresholds")
    parser.add_argument("--eval-tta", action="store_true", default=False)
    parser.add_argument("--save-overlays", action="store_true", default=False)
    parser.add_argument("--save-masks", action="store_true", default=False)
    parser.add_argument("--max-overlays", type=int, default=48)
    parser.add_argument("--min-box-area", type=int, default=8)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def infer_task_from_checkpoint(ckpt: Dict[str, object], explicit: str) -> str:
    if explicit:
        return explicit
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    task = args.get("task")
    if not task:
        raise SystemExit("--task is required because checkpoint args do not contain task")
    return str(task)


def thresholds_from_checkpoint(ckpt: Dict[str, object], specs, override: float) -> Dict[str, float]:
    if override > 0:
        return {spec.slug: float(override) for spec in specs}
    raw = ckpt.get("thresholds", {}) if isinstance(ckpt, dict) else {}
    thresholds = {}
    for spec in specs:
        thresholds[spec.slug] = float(raw.get(spec.slug, 0.5)) if isinstance(raw, dict) else 0.5
    return thresholds


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, ckpt = load_checkpoint_model(args.checkpoint, device=device)
    task = infer_task_from_checkpoint(ckpt, args.task)
    specs = target_specs_for_task(task)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    image_size = int(args.image_size or ckpt_args.get("image_size", 384))
    thresholds = thresholds_from_checkpoint(ckpt, specs, args.classification_threshold)
    output_dir = args.output_dir or args.output_csv.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    rows: List[Dict[str, object]] = []
    overlay_count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            masks = batch["mask"]
            cams, probs = fused_cam_with_tta(model, images, output_size=tuple(masks.shape[-2:]), eval_tta=args.eval_tta)
            cams_np = cams.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            priors_np = batch["prior"].cpu().numpy()
            for item_idx, crop_id in enumerate(batch["crop_id"]):
                row: Dict[str, object] = {
                    "slide_id": batch["slide_id"][item_idx],
                    "glomerulus_id": batch["glomerulus_id"][item_idx],
                    "crop_id": crop_id,
                    "split": args.split,
                    "image_path": batch["image_path"][item_idx],
                    "source_notes": "hfrm_cam_weakseg:{}".format(task),
                }
                class_probs = []
                class_areas = []
                class_boxes = {}
                for class_idx, spec in enumerate(specs):
                    prob = float(probs_np[item_idx, class_idx])
                    cam = cams_np[item_idx, class_idx]
                    prior = priors_np[item_idx, 0] if args.mask_use_prior and priors_np.ndim == 4 else None
                    gated_mask = cam_binary_mask(
                        cam,
                        mask_threshold=args.mask_threshold,
                        active=prob >= thresholds[spec.slug],
                        prior=prior,
                        use_prior=args.mask_use_prior,
                        min_area=args.mask_min_area,
                        open_radius=args.mask_open_radius,
                        close_radius=args.mask_close_radius,
                        fill_holes=args.mask_fill_holes,
                        keep_components=args.mask_keep_components,
                        topk_area=args.mask_topk_area,
                    )
                    boxes = connected_boxes(gated_mask, min_area=args.min_box_area)
                    row[spec.prob_field] = "{:.6f}".format(prob)
                    row[spec.area_field] = "{:.6f}".format(float(gated_mask.mean()))
                    row["{}_boxes".format(spec.slug)] = json.dumps(
                        [
                            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": area}
                            for x1, y1, x2, y2, area in boxes
                        ],
                        ensure_ascii=False,
                    )
                    class_probs.append(prob)
                    class_areas.append(float(gated_mask.mean()))
                    class_boxes[spec.slug] = boxes
                    if args.save_masks:
                        mask_path = output_dir / "masks" / "{}_{}.png".format(crop_id, spec.slug)
                        save_mask(mask_path, gated_mask)
                        row["{}_mask_path".format(spec.slug)] = str(mask_path)
                if task == "crescent":
                    row["crescent_prob"] = "{:.6f}".format(float(max(class_probs) if class_probs else 0.0))
                    row["crescent_weakseg_area"] = "{:.6f}".format(float(max(class_areas) if class_areas else 0.0))
                    best_idx = int(np.argmax(class_probs)) if class_probs else 0
                    if class_probs and class_probs[best_idx] >= thresholds[specs[best_idx].slug]:
                        row["crescent_subtype"] = specs[best_idx].slug
                    else:
                        row["crescent_subtype"] = ""
                if args.save_overlays and overlay_count < args.max_overlays:
                    overlay_path = output_dir / "overlays" / "{}_weakseg.png".format(crop_id)
                    save_evidence_overlay(
                        Path(str(batch["image_path"][item_idx])),
                        cams_np[item_idx],
                        probs_np[item_idx],
                        specs,
                        thresholds,
                        overlay_path,
                        mask_threshold=args.mask_threshold,
                        min_area=max(args.mask_min_area, args.min_box_area),
                        open_radius=args.mask_open_radius,
                        close_radius=args.mask_close_radius,
                        fill_holes=args.mask_fill_holes,
                        keep_components=args.mask_keep_components,
                        topk_area=args.mask_topk_area,
                    )
                    row["weakseg_overlay_path"] = str(overlay_path)
                    overlay_count += 1
                rows.append(row)

    write_csv_rows(rows, args.output_csv)
    positive_counts = {}
    for spec in specs:
        positive_counts[spec.slug] = sum(float(row[spec.prob_field]) >= thresholds[spec.slug] for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "task": task,
        "split": args.split,
        "data_root": str(args.data_root),
        "output_csv": str(args.output_csv),
        "samples": len(rows),
        "thresholds": thresholds,
        "positive_counts": positive_counts,
        "overlays": overlay_count,
    }
    save_json(summary, output_dir / "weakseg_inference_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
