import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hubmap_sam2.dataset import discover_prepared_samples, load_prepared_sample
from hubmap_sam2.metrics import BinaryMetricAccumulator, instance_level_metrics, instance_map_to_binary
from hubmap_sam2.prompts import prompts_from_binary_mask
from hubmap_sam2.visualization import save_prediction_preview


EVAL_MODES = ("oracle-point", "oracle-box", "oracle-point-box", "amg", "prior-mask")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a HuBMAP fine-tuned SAM2 checkpoint on the prepared tile dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=str, required=True, help="Prepared dataset root created by prepare_hubmap_sam2_dataset.py.")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"), help="Dataset split to evaluate.")
    parser.add_argument("--config", type=str, required=True, help="SAM2 training config used to instantiate the model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where metrics and previews will be saved.")
    parser.add_argument("--mode", type=str, default="oracle-point", choices=EVAL_MODES, help="Evaluation mode.")
    parser.add_argument("--device", type=str, default="", help="Torch device override, for example cuda or cpu.")
    parser.add_argument("--prior-mask-dir", type=str, default="", help="Directory containing coarse binary masks for mode=prior-mask.")
    parser.add_argument("--prompt-mode", type=str, default="point_box", choices=("point", "box", "point_box"), help="Prompt type used in prior-mask mode.")
    parser.add_argument("--min-component-area", type=int, default=32, help="Minimum connected-component area kept when generating prompts from coarse masks.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample limit for quick checks.")
    parser.add_argument("--preview-count", type=int, default=8, help="Number of preview images to save.")
    parser.add_argument("--min-mask-area", type=int, default=16, help="Minimum predicted mask area kept in the final instance map.")
    parser.add_argument("--points-per-side", type=int, default=24, help="AMG grid density.")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.75, help="AMG predicted IoU threshold.")
    parser.add_argument("--stability-score-thresh", type=float, default=0.9, help="AMG stability threshold.")
    parser.add_argument("--instance-iou-threshold", type=float, default=0.5, help="IoU threshold for object-level matching.")
    return parser.parse_args()


def write_csv(rows, output_path: Path):
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        mask = tifffile.imread(str(path))
    else:
        mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def resolve_prior_mask(prior_mask_dir: Path, sample_id: str) -> Path:
    candidates = [
        prior_mask_dir / f"{sample_id}.png",
        prior_mask_dir / f"{sample_id}.tif",
        prior_mask_dir / f"{sample_id}.tiff",
        prior_mask_dir / sample_id / "binary_mask.png",
        prior_mask_dir / sample_id / "binary_mask.tif",
        prior_mask_dir / sample_id / "binary_mask.tiff",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a prior mask for sample {sample_id} in {prior_mask_dir}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from hubmap_sam2.inference import (
        SAM2ImagePredictor,
        anns_to_instance_map,
        automatic_mask_generation,
        load_finetuned_model,
        oracle_prompts_from_annotation,
        predict_instance_masks,
        predictions_to_instance_map,
    )

    dataset_root = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_prepared_samples(dataset_root, args.split)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise FileNotFoundError(f"No prepared samples were found in {dataset_root / args.split}")

    model = load_finetuned_model(args.config, args.checkpoint, device=args.device or None)
    predictor = SAM2ImagePredictor(model) if args.mode.startswith("oracle") or args.mode == "prior-mask" else None
    prior_mask_dir = Path(args.prior_mask_dir) if args.prior_mask_dir else None
    if args.mode == "prior-mask" and prior_mask_dir is None:
        raise ValueError("--prior-mask-dir is required when --mode prior-mask")

    semantic_accumulator = BinaryMetricAccumulator()
    sample_rows = []
    instance_tp = 0.0
    instance_fp = 0.0
    instance_fn = 0.0

    preview_dir = output_dir / "previews"
    for index, sample in enumerate(samples):
        image, true_instance_map, metadata = load_prepared_sample(sample)
        prompt_points = []

        if args.mode.startswith("oracle"):
            prompt_mode = args.mode.replace("oracle-", "")
            prompts = oracle_prompts_from_annotation(true_instance_map, prompt_mode=prompt_mode)
            predictions = predict_instance_masks(
                predictor,
                image,
                prompts,
                prompt_mode=prompt_mode.replace("-", "_"),
                multimask_output=False,
            )
            pred_instance_map = predictions_to_instance_map(predictions, true_instance_map.shape, min_mask_area=args.min_mask_area)
            prompt_points = [prompt.point for prompt in prompts if prompt.point is not None]
        elif args.mode == "amg":
            anns = automatic_mask_generation(
                model,
                image,
                points_per_side=args.points_per_side,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_score_thresh,
                min_mask_region_area=args.min_mask_area,
            )
            pred_instance_map = anns_to_instance_map(anns, true_instance_map.shape, min_mask_area=args.min_mask_area)
            prompt_points = [tuple(ann["point_coords"][0]) for ann in anns[:16] if ann.get("point_coords")]
        else:
            prior_mask = load_mask(resolve_prior_mask(prior_mask_dir, sample.sample_id))
            prompts = prompts_from_binary_mask(prior_mask, min_component_area=args.min_component_area)
            predictions = predict_instance_masks(
                predictor,
                image,
                prompts,
                prompt_mode=args.prompt_mode,
                multimask_output=False,
            )
            pred_instance_map = predictions_to_instance_map(predictions, true_instance_map.shape, min_mask_area=args.min_mask_area)
            prompt_points = [prompt.point for prompt in prompts if prompt.point is not None]

        semantic_accumulator.update(instance_map_to_binary(pred_instance_map), instance_map_to_binary(true_instance_map))
        instance_metrics = instance_level_metrics(
            pred_instance_map,
            true_instance_map,
            iou_threshold=args.instance_iou_threshold,
        )
        instance_tp += instance_metrics["instance_tp"]
        instance_fp += instance_metrics["instance_fp"]
        instance_fn += instance_metrics["instance_fn"]

        semantic_metrics = BinaryMetricAccumulator()
        semantic_metrics.update(instance_map_to_binary(pred_instance_map), instance_map_to_binary(true_instance_map))
        row = {
            "sample_id": sample.sample_id,
            "slide_id": sample.slide_id,
            **semantic_metrics.compute(),
            **instance_metrics,
        }
        sample_rows.append(row)

        if index < args.preview_count:
            save_prediction_preview(
                image,
                true_instance_map,
                pred_instance_map,
                preview_dir / f"{index:03d}_{sample.sample_id}.png",
                metrics=row,
                prompt_points=prompt_points,
            )

        logging.info(
            "[%s/%s] %s dice=%.4f iou=%.4f instance_f1=%.4f",
            index + 1,
            len(samples),
            sample.sample_id,
            row["dice"],
            row["iou"],
            row["instance_f1"],
        )

    metrics = semantic_accumulator.compute()
    instance_precision = instance_tp / (instance_tp + instance_fp) if (instance_tp + instance_fp) else 0.0
    instance_recall = instance_tp / (instance_tp + instance_fn) if (instance_tp + instance_fn) else 0.0
    instance_f1 = (
        2 * instance_precision * instance_recall / (instance_precision + instance_recall)
        if (instance_precision + instance_recall)
        else 0.0
    )
    metrics.update(
        {
            "sample_count": len(sample_rows),
            "instance_tp": instance_tp,
            "instance_fp": instance_fp,
            "instance_fn": instance_fn,
            "instance_precision": instance_precision,
            "instance_recall": instance_recall,
            "instance_f1": instance_f1,
            "mode": args.mode,
            "checkpoint": args.checkpoint,
            "config": args.config,
        }
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    write_csv(sample_rows, output_dir / "per_sample_metrics.csv")

    logging.info("Saved evaluation summary to %s", output_dir / "metrics.json")


if __name__ == "__main__":
    main()
