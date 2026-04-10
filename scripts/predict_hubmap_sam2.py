import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hubmap_sam2.dataset import (
    bboxes_intersect,
    ensure_uint8_rgb,
    estimate_tissue_coverage,
    extract_tile,
    find_existing_path,
    generate_candidate_tiles_for_boxes,
    generate_candidate_tiles_for_records,
    generate_positions,
    load_polygon_records,
    open_slide_array,
    rasterize_records_to_mask,
    require_tifffile,
    save_palette_png,
)
from hubmap_sam2.prompts import component_boxes_from_binary_mask, prompts_from_binary_mask

try:
    import tifffile
except ImportError:  # pragma: no cover - optional at import time
    tifffile = None


PROMPT_SOURCES = ("amg", "mask")
TILE_SELECTIONS = ("auto", "grid", "roi", "prior-mask")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tile-level or whole-slide HuBMAP inference with a fine-tuned SAM2 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="SAM2 training config used to instantiate the model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to use for inference.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where predictions will be saved.")
    parser.add_argument("--device", type=str, default="", help="Torch device override, for example cuda or cpu.")
    parser.add_argument("--mode", type=str, default="tile", choices=("tile", "wsi"), help="Inference mode.")
    parser.add_argument("--prompt-source", type=str, default="amg", choices=PROMPT_SOURCES, help="How to generate prompts without GT.")
    parser.add_argument("--image", type=str, default="", help="Tile image path for tile mode.")
    parser.add_argument("--slide", type=str, default="", help="Whole-slide TIFF path for wsi mode.")
    parser.add_argument("--prior-mask", type=str, default="", help="Optional prior binary mask used when prompt-source=mask.")
    parser.add_argument("--anatomical-json", type=str, default="", help="Optional anatomical ROI JSON for wsi mode.")
    parser.add_argument("--roi-labels", nargs="*", default=[], help="Optional ROI labels to keep during wsi inference.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Patch size for wsi inference.")
    parser.add_argument("--stride", type=int, default=1024, help="Sliding stride for wsi inference.")
    parser.add_argument("--tile-selection", type=str, default="auto", choices=TILE_SELECTIONS, help="How to choose WSI tiles. 'auto' uses prior-mask boxes for mask prompting, ROI boxes when ROI labels are given, and otherwise falls back to full-grid scanning.")
    parser.add_argument("--tile-context-radius", type=int, default=0, help="Optional expansion radius in grid steps around selected WSI tiles.")
    parser.add_argument("--border-ignore", type=int, default=64, help="Ignore predictions too close to inner patch borders during wsi stitching.")
    parser.add_argument("--white-threshold", type=float, default=230.0, help="Threshold used for tissue filtering in wsi mode.")
    parser.add_argument("--min-tissue-coverage", type=float, default=0.05, help="Minimum tissue coverage required for a patch.")
    parser.add_argument("--min-roi-coverage", type=float, default=0.05, help="Minimum ROI coverage required for a patch when ROI labels are used.")
    parser.add_argument("--min-mask-area", type=int, default=16, help="Minimum mask area kept in outputs.")
    parser.add_argument("--min-component-area", type=int, default=32, help="Minimum prior-mask component area used to create prompts.")
    parser.add_argument("--prompt-mode", type=str, default="point_box", choices=("point", "box", "point_box"), help="Prompt type used with prior masks.")
    parser.add_argument("--points-per-side", type=int, default=24, help="AMG grid density.")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.75, help="AMG predicted IoU threshold.")
    parser.add_argument("--stability-score-thresh", type=float, default=0.9, help="AMG stability threshold.")
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        if tifffile is not None:
            return ensure_uint8_rgb(tifffile.imread(str(path)))
        return np.asarray(Image.open(path).convert("RGB"))
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        if tifffile is not None:
            mask = tifffile.imread(str(path))
        else:
            mask = np.asarray(Image.open(path))
    else:
        mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def keep_instance(mask: np.ndarray, tile_x: int, tile_y: int, patch_hw, full_hw, border_ignore: int) -> bool:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return False

    patch_h, patch_w = patch_hw
    full_h, full_w = full_hw
    touches_left = tile_x == 0
    touches_top = tile_y == 0
    touches_right = tile_x + patch_w >= full_w
    touches_bottom = tile_y + patch_h >= full_h

    if not touches_left and xs.min() < border_ignore:
        return False
    if not touches_top and ys.min() < border_ignore:
        return False
    if not touches_right and xs.max() >= patch_w - border_ignore:
        return False
    if not touches_bottom and ys.max() >= patch_h - border_ignore:
        return False
    return True


def save_tile_outputs(output_dir: Path, image: np.ndarray, instance_map: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_uint8_rgb(image)).save(output_dir / "image.png")
    save_palette_png(instance_map, output_dir / "instance_map.png")
    Image.fromarray((instance_map > 0).astype(np.uint8) * 255).save(output_dir / "binary_mask.png")


def resolve_wsi_tile_positions(args, full_w: int, full_h: int, roi_records, prior_mask_full):
    if args.tile_selection == "grid":
        x_positions = generate_positions(full_w, args.tile_size, args.stride)
        y_positions = generate_positions(full_h, args.tile_size, args.stride)
        return [(x, y) for y in y_positions for x in x_positions]

    if args.tile_selection in {"auto", "prior-mask"} and prior_mask_full is not None:
        component_boxes = component_boxes_from_binary_mask(
            prior_mask_full,
            min_component_area=args.min_component_area,
        )
        if component_boxes:
            return generate_candidate_tiles_for_boxes(
                component_boxes,
                width=full_w,
                height=full_h,
                tile_size=args.tile_size,
                stride=args.stride,
                expand_radius=args.tile_context_radius,
            )
        if args.tile_selection == "prior-mask":
            return []

    if args.tile_selection in {"auto", "roi"} and roi_records:
        return generate_candidate_tiles_for_records(
            roi_records,
            width=full_w,
            height=full_h,
            tile_size=args.tile_size,
            stride=args.stride,
            expand_radius=args.tile_context_radius,
        )

    x_positions = generate_positions(full_w, args.tile_size, args.stride)
    y_positions = generate_positions(full_h, args.tile_size, args.stride)
    return [(x, y) for y in y_positions for x in x_positions]


def run_tile_inference(args, model):
    from hubmap_sam2.inference import (
        SAM2AutomaticMaskGenerator,
        SAM2ImagePredictor,
        anns_to_instance_map,
        predict_instance_masks,
        predictions_to_instance_map,
    )
    if not args.image:
        raise ValueError("--image is required in tile mode")

    output_dir = Path(args.output_dir)
    image = load_image(Path(args.image))

    if args.prompt_source == "amg":
        generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_region_area=args.min_mask_area,
            output_mode="binary_mask",
            multimask_output=False,
        )
        anns = generator.generate(image)
        instance_map = anns_to_instance_map(anns, image.shape[:2], min_mask_area=args.min_mask_area)
    else:
        if not args.prior_mask:
            raise ValueError("--prior-mask is required when --prompt-source mask")
        prior_mask = load_mask(Path(args.prior_mask))
        prompts = prompts_from_binary_mask(prior_mask, min_component_area=args.min_component_area)
        predictor = SAM2ImagePredictor(model)
        predictions = predict_instance_masks(
            predictor,
            image,
            prompts,
            prompt_mode=args.prompt_mode,
            multimask_output=False,
        )
        instance_map = predictions_to_instance_map(predictions, image.shape[:2], min_mask_area=args.min_mask_area)

    save_tile_outputs(output_dir, image, instance_map)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": "tile",
                "prompt_source": args.prompt_source,
                "instance_count": int(len([value for value in np.unique(instance_map) if int(value) > 0])),
                "positive_pixels": int((instance_map > 0).sum()),
            },
            handle,
            indent=2,
        )


def run_wsi_inference(args, model):
    from hubmap_sam2.inference import (
        SAM2AutomaticMaskGenerator,
        SAM2ImagePredictor,
        anns_to_instance_map,
        predict_instance_masks,
        predictions_to_instance_map,
    )
    if not args.slide:
        raise ValueError("--slide is required in wsi mode")

    require_tifffile()
    slide_path = Path(args.slide)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_array = open_slide_array(slide_path)
    full_h, full_w = slide_array.shape[:2]
    global_instance_map = np.zeros((full_h, full_w), dtype=np.uint32)
    next_instance_id = 1

    generator = None
    predictor = None
    prior_mask_full = None

    if args.prompt_source == "amg":
        generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_region_area=args.min_mask_area,
            output_mode="binary_mask",
            multimask_output=False,
        )
    else:
        if not args.prior_mask:
            raise ValueError("--prior-mask is required when --prompt-source mask")
        prior_mask_full = load_mask(Path(args.prior_mask))
        if prior_mask_full.shape[:2] != (full_h, full_w):
            raise ValueError("Prior mask shape does not match the whole-slide image.")
        predictor = SAM2ImagePredictor(model)

    roi_records = []
    if args.anatomical_json and args.roi_labels:
        roi_records = load_polygon_records(Path(args.anatomical_json), target_labels=args.roi_labels)

    tile_positions = resolve_wsi_tile_positions(args, full_w, full_h, roi_records, prior_mask_full)
    logging.info("WSI inference will evaluate %s candidate tiles", len(tile_positions))

    for x, y in tile_positions:
        tile_bbox = (x, y, x + args.tile_size, y + args.tile_size)
        image_tile, actual_hw = extract_tile(slide_array, x, y, args.tile_size)
        if estimate_tissue_coverage(image_tile, args.white_threshold) < args.min_tissue_coverage:
            continue

        if roi_records:
            candidate_roi_records = [record for record in roi_records if bboxes_intersect(record.bbox, tile_bbox)]
            roi_mask = rasterize_records_to_mask(candidate_roi_records, tile_bbox, args.tile_size)
            if float((roi_mask > 0).mean()) < args.min_roi_coverage:
                continue

        patch_h, patch_w = actual_hw
        valid_shape = (patch_h, patch_w)

        if args.prompt_source == "amg":
            anns = generator.generate(image_tile)
            local_instance_map = anns_to_instance_map(anns, image_tile.shape[:2], min_mask_area=args.min_mask_area)
        else:
            prior_tile = prior_mask_full[y : y + patch_h, x : x + patch_w]
            if int((prior_tile > 0).sum()) == 0:
                continue
            padded_prior = np.zeros((args.tile_size, args.tile_size), dtype=prior_tile.dtype)
            padded_prior[:patch_h, :patch_w] = prior_tile
            prompts = prompts_from_binary_mask(padded_prior, min_component_area=args.min_component_area)
            if not prompts:
                continue
            predictions = predict_instance_masks(
                predictor,
                image_tile,
                prompts,
                prompt_mode=args.prompt_mode,
                multimask_output=False,
            )
            local_instance_map = predictions_to_instance_map(predictions, image_tile.shape[:2], min_mask_area=args.min_mask_area)

        local_ids = [int(value) for value in np.unique(local_instance_map[:patch_h, :patch_w]) if int(value) > 0]
        for local_id in local_ids:
            local_mask = local_instance_map[:patch_h, :patch_w] == local_id
            if not keep_instance(local_mask, x, y, valid_shape, (full_h, full_w), args.border_ignore):
                continue
            global_instance_map[y : y + patch_h, x : x + patch_w][local_mask] = next_instance_id
            next_instance_id += 1

    tifffile.imwrite(str(output_dir / "instance_map.tiff"), global_instance_map.astype(np.uint32))
    tifffile.imwrite(str(output_dir / "binary_mask.tiff"), (global_instance_map > 0).astype(np.uint8))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": "wsi",
                "prompt_source": args.prompt_source,
                "slide": str(slide_path),
                "instance_count": int(next_instance_id - 1),
                "positive_pixels": int((global_instance_map > 0).sum()),
            },
            handle,
            indent=2,
        )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from hubmap_sam2.inference import load_finetuned_model

    model = load_finetuned_model(args.config, args.checkpoint, device=args.device or None)

    if args.mode == "tile":
        run_tile_inference(args, model)
    else:
        run_wsi_inference(args, model)


if __name__ == "__main__":
    main()
