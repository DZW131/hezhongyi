import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hubmap_sam2.dataset import (
    PREPARED_IMAGE_NAME,
    PREPARED_MASK_NAME,
    assign_slide_splits,
    bboxes_intersect,
    ensure_uint8_rgb,
    estimate_tissue_coverage,
    extract_tile,
    filter_records_by_roi,
    find_existing_path,
    find_slide_ids,
    generate_positions,
    load_polygon_records,
    open_slide_array,
    rasterize_records_to_mask,
    resize_image,
    resize_mask,
    save_image_png,
    save_palette_png,
    write_csv,
    write_json,
)
from hubmap_sam2.prompts import instance_prompts_from_instance_map


MISSING_ROI_POLICIES = ("skip-slide", "ignore-roi", "error")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a single-frame SAM2 training dataset from HuBMAP TIFF + polygon annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing HuBMAP TIFF slides.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where the prepared SAM2 dataset will be written.")
    parser.add_argument("--annotations-dir", type=str, default="", help="Directory containing glomerulus JSON files. Defaults to images-dir.")
    parser.add_argument("--annotation-json-suffix", type=str, default=".json", help="Suffix for glomerulus polygon annotation JSON.")
    parser.add_argument("--anatomical-dir", type=str, default="", help="Directory containing anatomical JSON files. Defaults to images-dir.")
    parser.add_argument("--anatomical-json-suffix", type=str, default="-anatomical-structure.json", help="Suffix for anatomical ROI JSON.")
    parser.add_argument("--target-labels", nargs="+", default=["glomerulus"], help="Annotation labels treated as positive glomerulus instances.")
    parser.add_argument("--roi-labels", nargs="*", default=[], help="Optional anatomical labels to keep, for example Cortex.")
    parser.add_argument("--missing-roi-policy", choices=MISSING_ROI_POLICIES, default="skip-slide", help="Behavior when requested ROI annotations are missing.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Patch size before optional resize.")
    parser.add_argument("--stride", type=int, default=1024, help="Sliding-window stride.")
    parser.add_argument("--downsample", type=float, default=1.0, help="Optional downsample ratio applied after tiling.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio at slide level when split-csv is not provided.")
    parser.add_argument("--split-csv", type=str, default="", help="Optional CSV with columns slide_id,split to control train/val split.")
    parser.add_argument("--white-threshold", type=float, default=230.0, help="Threshold used to estimate white background.")
    parser.add_argument("--min-tissue-coverage", type=float, default=0.05, help="Minimum tissue coverage required to keep a tile.")
    parser.add_argument("--min-roi-coverage", type=float, default=0.05, help="Minimum ROI coverage required when roi-labels are used.")
    parser.add_argument("--min-positive-pixels", type=int, default=64, help="Minimum positive pixels required to keep a tile.")
    parser.add_argument("--max-instances-per-tile", type=int, default=254, help="Maximum glomerulus instances stored in a single palette PNG tile.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for slide splitting.")
    parser.add_argument("--limit-slides", type=int, default=0, help="Optional limit for quick experiments.")
    return parser.parse_args()


def write_list_file(sample_ids: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            handle.write(f"{sample_id}\n")


def resolve_roi_records(args, slide_id: str, anatomical_dir: Path):
    if not args.roi_labels:
        return []

    roi_json_path = anatomical_dir / f"{slide_id}{args.anatomical_json_suffix}"
    if not roi_json_path.exists():
        message = f"Missing anatomical JSON for slide {slide_id}"
        if args.missing_roi_policy == "ignore-roi":
            logging.warning("%s. Continuing without ROI filtering.", message)
            return []
        if args.missing_roi_policy == "skip-slide":
            logging.warning("%s. Skipping slide.", message)
            return None
        raise FileNotFoundError(message)

    roi_records = load_polygon_records(roi_json_path, target_labels=args.roi_labels)
    if roi_records:
        return roi_records

    message = f"ROI labels {args.roi_labels} were not found in {roi_json_path}"
    if args.missing_roi_policy == "ignore-roi":
        logging.warning("%s. Continuing without ROI filtering.", message)
        return []
    if args.missing_roi_policy == "skip-slide":
        logging.warning("%s. Skipping slide.", message)
        return None
    raise ValueError(message)


def save_sample(
    output_dir: Path,
    split: str,
    sample_id: str,
    image_tile,
    instance_map,
    metadata: Dict[str, object],
) -> None:
    image_path = output_dir / split / "JPEGImages" / sample_id / PREPARED_IMAGE_NAME
    mask_path = output_dir / split / "Annotations" / sample_id / PREPARED_MASK_NAME
    metadata_path = output_dir / split / "Metadata" / f"{sample_id}.json"

    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    save_image_png(image_tile, image_path)
    save_palette_png(instance_map, mask_path)
    write_json(metadata, metadata_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not (0.0 < args.downsample <= 1.0):
        raise ValueError("--downsample must be in the interval (0, 1].")
    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError("--tile-size and --stride must be positive integers.")

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else images_dir
    anatomical_dir = Path(args.anatomical_dir) if args.anatomical_dir else images_dir
    split_csv = Path(args.split_csv) if args.split_csv else None

    slide_ids = find_slide_ids(images_dir, limit_slides=args.limit_slides)
    slide_splits = assign_slide_splits(slide_ids, split_csv=split_csv, val_ratio=args.val_ratio, seed=args.seed)

    sample_rows: List[Dict[str, object]] = []
    slide_rows: List[Dict[str, object]] = []
    sample_ids_by_split = {"train": [], "val": []}

    for slide_id in slide_ids:
        image_path = find_existing_path(images_dir, slide_id, (".tif", ".tiff"))
        if image_path is None:
            logging.warning("Skipping slide %s because TIFF is missing.", slide_id)
            continue

        annotation_path = annotations_dir / f"{slide_id}{args.annotation_json_suffix}"
        if not annotation_path.exists():
            logging.warning("Skipping slide %s because annotation JSON is missing.", slide_id)
            continue

        records = load_polygon_records(annotation_path, target_labels=args.target_labels)
        if not records:
            logging.warning("Skipping slide %s because no glomerulus polygons were found.", slide_id)
            continue

        roi_records = resolve_roi_records(args, slide_id, anatomical_dir)
        if roi_records is None:
            continue
        records = filter_records_by_roi(records, roi_records)
        if not records:
            logging.warning("Skipping slide %s because ROI filtering removed all glomerulus instances.", slide_id)
            continue

        logging.info("Opening slide %s", image_path)
        slide_array = open_slide_array(image_path)
        height, width = slide_array.shape[:2]
        split = slide_splits[slide_id]

        x_positions = generate_positions(width, args.tile_size, args.stride)
        y_positions = generate_positions(height, args.tile_size, args.stride)
        positive_tile_count = 0

        for y in y_positions:
            for x in x_positions:
                tile_bbox = (x, y, x + args.tile_size, y + args.tile_size)
                candidate_records = [record for record in records if bboxes_intersect(record.bbox, tile_bbox)]
                candidate_roi_records = [record for record in roi_records if bboxes_intersect(record.bbox, tile_bbox)]

                image_tile, actual_hw = extract_tile(slide_array, x, y, args.tile_size)
                tissue_coverage = estimate_tissue_coverage(image_tile, args.white_threshold)
                if tissue_coverage < args.min_tissue_coverage:
                    continue

                if args.roi_labels:
                    roi_mask = rasterize_records_to_mask(candidate_roi_records, tile_bbox, args.tile_size)
                    roi_coverage = float((roi_mask > 0).mean())
                    if roi_coverage < args.min_roi_coverage:
                        continue
                else:
                    roi_coverage = 1.0

                if not candidate_records:
                    continue

                candidate_records = sorted(candidate_records, key=lambda record: record.area, reverse=True)
                candidate_records = candidate_records[: args.max_instances_per_tile]
                local_ids = {record.record_id: index + 1 for index, record in enumerate(candidate_records)}
                instance_map = rasterize_records_to_mask(candidate_records, tile_bbox, args.tile_size, value_lookup=local_ids)

                if int((instance_map > 0).sum()) < args.min_positive_pixels:
                    continue

                if args.downsample != 1.0:
                    output_size = max(1, int(round(args.tile_size * args.downsample)))
                    image_tile = resize_image(image_tile, output_size)
                    instance_map = resize_mask(instance_map, output_size)

                prompts = instance_prompts_from_instance_map(instance_map)
                if not prompts:
                    continue

                sample_id = f"{slide_id}_x{x:05d}_y{y:05d}"
                metadata = {
                    "sample_id": sample_id,
                    "slide_id": slide_id,
                    "split": split,
                    "tile_origin_xy": [int(x), int(y)],
                    "source_tile_size": int(args.tile_size),
                    "saved_image_shape": [int(image_tile.shape[0]), int(image_tile.shape[1])],
                    "actual_hw_before_padding": [int(actual_hw[0]), int(actual_hw[1])],
                    "tissue_coverage": tissue_coverage,
                    "roi_coverage": roi_coverage,
                    "positive_pixels": int((instance_map > 0).sum()),
                    "objects": [],
                }
                for prompt in prompts:
                    metadata["objects"].append(
                        {
                            "tile_object_id": int(prompt.object_id),
                            "visible_area": int(prompt.area),
                            "point_xy": [float(prompt.point[0]), float(prompt.point[1])] if prompt.point is not None else None,
                            "box_xyxy": [float(value) for value in prompt.box] if prompt.box is not None else None,
                        }
                    )

                save_sample(output_dir, split, sample_id, ensure_uint8_rgb(image_tile), instance_map, metadata)
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "slide_id": slide_id,
                        "split": split,
                        "x": x,
                        "y": y,
                        "object_count": len(prompts),
                        "positive_pixels": int((instance_map > 0).sum()),
                        "tissue_coverage": tissue_coverage,
                        "roi_coverage": roi_coverage,
                    }
                )
                sample_ids_by_split[split].append(sample_id)
                positive_tile_count += 1

        slide_rows.append(
            {
                "slide_id": slide_id,
                "split": split,
                "image_path": str(image_path),
                "height": height,
                "width": width,
                "filtered_instance_count": len(records),
                "positive_tile_count": positive_tile_count,
            }
        )
        logging.info("Prepared %s positive tiles for slide %s", positive_tile_count, slide_id)

    manifests_dir = output_dir / "manifests"
    write_csv(sample_rows, manifests_dir / "samples.csv")
    write_csv(slide_rows, manifests_dir / "slides.csv")
    write_json(
        {
            "slide_count": len(slide_rows),
            "sample_count": len(sample_rows),
            "train_samples": len(sample_ids_by_split["train"]),
            "val_samples": len(sample_ids_by_split["val"]),
        },
        manifests_dir / "summary.json",
    )
    write_list_file(sample_ids_by_split["train"], output_dir / "train" / "list.txt")
    write_list_file(sample_ids_by_split["val"], output_dir / "val" / "list.txt")

    logging.info("Finished. Prepared %s samples across %s slides.", len(sample_rows), len(slide_rows))


if __name__ == "__main__":
    main()
