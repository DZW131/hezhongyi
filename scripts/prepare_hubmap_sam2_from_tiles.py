import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hubmap_sam2.dataset import (
    PREPARED_MASK_NAME,
    infer_mask_format_from_cache,
    link_or_copy_file,
    load_mask_array,
    remap_mask_to_instance_map,
    save_palette_png,
    write_csv,
    write_json,
)
from hubmap_sam2.prompts import instance_prompts_from_instance_map


LINK_MODES = ("auto", "hardlink", "symlink", "copy")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adapt an existing HuBMAP tile dataset into the SAM2 prepared dataset format without re-reading WSIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=str, required=True, help="Existing tile dataset root, for example /root/datasets/HuBMAP_tiles_v2.")
    parser.add_argument("--output-dir", type=str, required=True, help="Prepared SAM2 dataset root to write.")
    parser.add_argument("--tiles-csv", type=str, default="", help="Optional override for manifests/tiles.csv.")
    parser.add_argument("--slides-csv", type=str, default="", help="Optional override for manifests/slides.csv.")
    parser.add_argument("--mask-format", choices=("auto", "binary", "instance"), default="auto", help="How to interpret source masks before converting them to SAM2 instance maps.")
    parser.add_argument("--min-instance-area", type=int, default=32, help="Minimum connected-component area kept as a SAM2 instance.")
    parser.add_argument("--max-instances-per-tile", type=int, default=254, help="Maximum number of instances stored in one palette PNG.")
    parser.add_argument("--min-positive-pixels", type=int, default=64, help="Minimum positive pixels required to keep an adapted tile.")
    parser.add_argument("--link-mode", choices=LINK_MODES, default="auto", help="How to materialize source images into the SAM2 dataset.")
    parser.add_argument("--limit-tiles", type=int, default=0, help="Optional limit for smoke tests.")
    parser.add_argument("--include-negative", action="store_true", help="Also adapt negative tiles. This is mainly for inspection and is not recommended for SAM2 training.")
    return parser.parse_args()


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_list_file(sample_ids: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            handle.write(f"{sample_id}\n")


def build_metadata(row: Dict[str, str], image_path: Path, mask_path: Path, instance_map, detected_mask_format: str) -> Dict[str, object]:
    sample_id = str(row["tile_id"])
    tile_size = int(float(row.get("saved_size") or row.get("tile_size") or 0))
    x = int(float(row.get("x") or 0))
    y = int(float(row.get("y") or 0))

    prompts = instance_prompts_from_instance_map(instance_map)
    metadata = {
        "sample_id": sample_id,
        "slide_id": str(row.get("slide_id", "")),
        "split": str(row.get("split", "")),
        "tile_origin_xy": [x, y],
        "source_tile_size": tile_size,
        "saved_image_shape": [int(instance_map.shape[0]), int(instance_map.shape[1])],
        "actual_hw_before_padding": [int(instance_map.shape[0]), int(instance_map.shape[1])],
        "source_image_path": str(image_path),
        "source_mask_path": str(mask_path),
        "source_mask_format": detected_mask_format,
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
    return metadata


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    tiles_csv = Path(args.tiles_csv) if args.tiles_csv else source_root / "manifests" / "tiles.csv"
    slides_csv = Path(args.slides_csv) if args.slides_csv else source_root / "manifests" / "slides.csv"

    tile_rows = read_csv_rows(tiles_csv)
    if args.limit_tiles > 0:
        tile_rows = tile_rows[: args.limit_tiles]
    source_slide_rows = read_csv_rows(slides_csv) if slides_csv.exists() else []
    source_slide_lookup = {str(row.get("slide_id", "")): row for row in source_slide_rows}

    inferred_mask_formats = {}
    for split in ("train", "val"):
        cache_format = infer_mask_format_from_cache(source_root / split / "masks")
        if cache_format:
            inferred_mask_formats[split] = cache_format
            logging.info("Detected %s masks for split %s from cache", cache_format, split)

    sample_rows = []
    slide_counts = defaultdict(lambda: {"samples": 0, "positive_pixels": 0})
    sample_ids_by_split = {"train": [], "val": []}
    link_modes_used = defaultdict(int)
    skipped_negative_tiles = 0

    for index, row in enumerate(tile_rows, start=1):
        split = str(row.get("split", "")).strip().lower()
        if split not in sample_ids_by_split:
            logging.warning("Skipping tile %s because split '%s' is unsupported.", row.get("tile_id"), split)
            continue

        is_positive = parse_bool(row.get("is_positive", "1"))
        if not args.include_negative and not is_positive:
            skipped_negative_tiles += 1
            continue

        sample_id = str(row["tile_id"])
        image_path = source_root / row["image_path"]
        mask_path = source_root / row["mask_path"]
        if not image_path.exists() or not mask_path.exists():
            logging.warning("Skipping tile %s because source files are missing.", sample_id)
            continue

        mask_array = load_mask_array(mask_path)
        mask_format = args.mask_format
        if mask_format == "auto":
            mask_format = inferred_mask_formats.get(split, "auto")
        instance_map, detected_mask_format = remap_mask_to_instance_map(
            mask_array,
            mask_format=mask_format,
            min_instance_area=args.min_instance_area,
            max_instances=args.max_instances_per_tile,
        )
        positive_pixels = int((instance_map > 0).sum())
        if positive_pixels < args.min_positive_pixels:
            continue

        metadata = build_metadata(row, image_path, mask_path, instance_map, detected_mask_format)
        if not metadata["objects"]:
            continue

        source_suffix = image_path.suffix.lower()
        target_image_path = output_dir / split / "JPEGImages" / sample_id / f"00000{source_suffix}"
        target_mask_path = output_dir / split / "Annotations" / sample_id / PREPARED_MASK_NAME
        target_metadata_path = output_dir / split / "Metadata" / f"{sample_id}.json"

        target_mask_path.parent.mkdir(parents=True, exist_ok=True)
        target_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        materialized_mode = link_or_copy_file(image_path, target_image_path, mode=args.link_mode)
        link_modes_used[materialized_mode] += 1
        save_palette_png(instance_map, target_mask_path)
        write_json(metadata, target_metadata_path)

        object_count = len(metadata["objects"])
        sample_rows.append(
            {
                "sample_id": sample_id,
                "slide_id": row.get("slide_id", ""),
                "split": split,
                "x": int(float(row.get("x") or 0)),
                "y": int(float(row.get("y") or 0)),
                "object_count": object_count,
                "positive_pixels": positive_pixels,
                "source_mask_format": detected_mask_format,
                "source_image_path": row["image_path"],
                "source_mask_path": row["mask_path"],
            }
        )
        sample_ids_by_split[split].append(sample_id)

        slide_id = str(row.get("slide_id", ""))
        slide_counts[(slide_id, split)]["samples"] += 1
        slide_counts[(slide_id, split)]["positive_pixels"] += positive_pixels

        if index % 500 == 0:
            logging.info("Adapted %s source tiles", index)

    slide_rows = []
    for (slide_id, split), counts in sorted(slide_counts.items(), key=lambda item: (item[0][1], item[0][0])):
        source_row = source_slide_lookup.get(slide_id, {})
        slide_rows.append(
            {
                "slide_id": slide_id,
                "split": split,
                "annotation_format": source_row.get("annotation_format", ""),
                "width": source_row.get("width", ""),
                "height": source_row.get("height", ""),
                "source_positive_tiles": source_row.get("positive_tiles", ""),
                "source_negative_tiles_kept": source_row.get("negative_tiles_kept", ""),
                "source_total_tiles_kept": source_row.get("total_tiles_kept", ""),
                "adapted_samples": counts["samples"],
                "adapted_positive_pixels": counts["positive_pixels"],
            }
        )

    manifests_dir = output_dir / "manifests"
    write_csv(sample_rows, manifests_dir / "samples.csv")
    write_csv(slide_rows, manifests_dir / "slides.csv")
    write_json(
        {
            "source_root": str(source_root),
            "source_tiles_csv": str(tiles_csv),
            "source_slides_csv": str(slides_csv) if slides_csv.exists() else None,
            "train_samples": len(sample_ids_by_split["train"]),
            "val_samples": len(sample_ids_by_split["val"]),
            "sample_count": len(sample_rows),
            "slide_count": len(slide_rows),
            "skipped_negative_tiles": skipped_negative_tiles,
            "min_instance_area": args.min_instance_area,
            "max_instances_per_tile": args.max_instances_per_tile,
            "min_positive_pixels": args.min_positive_pixels,
            "mask_format": args.mask_format,
            "link_mode_requested": args.link_mode,
            "link_modes_used": dict(link_modes_used),
        },
        manifests_dir / "summary.json",
    )
    write_list_file(sample_ids_by_split["train"], output_dir / "train" / "list.txt")
    write_list_file(sample_ids_by_split["val"], output_dir / "val" / "list.txt")

    logging.info(
        "Finished adapting %s samples from %s. train=%s val=%s",
        len(sample_rows),
        source_root,
        len(sample_ids_by_split["train"]),
        len(sample_ids_by_split["val"]),
    )


if __name__ == "__main__":
    main()
