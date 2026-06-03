import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from hzy_lesion_config import LesionTask, resolve_lesion_tasks
from prepare_hubmap_tiles import (
    IMAGE_SUFFIXES,
    assign_slide_splits,
    ensure_uint8_rgb,
    estimate_tissue_coverage,
    find_image_path,
    get_feature_label,
    open_slide_array,
    pad_image_tile,
    pad_mask_tile,
    point_to_tuple,
    polygon_rings_from_geometry,
    read_geojson_features,
    read_split_table,
    select_tiles,
    write_csv,
)


DEFAULT_IMAGES_DIR = "/root/datasets/HZY_HSPN_export_ds025/images"
DEFAULT_ANNOTATIONS_DIR = "/root/datasets/HZY_HSPN_export_ds025/annotations"
DEFAULT_OUTPUT_ROOT = "/root/datasets/HZY_HSPN_lesion_tasks"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare grouped multi-class tile datasets for HZY lesion segmentation tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Directory containing scene TIFF images")
    parser.add_argument("--annotations-dir", default=DEFAULT_ANNOTATIONS_DIR, help="Directory containing scene JSON annotations")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for grouped lesion tile datasets")
    parser.add_argument("--tasks", nargs="*", default=[],
                        help="Optional task slugs/names to prepare: proliferation, crescent, crescent_binary, other_lesions")
    parser.add_argument("--annotation-json-suffix", default=".json", help="Suffix for scene annotation JSON files")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for lesion crops")
    parser.add_argument("--stride", type=int, default=512, help="Sliding-window stride")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation slide ratio")
    parser.add_argument("--split-csv", default="", help="Optional slide split CSV with columns slide_id,split")
    parser.add_argument("--min-tissue-coverage", type=float, default=0.05, help="Minimum tissue coverage to keep a tile")
    parser.add_argument("--white-threshold", type=float, default=230.0,
                        help="Mean intensity threshold used to estimate white background pixels")
    parser.add_argument("--min-positive-pixels", type=int, default=16,
                        help="Minimum non-background task pixels required to keep a positive lesion tile")
    parser.add_argument("--negative-ratio", type=float, default=3.0,
                        help="How many negative tiles to keep per positive tile")
    parser.add_argument("--max-background-tiles-per-slide", type=int, default=50,
                        help="Negative tile cap for slides without positive tiles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for slide splits and negative sampling")
    parser.add_argument("--downsample", type=float, default=1.0, help="Optional downsample when saving tiles")
    parser.add_argument("--limit-slides", type=int, default=0, help="Optional slide cap for quick dry runs")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Remove an existing task output directory before writing")
    return parser.parse_args()


def validate_args(args):
    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)

    if args.tile_size <= 0 or args.stride <= 0:
        raise ValueError("tile-size and stride must both be positive integers.")
    if not (0 < args.downsample <= 1.0):
        raise ValueError("downsample must be in the interval (0, 1].")
    if args.val_ratio < 0:
        raise ValueError("val-ratio must be non-negative.")
    if args.min_tissue_coverage < 0:
        raise ValueError("min-tissue-coverage must be non-negative.")
    if args.min_positive_pixels < 0:
        raise ValueError("min-positive-pixels must be non-negative.")
    if args.negative_ratio < 0:
        raise ValueError("negative-ratio must be non-negative.")
    if args.max_background_tiles_per_slide < 0:
        raise ValueError("max-background-tiles-per-slide must be non-negative.")
    if not images_dir.exists():
        raise FileNotFoundError("Images directory does not exist: {}".format(images_dir))
    if not annotations_dir.exists():
        raise FileNotFoundError("Annotations directory does not exist: {}".format(annotations_dir))


def discover_slide_ids(images_dir: Path, limit_slides: int) -> List[str]:
    slide_ids = sorted(path.stem for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit_slides > 0:
        return slide_ids[:limit_slides]
    return slide_ids


def annotation_path_for_slide(annotations_dir: Path, slide_id: str, suffix: str) -> Path:
    path = annotations_dir / "{}{}".format(slide_id, suffix)
    if not path.exists():
        raise FileNotFoundError("Could not find annotation JSON for slide {}: {}".format(slide_id, path))
    return path


def rasterize_task_mask(image_shape, features: Sequence[Dict[str, object]], task: LesionTask):
    height, width = image_shape[:2]
    task_labels = {label.label for label in task.labels}
    class_map = task.class_map
    mask = np.zeros((height, width), dtype=np.uint8)
    occupied = np.zeros((height, width), dtype=bool)
    feature_counts = Counter()
    class_pixel_counts = Counter()
    overlap_pixels = 0

    features_by_label: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for feature in features:
        label = get_feature_label(feature)
        if label in task_labels:
            features_by_label[label].append(feature)
            feature_counts[label] += 1

    for label in [item.label for item in task.labels]:
        class_id = class_map[label]
        class_image = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(class_image)

        for feature in features_by_label.get(label, []):
            geometry = feature.get("geometry") or {}
            for polygon in polygon_rings_from_geometry(geometry):
                if not polygon:
                    continue

                outer_ring = polygon[0] if polygon else []
                if len(outer_ring) >= 3:
                    draw.polygon([point_to_tuple(point) for point in outer_ring], outline=1, fill=1)

                for hole in polygon[1:]:
                    if len(hole) >= 3:
                        draw.polygon([point_to_tuple(point) for point in hole], outline=0, fill=0)

        class_mask = np.asarray(class_image) > 0
        overlap_pixels += int(np.logical_and(occupied, class_mask).sum())
        mask[class_mask] = class_id
        occupied |= class_mask
        class_pixel_counts[label] = int(class_mask.sum())

    return mask, {
        "feature_counts": dict(feature_counts),
        "class_pixel_counts": dict(class_pixel_counts),
        "overlap_pixels": overlap_pixels,
    }


def generate_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def extract_crop(array: np.ndarray, x: int, y: int, tile_size: int):
    return array[y:y + tile_size, x:x + tile_size]


def resize_multiclass_tile_pair(image_tile: np.ndarray, mask_tile: np.ndarray, downsample: float):
    if downsample == 1.0:
        return image_tile, mask_tile.astype(np.uint8)

    output_size = max(1, int(round(image_tile.shape[0] * downsample)))
    image = Image.fromarray(image_tile).resize((output_size, output_size), resample=Image.BICUBIC)
    mask = Image.fromarray(mask_tile.astype(np.uint8)).resize((output_size, output_size), resample=Image.NEAREST)
    return np.asarray(image), np.asarray(mask).astype(np.uint8)


def collect_tile_records(slide_array, mask_array, slide_id: str, split: str, args, num_classes: int):
    height, width = mask_array.shape
    positive_tiles = []
    negative_tiles = []
    y_positions = generate_positions(height, args.tile_size, args.stride)
    x_positions = generate_positions(width, args.tile_size, args.stride)

    for y in y_positions:
        for x in x_positions:
            image_crop = extract_crop(slide_array, x, y, args.tile_size)
            mask_crop = extract_crop(mask_array, x, y, args.tile_size)

            tissue_coverage = estimate_tissue_coverage(image_crop, args.white_threshold)
            if tissue_coverage < args.min_tissue_coverage:
                continue

            positive_pixels = int((mask_crop > 0).sum())
            mask_coverage = positive_pixels / float(mask_crop.size)
            class_pixels = {
                str(class_id): int((mask_crop == class_id).sum())
                for class_id in range(1, num_classes)
            }
            record = {
                "slide_id": slide_id,
                "split": split,
                "x": x,
                "y": y,
                "width": int(image_crop.shape[1]),
                "height": int(image_crop.shape[0]),
                "tissue_coverage": round(tissue_coverage, 6),
                "mask_coverage": round(mask_coverage, 6),
                "positive_pixels": positive_pixels,
                "class_pixels": class_pixels,
                "is_positive": int(positive_pixels >= args.min_positive_pixels),
            }
            if record["is_positive"]:
                positive_tiles.append(record)
            else:
                negative_tiles.append(record)

    return positive_tiles, negative_tiles


def save_multiclass_tiles(slide_array, mask_array, selected_tiles, output_dir: Path, tile_size: int, downsample: float):
    rows = []
    for record in selected_tiles:
        x = record["x"]
        y = record["y"]

        image_crop = extract_crop(slide_array, x, y, tile_size)
        mask_crop = extract_crop(mask_array, x, y, tile_size)
        image_tile = pad_image_tile(image_crop, tile_size)
        mask_tile = pad_mask_tile(mask_crop, tile_size)
        image_tile, mask_tile = resize_multiclass_tile_pair(image_tile, mask_tile, downsample)

        tile_name = "{slide}_{x}_{y}".format(slide=record["slide_id"], x=x, y=y)
        image_path = output_dir / record["split"] / "images" / "{}.jpg".format(tile_name)
        mask_path = output_dir / record["split"] / "masks" / "{}.png".format(tile_name)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(ensure_uint8_rgb(image_tile)).save(image_path, quality=95)
        Image.fromarray(mask_tile.astype(np.uint8)).save(mask_path)

        row = dict(record)
        row["tile_id"] = tile_name
        row["tile_size"] = tile_size
        row["saved_size"] = image_tile.shape[0]
        row["class_pixels"] = json.dumps(record["class_pixels"], ensure_ascii=False)
        row["image_path"] = str(image_path.relative_to(output_dir))
        row["mask_path"] = str(mask_path.relative_to(output_dir))
        rows.append(row)

    return rows


def write_mask_value_cache(mask_dir: Path, num_ids: int, num_classes: int):
    mask_dir.mkdir(parents=True, exist_ok=True)
    cache_path = mask_dir / ".mask_values_cache.json"
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mask_suffix": "",
                "num_ids": num_ids,
                "mask_values": list(range(num_classes)),
            },
            handle,
            indent=2,
        )


def ensure_split_dirs(output_dir: Path):
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def prepare_one_task(args, task: LesionTask, slide_ids: Sequence[str], split_map: Dict[str, str]):
    output_dir = Path(args.output_root) / task.slug
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError("Output directory already exists. Re-run with --overwrite: {}".format(output_dir))
        shutil.rmtree(output_dir)
    ensure_split_dirs(output_dir)

    assignments = assign_slide_splits(slide_ids, split_map, args.val_ratio, args.seed)
    tile_rows = []
    slide_rows = []
    global_feature_counts = Counter()
    global_class_pixels = Counter()
    total_overlap_pixels = 0

    rng = __import__("random").Random(args.seed)
    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)

    for slide_id in tqdm(slide_ids, desc="Task {} slides".format(task.slug), unit="slide"):
        split = assignments.get(slide_id, "train")
        image_path = find_image_path(images_dir, slide_id)
        annotation_path = annotation_path_for_slide(annotations_dir, slide_id, args.annotation_json_suffix)

        slide_array = open_slide_array(image_path)
        features = read_geojson_features(annotation_path)
        mask_array, mask_stats = rasterize_task_mask(slide_array.shape, features, task)

        global_feature_counts.update(mask_stats["feature_counts"])
        global_class_pixels.update(mask_stats["class_pixel_counts"])
        total_overlap_pixels += int(mask_stats["overlap_pixels"])

        positive_tiles, negative_tiles = collect_tile_records(
            slide_array=slide_array,
            mask_array=mask_array,
            slide_id=slide_id,
            split=split,
            args=args,
            num_classes=task.num_classes,
        )
        selected_tiles = select_tiles(
            positive_tiles=positive_tiles,
            negative_tiles=negative_tiles,
            rng=rng,
            negative_ratio=args.negative_ratio,
            max_background_tiles=args.max_background_tiles_per_slide,
        )
        rows = save_multiclass_tiles(slide_array, mask_array, selected_tiles, output_dir, args.tile_size, args.downsample)
        tile_rows.extend(rows)
        slide_rows.append(
            {
                "slide_id": slide_id,
                "split": split,
                "width": int(slide_array.shape[1]),
                "height": int(slide_array.shape[0]),
                "positive_tiles": len(positive_tiles),
                "negative_tiles_kept": sum(1 for row in rows if not row["is_positive"]),
                "total_tiles_kept": len(rows),
                "feature_counts": json.dumps(mask_stats["feature_counts"], ensure_ascii=False),
                "class_pixel_counts": json.dumps(mask_stats["class_pixel_counts"], ensure_ascii=False),
                "overlap_pixels": int(mask_stats["overlap_pixels"]),
                "status": "processed",
            }
        )

        del slide_array
        del mask_array

    manifests_dir = output_dir / "manifests"
    write_csv(tile_rows, manifests_dir / "tiles.csv")
    write_csv(slide_rows, manifests_dir / "slides.csv")

    train_count = sum(1 for row in tile_rows if row.get("split") == "train")
    val_count = sum(1 for row in tile_rows if row.get("split") == "val")
    write_mask_value_cache(output_dir / "train" / "masks", train_count, task.num_classes)
    write_mask_value_cache(output_dir / "val" / "masks", val_count, task.num_classes)
    if train_count == 0 or val_count == 0:
        logging.warning(
            "Task %s produced train=%s and val=%s tiles. Training requires non-empty train and val splits.",
            task.slug,
            train_count,
            val_count,
        )

    summary = {
        "task_name": task.name,
        "task_slug": task.slug,
        "num_classes": task.num_classes,
        "class_mapping": task.class_mapping,
        "label_counts_expected": {label.label: label.count for label in task.labels},
        "feature_counts_observed": dict(global_feature_counts),
        "class_pixel_counts": dict(global_class_pixels),
        "overlap_pixels": int(total_overlap_pixels),
        "slides": len(slide_ids),
        "train_tiles": train_count,
        "val_tiles": val_count,
        "positive_tiles": sum(int(row.get("is_positive", 0)) for row in tile_rows),
        "negative_tiles": sum(1 for row in tile_rows if not row.get("is_positive")),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "downsample": args.downsample,
    }
    with (manifests_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if total_overlap_pixels > 0:
        logging.warning(
            "Task %s has %s cross-class overlap pixels. Later class IDs overwrite earlier class IDs.",
            task.slug,
            total_overlap_pixels,
        )
    return summary


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_args(args)
    images_dir = Path(args.images_dir)
    split_csv = Path(args.split_csv) if args.split_csv else None
    split_map = read_split_table(split_csv)
    slide_ids = discover_slide_ids(images_dir, args.limit_slides)
    tasks = resolve_lesion_tasks(args.tasks)

    index_rows = []
    for task in tasks:
        logging.info("Preparing task %s (%s)", task.name, task.slug)
        summary = prepare_one_task(args, task, slide_ids, split_map)
        index_rows.append(
            {
                "task_name": task.name,
                "task_slug": task.slug,
                "num_classes": task.num_classes,
                "output_dir": str(Path(args.output_root) / task.slug),
                "train_tiles": summary["train_tiles"],
                "val_tiles": summary["val_tiles"],
                "positive_tiles": summary["positive_tiles"],
                "negative_tiles": summary["negative_tiles"],
                "overlap_pixels": summary["overlap_pixels"],
            }
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(index_rows, output_root / "lesion_tasks.csv")
    with (output_root / "lesion_tasks.json").open("w", encoding="utf-8") as handle:
        json.dump(index_rows, handle, ensure_ascii=False, indent=2)

    print("Wrote task index:", output_root / "lesion_tasks.csv")


if __name__ == "__main__":
    main()
