import argparse
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from hzy_lesion_config import GLOMERULUS_LABELS, LesionTask, resolve_lesion_tasks
from prepare_hubmap_tiles import (
    IMAGE_SUFFIXES,
    assign_slide_splits,
    ensure_uint8_rgb,
    find_image_path,
    get_feature_label,
    normalize_label,
    open_slide_array,
    point_to_tuple,
    polygon_rings_from_geometry,
    read_geojson_features,
    read_split_table,
    select_tiles,
    write_csv,
)
from prepare_hzy_lesion_tiles import annotation_path_for_slide, rasterize_task_mask, write_mask_value_cache


DEFAULT_IMAGES_DIR = "/root/datasets/HZY_HSPN_export_ds025/images"
DEFAULT_ANNOTATIONS_DIR = "/root/datasets/HZY_HSPN_export_ds025/annotations"
DEFAULT_OUTPUT_ROOT = "/root/datasets/HZY_HSPN_glomerulus_lesion_tasks"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare per-glomerulus crops for grouped HZY lesion segmentation tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Directory containing scene TIFF images")
    parser.add_argument("--annotations-dir", default=DEFAULT_ANNOTATIONS_DIR, help="Directory containing scene JSON annotations")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for glomerulus-crop lesion datasets")
    parser.add_argument("--tasks", nargs="*", default=[],
                        help="Optional task slugs/names to prepare: proliferation, crescent, crescent_binary, other_lesions")
    parser.add_argument("--glomerulus-labels", nargs="*", default=[],
                        help="Optional glomerulus labels to crop. Defaults to configured non-discarded and discarded labels")
    parser.add_argument("--annotation-json-suffix", default=".json", help="Suffix for scene annotation JSON files")
    parser.add_argument("--crop-size", type=int, default=512, help="Saved crop size for images and masks")
    parser.add_argument("--margin", type=int, default=96, help="Source-pixel margin around each glomerulus bbox")
    parser.add_argument("--min-source-crop-size", type=int, default=384,
                        help="Minimum source crop side before resizing")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation slide ratio")
    parser.add_argument("--split-csv", default="", help="Optional slide split CSV with columns slide_id,split")
    parser.add_argument("--min-positive-pixels", type=int, default=8,
                        help="Minimum non-background lesion pixels after resizing to mark a crop positive")
    parser.add_argument("--negative-ratio", type=float, default=1.0,
                        help="How many negative glomerulus crops to keep per positive crop")
    parser.add_argument("--max-background-crops-per-slide", type=int, default=40,
                        help="Negative crop cap for slides without positive crops")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits and negative sampling")
    parser.add_argument("--limit-slides", type=int, default=0, help="Optional slide cap for quick dry runs")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Remove an existing task output directory before writing")
    return parser.parse_args()


def validate_args(args):
    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    if args.crop_size <= 0:
        raise ValueError("crop-size must be a positive integer.")
    if args.margin < 0:
        raise ValueError("margin must be non-negative.")
    if args.min_source_crop_size <= 0:
        raise ValueError("min-source-crop-size must be a positive integer.")
    if args.val_ratio < 0:
        raise ValueError("val-ratio must be non-negative.")
    if args.min_positive_pixels < 0:
        raise ValueError("min-positive-pixels must be non-negative.")
    if args.negative_ratio < 0:
        raise ValueError("negative-ratio must be non-negative.")
    if args.max_background_crops_per_slide < 0:
        raise ValueError("max-background-crops-per-slide must be non-negative.")
    if not images_dir.exists():
        raise FileNotFoundError("Images directory does not exist: {}".format(images_dir))
    if not annotations_dir.exists():
        raise FileNotFoundError("Annotations directory does not exist: {}".format(annotations_dir))


def discover_slide_ids(images_dir: Path, limit_slides: int) -> List[str]:
    slide_ids = sorted(path.stem for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if limit_slides > 0:
        return slide_ids[:limit_slides]
    return slide_ids


def resolve_glomerulus_label_set(requested_labels: Iterable[str]):
    labels = list(requested_labels) or [item.label for item in GLOMERULUS_LABELS]
    return {normalize_label(label) for label in labels}


def geometry_points(geometry: Dict[str, object]):
    for polygon in polygon_rings_from_geometry(geometry):
        for ring in polygon:
            for point in ring:
                yield point_to_tuple(point)


def geometry_bbox(geometry: Dict[str, object]) -> Optional[Tuple[int, int, int, int]]:
    points = list(geometry_points(geometry))
    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def crop_box_from_bbox(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin: int,
    min_source_crop_size: int,
) -> Tuple[int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    width = max(1, x_max - x_min + 1)
    height = max(1, y_max - y_min + 1)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    side = int(round(max(width, height) + 2 * margin))
    side = max(side, min_source_crop_size, 1)

    crop_x = int(round(center_x - side / 2.0))
    crop_y = int(round(center_y - side / 2.0))
    return crop_x, crop_y, side


def crop_with_padding(array: np.ndarray, crop_x: int, crop_y: int, side: int, fill_value: int):
    height, width = array.shape[:2]
    if array.ndim == 2:
        output = np.full((side, side), fill_value, dtype=array.dtype)
    else:
        output = np.full((side, side, array.shape[2]), fill_value, dtype=array.dtype)

    src_x0 = max(crop_x, 0)
    src_y0 = max(crop_y, 0)
    src_x1 = min(crop_x + side, width)
    src_y1 = min(crop_y + side, height)

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return output

    dst_x0 = src_x0 - crop_x
    dst_y0 = src_y0 - crop_y
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    output[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return output


def resize_crop_pair(image_crop: np.ndarray, mask_crop: np.ndarray, crop_size: int):
    image = Image.fromarray(ensure_uint8_rgb(image_crop)).resize((crop_size, crop_size), resample=Image.BICUBIC)
    mask = Image.fromarray(mask_crop.astype(np.uint8)).resize((crop_size, crop_size), resample=Image.NEAREST)
    return np.asarray(image), np.asarray(mask).astype(np.uint8)


def collect_glomerulus_records(
    slide_array: np.ndarray,
    task_mask: np.ndarray,
    features: Sequence[Dict[str, object]],
    slide_id: str,
    split: str,
    glomerulus_label_set,
    args,
    num_classes: int,
):
    height, width = task_mask.shape
    positive_crops = []
    negative_crops = []
    glomerulus_count = 0
    skipped_without_bbox = 0

    for feature in features:
        label = get_feature_label(feature)
        if label not in glomerulus_label_set:
            continue

        bbox = geometry_bbox(feature.get("geometry") or {})
        if bbox is None:
            skipped_without_bbox += 1
            continue

        crop_x, crop_y, source_side = crop_box_from_bbox(
            bbox=bbox,
            image_width=width,
            image_height=height,
            margin=args.margin,
            min_source_crop_size=args.min_source_crop_size,
        )
        mask_crop = crop_with_padding(task_mask, crop_x, crop_y, source_side, fill_value=0)
        _, resized_mask = resize_crop_pair(
            np.zeros((source_side, source_side, 3), dtype=np.uint8),
            mask_crop,
            args.crop_size,
        )

        positive_pixels = int((resized_mask > 0).sum())
        class_pixels = {
            str(class_id): int((resized_mask == class_id).sum())
            for class_id in range(1, num_classes)
        }
        glomerulus_count += 1
        record = {
            "slide_id": slide_id,
            "split": split,
            "glomerulus_index": glomerulus_count,
            "glomerulus_label": label,
            "x": crop_x,
            "y": crop_y,
            "bbox_x_min": bbox[0],
            "bbox_y_min": bbox[1],
            "bbox_x_max": bbox[2],
            "bbox_y_max": bbox[3],
            "crop_x": crop_x,
            "crop_y": crop_y,
            "source_crop_size": source_side,
            "saved_size": args.crop_size,
            "positive_pixels": positive_pixels,
            "mask_coverage": round(positive_pixels / float(resized_mask.size), 6),
            "class_pixels": class_pixels,
            "is_positive": int(positive_pixels >= args.min_positive_pixels),
        }
        if record["is_positive"]:
            positive_crops.append(record)
        else:
            negative_crops.append(record)

    return positive_crops, negative_crops, glomerulus_count, skipped_without_bbox


def save_glomerulus_crops(
    slide_array: np.ndarray,
    task_mask: np.ndarray,
    selected_crops,
    output_dir: Path,
    crop_size: int,
):
    rows = []
    for record in selected_crops:
        crop_x = int(record["crop_x"])
        crop_y = int(record["crop_y"])
        source_side = int(record["source_crop_size"])
        image_crop = crop_with_padding(slide_array, crop_x, crop_y, source_side, fill_value=255)
        mask_crop = crop_with_padding(task_mask, crop_x, crop_y, source_side, fill_value=0)
        image_tile, mask_tile = resize_crop_pair(image_crop, mask_crop, crop_size)

        crop_name = "{slide}_g{index:04d}_{x}_{y}".format(
            slide=record["slide_id"],
            index=int(record["glomerulus_index"]),
            x=crop_x,
            y=crop_y,
        )
        image_path = output_dir / record["split"] / "images" / "{}.jpg".format(crop_name)
        mask_path = output_dir / record["split"] / "masks" / "{}.png".format(crop_name)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(ensure_uint8_rgb(image_tile)).save(image_path, quality=95)
        Image.fromarray(mask_tile.astype(np.uint8)).save(mask_path)

        row = dict(record)
        row["crop_id"] = crop_name
        row["class_pixels"] = json.dumps(record["class_pixels"], ensure_ascii=False)
        row["image_path"] = str(image_path.relative_to(output_dir))
        row["mask_path"] = str(mask_path.relative_to(output_dir))
        rows.append(row)
    return rows


def ensure_split_dirs(output_dir: Path):
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def prepare_one_task(args, task: LesionTask, slide_ids: Sequence[str], split_map: Dict[str, str], glomerulus_label_set):
    output_dir = Path(args.output_root) / task.slug
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError("Output directory already exists. Re-run with --overwrite: {}".format(output_dir))
        shutil.rmtree(output_dir)
    ensure_split_dirs(output_dir)

    assignments = assign_slide_splits(slide_ids, split_map, args.val_ratio, args.seed)
    rng = random.Random(args.seed)
    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    crop_rows = []
    slide_rows = []
    global_feature_counts = Counter()
    global_class_pixels = Counter()
    total_overlap_pixels = 0
    total_glomeruli = 0

    for slide_id in tqdm(slide_ids, desc="Task {} glomeruli".format(task.slug), unit="slide"):
        split = assignments.get(slide_id, "train")
        image_path = find_image_path(images_dir, slide_id)
        annotation_path = annotation_path_for_slide(annotations_dir, slide_id, args.annotation_json_suffix)

        slide_array = open_slide_array(image_path)
        features = read_geojson_features(annotation_path)
        task_mask, mask_stats = rasterize_task_mask(slide_array.shape, features, task)

        global_feature_counts.update(mask_stats["feature_counts"])
        global_class_pixels.update(mask_stats["class_pixel_counts"])
        total_overlap_pixels += int(mask_stats["overlap_pixels"])

        positive_crops, negative_crops, glomerulus_count, skipped_without_bbox = collect_glomerulus_records(
            slide_array=slide_array,
            task_mask=task_mask,
            features=features,
            slide_id=slide_id,
            split=split,
            glomerulus_label_set=glomerulus_label_set,
            args=args,
            num_classes=task.num_classes,
        )
        selected_crops = select_tiles(
            positive_tiles=positive_crops,
            negative_tiles=negative_crops,
            rng=rng,
            negative_ratio=args.negative_ratio,
            max_background_tiles=args.max_background_crops_per_slide,
        )
        rows = save_glomerulus_crops(slide_array, task_mask, selected_crops, output_dir, args.crop_size)
        crop_rows.extend(rows)
        total_glomeruli += glomerulus_count
        slide_rows.append(
            {
                "slide_id": slide_id,
                "split": split,
                "width": int(slide_array.shape[1]),
                "height": int(slide_array.shape[0]),
                "glomeruli": glomerulus_count,
                "positive_crops": len(positive_crops),
                "negative_crops_kept": sum(1 for row in rows if not row["is_positive"]),
                "total_crops_kept": len(rows),
                "skipped_without_bbox": skipped_without_bbox,
                "feature_counts": json.dumps(mask_stats["feature_counts"], ensure_ascii=False),
                "class_pixel_counts": json.dumps(mask_stats["class_pixel_counts"], ensure_ascii=False),
                "overlap_pixels": int(mask_stats["overlap_pixels"]),
            }
        )
        del slide_array
        del task_mask

    manifests_dir = output_dir / "manifests"
    write_csv(crop_rows, manifests_dir / "crops.csv")
    write_csv(slide_rows, manifests_dir / "slides.csv")

    train_count = sum(1 for row in crop_rows if row.get("split") == "train")
    val_count = sum(1 for row in crop_rows if row.get("split") == "val")
    write_mask_value_cache(output_dir / "train" / "masks", train_count, task.num_classes)
    write_mask_value_cache(output_dir / "val" / "masks", val_count, task.num_classes)

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
        "glomeruli": total_glomeruli,
        "train_crops": train_count,
        "val_crops": val_count,
        "train_tiles": train_count,
        "val_tiles": val_count,
        "positive_crops": sum(int(row.get("is_positive", 0)) for row in crop_rows),
        "positive_tiles": sum(int(row.get("is_positive", 0)) for row in crop_rows),
        "negative_crops": sum(1 for row in crop_rows if not row.get("is_positive")),
        "crop_size": args.crop_size,
        "margin": args.margin,
        "min_source_crop_size": args.min_source_crop_size,
        "negative_ratio": args.negative_ratio,
    }
    with (manifests_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if train_count == 0 or val_count == 0:
        logging.warning(
            "Task %s produced train=%s and val=%s crops. Training requires non-empty train and val splits.",
            task.slug,
            train_count,
            val_count,
        )
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
    glomerulus_label_set = resolve_glomerulus_label_set(args.glomerulus_labels)

    index_rows = []
    for task in tasks:
        logging.info("Preparing glomerulus crops for task %s (%s)", task.name, task.slug)
        summary = prepare_one_task(args, task, slide_ids, split_map, glomerulus_label_set)
        index_rows.append(
            {
                "task_name": task.name,
                "task_slug": task.slug,
                "num_classes": task.num_classes,
                "output_dir": str(Path(args.output_root) / task.slug),
                "glomeruli": summary["glomeruli"],
                "train_crops": summary["train_crops"],
                "val_crops": summary["val_crops"],
                "positive_crops": summary["positive_crops"],
                "negative_crops": summary["negative_crops"],
                "overlap_pixels": summary["overlap_pixels"],
            }
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(index_rows, output_root / "glomerulus_lesion_tasks.csv")
    with (output_root / "glomerulus_lesion_tasks.json").open("w", encoding="utf-8") as handle:
        json.dump(index_rows, handle, ensure_ascii=False, indent=2)

    print("Wrote glomerulus-crop task index:", output_root / "glomerulus_lesion_tasks.csv")


if __name__ == "__main__":
    main()
