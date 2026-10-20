"""Create a slide-disjoint DPA-P2PNet dataset for intrinsic-cell detection.

The source annotations store point locations in CZI mosaic-local coordinates.
This script clips one square crop around each annotated glomerulus, rescales it
to a fixed size and transforms the three cell-point classes accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASSES = ["endothelial_cell", "mesangial_cell", "podocyte"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--padding", type=float, default=1.20)
    parser.add_argument("--minimum-side", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_splits(slide_ids: list[str], seed: int) -> dict[str, str]:
    """Use a 68/16/16 slide-level split; no crop from a held-out slide leaks."""
    shuffled = sorted(slide_ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * 0.68)
    n_val = round(n * 0.16)
    split = {}
    for slide_id in shuffled[:n_train]:
        split[slide_id] = "train"
    for slide_id in shuffled[n_train:n_train + n_val]:
        split[slide_id] = "val"
    for slide_id in shuffled[n_train + n_val:]:
        split[slide_id] = "test"
    return split


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image).squeeze()
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError("Expected a 2D/3D CZI crop, got {}".format(image.shape))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    return image[..., :3].astype(np.uint8, copy=False)


def polygon_bounds(feature: dict) -> tuple[float, float, float, float]:
    points = feature["geometry"]["coordinates"][0]
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def copy_crop(
    czi,
    mosaic_box,
    feature: dict,
    points: list[dict[str, str]],
    output_path: Path,
    image_size: int,
    padding: float,
    minimum_side: int,
) -> tuple[dict[str, list[list[float]]], dict]:
    min_x, min_y, max_x, max_y = polygon_bounds(feature)
    side = max(float(minimum_side), (max_x - min_x) * padding, (max_y - min_y) * padding)
    side = int(np.ceil(side))
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    local_x = int(round(center_x - side / 2))
    local_y = int(round(center_y - side / 2))
    local_x = max(0, min(local_x, int(mosaic_box.w) - side))
    local_y = max(0, min(local_y, int(mosaic_box.h) - side))
    if local_x < 0 or local_y < 0:
        raise ValueError("ROI crop is larger than its CZI mosaic")

    region = (int(mosaic_box.x) + local_x, int(mosaic_box.y) + local_y, side, side)
    scale_factor = image_size / side
    image = ensure_rgb(czi.read_mosaic(region=region, C=0, scale_factor=scale_factor))
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Could not write {}".format(output_path))

    annotation = {"classes": CLASSES}
    for label in CLASSES:
        annotation[label] = []
    scale = image_size / side
    for point in points:
        x, y = float(point["x"]), float(point["y"])
        if local_x <= x < local_x + side and local_y <= y < local_y + side:
            annotation[point["class_slug"]].append([
                round((x - local_x) * scale, 4), round((y - local_y) * scale, 4),
            ])
    meta = {
        "local_x": local_x,
        "local_y": local_y,
        "side_raw_px": side,
        "num_points": sum(len(annotation[label]) for label in CLASSES),
    }
    return annotation, meta


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit("Output directory is not empty; pass --overwrite to recreate it.")
        shutil.rmtree(args.output_dir)

    try:
        from aicspylibczi import CziFile
    except ImportError as exc:
        raise SystemExit("aicspylibczi is required to read CZI crops: {}".format(exc)) from exc

    slides = {row["slide_id"]: row for row in read_csv(args.export_dir / "slides.csv")}
    points_by_slide: dict[str, list[dict[str, str]]] = defaultdict(list)
    for point in read_csv(args.export_dir / "cell_points.csv"):
        points_by_slide[point["slide_id"]].append(point)
    features = json.loads((args.export_dir / "glomerulus_rois.geojson").read_text(encoding="utf-8"))["features"]
    features = [item for item in features if item["properties"]["slide_id"] in points_by_slide]
    slide_ids = sorted({item["properties"]["slide_id"] for item in features})
    splits = choose_splits(slide_ids, args.seed)

    for split in ("train", "val", "heldout_test"):
        (args.output_dir / "{}_image".format(split)).mkdir(parents=True)
        (args.output_dir / "{}_point".format(split)).mkdir(parents=True)

    rows: list[dict] = []
    czi_cache: dict[str, tuple[object, object]] = {}
    roi_numbers: Counter[str] = Counter()
    for feature in features:
        props = feature["properties"]
        slide_id = props["slide_id"]
        split = splits[slide_id]
        output_split = "heldout_test" if split == "test" else split
        if slide_id not in czi_cache:
            czi = CziFile(slides[slide_id]["czi_path"])
            czi_cache[slide_id] = (czi, czi.get_mosaic_bounding_box())
        czi, mosaic_box = czi_cache[slide_id]
        roi_numbers[slide_id] += 1
        stem = "{}_roi{:03d}".format(slide_id, roi_numbers[slide_id])
        image_path = args.output_dir / "{}_image".format(output_split) / (stem + ".jpg")
        annotation, crop_meta = copy_crop(
            czi, mosaic_box, feature, points_by_slide[slide_id], image_path,
            args.image_size, args.padding, args.minimum_side,
        )
        (args.output_dir / "{}_point".format(output_split) / (stem + ".json")).write_text(
            json.dumps(annotation, ensure_ascii=False), encoding="utf-8"
        )
        rows.append({
            "crop_id": stem,
            "split": split,
            "slide_id": slide_id,
            "case_id": props["case_id"],
            "roi_mark_id": props["mark_id"],
            "roi_label": props["label_cn"],
            **crop_meta,
            **{"{}_points".format(name): len(annotation[name]) for name in CLASSES},
        })

    # The upstream loader uses test_* as its validation directory.  The
    # slide-disjoint test split is intentionally kept separate for final use.
    for suffix in ("image", "point"):
        shutil.copytree(args.output_dir / "val_{}".format(suffix), args.output_dir / "test_{}".format(suffix))

    np.save(args.output_dir / "mean_std.npy", np.array([[0.75, 0.58, 0.70], [0.18, 0.20, 0.17]], dtype=np.float32))
    fields = ["crop_id", "split", "slide_id", "case_id", "roi_mark_id", "roi_label", "local_x", "local_y", "side_raw_px", "num_points"] + ["{}_points".format(name) for name in CLASSES]
    write_csv(args.output_dir / "manifest.csv", rows, fields)
    summary = {
        "classes": CLASSES,
        "image_size": args.image_size,
        "slide_split": {name: sorted(slide for slide, value in splits.items() if value == name) for name in ("train", "val", "test")},
        "crop_count": dict(Counter(row["split"] for row in rows)),
        "point_count": {name: sum(int(row["{}_points".format(name)]) for row in rows) for name in CLASSES},
        "notes": "train.py reads test_* as validation; held-out test_* remains separate in this export.",
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
