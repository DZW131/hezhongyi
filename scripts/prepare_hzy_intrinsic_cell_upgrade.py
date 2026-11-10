"""Prepare the 43-case cell upgrade while preserving legacy held-out cases."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

import cv2
import numpy as np

from prepare_hzy_intrinsic_cell_p2pnet import CLASSES, copy_crop, polygon_bounds


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_key(value):
    match = re.search(r"(20\d{2})0*(\d+)", str(value))
    if not match:
        raise ValueError("Cannot derive a case key from " + str(value))
    return match.group(1) + str(int(match.group(2)))


def assign_splits(slide_ids, legacy, diseases, seed):
    assignments = {}
    legacy_map = {case_key(slide): split for split, slides in legacy.items() for slide in slides}
    if sum(len(x) for x in legacy.values()) != len(legacy_map):
        raise ValueError("Legacy case IDs overlap across splits")
    groups = defaultdict(list)
    for slide in sorted(slide_ids):
        key = case_key(slide)
        if key in legacy_map:
            assignments[slide] = legacy_map[key]
        else:
            groups[diseases.get(key, "unknown")].append(slide)
    rng = random.Random(seed)
    for slides in groups.values():
        rng.shuffle(slides)
    remaining = sum(map(len, groups.values()))
    counts = {"val": round(remaining * 0.16), "test": round(remaining * 0.16)}
    per_group = Counter()
    for split in ("test", "val"):
        for _ in range(counts[split]):
            eligible = [g for g in groups if groups[g]]
            group = max(eligible, key=lambda g: (len(groups[g]) / (1 + per_group[(g, split)]), g))
            assignments[groups[group].pop()] = split
            per_group[(group, split)] += 1
    for slides in groups.values():
        assignments.update({slide: "train" for slide in slides})
    return assignments


def crop_geometry(feature):
    x1, y1, x2, y2 = polygon_bounds(feature)
    side = int(np.ceil(max(1200.0, (x2 - x1) * 1.2, (y2 - y1) * 1.2)))
    return side


def make_roi_annotation(feature, points, local_x, local_y, side, image_size):
    scale = image_size / side
    polygon = (np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64) - [local_x, local_y]) * scale
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    annotation = {"classes": CLASSES, **{name: [] for name in CLASSES}}
    seen = set()
    for point in points:
        x = (float(point["x"]) - local_x) * scale
        y = (float(point["y"]) - local_y) * scale
        if not (0 <= x < image_size and 0 <= y < image_size):
            continue
        # Background outside the annotated glomerulus is unknown, not negative.
        if cv2.pointPolygonTest(polygon.astype(np.float32), (x, y), False) < 0:
            continue
        key = (point["class_slug"], round(x, 3), round(y, 3))
        if key not in seen:
            annotation[point["class_slug"]].append([round(x, 4), round(y, 4)])
            seen.add(key)
    annotation["roi_polygon"] = polygon.round(4).tolist()
    point_classes = defaultdict(set)
    for label in CLASSES:
        for x, y in annotation[label]:
            point_classes[(round(x, 3), round(y, 3))].add(label)
    conflicts = {key for key, labels in point_classes.items() if len(labels) > 1}
    for label in CLASSES:
        annotation[label] = [p for p in annotation[label] if (round(p[0], 3), round(p[1], 3)) not in conflicts]
    annotation["ambiguous_points"] = [list(point) for point in sorted(conflicts)]
    for x, y in conflicts:
        cv2.circle(mask, (int(round(x)), int(round(y))), 3, 0, -1)
    return annotation, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--legacy-dataset", type=Path, required=True)
    parser.add_argument("--score-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--image-size", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import openpyxl

    workbook = openpyxl.load_workbook(args.score_xlsx, read_only=True, data_only=True)
    diseases = {}
    for row in list(workbook.worksheets[1].values)[1:]:
        if row[0]:
            diseases[case_key(row[0])] = str(row[4] or "unknown").strip()
    workbook.close()
    slides = {row["slide_id"]: row for row in read_csv(args.export_dir / "slides.csv")}
    features = json.loads((args.export_dir / "glomerulus_rois.geojson").read_text(encoding="utf-8"))["features"]
    points = defaultdict(list)
    for point in read_csv(args.export_dir / "cell_points.csv"):
        points[point["slide_id"]].append(point)
    legacy = json.loads((args.legacy_dataset / "dataset_summary.json").read_text())
    assignments = assign_splits(slides, legacy["slide_split"], diseases, args.seed)
    split_rows = [{"slide_id": slide, "case_key": case_key(slide), "split": split,
                   "disease": diseases.get(case_key(slide), "unknown"),
                   "legacy_split": next((name for name, ids in legacy["slide_split"].items() if slide in ids), "new")}
                  for slide, split in sorted(assignments.items())]
    split_path = args.output_dir / "case_splits.csv"
    if split_path.exists() and read_csv(split_path) != [{k: str(v) for k, v in r.items()} for r in split_rows]:
        raise ValueError("Refusing to change an existing frozen split")
    write_csv(split_path, split_rows)
    old_rows = {(row["slide_id"], row["roi_mark_id"]): row
                for row in read_csv(args.legacy_dataset / "manifest.csv")}
    for name in ("train", "val", "test"):
        for suffix in ("image", "point", "roi"):
            (args.output_dir / (name + "_" + suffix)).mkdir(exist_ok=True)
    czi_cache = {}
    roi_numbers = Counter()
    rows = []
    reused = 0
    train_sum = np.zeros(3, dtype=np.float64)
    train_sq = np.zeros(3, dtype=np.float64)
    train_pixels = 0
    for feature in features:
        props = feature["properties"]
        slide = props["slide_id"]
        split = assignments[slide]
        roi_numbers[slide] += 1
        stem = "{}_roi{:03d}".format(slide, roi_numbers[slide])
        image_path = args.output_dir / (split + "_image") / (stem + ".jpg")
        annotation_path = args.output_dir / (split + "_point") / (stem + ".json")
        old = old_rows.get((slide, str(props["mark_id"])))
        side = crop_geometry(feature)
        if old and int(old["side_raw_px"]) == side:
            local_x, local_y = int(old["local_x"]), int(old["local_y"])
            prefix = "heldout_test" if old["split"] == "test" else old["split"]
            source = args.legacy_dataset / (prefix + "_image") / (old["crop_id"] + ".jpg")
            if not image_path.exists():
                shutil.copy2(source, image_path)
            reused += 1
        elif image_path.exists() and annotation_path.exists():
            existing = json.loads(annotation_path.read_text(encoding="utf-8"))
            local_x, local_y = existing["local_origin"]
            side = existing["side_raw_px"]
        else:
            from aicspylibczi import CziFile
            if slide not in czi_cache:
                czi = CziFile(slides[slide]["czi_path"])
                czi_cache[slide] = (czi, czi.get_mosaic_bounding_box())
            czi, box = czi_cache[slide]
            _, meta = copy_crop(czi, box, feature, [], image_path, args.image_size, 1.2, 1200)
            local_x, local_y, side = meta["local_x"], meta["local_y"], meta["side_raw_px"]
        annotation, mask = make_roi_annotation(feature, points[slide], local_x, local_y, side, args.image_size)
        annotation.update({"local_origin": [local_x, local_y], "side_raw_px": side})
        annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
        cv2.imwrite(str(args.output_dir / (split + "_roi") / (stem + ".png")), mask)
        row = {"crop_id": stem, "slide_id": slide, "case_key": case_key(slide),
               "split": split, "disease": diseases.get(case_key(slide), "unknown"),
               "roi_mark_id": str(props["mark_id"]), "local_x": local_x, "local_y": local_y,
               "side_raw_px": side, "valid_area_pixels": int((mask > 0).sum()),
               **{name + "_points": len(annotation[name]) for name in CLASSES}}
        rows.append(row)
        if split == "train":
            pixels = cv2.imread(str(image_path))[..., ::-1].reshape(-1, 3)[::16].astype(np.float64) / 255
            train_sum += pixels.sum(0)
            train_sq += (pixels * pixels).sum(0)
            train_pixels += len(pixels)
        if len(rows) % 25 == 0:
            print(json.dumps({"processed": len(rows), "total": len(features), "reused": reused}), flush=True)
    write_csv(args.output_dir / "manifest.csv", rows)
    # Keep the pretrained normalization for both models; record new train-only statistics separately.
    shutil.copy2(args.legacy_dataset / "mean_std.npy", args.output_dir / "mean_std.npy")
    mean = train_sum / train_pixels
    std = np.sqrt(np.maximum(train_sq / train_pixels - mean * mean, 1e-8))
    summary = {"classes": CLASSES, "image_size": args.image_size, "seed": args.seed,
               "slide_split": {name: [r["slide_id"] for r in split_rows if r["split"] == name]
                               for name in ("train", "val", "test")},
               "crop_count": dict(Counter(r["split"] for r in rows)),
               "case_count": dict(Counter(r["split"] for r in split_rows)),
               "class_counts": {split: {name: sum(r[name + "_points"] for r in rows if r["split"] == split)
                                        for name in CLASSES} for split in ("train", "val", "test")},
               "reused_legacy_crops": reused, "new_case_count": sum(r["legacy_split"] == "new" for r in split_rows),
               "legacy_cases_missing_from_new": sorted(set(old_rows_key[0] for old_rows_key in old_rows) - set(slides)),
               "train_rgb_mean": mean.tolist(), "train_rgb_std": std.tolist(),
               "normalization_used": np.load(args.output_dir / "mean_std.npy").tolist(),
               "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
               "annotation_policy": "Only points inside the annotated ROI; ignore unannotated surroundings in loss/evaluation."}
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
