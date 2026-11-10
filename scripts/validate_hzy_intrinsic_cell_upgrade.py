"""Audit prepared cell crops, coordinates, annotation masks and case splits."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from hzy_intrinsic_cell_upgrade import CLASSES, read_rows


def audit(root):
    root = Path(root)
    summary = json.loads((root / "dataset_summary.json").read_text())
    rows = read_rows(root / "manifest.csv")
    case_splits = {}
    errors = []
    boundary_points = 0
    point_total = 0
    conflicts = 0
    per_split = Counter()
    for row in rows:
        case = row["case_key"]
        if case in case_splits and case_splits[case] != row["split"]:
            errors.append("Case split leakage: " + case)
        case_splits[case] = row["split"]
        split, stem = row["split"], row["crop_id"]
        image = cv2.imread(str(root / (split + "_image") / (stem + ".jpg")))
        mask = cv2.imread(str(root / (split + "_roi") / (stem + ".png")), cv2.IMREAD_GRAYSCALE)
        annotation = json.loads((root / (split + "_point") / (stem + ".json")).read_text())
        if image is None or image.shape != (summary["image_size"], summary["image_size"], 3):
            errors.append("Invalid image dimensions: " + stem)
            continue
        if mask is None or not np.any(mask):
            errors.append("Invalid supervision ROI: " + stem)
            continue
        if annotation["classes"] != CLASSES:
            errors.append("Invalid class order: " + stem)
        seen = {}
        for name in CLASSES:
            points = np.asarray(annotation[name]).reshape(-1, 2)
            if not np.isfinite(points).all():
                errors.append("Non-finite points: " + stem)
            for x, y in points:
                if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                    errors.append("Out-of-bounds point: " + stem)
                    continue
                point_total += 1
                boundary_points += int(mask[int(y), int(x)] == 0)
                key = (round(x, 3), round(y, 3))
                conflicts += int(key in seen and seen[key] != name)
                seen[key] = name
            if len(points) != int(row[name + "_points"]):
                errors.append("Point count mismatch: " + stem)
        per_split[split] += 1
    if dict(per_split) != summary["crop_count"]:
        errors.append("Manifest/summary crop count mismatch")
    mean_std = np.load(root / "mean_std.npy")
    if mean_std.shape != (2, 3) or not np.all(mean_std[1] > 0):
        errors.append("Invalid normalization statistics")
    result = {"cases": len(case_splits), "crops": len(rows), "point_instances": point_total,
              "crop_count": dict(per_split), "boundary_rounding_points": boundary_points,
              "same_coordinate_class_conflicts": conflicts, "errors": errors,
              "split_sha256": summary["split_sha256"], "passed": not errors and conflicts == 0}
    (root / "quality_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset)
    print(json.dumps(result, indent=2))
    raise SystemExit(not result["passed"])
