"""Export HZY task-2 intrinsic-cell point labels from slice.db files.

The HZY annotation client stores cell labels as ``spot`` marks in a SQLite
database.  This exporter produces a compact, model-agnostic point table plus
glomerulus ROI polygons, without modifying the raw CZI or database files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CELL_LABELS = {
    "内皮细胞": "endothelial_cell",
    "肾小球系膜细胞": "mesangial_cell",
    "足细胞": "podocyte",
}
GLOMERULUS_LABELS = {"未废弃肾小球", "废弃肾小球"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sanitize_identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_value).strip("_")
    return cleaned or "slide"


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    return cursor.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone() is not None


def group_names(cursor: sqlite3.Cursor) -> Dict[int, str]:
    if not table_exists(cursor, "MarkGroup"):
        return {}
    return {
        int(group_id): str(name).strip()
        for group_id, name in cursor.execute("select id, groupName from MarkGroup")
        if group_id is not None and name
    }


def parse_group_ids(value: object) -> List[int]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        parsed = [parsed]
    result = []
    for item in parsed:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def parse_xy(value: object) -> List[Tuple[float, float]]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    xs, ys = parsed.get("x"), parsed.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list):
        return []
    points = []
    for x, y in zip(xs, ys):
        try:
            points.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    return points


def mark_rows(cursor: sqlite3.Cursor) -> Iterable[Dict[str, object]]:
    table = "Mark_label_None"
    if not table_exists(cursor, table):
        return
    columns = {row[1] for row in cursor.execute("pragma table_info({})".format(table))}
    wanted = ["id", "position", "method", "radius", "groupId"]
    selected = [name for name in wanted if name in columns]
    for row in cursor.execute("select {} from {}".format(", ".join(selected), table)):
        yield dict(zip(selected, row))


def find_db(czi_path: Path) -> Optional[Path]:
    same_dir = czi_path.parent / "slice.db"
    return same_dir if same_dir.exists() else None


def case_id_for(czi_path: Path, raw_root: Path) -> str:
    relative = czi_path.relative_to(raw_root)
    return relative.parts[0] if len(relative.parts) > 1 else czi_path.parent.name


def export_slide(
    czi_path: Path,
    raw_root: Path,
    used_ids: set[str],
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    db_path = find_db(czi_path)
    case_id = case_id_for(czi_path, raw_root)
    base_id = sanitize_identifier(czi_path.stem)
    slide_id = base_id
    suffix = 2
    while slide_id in used_ids:
        slide_id = "{}_{}".format(base_id, suffix)
        suffix += 1
    used_ids.add(slide_id)

    points: List[Dict[str, object]] = []
    rois: List[Dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    ignored_marks = 0
    if db_path is not None:
        connection = sqlite3.connect(str(db_path))
        cursor = connection.cursor()
        groups = group_names(cursor)
        for mark in mark_rows(cursor):
            labels = [groups[group_id] for group_id in parse_group_ids(mark.get("groupId")) if group_id in groups]
            if not labels:
                ignored_marks += 1
                continue
            coordinates = parse_xy(mark.get("position"))
            if not coordinates:
                ignored_marks += 1
                continue
            method = str(mark.get("method") or "")
            for label in labels:
                if label in CELL_LABELS and method == "spot":
                    x, y = coordinates[0]
                    label_counts[label] += 1
                    points.append(
                        {
                            "slide_id": slide_id,
                            "case_id": case_id,
                            "czi_path": str(czi_path),
                            "db_path": str(db_path),
                            "mark_id": mark.get("id"),
                            "label_cn": label,
                            "class_slug": CELL_LABELS[label],
                            "x": "{:.6f}".format(x),
                            "y": "{:.6f}".format(y),
                            "radius": "{:.6f}".format(float(mark.get("radius") or 0.0)),
                            "coordinate_frame": "mosaic_local",
                        }
                    )
                elif label in GLOMERULUS_LABELS and len(coordinates) >= 3:
                    ring = [[x, y] for x, y in coordinates]
                    if ring[0] != ring[-1]:
                        ring.append(ring[0])
                    rois.append(
                        {
                            "type": "Feature",
                            "properties": {
                                "slide_id": slide_id,
                                "case_id": case_id,
                                "mark_id": mark.get("id"),
                                "label_cn": label,
                                "source_method": method,
                            },
                            "geometry": {"type": "Polygon", "coordinates": [ring]},
                        }
                    )
        connection.close()

    slide = {
        "slide_id": slide_id,
        "case_id": case_id,
        "czi_path": str(czi_path),
        "db_path": str(db_path) if db_path else "",
        "has_annotation_db": int(db_path is not None),
        "endothelial_cell_points": label_counts["内皮细胞"],
        "mesangial_cell_points": label_counts["肾小球系膜细胞"],
        "podocyte_points": label_counts["足细胞"],
        "glomerulus_rois": len(rois),
        "ignored_marks": ignored_marks,
    }
    return slide, points, rois


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("Output directory is not empty; pass --overwrite to replace generated files.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    czi_paths = sorted(args.raw_root.rglob("*.czi"))
    if not czi_paths:
        raise SystemExit("No CZI files found under {}".format(args.raw_root))

    used_ids: set[str] = set()
    slides: List[Dict[str, object]] = []
    points: List[Dict[str, object]] = []
    rois: List[Dict[str, object]] = []
    for czi_path in czi_paths:
        slide, slide_points, slide_rois = export_slide(czi_path, args.raw_root, used_ids)
        slides.append(slide)
        points.extend(slide_points)
        rois.extend(slide_rois)

    write_csv(
        args.output_dir / "slides.csv",
        slides,
        [
            "slide_id", "case_id", "czi_path", "db_path", "has_annotation_db",
            "endothelial_cell_points", "mesangial_cell_points", "podocyte_points",
            "glomerulus_rois", "ignored_marks",
        ],
    )
    write_csv(
        args.output_dir / "cell_points.csv",
        points,
        ["slide_id", "case_id", "czi_path", "db_path", "mark_id", "label_cn", "class_slug", "x", "y", "radius", "coordinate_frame"],
    )
    (args.output_dir / "glomerulus_rois.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": rois}, ensure_ascii=False),
        encoding="utf-8",
    )
    counts = Counter(point["class_slug"] for point in points)
    summary = {
        "czi_slides": len(slides),
        "slides_with_annotation_db": sum(int(row["has_annotation_db"]) for row in slides),
        "slides_with_cell_points": sum(
            int(row["endothelial_cell_points"]) + int(row["mesangial_cell_points"]) + int(row["podocyte_points"]) > 0
            for row in slides
        ),
        "cell_points": dict(counts),
        "cell_points_total": len(points),
        "glomerulus_rois": len(rois),
        "coordinate_frame": "mosaic_local; add the CZI mosaic bounding-box origin when reading a region",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
