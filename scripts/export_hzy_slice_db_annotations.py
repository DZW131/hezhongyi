import argparse
import csv
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Mark_label_None contains the structured pathology labels in the current HZY export.
# Mark_human mostly contains unlabeled yellow freehand traces, so keep it opt-in.
DEFAULT_MARK_TABLES = ("Mark_label_None",)
DEFAULT_LABEL_ALIASES = {
    "废弃小球": "废弃肾小球",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export Henan TCM Hospital slice.db polygon annotations to GeoJSON files "
            "that can be consumed by prepare_hubmap_tiles.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", required=True, help="Root directory containing uploaded HZY HSPN raw data")
    parser.add_argument("--output-dir", required=True, help="Directory where annotations and manifest will be written")
    parser.add_argument(
        "--mark-tables",
        nargs="+",
        default=list(DEFAULT_MARK_TABLES),
        help="SQLite mark tables to export",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not write annotation JSON files for slides without valid polygon marks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotation JSON and manifest files",
    )
    parser.add_argument(
        "--coordinate-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to exported polygon coordinates. Use the same value as "
            "convert_hzy_czi_to_tiff.py --downsample when converting downsampled TIFFs."
        ),
    )
    return parser.parse_args()


def sanitize_slide_id(raw_name: str, used: set) -> str:
    normalized = unicodedata.normalize("NFKD", raw_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_name).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "slide"

    candidate = cleaned
    suffix = 2
    while candidate in used:
        candidate = "{}_{}".format(cleaned, suffix)
        suffix += 1
    used.add(candidate)
    return candidate


def discover_slice_dbs(raw_root: Path) -> List[Path]:
    return sorted(raw_root.rglob("slice.db"))


def find_czi_near_db(db_path: Path) -> Optional[Path]:
    czi_files = sorted(db_path.parent.glob("*.czi"))
    if not czi_files:
        czi_files = sorted(db_path.parent.rglob("*.czi"))
    return czi_files[0] if czi_files else None


def table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    row = cursor.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def load_mark_groups(cursor: sqlite3.Cursor) -> Dict[int, str]:
    if not table_exists(cursor, "MarkGroup"):
        return {}
    return {
        int(row[0]): str(row[1])
        for row in cursor.execute("select id, groupName from MarkGroup")
        if row[0] is not None and row[1]
    }


def parse_group_ids(raw_group_id) -> List[int]:
    if raw_group_id is None:
        return []

    if isinstance(raw_group_id, int):
        return [raw_group_id]

    text = str(raw_group_id).strip()
    if not text or text.lower() in {"null", "none"}:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        return [int(value) for value in parsed if isinstance(value, (int, float, str)) and str(value).strip()]
    if isinstance(parsed, (int, float, str)) and str(parsed).strip():
        return [int(parsed)]
    return []


def normalize_label(label: Optional[str]) -> str:
    if not label:
        return ""
    label = str(label).strip()
    return DEFAULT_LABEL_ALIASES.get(label, label)


def labels_for_mark(group_ids: Sequence[int], group_names: Dict[int, str], remark, stroke_color) -> List[str]:
    labels = [normalize_label(group_names.get(group_id)) for group_id in group_ids]
    labels = [label for label in labels if label]
    if labels:
        return labels

    remark_label = normalize_label(remark)
    if remark_label:
        return [remark_label]

    color_label = normalize_label(stroke_color)
    return [color_label] if color_label else []


def parse_polygon_position(position_text: str, coordinate_scale: float = 1.0) -> Optional[List[List[float]]]:
    if not position_text:
        return None

    try:
        data = json.loads(position_text)
    except json.JSONDecodeError:
        return None

    points: List[List[float]] = []
    if isinstance(data, dict) and isinstance(data.get("x"), list) and isinstance(data.get("y"), list):
        for x_value, y_value in zip(data["x"], data["y"]):
            try:
                points.append([float(x_value) * coordinate_scale, float(y_value) * coordinate_scale])
            except (TypeError, ValueError):
                continue
    elif isinstance(data, list):
        for point in data:
            if isinstance(point, dict):
                x_value = point.get("x")
                y_value = point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x_value, y_value = point[0], point[1]
            else:
                continue
            try:
                points.append([float(x_value) * coordinate_scale, float(y_value) * coordinate_scale])
            except (TypeError, ValueError):
                continue

    if len(points) < 3:
        return None

    deduped: List[List[float]] = []
    for point in points:
        if not deduped or point != deduped[-1]:
            deduped.append(point)

    if len(deduped) < 3:
        return None

    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])
    return deduped


def iter_mark_rows(cursor: sqlite3.Cursor, mark_tables: Iterable[str]):
    for table_name in mark_tables:
        if not table_exists(cursor, table_name):
            continue

        columns = {row[1] for row in cursor.execute("pragma table_info({})".format(table_name))}
        required = {"id", "position"}
        if not required.issubset(columns):
            continue

        optional_columns = ["remark", "strokeColor", "groupId", "method", "markType"]
        select_columns = ["id", "position"] + [
            column for column in optional_columns if column in columns
        ]
        query = "select {} from {}".format(", ".join(select_columns), table_name)

        for row in cursor.execute(query):
            payload = dict(zip(select_columns, row))
            payload["source_table"] = table_name
            yield payload


def export_db_annotations(
    db_path: Path,
    czi_path: Optional[Path],
    slide_id: str,
    mark_tables: Sequence[str],
    coordinate_scale: float,
):
    connection = sqlite3.connect(str(db_path))
    cursor = connection.cursor()
    group_names = load_mark_groups(cursor)

    features = []
    label_counts: Counter = Counter()
    skipped_rows = 0

    for row in iter_mark_rows(cursor, mark_tables):
        polygon = parse_polygon_position(row.get("position"), coordinate_scale=coordinate_scale)
        if polygon is None:
            skipped_rows += 1
            continue

        group_ids = parse_group_ids(row.get("groupId"))
        labels = labels_for_mark(group_ids, group_names, row.get("remark"), row.get("strokeColor"))
        if not labels:
            skipped_rows += 1
            continue

        for label in labels:
            label_counts[label] += 1
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "classification": {
                            "name": label,
                        },
                        "name": label,
                        "source_table": row.get("source_table"),
                        "mark_id": row.get("id"),
                        "group_ids": group_ids,
                        "remark": row.get("remark"),
                        "strokeColor": row.get("strokeColor"),
                        "method": row.get("method"),
                        "markType": row.get("markType"),
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon],
                    },
                }
            )

    connection.close()

    feature_collection = {
        "type": "FeatureCollection",
        "metadata": {
            "slide_id": slide_id,
            "source_db": str(db_path),
            "source_czi": str(czi_path) if czi_path else "",
            "mark_tables": list(mark_tables),
            "coordinate_scale": coordinate_scale,
            "label_counts": dict(label_counts),
            "skipped_rows": skipped_rows,
        },
        "features": features,
    }
    return feature_collection, label_counts, skipped_rows


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    annotations_dir = output_dir / "annotations"
    manifest_path = output_dir / "hzy_hspn_manifest.csv"
    summary_path = output_dir / "hzy_hspn_annotation_summary.json"

    if not raw_root.exists():
        raise FileNotFoundError("Raw root does not exist: {}".format(raw_root))

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "Output directory already contains files. Re-run with --overwrite: {}".format(output_dir)
        )

    annotations_dir.mkdir(parents=True, exist_ok=True)
    used_slide_ids = set()
    manifest_rows: List[dict] = []
    global_label_counts: Counter = Counter()

    for db_path in discover_slice_dbs(raw_root):
        czi_path = find_czi_near_db(db_path)
        raw_slide_name = czi_path.stem if czi_path else db_path.parent.name
        slide_id = sanitize_slide_id(raw_slide_name, used_slide_ids)

        feature_collection, label_counts, skipped_rows = export_db_annotations(
            db_path=db_path,
            czi_path=czi_path,
            slide_id=slide_id,
            mark_tables=args.mark_tables,
            coordinate_scale=args.coordinate_scale,
        )

        annotation_path = annotations_dir / "{}.json".format(slide_id)
        feature_count = len(feature_collection["features"])
        if feature_count > 0 or not args.skip_empty:
            with annotation_path.open("w", encoding="utf-8") as handle:
                json.dump(feature_collection, handle, ensure_ascii=False, indent=2)

        global_label_counts.update(label_counts)
        manifest_rows.append(
            {
                "slide_id": slide_id,
                "czi_path": str(czi_path) if czi_path else "",
                "db_path": str(db_path),
                "annotation_path": str(annotation_path),
                "feature_count": feature_count,
                "skipped_rows": skipped_rows,
                "label_counts": json.dumps(dict(label_counts), ensure_ascii=False),
            }
        )

    write_csv(manifest_rows, manifest_path)
    summary = {
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "slide_count": len(manifest_rows),
        "feature_count": sum(int(row["feature_count"]) for row in manifest_rows),
        "coordinate_scale": args.coordinate_scale,
        "label_counts": dict(global_label_counts),
        "manifest": str(manifest_path),
        "annotations_dir": str(annotations_dir),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("Exported {} slides".format(summary["slide_count"]))
    print("Exported {} polygon features".format(summary["feature_count"]))
    print("Manifest:", manifest_path)
    print("Annotations:", annotations_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
