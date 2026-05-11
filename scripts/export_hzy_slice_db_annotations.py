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
    parser.add_argument(
        "--split-scenes",
        action="store_true",
        help=(
            "Split each CZI into scene-level annotation JSON files. HZY DB coordinates are "
            "treated as mosaic-global coordinates and shifted into each scene coordinate frame."
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


def polygon_bounds(polygon: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_intersects_box(polygon: Sequence[Sequence[float]], width: float, height: float) -> bool:
    min_x, min_y, max_x, max_y = polygon_bounds(polygon)
    return max_x >= 0 and max_y >= 0 and min_x <= width and min_y <= height


def bbox_to_dict(box) -> Dict[str, int]:
    return {
        "x": int(box.x),
        "y": int(box.y),
        "w": int(box.w),
        "h": int(box.h),
    }


def load_czi_scene_layout(czi_path: Path):
    try:
        from aicspylibczi import CziFile
    except ImportError as exc:
        raise ImportError(
            "--split-scenes requires aicspylibczi. Install it on the conversion server with "
            "`pip install aicspylibczi`."
        ) from exc

    czi = CziFile(str(czi_path))
    scene_boxes = czi.get_all_scene_bounding_boxes()
    mosaic_box = bbox_to_dict(czi.get_mosaic_bounding_box())

    scenes = []
    for scene_index, box in sorted(scene_boxes.items(), key=lambda item: int(item[0])):
        scene_box = bbox_to_dict(box)
        scenes.append(
            {
                "scene_index": int(scene_index),
                "scene_x": scene_box["x"],
                "scene_y": scene_box["y"],
                "scene_width": scene_box["w"],
                "scene_height": scene_box["h"],
                "mosaic_x": mosaic_box["x"],
                "mosaic_y": mosaic_box["y"],
                "mosaic_width": mosaic_box["w"],
                "mosaic_height": mosaic_box["h"],
                "scene_offset_x": scene_box["x"] - mosaic_box["x"],
                "scene_offset_y": scene_box["y"] - mosaic_box["y"],
            }
        )
    return scenes


def transform_polygon_to_scene(
    polygon: Sequence[Sequence[float]],
    scene: Dict[str, int],
    coordinate_scale: float,
) -> List[List[float]]:
    offset_x = float(scene["scene_offset_x"])
    offset_y = float(scene["scene_offset_y"])
    return [
        [
            (float(x) - offset_x) * coordinate_scale,
            (float(y) - offset_y) * coordinate_scale,
        ]
        for x, y in polygon
    ]


def make_feature(row: Dict[str, object], label: str, group_ids: Sequence[int], polygon: List[List[float]]):
    return {
        "type": "Feature",
        "properties": {
            "classification": {
                "name": label,
            },
            "name": label,
            "source_table": row.get("source_table"),
            "mark_id": row.get("id"),
            "group_ids": list(group_ids),
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
            features.append(make_feature(row, label, group_ids, polygon))

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


def export_db_annotations_by_scene(
    db_path: Path,
    czi_path: Path,
    slide_id: str,
    mark_tables: Sequence[str],
    coordinate_scale: float,
):
    scenes = load_czi_scene_layout(czi_path)
    scene_payloads = {
        scene["scene_index"]: {
            "scene": scene,
            "features": [],
            "label_counts": Counter(),
        }
        for scene in scenes
    }

    connection = sqlite3.connect(str(db_path))
    cursor = connection.cursor()
    group_names = load_mark_groups(cursor)
    skipped_rows = 0

    for row in iter_mark_rows(cursor, mark_tables):
        polygon = parse_polygon_position(row.get("position"), coordinate_scale=1.0)
        if polygon is None:
            skipped_rows += 1
            continue

        group_ids = parse_group_ids(row.get("groupId"))
        labels = labels_for_mark(group_ids, group_names, row.get("remark"), row.get("strokeColor"))
        if not labels:
            skipped_rows += 1
            continue

        for scene in scenes:
            scene_polygon = transform_polygon_to_scene(
                polygon=polygon,
                scene=scene,
                coordinate_scale=coordinate_scale,
            )
            scene_width = float(scene["scene_width"]) * coordinate_scale
            scene_height = float(scene["scene_height"]) * coordinate_scale
            if not polygon_intersects_box(scene_polygon, width=scene_width, height=scene_height):
                continue

            payload = scene_payloads[scene["scene_index"]]
            for label in labels:
                payload["label_counts"][label] += 1
                payload["features"].append(make_feature(row, label, group_ids, scene_polygon))

    connection.close()

    outputs = []
    for scene in scenes:
        payload = scene_payloads[scene["scene_index"]]
        scene_slide_id = "{}_s{}".format(slide_id, scene["scene_index"])
        feature_collection = {
            "type": "FeatureCollection",
            "metadata": {
                "slide_id": scene_slide_id,
                "parent_slide_id": slide_id,
                "source_db": str(db_path),
                "source_czi": str(czi_path),
                "mark_tables": list(mark_tables),
                "coordinate_scale": coordinate_scale,
                "coordinate_system": "scene_local_from_mosaic_global",
                "scene": scene,
                "label_counts": dict(payload["label_counts"]),
                "skipped_rows": skipped_rows,
            },
            "features": payload["features"],
        }
        outputs.append(
            {
                "slide_id": scene_slide_id,
                "feature_collection": feature_collection,
                "label_counts": payload["label_counts"],
                "skipped_rows": skipped_rows,
                "scene": scene,
            }
        )
    return outputs


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
    source_slide_count = 0

    for db_path in discover_slice_dbs(raw_root):
        czi_path = find_czi_near_db(db_path)
        raw_slide_name = czi_path.stem if czi_path else db_path.parent.name
        slide_id = sanitize_slide_id(raw_slide_name, used_slide_ids)
        source_slide_count += 1

        if args.split_scenes:
            if czi_path is None:
                raise FileNotFoundError("--split-scenes requires a CZI next to {}".format(db_path))

            scene_outputs = export_db_annotations_by_scene(
                db_path=db_path,
                czi_path=czi_path,
                slide_id=slide_id,
                mark_tables=args.mark_tables,
                coordinate_scale=args.coordinate_scale,
            )

            for scene_output in scene_outputs:
                scene_slide_id = scene_output["slide_id"]
                feature_collection = scene_output["feature_collection"]
                label_counts = scene_output["label_counts"]
                skipped_rows = scene_output["skipped_rows"]
                scene = scene_output["scene"]
                annotation_path = annotations_dir / "{}.json".format(scene_slide_id)
                feature_count = len(feature_collection["features"])

                if feature_count > 0 or not args.skip_empty:
                    with annotation_path.open("w", encoding="utf-8") as handle:
                        json.dump(feature_collection, handle, ensure_ascii=False, indent=2)

                    global_label_counts.update(label_counts)
                    manifest_rows.append(
                        {
                            "slide_id": scene_slide_id,
                            "parent_slide_id": slide_id,
                            "czi_path": str(czi_path),
                            "db_path": str(db_path),
                            "annotation_path": str(annotation_path),
                            "feature_count": feature_count,
                            "skipped_rows": skipped_rows,
                            "label_counts": json.dumps(dict(label_counts), ensure_ascii=False),
                            "coordinate_system": "scene_local_from_mosaic_global",
                            "scene_index": scene["scene_index"],
                            "scene_x": scene["scene_x"],
                            "scene_y": scene["scene_y"],
                            "scene_width": scene["scene_width"],
                            "scene_height": scene["scene_height"],
                            "mosaic_x": scene["mosaic_x"],
                            "mosaic_y": scene["mosaic_y"],
                            "mosaic_width": scene["mosaic_width"],
                            "mosaic_height": scene["mosaic_height"],
                            "scene_offset_x": scene["scene_offset_x"],
                            "scene_offset_y": scene["scene_offset_y"],
                        }
                    )
            continue

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
                    "coordinate_system": "db_native",
                }
            )

    write_csv(manifest_rows, manifest_path)
    summary = {
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "slide_count": len(manifest_rows),
        "source_slide_count": source_slide_count,
        "feature_count": sum(int(row["feature_count"]) for row in manifest_rows),
        "coordinate_scale": args.coordinate_scale,
        "split_scenes": args.split_scenes,
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
