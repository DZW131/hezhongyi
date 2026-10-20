"""Build WSI-level pathology scorer input from per-glomerulus inference tables.

This is the bridge between model inference and ``score_hspn_pathology.py``:

1. Start from the glomerulus crop manifest produced by
   ``prepare_hzy_glomerulus_lesion_crops.py``.
2. Merge optional inference CSVs for glomerulus-level proliferation
   classification, crescent detection, crescent subtype classification, other
   lesions, and manual overrides.
3. Write ``per_glomerulus_predictions.csv`` in the scorer's expected schema.
4. Optionally run the ISKDC/Oxford scorer immediately.
"""

import csv
import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_hspn_pathology import (  # noqa: E402
    Thresholds,
    group_and_score,
    load_glomeruli,
    write_report_outputs,
)


SCORER_FIELDS = [
    "slide_id",
    "glomerulus_id",
    "crop_id",
    "split",
    "x1",
    "y1",
    "x2",
    "y2",
    "mesangial_prob",
    "endocapillary_prob",
    "segmental_sclerosis_prob",
    "adhesion_prob",
    "cellular_crescent_prob",
    "fibrocellular_crescent_prob",
    "fibrous_crescent_prob",
    "crescent_prob",
    "fibrinoid_necrosis_prob",
    "fibrin_thrombus_prob",
    "membranoproliferative_pattern",
    "ifta_percent",
    "image_path",
    "source_notes",
]

DIRECT_SCORER_FIELDS = {
    "mesangial_prob",
    "endocapillary_prob",
    "segmental_sclerosis_prob",
    "adhesion_prob",
    "cellular_crescent_prob",
    "fibrocellular_crescent_prob",
    "fibrous_crescent_prob",
    "crescent_prob",
    "fibrinoid_necrosis_prob",
    "fibrin_thrombus_prob",
    "membranoproliferative_pattern",
    "ifta_percent",
}

ID_COLUMNS = (
    "crop_id",
    "case_id",
    "name",
    "image_name",
    "filename",
    "file_name",
    "source_stem",
    "source_image",
    "image",
    "image_path",
    "roi_id",
    "glomerulus_id",
)

PROB_COLUMNS = ("prob_positive", "positive_prob", "prob", "score", "max_score", "confidence", "pred_score", "bin_pred")
MESANGIAL_LABELS = {"yesm", "m1", "mesangial", "mesangial_positive", "mesangial_hypercellularity"}
ENDOCAPILLARY_LABELS = {"yese", "e1", "endocapillary", "endocapillary_positive", "endocapillary_hypercellularity"}
NEGATIVE_LABELS = {
    "nom",
    "noe",
    "m0",
    "e0",
    "normal",
    "negative",
    "neg",
    "background",
    "none",
    "nan",
    "no_proliferation",
}


def clean_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [{clean_key(key): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(rows: Sequence[Dict[str, object]], path: Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    if number > 1.0 and number <= 100.0:
        return number / 100.0
    return number


def first_value(row: Dict[str, str], columns: Iterable[str]) -> Optional[str]:
    for column in columns:
        key = clean_key(column)
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def path_stem(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    return Path(text).stem


def normalize_identifier(value: object) -> str:
    stem = path_stem(value)
    return stem or str(value or "").strip()


def crop_aliases(row: Dict[str, object]) -> List[str]:
    aliases = []
    for key in ("crop_id", "glomerulus_id", "image_path", "mask_path", "source_image", "image"):
        value = row.get(key)
        if value not in (None, ""):
            aliases.append(str(value))
            aliases.append(normalize_identifier(value))
    slide_id = row.get("slide_id")
    glomerulus_id = row.get("glomerulus_id")
    if slide_id not in (None, "") and glomerulus_id not in (None, ""):
        aliases.append(f"{slide_id}::{glomerulus_id}")
    return sorted({alias for alias in aliases if alias})


def add_note(row: Dict[str, object], note: str) -> None:
    if not note:
        return
    current = str(row.get("source_notes", "") or "")
    pieces = [piece for piece in current.split(";") if piece]
    if note not in pieces:
        pieces.append(note)
    row["source_notes"] = ";".join(pieces)


def set_max_prob(row: Dict[str, object], field: str, value: object, note: str = "") -> None:
    number = parse_float(value)
    if number is None:
        return
    old = parse_float(row.get(field))
    if old is None or number > old:
        row[field] = f"{number:.6f}"
        add_note(row, note)


def set_direct_value(row: Dict[str, object], field: str, value: object, note: str = "") -> None:
    if value in (None, ""):
        return
    row[field] = value
    add_note(row, note)


def bool_or_score(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "positive", "pos", "present", "有", "是", "阳性"}:
        return 1.0
    if text in {"0", "false", "no", "n", "negative", "neg", "absent", "无", "否", "阴性", ""}:
        return 0.0
    return parse_float(value)


def best_probability(row: Dict[str, str], extra_columns: Sequence[str] = ()) -> Optional[float]:
    for column in tuple(extra_columns) + PROB_COLUMNS:
        value = first_value(row, (column,))
        number = bool_or_score(value)
        if number is not None:
            return number
    return None


def label_probability(row: Dict[str, str], target: str) -> Optional[float]:
    """Parse weakly supervised classifier labels such as yesM/noM or yesE/noE."""
    labels = (
        "pred_class",
        "class_slug",
        "class_name",
        "label",
        "prediction",
        "pred_label_name",
        "pred",
    )
    positives = MESANGIAL_LABELS if target == "mesangial" else ENDOCAPILLARY_LABELS
    for column in labels:
        value = first_value(row, (column,))
        if value is None:
            continue
        text = clean_key(str(value))
        if text in positives:
            return 1.0
        if text in NEGATIVE_LABELS:
            return 0.0
    return None


def classification_probability(row: Dict[str, str], target: str) -> Optional[float]:
    target_columns = {
        "mesangial": ("mesangial_prob", "prob_mesangial", "m_prob"),
        "endocapillary": ("endocapillary_prob", "prob_endocapillary", "e_prob"),
    }[target]
    explicit = first_value(row, target_columns)
    number = bool_or_score(explicit)
    if number is not None:
        return number
    label_value = label_probability(row, target)
    if label_value is not None:
        return label_value
    return best_probability(row, extra_columns=("proliferation_prob", "prob_proliferation"))


def load_manifest_rows(manifest_csv: Path, default_split: str = "") -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    rows = []
    alias_to_row: Dict[str, Dict[str, object]] = {}
    for raw in read_csv(manifest_csv):
        crop_id = first_value(raw, ("crop_id", "case_id", "name", "image_name")) or normalize_identifier(
            first_value(raw, ("image_path", "image"))
        )
        if not crop_id:
            raise ValueError("Manifest row is missing crop_id/image_path: {}".format(raw))
        glom_id = first_value(raw, ("glomerulus_id",)) or "g{:04d}".format(int(float(first_value(raw, ("glomerulus_index",)) or len(rows) + 1)))
        x1 = first_value(raw, ("bbox_x_min", "x1", "x_min", "xmin"))
        y1 = first_value(raw, ("bbox_y_min", "y1", "y_min", "ymin"))
        x2 = first_value(raw, ("bbox_x_max", "x2", "x_max", "xmax"))
        y2 = first_value(raw, ("bbox_y_max", "y2", "y_max", "ymax"))
        if None in (x1, y1, x2, y2):
            crop_x = parse_float(first_value(raw, ("crop_x", "x")))
            crop_y = parse_float(first_value(raw, ("crop_y", "y")))
            side = parse_float(first_value(raw, ("source_crop_size", "saved_size", "crop_size")))
            if None not in (crop_x, crop_y, side):
                x1, y1, x2, y2 = crop_x, crop_y, crop_x + side, crop_y + side

        row: Dict[str, object] = {field: "" for field in SCORER_FIELDS}
        row.update(
            {
                "slide_id": first_value(raw, ("slide_id", "wsi_id", "case_id")) or "unknown_slide",
                "glomerulus_id": str(glom_id),
                "crop_id": crop_id,
                "split": first_value(raw, ("split",)) or default_split,
                "x1": x1 if x1 is not None else "",
                "y1": y1 if y1 is not None else "",
                "x2": x2 if x2 is not None else "",
                "y2": y2 if y2 is not None else "",
                "image_path": first_value(raw, ("slide_image_path", "wsi_path", "source_image", "image_path")) or "",
            }
        )
        rows.append(row)
        for alias in crop_aliases({**raw, **row}):
            alias_to_row[alias] = row
    return rows, alias_to_row


def find_target_row(row: Dict[str, str], alias_to_row: Dict[str, Dict[str, object]], roi_to_crop: Optional[Dict[str, str]] = None) -> Optional[Dict[str, object]]:
    slide_id = first_value(row, ("slide_id", "wsi_id", "case_id"))
    glomerulus_id = first_value(row, ("glomerulus_id", "glomerulus_index"))
    if slide_id and glomerulus_id:
        target = alias_to_row.get(f"{slide_id}::{glomerulus_id}")
        if target is not None:
            return target

    for column in ID_COLUMNS:
        if column == "glomerulus_id" and slide_id:
            continue
        value = first_value(row, (column,))
        if not value:
            continue
        candidates = [value, normalize_identifier(value)]
        if roi_to_crop:
            for candidate in list(candidates):
                mapped = roi_to_crop.get(candidate) or roi_to_crop.get(normalize_identifier(candidate))
                if mapped:
                    candidates.extend([mapped, normalize_identifier(mapped)])
        for candidate in candidates:
            if candidate in alias_to_row:
                return alias_to_row[candidate]
    return None


def load_roi_to_crop(roi_manifest_csv: Optional[Path]) -> Dict[str, str]:
    if not roi_manifest_csv:
        return {}
    mapping: Dict[str, str] = {}
    for row in read_csv(roi_manifest_csv):
        roi_id = first_value(row, ("roi_id",))
        source = first_value(row, ("source_stem", "crop_id", "source_image", "source_mask"))
        if roi_id and source:
            mapping[roi_id] = normalize_identifier(source)
            mapping[normalize_identifier(roi_id)] = normalize_identifier(source)
    return mapping


def merge_direct_columns(
    table_csv: Optional[Path],
    alias_to_row: Dict[str, Dict[str, object]],
    note: str,
    roi_to_crop: Optional[Dict[str, str]] = None,
) -> int:
    if not table_csv:
        return 0
    matched = 0
    for row in read_csv(table_csv):
        target = find_target_row(row, alias_to_row, roi_to_crop=roi_to_crop)
        if target is None:
            continue
        matched += 1
        for field in DIRECT_SCORER_FIELDS:
            if field in row and row[field] != "":
                if field.endswith("_prob") or field == "ifta_percent":
                    set_max_prob(target, field, row[field], note)
                else:
                    set_direct_value(target, field, row[field], note)
    return matched


def merge_proliferation(
    table_csv: Optional[Path],
    alias_to_row: Dict[str, Dict[str, object]],
    binary_target: str,
    note: str = "proliferation_classification",
) -> int:
    if not table_csv:
        return 0
    matched = 0
    for row in read_csv(table_csv):
        target = find_target_row(row, alias_to_row)
        if target is None:
            continue
        matched += 1

        if any(key in row for key in ("mesangial_prob", "prob_mesangial", "m_prob")):
            set_max_prob(target, "mesangial_prob", first_value(row, ("mesangial_prob", "prob_mesangial", "m_prob")), note)
        if any(key in row for key in ("endocapillary_prob", "prob_endocapillary", "e_prob")):
            set_max_prob(
                target,
                "endocapillary_prob",
                first_value(row, ("endocapillary_prob", "prob_endocapillary", "e_prob")),
                note,
            )

        generic = best_probability(row, extra_columns=("proliferation_prob", "prob_proliferation"))
        if generic is not None:
            if binary_target in {"mesangial", "both"}:
                set_max_prob(target, "mesangial_prob", generic, f"{note}_binary")
            if binary_target in {"endocapillary", "both"}:
                set_max_prob(target, "endocapillary_prob", generic, f"{note}_binary")
    return matched


def merge_proliferation_classifier(
    table_csv: Optional[Path],
    alias_to_row: Dict[str, Dict[str, object]],
    target_name: str,
) -> int:
    if not table_csv:
        return 0
    output_field = "mesangial_prob" if target_name == "mesangial" else "endocapillary_prob"
    matched = 0
    for row in read_csv(table_csv):
        target = find_target_row(row, alias_to_row)
        if target is None:
            continue
        matched += 1
        value = classification_probability(row, target_name)
        if value is not None:
            set_max_prob(target, output_field, value, f"{target_name}_weak_classifier")
    return matched


def merge_crescent_detection(table_csv: Optional[Path], alias_to_row: Dict[str, Dict[str, object]]) -> int:
    if not table_csv:
        return 0
    matched = 0
    for row in read_csv(table_csv):
        target = find_target_row(row, alias_to_row)
        if target is None:
            continue
        matched += 1
        prob = best_probability(row, extra_columns=("crescent_prob", "prob_crescent"))
        if prob is not None:
            set_max_prob(target, "crescent_prob", prob, "crescent_detection")
        for field, aliases in {
            "cellular_crescent_prob": ("cellular_crescent_prob", "prob_cellular_crescent"),
            "fibrocellular_crescent_prob": ("fibrocellular_crescent_prob", "prob_fibrocellular_crescent"),
            "fibrous_crescent_prob": ("fibrous_crescent_prob", "prob_fibrous_crescent"),
        }.items():
            value = first_value(row, aliases)
            if value is not None:
                set_max_prob(target, field, value, "crescent_detection")
    return matched


def merge_crescent_subtypes(
    table_csv: Optional[Path],
    alias_to_row: Dict[str, Dict[str, object]],
    roi_to_crop: Dict[str, str],
) -> int:
    if not table_csv:
        return 0
    matched = 0
    class_fields = {
        "cellular_crescent": "cellular_crescent_prob",
        "fibrocellular_crescent": "fibrocellular_crescent_prob",
        "fibrous_crescent": "fibrous_crescent_prob",
    }
    for row in read_csv(table_csv):
        target = find_target_row(row, alias_to_row, roi_to_crop=roi_to_crop)
        if target is None:
            continue
        matched += 1
        for class_slug, field in class_fields.items():
            value = first_value(row, (f"prob_{class_slug}", field))
            if value is not None:
                set_max_prob(target, field, value, "crescent_subtype")
                set_max_prob(target, "crescent_prob", value, "crescent_subtype")
        pred_class = first_value(row, ("pred_class", "class_slug", "crescent_subtype"))
        pred_prob = best_probability(row)
        if pred_class:
            pred_class = clean_key(pred_class)
            field = class_fields.get(pred_class)
            if field and pred_prob is not None:
                set_max_prob(target, field, pred_prob, "crescent_subtype")
                set_max_prob(target, "crescent_prob", pred_prob, "crescent_subtype")
    return matched


def merge_other_lesions(table_csv: Optional[Path], alias_to_row: Dict[str, Dict[str, object]]) -> int:
    if not table_csv:
        return 0
    matched = 0
    aliases = {
        "segmental_sclerosis_prob": ("segmental_sclerosis_prob", "sclerosis_prob", "s_prob"),
        "adhesion_prob": ("adhesion_prob", "capsular_adhesion_prob", "segmental_capsular_adhesion_prob"),
        "fibrinoid_necrosis_prob": ("fibrinoid_necrosis_prob", "necrosis_prob"),
        "fibrin_thrombus_prob": ("fibrin_thrombus_prob", "thrombus_prob"),
        "membranoproliferative_pattern": ("membranoproliferative_pattern", "mpgn_pattern", "iskdc_vi"),
        "ifta_percent": ("ifta_percent", "t_percent"),
    }
    for row in read_csv(table_csv):
        target = find_target_row(row, alias_to_row)
        if target is None:
            continue
        matched += 1
        for field, columns in aliases.items():
            value = first_value(row, columns)
            if value is None:
                continue
            if field.endswith("_prob") or field == "ifta_percent":
                set_max_prob(target, field, value, "other_lesions")
            else:
                set_direct_value(target, field, value, "other_lesions")
    return matched


def merge_ifta_by_slide(table_csv: Optional[Path], rows: Sequence[Dict[str, object]]) -> int:
    """Broadcast a slide-level tubular-atrophy burden to its glomerular rows.

    The tubular-atrophy model operates on tissue scenes rather than isolated
    glomerulus crops.  Its scene/slide burden therefore needs to be attached
    to every glomerular record from the same slide before the pathology scorer
    can produce one consistent T field for that slide.
    """
    if not table_csv:
        return 0
    values: Dict[str, object] = {}
    for source in read_csv(table_csv):
        slide_id = first_value(source, ("slide_id", "wsi_id", "case_id", "scene_id"))
        value = first_value(
            source,
            (
                "ifta_percent",
                "tubular_atrophy_target_percent",
                "tubular_atrophy_percent",
                "t_percent",
            ),
        )
        if slide_id and parse_float(value) is not None:
            values[str(slide_id)] = value
            scene_path = first_value(source, ("scene_path", "image_path", "wsi_path"))
            if scene_path:
                values[normalize_identifier(scene_path)] = value

    matched = 0
    for row in rows:
        value = values.get(str(row.get("slide_id", "")))
        if value is None:
            continue
        set_direct_value(row, "ifta_percent", value, "tubular_atrophy_scene_model")
        matched += 1
    return matched


def rows_with_any_evidence(rows: Sequence[Dict[str, object]]) -> int:
    evidence_fields = [
        field
        for field in SCORER_FIELDS
        if field.endswith("_prob") or field in {"membranoproliferative_pattern", "ifta_percent"}
    ]
    return sum(any(row.get(field) not in (None, "") for field in evidence_fields) for row in rows)


def run_scorer(
    per_glomerulus_csv: Path,
    score_output_dir: Path,
    images_dir: Optional[Path],
    min_glomeruli: int,
    generic_crescent_active: bool,
    thresholds: Thresholds,
    iskdc_crescent_mode: str,
) -> None:
    glomeruli = load_glomeruli(per_glomerulus_csv, thresholds, generic_crescent_active=generic_crescent_active)
    scores = group_and_score(glomeruli, min_glomeruli=min_glomeruli, iskdc_crescent_mode=iskdc_crescent_mode)
    write_report_outputs(scores, score_output_dir, images_dir=images_dir)


def write_template(path: Path) -> None:
    write_csv(
        [
            {
                "crop_id": "2026001_s0_g0001_100_200",
                "prob_positive": "0.82",
                "pred_class": "",
                "mesangial_prob": "",
                "endocapillary_prob": "",
            }
        ],
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate HSPN model inference outputs into pathology scorer input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest-csv", type=Path, help="Glomerulus crop manifest, usually <task>/manifests/crops.csv")
    parser.add_argument("--output-csv", type=Path, default=Path("reports/per_glomerulus_predictions.csv"))
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional aggregation summary JSON path")
    parser.add_argument("--write-template", type=Path, default=None, help="Write a small prediction CSV template and exit")
    parser.add_argument("--default-split", default="", help="Split value if the manifest does not include one")
    parser.add_argument(
        "--proliferation-csv",
        type=Path,
        default=None,
        help="Generic glomerulus-level proliferation classifier CSV; no lesion box/mask is required",
    )
    parser.add_argument("--mesangial-classifier-csv", type=Path, default=None, help="Weakly supervised M/mesangial classifier CSV")
    parser.add_argument("--endocapillary-classifier-csv", type=Path, default=None, help="Weakly supervised E/endocapillary classifier CSV")
    parser.add_argument(
        "--proliferation-binary-target",
        choices=("mesangial", "endocapillary", "both"),
        default="mesangial",
        help="How to map a generic proliferation probability when M/E are not separated",
    )
    parser.add_argument("--crescent-detection-csv", type=Path, default=None, help="Crescent detection/presence inference CSV")
    parser.add_argument("--crescent-subtype-csv", type=Path, default=None, help="Crescent subtype classifier prediction CSV")
    parser.add_argument("--crescent-roi-manifest-csv", type=Path, default=None, help="ROI manifest for mapping subtype ROI ids back to crop ids")
    parser.add_argument("--other-lesion-csv", type=Path, default=None, help="Sclerosis/adhesion/necrosis/IFTA inference CSV")
    parser.add_argument(
        "--tubular-atrophy-slide-csv",
        type=Path,
        default=None,
        help="Scene/slide-level tubular-atrophy burden CSV; mapped to IFTA scorer input by slide_id.",
    )
    parser.add_argument(
        "--weakseg-csv",
        type=Path,
        action="append",
        default=[],
        help="One or more HFRM-CAM weak segmentation CSVs with scorer probability fields.",
    )
    parser.add_argument("--manual-csv", type=Path, default=None, help="Manual override CSV with scorer field names")
    parser.add_argument("--score-output-dir", type=Path, default=None, help="If set, run score_hspn_pathology outputs here")
    parser.add_argument("--score-images-dir", type=Path, default=None, help="Optional slide preview image directory for evidence PNG")
    parser.add_argument("--min-glomeruli", type=int, default=8)
    parser.add_argument("--mesangial-threshold", type=float, default=0.35,
                        help="Mesangial M classifier threshold; default follows the 2026-08-22 M-only focal calibration.")
    parser.add_argument("--endocapillary-threshold", type=float, default=0.20,
                        help="Endocapillary E classifier threshold; default follows the 2026-08-22 E transfer calibration.")
    parser.add_argument("--crescent-threshold", type=float, default=0.30,
                        help="Generic crescent detection threshold; default follows the balanced Faster R-CNN box F1 operating point.")
    parser.add_argument("--cellular-crescent-threshold", type=float, default=0.5)
    parser.add_argument("--fibrocellular-crescent-threshold", type=float, default=0.5)
    parser.add_argument("--fibrous-crescent-threshold", type=float, default=0.5)
    parser.add_argument("--iskdc-crescent-mode", choices=("all", "active"), default="all",
                        help="Use all crescent subtypes for ISKDC IV/V by default; Oxford C still uses active crescents.")
    parser.add_argument("--generic-crescent-active", action="store_true", default=True)
    parser.add_argument("--no-generic-crescent-active", dest="generic_crescent_active", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_template:
        write_template(args.write_template)
        print(f"Wrote prediction table template: {args.write_template}")
        return
    if args.manifest_csv is None:
        raise SystemExit("--manifest-csv is required unless --write-template is used.")

    rows, alias_to_row = load_manifest_rows(args.manifest_csv, default_split=args.default_split)
    roi_to_crop = load_roi_to_crop(args.crescent_roi_manifest_csv)
    matches = {
        "proliferation": merge_proliferation(args.proliferation_csv, alias_to_row, args.proliferation_binary_target),
        "mesangial_classifier": merge_proliferation_classifier(args.mesangial_classifier_csv, alias_to_row, "mesangial"),
        "endocapillary_classifier": merge_proliferation_classifier(args.endocapillary_classifier_csv, alias_to_row, "endocapillary"),
        "crescent_detection": merge_crescent_detection(args.crescent_detection_csv, alias_to_row),
        "crescent_subtype": merge_crescent_subtypes(args.crescent_subtype_csv, alias_to_row, roi_to_crop),
        "other_lesions": merge_other_lesions(args.other_lesion_csv, alias_to_row),
        "tubular_atrophy_slide": merge_ifta_by_slide(args.tubular_atrophy_slide_csv, rows),
        "weakseg": sum(merge_direct_columns(path, alias_to_row, "hfrm_cam_weakseg") for path in args.weakseg_csv),
        "manual": merge_direct_columns(args.manual_csv, alias_to_row, "manual_override"),
    }

    write_csv(rows, args.output_csv, fieldnames=SCORER_FIELDS)
    summary = {
        "manifest_csv": str(args.manifest_csv),
        "output_csv": str(args.output_csv),
        "glomeruli": len(rows),
        "with_any_evidence": rows_with_any_evidence(rows),
        "matched_rows": matches,
        "proliferation_binary_target": args.proliferation_binary_target,
        "thresholds": {
            "mesangial": args.mesangial_threshold,
            "endocapillary": args.endocapillary_threshold,
            "crescent": args.crescent_threshold,
            "cellular_crescent": args.cellular_crescent_threshold,
            "fibrocellular_crescent": args.fibrocellular_crescent_threshold,
            "fibrous_crescent": args.fibrous_crescent_threshold,
        },
        "iskdc_crescent_mode": args.iskdc_crescent_mode,
        "generic_crescent_active": args.generic_crescent_active,
    }
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if args.score_output_dir:
        thresholds = Thresholds(
            mesangial=args.mesangial_threshold,
            endocapillary=args.endocapillary_threshold,
            crescent=args.crescent_threshold,
            cellular_crescent=args.cellular_crescent_threshold,
            fibrocellular_crescent=args.fibrocellular_crescent_threshold,
            fibrous_crescent=args.fibrous_crescent_threshold,
        )
        run_scorer(
            args.output_csv,
            score_output_dir=args.score_output_dir,
            images_dir=args.score_images_dir,
            min_glomeruli=args.min_glomeruli,
            generic_crescent_active=args.generic_crescent_active,
            thresholds=thresholds,
            iskdc_crescent_mode=args.iskdc_crescent_mode,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
