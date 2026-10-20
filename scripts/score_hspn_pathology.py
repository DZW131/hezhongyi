"""Aggregate glomerulus-level HSPN predictions into ISKDC and Oxford scores.

The script expects one row per glomerulus, or multiple rows that share the
same slide/glomerulus id. Values may be booleans, labels, or probabilities.
It writes slide-level JSON/CSV reports, a glomerulus evidence CSV, and one
evidence PNG per slide.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


MESANGIAL_ALIASES = (
    "mesangial_hypercellularity",
    "mesangial",
    "mesangial_positive",
    "mesangial_prob",
    "prob_mesangial",
    "m_prob",
)
ENDOCAPILLARY_ALIASES = (
    "endocapillary_hypercellularity",
    "endocapillary",
    "endocapillary_positive",
    "endocapillary_prob",
    "prob_endocapillary",
    "e_prob",
)
SEGMENTAL_SCLEROSIS_ALIASES = (
    "segmental_sclerosis",
    "sclerosis",
    "segmental_sclerosis_prob",
    "sclerosis_prob",
    "s_prob",
)
CAPSULAR_ADHESION_ALIASES = (
    "segmental_capsular_adhesion",
    "capsular_adhesion",
    "adhesion",
    "adhesion_prob",
)
CELLULAR_CRESCENT_ALIASES = (
    "cellular_crescent",
    "cellular_crescent_prob",
    "prob_cellular_crescent",
)
FIBROCELLULAR_CRESCENT_ALIASES = (
    "fibrocellular_crescent",
    "fibrocellular_crescent_prob",
    "prob_fibrocellular_crescent",
)
FIBROUS_CRESCENT_ALIASES = (
    "fibrous_crescent",
    "fibrous_crescent_prob",
    "prob_fibrous_crescent",
)
GENERIC_CRESCENT_ALIASES = (
    "crescent",
    "crescent_positive",
    "crescent_prob",
    "prob_crescent",
)
NECROSIS_ALIASES = (
    "fibrinoid_necrosis",
    "necrosis",
    "fibrinoid_necrosis_prob",
    "necrosis_prob",
)
FIBRIN_THROMBUS_ALIASES = (
    "fibrin_thrombus",
    "thrombus",
    "fibrin_thrombus_prob",
    "thrombus_prob",
)
MPGN_ALIASES = (
    "membranoproliferative_pattern",
    "membranoproliferative",
    "mpgn_pattern",
    "iskdc_vi",
)
IFTA_ALIASES = (
    "ifta_percent",
    "t_percent",
    "tubular_atrophy_interstitial_fibrosis_percent",
    "tubular_atrophy_percent",
    "interstitial_fibrosis_percent",
)


TRUE_VALUES = {"1", "true", "t", "yes", "y", "positive", "pos", "present", "有", "是", "阳性"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "negative", "neg", "absent", "无", "否", "阴性", ""}
ACTIVE_SUBTYPES = {"cellular", "fibrocellular", "cellular_crescent", "fibrocellular_crescent"}
FIBROUS_SUBTYPES = {"fibrous", "fibrous_crescent"}


@dataclass
class Thresholds:
    mesangial: float = 0.5
    endocapillary: float = 0.5
    sclerosis: float = 0.5
    adhesion: float = 0.5
    crescent: float = 0.5
    cellular_crescent: float = 0.5
    fibrocellular_crescent: float = 0.5
    fibrous_crescent: float = 0.5
    necrosis: float = 0.5
    thrombus: float = 0.5
    mpgn: float = 0.5


@dataclass
class GlomerulusEvidence:
    slide_id: str
    glomerulus_id: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    image_path: Optional[str] = None
    mesangial: bool = False
    endocapillary: bool = False
    segmental_sclerosis: bool = False
    capsular_adhesion: bool = False
    cellular_crescent: bool = False
    fibrocellular_crescent: bool = False
    fibrous_crescent: bool = False
    generic_crescent: bool = False
    fibrinoid_necrosis: bool = False
    fibrin_thrombus: bool = False
    membranoproliferative_pattern: bool = False
    ifta_percent: Optional[float] = None
    source_rows: int = 0

    @property
    def active_crescent(self) -> bool:
        return self.cellular_crescent or self.fibrocellular_crescent

    @property
    def any_crescent(self) -> bool:
        return self.active_crescent or self.fibrous_crescent or self.generic_crescent

    @property
    def segmental_or_necrotic(self) -> bool:
        return (
            self.segmental_sclerosis
            or self.capsular_adhesion
            or self.fibrinoid_necrosis
            or self.fibrin_thrombus
            or self.any_crescent
        )


@dataclass
class SlideScore:
    slide_id: str
    glomeruli: List[GlomerulusEvidence]
    total_glomeruli: int
    counts: Dict[str, int]
    percentages: Dict[str, float]
    oxford: Dict[str, str]
    iskdc: Dict[str, str]
    warnings: List[str] = field(default_factory=list)


def clean_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    return {clean_key(key): (value.strip() if isinstance(value, str) else value) for key, value in row.items()}


def first_value(row: Dict[str, str], aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        key = clean_key(alias)
        if key in row and row[key] is not None:
            value = row[key]
            if not isinstance(value, str) or value.strip() != "":
                return value
    return None


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number


def parse_fraction(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    had_percent = text.endswith("%")
    number = parse_float(value)
    if number is None:
        return None
    if had_percent or (number > 1.0 and number <= 100.0):
        return number / 100.0
    return number


def parse_bool_or_prob(value: object, threshold: float) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    number = parse_fraction(value)
    if number is None:
        return False
    return number >= threshold


def is_positive(row: Dict[str, str], aliases: Sequence[str], threshold: float) -> bool:
    value = first_value(row, aliases)
    return parse_bool_or_prob(value, threshold)


def parse_bbox(row: Dict[str, str]) -> Optional[Tuple[float, float, float, float]]:
    bbox = first_value(row, ("bbox", "box", "glomerulus_bbox"))
    if bbox:
        parts = [part.strip() for part in str(bbox).replace(";", ",").split(",")]
        if len(parts) == 4:
            parsed = [parse_float(part) for part in parts]
            if all(value is not None for value in parsed):
                x1, y1, x2, y2 = [float(value) for value in parsed]
                return x1, y1, x2, y2

    x1 = parse_float(first_value(row, ("x1", "xmin", "x_min", "left")))
    y1 = parse_float(first_value(row, ("y1", "ymin", "y_min", "top")))
    x2 = parse_float(first_value(row, ("x2", "xmax", "x_max", "right")))
    y2 = parse_float(first_value(row, ("y2", "ymax", "y_max", "bottom")))
    if None not in (x1, y1, x2, y2):
        return float(x1), float(y1), float(x2), float(y2)

    x = parse_float(first_value(row, ("x", "center_x")))
    y = parse_float(first_value(row, ("y", "center_y")))
    w = parse_float(first_value(row, ("w", "width")))
    h = parse_float(first_value(row, ("h", "height")))
    if None not in (x, y, w, h):
        if clean_key("center_x") in row or clean_key("center_y") in row:
            return float(x - w / 2), float(y - h / 2), float(x + w / 2), float(y + h / 2)
        return float(x), float(y), float(x + w), float(y + h)
    return None


def union_bbox(a: Optional[Tuple[float, float, float, float]], b: Optional[Tuple[float, float, float, float]]):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def row_to_glomerulus(row: Dict[str, str], index: int, thresholds: Thresholds, generic_crescent_active: bool) -> GlomerulusEvidence:
    row = normalize_row(row)
    slide_id = str(first_value(row, ("slide_id", "wsi_id", "case_id", "image_id")) or "unknown_slide")
    glomerulus_id = str(first_value(row, ("glomerulus_id", "glom_id", "instance_id", "roi_id")) or f"row_{index:06d}")

    subtype_raw = first_value(row, ("crescent_subtype", "subtype", "crescent_class", "pred_crescent_subtype"))
    subtype = clean_key(str(subtype_raw)) if subtype_raw else ""

    cellular = is_positive(row, CELLULAR_CRESCENT_ALIASES, thresholds.cellular_crescent)
    fibrocellular = is_positive(row, FIBROCELLULAR_CRESCENT_ALIASES, thresholds.fibrocellular_crescent)
    fibrous = is_positive(row, FIBROUS_CRESCENT_ALIASES, thresholds.fibrous_crescent)
    generic = is_positive(row, GENERIC_CRESCENT_ALIASES, thresholds.crescent)

    if subtype in ACTIVE_SUBTYPES:
        if "fibrocellular" in subtype:
            fibrocellular = True
        else:
            cellular = True
        generic = True
    elif subtype in FIBROUS_SUBTYPES:
        fibrous = True
        generic = True

    if generic and generic_crescent_active and not (cellular or fibrocellular or fibrous):
        cellular = True

    ifta = parse_fraction(first_value(row, IFTA_ALIASES))

    return GlomerulusEvidence(
        slide_id=slide_id,
        glomerulus_id=glomerulus_id,
        bbox=parse_bbox(row),
        image_path=first_value(row, ("image_path", "wsi_path", "slide_path")),
        mesangial=is_positive(row, MESANGIAL_ALIASES, thresholds.mesangial),
        endocapillary=is_positive(row, ENDOCAPILLARY_ALIASES, thresholds.endocapillary),
        segmental_sclerosis=is_positive(row, SEGMENTAL_SCLEROSIS_ALIASES, thresholds.sclerosis),
        capsular_adhesion=is_positive(row, CAPSULAR_ADHESION_ALIASES, thresholds.adhesion),
        cellular_crescent=cellular,
        fibrocellular_crescent=fibrocellular,
        fibrous_crescent=fibrous,
        generic_crescent=generic,
        fibrinoid_necrosis=is_positive(row, NECROSIS_ALIASES, thresholds.necrosis),
        fibrin_thrombus=is_positive(row, FIBRIN_THROMBUS_ALIASES, thresholds.thrombus),
        membranoproliferative_pattern=is_positive(row, MPGN_ALIASES, thresholds.mpgn),
        ifta_percent=ifta,
        source_rows=1,
    )


def merge_glomeruli(items: Iterable[GlomerulusEvidence]) -> GlomerulusEvidence:
    items = list(items)
    first = items[0]
    merged = GlomerulusEvidence(slide_id=first.slide_id, glomerulus_id=first.glomerulus_id)
    merged.image_path = first.image_path
    for item in items:
        merged.bbox = union_bbox(merged.bbox, item.bbox)
        merged.image_path = merged.image_path or item.image_path
        merged.mesangial = merged.mesangial or item.mesangial
        merged.endocapillary = merged.endocapillary or item.endocapillary
        merged.segmental_sclerosis = merged.segmental_sclerosis or item.segmental_sclerosis
        merged.capsular_adhesion = merged.capsular_adhesion or item.capsular_adhesion
        merged.cellular_crescent = merged.cellular_crescent or item.cellular_crescent
        merged.fibrocellular_crescent = merged.fibrocellular_crescent or item.fibrocellular_crescent
        merged.fibrous_crescent = merged.fibrous_crescent or item.fibrous_crescent
        merged.generic_crescent = merged.generic_crescent or item.generic_crescent
        merged.fibrinoid_necrosis = merged.fibrinoid_necrosis or item.fibrinoid_necrosis
        merged.fibrin_thrombus = merged.fibrin_thrombus or item.fibrin_thrombus
        merged.membranoproliferative_pattern = (
            merged.membranoproliferative_pattern or item.membranoproliferative_pattern
        )
        merged.ifta_percent = item.ifta_percent if item.ifta_percent is not None else merged.ifta_percent
        merged.source_rows += item.source_rows
    return merged


def load_glomeruli(csv_path: Path, thresholds: Thresholds, generic_crescent_active: bool) -> List[GlomerulusEvidence]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    raw_items = [row_to_glomerulus(row, index, thresholds, generic_crescent_active) for index, row in enumerate(rows)]
    grouped: Dict[Tuple[str, str], List[GlomerulusEvidence]] = defaultdict(list)
    for item in raw_items:
        grouped[(item.slide_id, item.glomerulus_id)].append(item)
    return [merge_glomeruli(group) for group in grouped.values()]


def ratio(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def score_oxford(glomeruli: Sequence[GlomerulusEvidence]) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, float], List[str]]:
    total = len(glomeruli)
    counts = {
        "mesangial": sum(item.mesangial for item in glomeruli),
        "endocapillary": sum(item.endocapillary for item in glomeruli),
        "segmental_sclerosis_or_adhesion": sum(item.segmental_sclerosis or item.capsular_adhesion for item in glomeruli),
        "cellular_crescent": sum(item.cellular_crescent for item in glomeruli),
        "fibrocellular_crescent": sum(item.fibrocellular_crescent for item in glomeruli),
        "fibrous_crescent": sum(item.fibrous_crescent for item in glomeruli),
        "generic_crescent": sum(item.generic_crescent for item in glomeruli),
        "active_crescent": sum(item.active_crescent for item in glomeruli),
        "any_crescent": sum(item.any_crescent for item in glomeruli),
    }
    percentages = {key: ratio(value, total) for key, value in counts.items()}
    ifta_values = [item.ifta_percent for item in glomeruli if item.ifta_percent is not None]
    ifta = max(ifta_values) if ifta_values else None

    m_score = "M1" if percentages["mesangial"] >= 0.5 else "M0"
    e_score = "E1" if counts["endocapillary"] > 0 else "E0"
    s_score = "S1" if counts["segmental_sclerosis_or_adhesion"] > 0 else "S0"
    if ifta is None:
        t_score = "T_NA"
    elif ifta <= 0.25:
        t_score = "T0"
    elif ifta <= 0.50:
        t_score = "T1"
    else:
        t_score = "T2"

    active_ratio = percentages["active_crescent"]
    if counts["active_crescent"] == 0:
        c_score = "C0"
    elif active_ratio < 0.25:
        c_score = "C1"
    else:
        c_score = "C2"

    warnings = []
    if t_score == "T_NA":
        warnings.append("Oxford T score is unavailable because IFTA/tubulointerstitial fibrosis percent was not provided.")

    oxford = {
        "M": m_score,
        "E": e_score,
        "S": s_score,
        "T": t_score,
        "C": c_score,
        "MEST_C": f"{m_score}{e_score}{s_score}{t_score}{c_score}",
    }
    if ifta is not None:
        percentages["ifta"] = ifta
    return oxford, counts, percentages, warnings


def score_iskdc(
    glomeruli: Sequence[GlomerulusEvidence],
    vi_overrides: bool = True,
    crescent_mode: str = "all",
) -> Dict[str, str]:
    total = len(glomeruli)
    mesangial_count = sum(item.mesangial for item in glomeruli)
    endocapillary_count = sum(item.endocapillary for item in glomeruli)
    if crescent_mode == "active":
        crescent_count = sum(item.active_crescent for item in glomeruli)
    else:
        crescent_count = sum(item.any_crescent for item in glomeruli)
    segmental_count = sum(item.segmental_or_necrotic for item in glomeruli)
    mpgn_count = sum(item.membranoproliferative_pattern for item in glomeruli)

    mesangial_ratio = ratio(mesangial_count, total)
    crescent_ratio = ratio(crescent_count, total)

    if vi_overrides and mpgn_count > 0:
        grade = "VI"
        description = "membranoproliferative-like lesion"
    elif crescent_ratio >= 0.75:
        grade = "V"
        description = "crescentic pattern with >=75% glomeruli involved"
    elif crescent_ratio >= 0.50:
        grade = "IV"
        description = "diffuse crescentic pattern with 50-75% glomeruli involved"
    elif segmental_count > 0 or endocapillary_count > 0:
        if mesangial_ratio >= 0.50:
            grade = "IIIb"
            description = "diffuse mesangial proliferative background with <50% segmental/crescentic lesions"
        else:
            grade = "IIIa"
            description = "focal mesangial proliferative background with <50% segmental/crescentic lesions"
    elif mesangial_count > 0:
        grade = "II"
        description = "mesangial proliferative pattern"
    else:
        grade = "I"
        description = "minimal glomerular abnormality by available model evidence"

    return {
        "grade": grade,
        "description": description,
        "crescent_mode": crescent_mode,
        "crescent_percent": f"{crescent_ratio:.4f}",
        "mesangial_percent": f"{mesangial_ratio:.4f}",
    }


def score_slide(
    slide_id: str,
    glomeruli: Sequence[GlomerulusEvidence],
    min_glomeruli: int = 8,
    iskdc_vi_overrides: bool = True,
    iskdc_crescent_mode: str = "all",
) -> SlideScore:
    oxford, counts, percentages, warnings = score_oxford(glomeruli)
    iskdc = score_iskdc(glomeruli, vi_overrides=iskdc_vi_overrides, crescent_mode=iskdc_crescent_mode)
    total = len(glomeruli)

    counts.update(
        {
            "iskdc_segmental_or_necrotic": sum(item.segmental_or_necrotic for item in glomeruli),
            "fibrinoid_necrosis": sum(item.fibrinoid_necrosis for item in glomeruli),
            "fibrin_thrombus": sum(item.fibrin_thrombus for item in glomeruli),
            "membranoproliferative_pattern": sum(item.membranoproliferative_pattern for item in glomeruli),
        }
    )
    percentages.update({key: ratio(value, total) for key, value in counts.items() if key not in percentages})

    if total < min_glomeruli:
        warnings.append(f"Only {total} glomeruli were scored; review reliability for WSI-level pathology grading.")
    if counts["mesangial"] > 0 and counts["mesangial"] + counts["endocapillary"] < total * 0.1:
        warnings.append("Proliferation model evidence is sparse; review negative and hard-negative cases.")

    return SlideScore(
        slide_id=slide_id,
        glomeruli=list(glomeruli),
        total_glomeruli=total,
        counts=counts,
        percentages=percentages,
        oxford=oxford,
        iskdc=iskdc,
        warnings=warnings,
    )


def group_and_score(
    glomeruli: Sequence[GlomerulusEvidence],
    min_glomeruli: int = 8,
    iskdc_vi_overrides: bool = True,
    iskdc_crescent_mode: str = "all",
) -> List[SlideScore]:
    by_key: Dict[Tuple[str, str], List[GlomerulusEvidence]] = defaultdict(list)
    for item in glomeruli:
        by_key[(item.slide_id, item.glomerulus_id)].append(item)

    by_slide: Dict[str, List[GlomerulusEvidence]] = defaultdict(list)
    for (slide_id, _), items in by_key.items():
        by_slide[slide_id].append(merge_glomeruli(items))

    return [
        score_slide(
            slide_id,
            items,
            min_glomeruli=min_glomeruli,
            iskdc_vi_overrides=iskdc_vi_overrides,
            iskdc_crescent_mode=iskdc_crescent_mode,
        )
        for slide_id, items in sorted(by_slide.items())
    ]


def bool_text(value: bool) -> str:
    return "1" if value else "0"


def write_summary_csv(scores: Sequence[SlideScore], path: Path) -> None:
    fields = [
        "slide_id",
        "total_glomeruli",
        "ISKDC",
        "ISKDC_description",
        "Oxford_MEST_C",
        "M",
        "E",
        "S",
        "T",
        "C",
        "mesangial_count",
        "mesangial_percent",
        "endocapillary_count",
        "endocapillary_percent",
        "segmental_sclerosis_or_adhesion_count",
        "segmental_sclerosis_or_adhesion_percent",
        "active_crescent_count",
        "active_crescent_percent",
        "any_crescent_count",
        "any_crescent_percent",
        "cellular_crescent_count",
        "cellular_crescent_percent",
        "fibrocellular_crescent_count",
        "fibrocellular_crescent_percent",
        "fibrous_crescent_count",
        "fibrous_crescent_percent",
        "generic_crescent_count",
        "generic_crescent_percent",
        "ifta_percent",
        "warnings",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "slide_id": score.slide_id,
                    "total_glomeruli": score.total_glomeruli,
                    "ISKDC": score.iskdc["grade"],
                    "ISKDC_description": score.iskdc["description"],
                    "Oxford_MEST_C": score.oxford["MEST_C"],
                    "M": score.oxford["M"],
                    "E": score.oxford["E"],
                    "S": score.oxford["S"],
                    "T": score.oxford["T"],
                    "C": score.oxford["C"],
                    "mesangial_count": score.counts["mesangial"],
                    "mesangial_percent": f"{score.percentages['mesangial']:.4f}",
                    "endocapillary_count": score.counts["endocapillary"],
                    "endocapillary_percent": f"{score.percentages['endocapillary']:.4f}",
                    "segmental_sclerosis_or_adhesion_count": score.counts["segmental_sclerosis_or_adhesion"],
                    "segmental_sclerosis_or_adhesion_percent": f"{score.percentages['segmental_sclerosis_or_adhesion']:.4f}",
                    "active_crescent_count": score.counts["active_crescent"],
                    "active_crescent_percent": f"{score.percentages['active_crescent']:.4f}",
                    "any_crescent_count": score.counts["any_crescent"],
                    "any_crescent_percent": f"{score.percentages['any_crescent']:.4f}",
                    "cellular_crescent_count": score.counts["cellular_crescent"],
                    "cellular_crescent_percent": f"{score.percentages['cellular_crescent']:.4f}",
                    "fibrocellular_crescent_count": score.counts["fibrocellular_crescent"],
                    "fibrocellular_crescent_percent": f"{score.percentages['fibrocellular_crescent']:.4f}",
                    "fibrous_crescent_count": score.counts["fibrous_crescent"],
                    "fibrous_crescent_percent": f"{score.percentages['fibrous_crescent']:.4f}",
                    "generic_crescent_count": score.counts["generic_crescent"],
                    "generic_crescent_percent": f"{score.percentages['generic_crescent']:.4f}",
                    "ifta_percent": f"{score.percentages['ifta']:.4f}" if "ifta" in score.percentages else "",
                    "warnings": " | ".join(score.warnings),
                }
            )


def write_iskdc_csv(scores: Sequence[SlideScore], path: Path) -> None:
    fields = [
        "slide_id",
        "total_glomeruli",
        "ISKDC",
        "ISKDC_description",
        "crescent_mode",
        "mesangial_count",
        "mesangial_percent",
        "endocapillary_count",
        "endocapillary_percent",
        "iskdc_segmental_or_necrotic_count",
        "iskdc_segmental_or_necrotic_percent",
        "any_crescent_count",
        "any_crescent_percent",
        "cellular_crescent_count",
        "cellular_crescent_percent",
        "fibrocellular_crescent_count",
        "fibrocellular_crescent_percent",
        "fibrous_crescent_count",
        "fibrous_crescent_percent",
        "generic_crescent_count",
        "generic_crescent_percent",
        "fibrinoid_necrosis_count",
        "fibrin_thrombus_count",
        "membranoproliferative_pattern_count",
        "warnings",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "slide_id": score.slide_id,
                    "total_glomeruli": score.total_glomeruli,
                    "ISKDC": score.iskdc["grade"],
                    "ISKDC_description": score.iskdc["description"],
                    "crescent_mode": score.iskdc["crescent_mode"],
                    "mesangial_count": score.counts["mesangial"],
                    "mesangial_percent": f"{score.percentages['mesangial']:.4f}",
                    "endocapillary_count": score.counts["endocapillary"],
                    "endocapillary_percent": f"{score.percentages['endocapillary']:.4f}",
                    "iskdc_segmental_or_necrotic_count": score.counts["iskdc_segmental_or_necrotic"],
                    "iskdc_segmental_or_necrotic_percent": f"{score.percentages['iskdc_segmental_or_necrotic']:.4f}",
                    "any_crescent_count": score.counts["any_crescent"],
                    "any_crescent_percent": f"{score.percentages['any_crescent']:.4f}",
                    "cellular_crescent_count": score.counts["cellular_crescent"],
                    "cellular_crescent_percent": f"{score.percentages['cellular_crescent']:.4f}",
                    "fibrocellular_crescent_count": score.counts["fibrocellular_crescent"],
                    "fibrocellular_crescent_percent": f"{score.percentages['fibrocellular_crescent']:.4f}",
                    "fibrous_crescent_count": score.counts["fibrous_crescent"],
                    "fibrous_crescent_percent": f"{score.percentages['fibrous_crescent']:.4f}",
                    "generic_crescent_count": score.counts["generic_crescent"],
                    "generic_crescent_percent": f"{score.percentages['generic_crescent']:.4f}",
                    "fibrinoid_necrosis_count": score.counts["fibrinoid_necrosis"],
                    "fibrin_thrombus_count": score.counts["fibrin_thrombus"],
                    "membranoproliferative_pattern_count": score.counts["membranoproliferative_pattern"],
                    "warnings": " | ".join(score.warnings),
                }
            )


def write_oxford_csv(scores: Sequence[SlideScore], path: Path) -> None:
    fields = [
        "slide_id",
        "total_glomeruli",
        "Oxford_MEST_C",
        "M",
        "E",
        "S",
        "T",
        "C",
        "mesangial_count",
        "mesangial_percent",
        "endocapillary_count",
        "endocapillary_percent",
        "segmental_sclerosis_or_adhesion_count",
        "segmental_sclerosis_or_adhesion_percent",
        "active_crescent_count",
        "active_crescent_percent",
        "cellular_crescent_count",
        "cellular_crescent_percent",
        "fibrocellular_crescent_count",
        "fibrocellular_crescent_percent",
        "fibrous_crescent_review_count",
        "fibrous_crescent_review_percent",
        "ifta_percent",
        "warnings",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "slide_id": score.slide_id,
                    "total_glomeruli": score.total_glomeruli,
                    "Oxford_MEST_C": score.oxford["MEST_C"],
                    "M": score.oxford["M"],
                    "E": score.oxford["E"],
                    "S": score.oxford["S"],
                    "T": score.oxford["T"],
                    "C": score.oxford["C"],
                    "mesangial_count": score.counts["mesangial"],
                    "mesangial_percent": f"{score.percentages['mesangial']:.4f}",
                    "endocapillary_count": score.counts["endocapillary"],
                    "endocapillary_percent": f"{score.percentages['endocapillary']:.4f}",
                    "segmental_sclerosis_or_adhesion_count": score.counts["segmental_sclerosis_or_adhesion"],
                    "segmental_sclerosis_or_adhesion_percent": f"{score.percentages['segmental_sclerosis_or_adhesion']:.4f}",
                    "active_crescent_count": score.counts["active_crescent"],
                    "active_crescent_percent": f"{score.percentages['active_crescent']:.4f}",
                    "cellular_crescent_count": score.counts["cellular_crescent"],
                    "cellular_crescent_percent": f"{score.percentages['cellular_crescent']:.4f}",
                    "fibrocellular_crescent_count": score.counts["fibrocellular_crescent"],
                    "fibrocellular_crescent_percent": f"{score.percentages['fibrocellular_crescent']:.4f}",
                    "fibrous_crescent_review_count": score.counts["fibrous_crescent"],
                    "fibrous_crescent_review_percent": f"{score.percentages['fibrous_crescent']:.4f}",
                    "ifta_percent": f"{score.percentages['ifta']:.4f}" if "ifta" in score.percentages else "",
                    "warnings": " | ".join(score.warnings),
                }
            )


def write_glomerulus_csv(scores: Sequence[SlideScore], path: Path) -> None:
    fields = [
        "slide_id",
        "glomerulus_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "mesangial",
        "endocapillary",
        "segmental_sclerosis",
        "capsular_adhesion",
        "cellular_crescent",
        "fibrocellular_crescent",
        "fibrous_crescent",
        "generic_crescent",
        "active_crescent",
        "any_crescent",
        "fibrinoid_necrosis",
        "fibrin_thrombus",
        "membranoproliferative_pattern",
        "ifta_percent",
        "source_rows",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for score in scores:
            for item in score.glomeruli:
                bbox = item.bbox or ("", "", "", "")
                writer.writerow(
                    {
                        "slide_id": item.slide_id,
                        "glomerulus_id": item.glomerulus_id,
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "mesangial": bool_text(item.mesangial),
                        "endocapillary": bool_text(item.endocapillary),
                        "segmental_sclerosis": bool_text(item.segmental_sclerosis),
                        "capsular_adhesion": bool_text(item.capsular_adhesion),
                        "cellular_crescent": bool_text(item.cellular_crescent),
                        "fibrocellular_crescent": bool_text(item.fibrocellular_crescent),
                        "fibrous_crescent": bool_text(item.fibrous_crescent),
                        "generic_crescent": bool_text(item.generic_crescent),
                        "active_crescent": bool_text(item.active_crescent),
                        "any_crescent": bool_text(item.any_crescent),
                        "fibrinoid_necrosis": bool_text(item.fibrinoid_necrosis),
                        "fibrin_thrombus": bool_text(item.fibrin_thrombus),
                        "membranoproliferative_pattern": bool_text(item.membranoproliferative_pattern),
                        "ifta_percent": "" if item.ifta_percent is None else f"{item.ifta_percent:.4f}",
                        "source_rows": item.source_rows,
                    }
                )


def scores_to_json(scores: Sequence[SlideScore]) -> List[Dict[str, object]]:
    payload = []
    for score in scores:
        payload.append(
            {
                "slide_id": score.slide_id,
                "total_glomeruli": score.total_glomeruli,
                "counts": score.counts,
                "percentages": score.percentages,
                "oxford": score.oxford,
                "iskdc": score.iskdc,
                "warnings": score.warnings,
                "glomeruli": [
                    {
                        "glomerulus_id": item.glomerulus_id,
                        "bbox": item.bbox,
                        "mesangial": item.mesangial,
                        "endocapillary": item.endocapillary,
                        "segmental_sclerosis": item.segmental_sclerosis,
                        "capsular_adhesion": item.capsular_adhesion,
                        "cellular_crescent": item.cellular_crescent,
                        "fibrocellular_crescent": item.fibrocellular_crescent,
                        "fibrous_crescent": item.fibrous_crescent,
                        "generic_crescent": item.generic_crescent,
                        "active_crescent": item.active_crescent,
                        "any_crescent": item.any_crescent,
                        "fibrinoid_necrosis": item.fibrinoid_necrosis,
                        "fibrin_thrombus": item.fibrin_thrombus,
                        "membranoproliferative_pattern": item.membranoproliferative_pattern,
                        "ifta_percent": item.ifta_percent,
                    }
                    for item in score.glomeruli
                ],
            }
        )
    return payload


def select_fields(mapping: Dict[str, object], fields: Sequence[str]) -> Dict[str, object]:
    return {field: mapping[field] for field in fields if field in mapping}


def scores_to_iskdc_json(scores: Sequence[SlideScore]) -> List[Dict[str, object]]:
    count_fields = (
        "mesangial",
        "endocapillary",
        "iskdc_segmental_or_necrotic",
        "any_crescent",
        "cellular_crescent",
        "fibrocellular_crescent",
        "fibrous_crescent",
        "generic_crescent",
        "fibrinoid_necrosis",
        "fibrin_thrombus",
        "membranoproliferative_pattern",
    )
    payload = []
    for score in scores:
        payload.append(
            {
                "slide_id": score.slide_id,
                "total_glomeruli": score.total_glomeruli,
                "iskdc": score.iskdc,
                "counts": select_fields(score.counts, count_fields),
                "percentages": select_fields(score.percentages, count_fields),
                "warnings": score.warnings,
                "glomeruli": [
                    {
                        "glomerulus_id": item.glomerulus_id,
                        "bbox": item.bbox,
                        "mesangial": item.mesangial,
                        "endocapillary": item.endocapillary,
                        "segmental_or_necrotic": item.segmental_or_necrotic,
                        "cellular_crescent": item.cellular_crescent,
                        "fibrocellular_crescent": item.fibrocellular_crescent,
                        "fibrous_crescent": item.fibrous_crescent,
                        "generic_crescent": item.generic_crescent,
                        "any_crescent": item.any_crescent,
                        "fibrinoid_necrosis": item.fibrinoid_necrosis,
                        "fibrin_thrombus": item.fibrin_thrombus,
                        "membranoproliferative_pattern": item.membranoproliferative_pattern,
                    }
                    for item in score.glomeruli
                ],
            }
        )
    return payload


def scores_to_oxford_json(scores: Sequence[SlideScore]) -> List[Dict[str, object]]:
    count_fields = (
        "mesangial",
        "endocapillary",
        "segmental_sclerosis_or_adhesion",
        "active_crescent",
        "cellular_crescent",
        "fibrocellular_crescent",
        "fibrous_crescent",
        "ifta",
    )
    payload = []
    for score in scores:
        payload.append(
            {
                "slide_id": score.slide_id,
                "total_glomeruli": score.total_glomeruli,
                "oxford": score.oxford,
                "counts": select_fields(score.counts, count_fields),
                "percentages": select_fields(score.percentages, count_fields),
                "warnings": score.warnings,
                "glomeruli": [
                    {
                        "glomerulus_id": item.glomerulus_id,
                        "bbox": item.bbox,
                        "mesangial": item.mesangial,
                        "endocapillary": item.endocapillary,
                        "segmental_sclerosis": item.segmental_sclerosis,
                        "capsular_adhesion": item.capsular_adhesion,
                        "cellular_crescent": item.cellular_crescent,
                        "fibrocellular_crescent": item.fibrocellular_crescent,
                        "fibrous_crescent_review": item.fibrous_crescent,
                        "active_crescent": item.active_crescent,
                        "ifta_percent": item.ifta_percent,
                    }
                    for item in score.glomeruli
                ],
            }
        )
    return payload


def find_slide_image(score: SlideScore, images_dir: Optional[Path]) -> Optional[Path]:
    for item in score.glomeruli:
        if item.image_path:
            path = Path(item.image_path)
            if path.exists():
                return path
            if images_dir and (images_dir / item.image_path).exists():
                return images_dir / item.image_path
    if images_dir:
        for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            candidate = images_dir / f"{score.slide_id}{suffix}"
            if candidate.exists():
                return candidate
    return None


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def color_for_glomerulus(item: GlomerulusEvidence) -> Tuple[int, int, int]:
    if item.active_crescent:
        return (220, 45, 45)
    if item.fibrous_crescent:
        return (235, 135, 25)
    if item.fibrinoid_necrosis or item.fibrin_thrombus:
        return (145, 55, 190)
    if item.segmental_sclerosis or item.capsular_adhesion:
        return (45, 100, 210)
    if item.mesangial or item.endocapillary:
        return (30, 150, 75)
    return (140, 140, 140)


def draw_bar(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], width: int, label: str, value: float, color: Tuple[int, int, int], font) -> None:
    x, y = xy
    draw.text((x, y), label, fill=(30, 30, 30), font=font)
    bar_x = x + 210
    bar_y = y + 4
    draw.rectangle((bar_x, bar_y, bar_x + width, bar_y + 18), outline=(210, 210, 210), width=1)
    fill_width = int(max(0.0, min(1.0, value)) * width)
    draw.rectangle((bar_x, bar_y, bar_x + fill_width, bar_y + 18), fill=color)
    draw.text((bar_x + width + 12, y), f"{value * 100:.1f}%", fill=(30, 30, 30), font=font)


def evidence_titles(score: SlideScore, mode: str) -> Tuple[str, str]:
    if mode == "iskdc":
        return (
            f"HSPN ISKDC evidence: {score.slide_id}",
            f"ISKDC {score.iskdc['grade']} | glomeruli={score.total_glomeruli}",
        )
    if mode == "oxford":
        return (
            f"HSPN Oxford MEST-C evidence: {score.slide_id}",
            f"Oxford {score.oxford['MEST_C']} | glomeruli={score.total_glomeruli}",
        )
    return (
        f"HSPN pathology evidence: {score.slide_id}",
        f"ISKDC {score.iskdc['grade']} | Oxford {score.oxford['MEST_C']} | glomeruli={score.total_glomeruli}",
    )


def evidence_bars(score: SlideScore, mode: str) -> List[Tuple[str, float, Tuple[int, int, int]]]:
    if mode == "iskdc":
        return [
            ("Mesangial", score.percentages["mesangial"], (30, 150, 75)),
            ("Endocapillary", score.percentages["endocapillary"], (30, 150, 75)),
            ("Segmental/necro", score.percentages["iskdc_segmental_or_necrotic"], (145, 55, 190)),
            ("Any crescents", score.percentages["any_crescent"], (235, 135, 25)),
            ("MPGN pattern", score.percentages["membranoproliferative_pattern"], (80, 80, 80)),
        ]
    if mode == "oxford":
        bars = [
            ("Mesangial", score.percentages["mesangial"], (30, 150, 75)),
            ("Endocapillary", score.percentages["endocapillary"], (30, 150, 75)),
            ("Sclerosis/adhesion", score.percentages["segmental_sclerosis_or_adhesion"], (45, 100, 210)),
            ("Active crescents", score.percentages["active_crescent"], (220, 45, 45)),
        ]
        if "ifta" in score.percentages:
            bars.append(("IFTA", score.percentages["ifta"], (90, 90, 90)))
        return bars
    return [
        ("Mesangial", score.percentages["mesangial"], (30, 150, 75)),
        ("Endocapillary", score.percentages["endocapillary"], (30, 150, 75)),
        ("Sclerosis/adhesion", score.percentages["segmental_sclerosis_or_adhesion"], (45, 100, 210)),
        ("Active crescents", score.percentages["active_crescent"], (220, 45, 45)),
        ("Any crescents", score.percentages["any_crescent"], (235, 135, 25)),
    ]


def evidence_count_lines(score: SlideScore, mode: str) -> List[str]:
    if mode == "iskdc":
        return [
            f"Mesangial: {score.counts['mesangial']}/{score.total_glomeruli}",
            f"Endocapillary: {score.counts['endocapillary']}/{score.total_glomeruli}",
            f"Segmental/necro: {score.counts['iskdc_segmental_or_necrotic']}/{score.total_glomeruli}",
            f"Any crescent: {score.counts['any_crescent']}/{score.total_glomeruli}",
            f"Cellular crescent: {score.counts['cellular_crescent']}/{score.total_glomeruli}",
            f"Fibrocellular crescent: {score.counts['fibrocellular_crescent']}/{score.total_glomeruli}",
            f"Fibrous crescent: {score.counts['fibrous_crescent']}/{score.total_glomeruli}",
            f"MPGN pattern: {score.counts['membranoproliferative_pattern']}/{score.total_glomeruli}",
        ]
    if mode == "oxford":
        return [
            f"M: {score.counts['mesangial']}/{score.total_glomeruli}",
            f"E: {score.counts['endocapillary']}/{score.total_glomeruli}",
            f"S/adhesion: {score.counts['segmental_sclerosis_or_adhesion']}/{score.total_glomeruli}",
            f"Active crescent: {score.counts['active_crescent']}/{score.total_glomeruli}",
            f"Cellular crescent: {score.counts['cellular_crescent']}/{score.total_glomeruli}",
            f"Fibrocellular crescent: {score.counts['fibrocellular_crescent']}/{score.total_glomeruli}",
            f"Fibrous review only: {score.counts['fibrous_crescent']}/{score.total_glomeruli}",
            "IFTA: {}".format(f"{score.percentages['ifta'] * 100:.1f}%" if "ifta" in score.percentages else "not provided"),
        ]
    return [
        f"M: {score.counts['mesangial']}/{score.total_glomeruli}",
        f"E: {score.counts['endocapillary']}/{score.total_glomeruli}",
        f"S/adhesion: {score.counts['segmental_sclerosis_or_adhesion']}/{score.total_glomeruli}",
        f"Active crescent: {score.counts['active_crescent']}/{score.total_glomeruli}",
        f"Any crescent: {score.counts['any_crescent']}/{score.total_glomeruli}",
        f"Cellular crescent: {score.counts['cellular_crescent']}/{score.total_glomeruli}",
        f"Fibrocellular crescent: {score.counts['fibrocellular_crescent']}/{score.total_glomeruli}",
        f"Fibrous crescent: {score.counts['fibrous_crescent']}/{score.total_glomeruli}",
        f"Necrosis/thrombus: {score.counts['fibrinoid_necrosis'] + score.counts['fibrin_thrombus']}/{score.total_glomeruli}",
    ]


def evidence_rule_lines(score: SlideScore, mode: str) -> List[str]:
    if mode == "iskdc":
        return [
            f"ISKDC: {score.iskdc['grade']}",
            score.iskdc["description"],
            f"Crescent mode: {score.iskdc['crescent_mode']}",
        ]
    if mode == "oxford":
        return [
            f"Oxford: {score.oxford['MEST_C']}",
            f"M={score.oxford['M']} E={score.oxford['E']} S={score.oxford['S']} T={score.oxford['T']} C={score.oxford['C']}",
            "C uses cellular + fibrocellular crescents.",
        ]
    return [
        f"Oxford: {score.oxford['MEST_C']}",
        f"ISKDC: {score.iskdc['grade']}",
        score.iskdc["description"],
    ]


def render_evidence(
    score: SlideScore,
    output_path: Path,
    images_dir: Optional[Path],
    preview_max_size: int = 780,
    mode: str = "combined",
) -> None:
    canvas = Image.new("RGB", (1320, 900), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    header_font = load_font(22)
    body_font = load_font(17)
    small_font = load_font(13)

    title, subtitle = evidence_titles(score, mode)
    draw.text((28, 22), title, fill=(20, 35, 55), font=title_font)
    draw.text((28, 68), subtitle, fill=(40, 40, 40), font=header_font)

    image_path = find_slide_image(score, images_dir)
    overlay_box = (28, 115, 760, 840)
    if image_path and image_path.exists():
        try:
            image = Image.open(image_path).convert("RGB")
            original_w, original_h = image.size
            image.thumbnail((overlay_box[2] - overlay_box[0], overlay_box[3] - overlay_box[1]))
            image_x = overlay_box[0]
            image_y = overlay_box[1]
            canvas.paste(image, (image_x, image_y))
            scale_x = image.width / original_w
            scale_y = image.height / original_h
            for item in score.glomeruli:
                if item.bbox is None:
                    continue
                color = color_for_glomerulus(item)
                x1, y1, x2, y2 = item.bbox
                box = (
                    image_x + int(x1 * scale_x),
                    image_y + int(y1 * scale_y),
                    image_x + int(x2 * scale_x),
                    image_y + int(y2 * scale_y),
                )
                draw.rectangle(box, outline=color, width=3)
            draw.text((image_x, image_y + image.height + 8), str(image_path), fill=(80, 80, 80), font=small_font)
        except Exception as exc:  # pragma: no cover - defensive visualization fallback
            draw.rectangle(overlay_box, outline=(210, 210, 210), width=1)
            draw.text((44, 140), f"Could not render slide image: {exc}", fill=(150, 50, 50), font=body_font)
    else:
        draw.rectangle(overlay_box, outline=(210, 210, 210), width=1)
        draw.text((44, 140), "No slide image was provided; showing count-based evidence only.", fill=(70, 70, 70), font=body_font)

    legend_x = 790
    y = 120
    draw.text((legend_x, y), "Key evidence ratios", fill=(20, 35, 55), font=header_font)
    y += 46
    for label, value, color in evidence_bars(score, mode):
        draw_bar(draw, (legend_x, y), 230, label, value, color, body_font)
        y += 36
    y += 50

    draw.text((legend_x, y), "Counts", fill=(20, 35, 55), font=header_font)
    y += 38
    for line in evidence_count_lines(score, mode):
        draw.text((legend_x, y), line, fill=(35, 35, 35), font=body_font)
        y += 28

    y += 18
    draw.text((legend_x, y), "Rule output", fill=(20, 35, 55), font=header_font)
    y += 36
    for line in evidence_rule_lines(score, mode):
        draw.text((legend_x, y), line, fill=(35, 35, 35), font=body_font)
        y += 28

    if score.warnings:
        y += 16
        draw.text((legend_x, y), "Warnings", fill=(160, 70, 0), font=header_font)
        y += 34
        for warning in score.warnings[:5]:
            draw.text((legend_x, y), f"- {warning}", fill=(120, 60, 0), font=small_font)
            y += 22

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_json(scores: Sequence[SlideScore], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(scores_to_json(scores), handle, indent=2, ensure_ascii=False)


def write_iskdc_json(scores: Sequence[SlideScore], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(scores_to_iskdc_json(scores), handle, indent=2, ensure_ascii=False)


def write_oxford_json(scores: Sequence[SlideScore], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(scores_to_oxford_json(scores), handle, indent=2, ensure_ascii=False)


def write_report_outputs(
    scores: Sequence[SlideScore],
    output_dir: Path,
    images_dir: Optional[Path],
    preview_max_size: int = 780,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_iskdc_csv(scores, output_dir / "iskdc_scores.csv")
    write_oxford_csv(scores, output_dir / "oxford_scores.csv")
    write_iskdc_json(scores, output_dir / "iskdc_scores.json")
    write_oxford_json(scores, output_dir / "oxford_scores.json")
    write_glomerulus_csv(scores, output_dir / "glomerulus_evidence.csv")

    # Backward-compatible combined outputs for old notebooks and report links.
    write_summary_csv(scores, output_dir / "slide_scores.csv")
    write_json(scores, output_dir / "slide_scores.json")

    for score in scores:
        render_evidence(
            score,
            output_dir / "iskdc_evidence" / f"{score.slide_id}_iskdc_evidence.png",
            images_dir=images_dir,
            preview_max_size=preview_max_size,
            mode="iskdc",
        )
        render_evidence(
            score,
            output_dir / "oxford_evidence" / f"{score.slide_id}_oxford_evidence.png",
            images_dir=images_dir,
            preview_max_size=preview_max_size,
            mode="oxford",
        )
        render_evidence(
            score,
            output_dir / "evidence" / f"{score.slide_id}_evidence.png",
            images_dir=images_dir,
            preview_max_size=preview_max_size,
            mode="combined",
        )


def write_template(path: Path) -> None:
    fields = [
        "slide_id",
        "glomerulus_id",
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
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score HSPN WSI predictions with ISKDC and Oxford MEST-C rules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", type=Path, help="Per-glomerulus prediction CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hspn_pathology_scores"), help="Output directory.")
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional directory containing slide preview images.")
    parser.add_argument("--write-template", type=Path, default=None, help="Write an empty input CSV template and exit.")
    parser.add_argument("--generic-crescent-active", action="store_true", default=True,
                        help="Treat generic crescent detections without subtype as active crescents for Oxford C.")
    parser.add_argument("--no-generic-crescent-active", dest="generic_crescent_active", action="store_false")
    parser.add_argument("--iskdc-crescent-mode", choices=("all", "active"), default="all",
                        help="Crescent denominator for ISKDC IV/V rules.")
    parser.add_argument("--no-iskdc-vi-override", dest="iskdc_vi_overrides", action="store_false", default=True,
                        help="Do not let membranoproliferative pattern override the ISKDC grade to VI.")
    parser.add_argument("--min-glomeruli", type=int, default=8, help="Warn below this scored glomerulus count.")
    parser.add_argument("--preview-max-size", type=int, default=780, help="Maximum preview size inside evidence PNG.")
    parser.add_argument("--mesangial-threshold", type=float, default=0.5)
    parser.add_argument("--endocapillary-threshold", type=float, default=0.5)
    parser.add_argument("--sclerosis-threshold", type=float, default=0.5)
    parser.add_argument("--adhesion-threshold", type=float, default=0.5)
    parser.add_argument("--crescent-threshold", type=float, default=0.5)
    parser.add_argument("--cellular-crescent-threshold", type=float, default=0.5)
    parser.add_argument("--fibrocellular-crescent-threshold", type=float, default=0.5)
    parser.add_argument("--fibrous-crescent-threshold", type=float, default=0.5)
    parser.add_argument("--necrosis-threshold", type=float, default=0.5)
    parser.add_argument("--thrombus-threshold", type=float, default=0.5)
    parser.add_argument("--mpgn-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_template:
        write_template(args.write_template)
        print(f"Wrote input template: {args.write_template}")
        return
    if args.input_csv is None:
        raise SystemExit("--input-csv is required unless --write-template is used.")

    thresholds = Thresholds(
        mesangial=args.mesangial_threshold,
        endocapillary=args.endocapillary_threshold,
        sclerosis=args.sclerosis_threshold,
        adhesion=args.adhesion_threshold,
        crescent=args.crescent_threshold,
        cellular_crescent=args.cellular_crescent_threshold,
        fibrocellular_crescent=args.fibrocellular_crescent_threshold,
        fibrous_crescent=args.fibrous_crescent_threshold,
        necrosis=args.necrosis_threshold,
        thrombus=args.thrombus_threshold,
        mpgn=args.mpgn_threshold,
    )
    glomeruli = load_glomeruli(args.input_csv, thresholds, generic_crescent_active=args.generic_crescent_active)
    scores = group_and_score(
        glomeruli,
        min_glomeruli=args.min_glomeruli,
        iskdc_vi_overrides=args.iskdc_vi_overrides,
        iskdc_crescent_mode=args.iskdc_crescent_mode,
    )

    write_report_outputs(scores, args.output_dir, images_dir=args.images_dir, preview_max_size=args.preview_max_size)
    print(f"Wrote {len(scores)} slide score(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
