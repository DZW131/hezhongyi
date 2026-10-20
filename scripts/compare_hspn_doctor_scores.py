"""Compare frozen HSPN model scores with case-level reference labels.

This utility intentionally performs no threshold fitting.  It maps scene-level
glomerulus predictions to the doctor-reference case manifest, merges all scenes
belonging to a case, and applies the existing ISKDC/Oxford scoring rules once
per case.  Tubular-atrophy burdens are pooled by tissue-pixel weight.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_hspn_pathology import (  # noqa: E402
    Thresholds,
    group_and_score,
    load_glomeruli,
    write_glomerulus_csv,
    write_iskdc_csv,
    write_iskdc_json,
    write_json,
    write_oxford_csv,
    write_oxford_json,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--tubular-csv", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesangial-threshold", type=float, default=0.35)
    parser.add_argument("--endocapillary-threshold", type=float, default=0.20)
    parser.add_argument("--crescent-threshold", type=float, default=0.30)
    parser.add_argument("--min-glomeruli", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def normalize_iskdc(value: object) -> str:
    text = str(value or "").strip().upper()
    text = text.replace("HSPN", "").replace(" ", "")
    for source, target in (
        ("Ⅵ", "VI"), ("Ⅴ", "V"), ("Ⅳ", "IV"),
        ("Ⅲ", "III"), ("Ⅱ", "II"), ("Ⅰ", "I"),
    ):
        text = text.replace(source, target)
    text = text.replace("IIIA", "IIIa").replace("IIIB", "IIIb")
    text = text.replace("IIA", "II").replace("IIB", "II")
    return text


def scene_case_mapper(manifest: Sequence[Dict[str, str]]):
    candidates = sorted(
        [(str(row["czi_stem"]), str(row["case_id"])) for row in manifest],
        key=lambda item: len(item[0]),
        reverse=True,
    )

    def map_scene(scene_id: str) -> Optional[str]:
        text = str(scene_id or "")
        for stem, case_id in candidates:
            if text.startswith(stem + "_") or text == stem:
                return case_id
        return None

    return map_scene


def pooled_tubular_burden(rows: Sequence[Dict[str, str]], map_scene) -> Dict[str, float]:
    weighted_target: Dict[str, float] = defaultdict(float)
    weighted_tissue: Dict[str, float] = defaultdict(float)
    for row in rows:
        case_id = map_scene(row.get("slide_id", ""))
        if not case_id:
            continue
        tissue = number(row.get("tissue_pixels"))
        burden = number(row.get("tubular_atrophy_target_percent"))
        if tissue <= 0:
            continue
        weighted_target[case_id] += burden * tissue
        weighted_tissue[case_id] += tissue
    return {
        case_id: weighted_target[case_id] / weighted_tissue[case_id]
        for case_id in weighted_tissue
        if weighted_tissue[case_id] > 0
    }


def binary_metrics(rows: Sequence[Dict[str, object]], field: str) -> Dict[str, object]:
    usable = [row for row in rows if row.get("{}_available".format(field), "1") == "1"]
    pairs = [(str(row["doctor_" + field]), str(row["model_" + field])) for row in usable]
    if not pairs:
        return {"n": 0, "accuracy": "", "sensitivity": "", "specificity": "", "f1": ""}
    negative = "{}0".format(field.upper())
    tp = sum(a != negative and b != negative for a, b in pairs)
    tn = sum(a == negative and b == negative for a, b in pairs)
    fp = sum(a == negative and b != negative for a, b in pairs)
    fn = sum(a != negative and b == negative for a, b in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    observed = (tp + tn) / len(pairs)
    actual_positive = (tp + fn) / len(pairs)
    predicted_positive = (tp + fp) / len(pairs)
    expected = actual_positive * predicted_positive + (1 - actual_positive) * (1 - predicted_positive)
    return {
        "n": len(pairs),
        "accuracy": observed,
        "sensitivity": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else 0.0,
    }


def multiclass_metrics(rows: Sequence[Dict[str, object]], truth: str, pred: str, labels: Sequence[str]) -> Dict[str, object]:
    pairs = [(str(row[truth]), str(row[pred])) for row in rows if row[truth] and row[pred]]
    if not pairs:
        return {"n": 0, "accuracy": "", "balanced_accuracy": "", "macro_f1": "", "weighted_f1": ""}
    f1_by_label: Dict[str, float] = {}
    recall_by_label: Dict[str, float] = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in pairs)
        fp = sum(a != label and b == label for a, b in pairs)
        fn = sum(a == label and b != label for a, b in pairs)
        denom = 2 * tp + fp + fn
        f1_by_label[label] = (2 * tp / denom) if denom else 0.0
        recall_by_label[label] = tp / (tp + fn) if tp + fn else 0.0
    observed = sum(a == b for a, b in pairs) / len(pairs)
    actual_counts = Counter(a for a, _ in pairs)
    pred_counts = Counter(b for _, b in pairs)
    supported = [label for label in labels if actual_counts[label] > 0]
    expected = sum(actual_counts[label] * pred_counts[label] for label in labels) / (len(pairs) ** 2)
    return {
        "n": len(pairs),
        "accuracy": observed,
        "balanced_accuracy": sum(recall_by_label[label] for label in supported) / len(supported),
        "macro_f1": sum(f1_by_label[label] for label in supported) / len(supported),
        "weighted_f1": sum(f1_by_label[label] * actual_counts[label] for label in supported) / len(pairs),
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else 0.0,
    }


def confusion_rows(rows: Sequence[Dict[str, object]], truth: str, pred: str, labels: Sequence[str]) -> List[Dict[str, object]]:
    pairs = [(str(row[truth]), str(row[pred])) for row in rows if row[truth] and row[pred]]
    return [
        {"doctor_label": actual, "model_label": predicted, "count": sum(a == actual and b == predicted for a, b in pairs)}
        for actual in labels for predicted in labels
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(args.case_manifest)
    by_case = {row["case_id"]: row for row in manifest}
    map_scene = scene_case_mapper(manifest)

    ifta_by_case: Dict[str, float] = {}
    tubular_rows: List[Dict[str, str]] = []
    for tubular_path in args.tubular_csv:
        if tubular_path.exists():
            tubular_rows.extend(read_csv(tubular_path))
    if tubular_rows:
        ifta_by_case = pooled_tubular_burden(tubular_rows, map_scene)

    case_prediction_rows: List[Dict[str, str]] = []
    unmatched_scenes = Counter()
    prediction_rows: List[Dict[str, str]] = []
    for prediction_path in args.prediction_csv:
        prediction_rows.extend(read_csv(prediction_path))
    for row in prediction_rows:
        case_id = map_scene(row.get("slide_id", ""))
        if not case_id:
            unmatched_scenes[row.get("slide_id", "")] += 1
            continue
        merged = dict(row)
        merged["source_scene_id"] = merged["slide_id"]
        merged["slide_id"] = case_id
        if case_id in ifta_by_case:
            merged["ifta_percent"] = "{:.8f}".format(ifta_by_case[case_id])
        case_prediction_rows.append(merged)

    case_prediction_path = args.output_dir / "per_glomerulus_case_predictions.csv"
    fields = list(case_prediction_rows[0].keys()) if case_prediction_rows else []
    write_csv(case_prediction_path, case_prediction_rows, fields)

    thresholds = Thresholds(
        mesangial=args.mesangial_threshold,
        endocapillary=args.endocapillary_threshold,
        crescent=args.crescent_threshold,
    )
    scores = group_and_score(
        load_glomeruli(case_prediction_path, thresholds, generic_crescent_active=False),
        min_glomeruli=args.min_glomeruli,
        iskdc_crescent_mode="all",
    )
    score_dir = args.output_dir / "scores"
    score_dir.mkdir(exist_ok=True)
    write_iskdc_csv(scores, score_dir / "iskdc_scores.csv")
    write_oxford_csv(scores, score_dir / "oxford_scores.csv")
    write_iskdc_json(scores, score_dir / "iskdc_scores.json")
    write_oxford_json(scores, score_dir / "oxford_scores.json")
    write_glomerulus_csv(scores, score_dir / "glomerulus_evidence.csv")
    write_summary_csv(scores, score_dir / "slide_scores.csv")
    write_json(scores, score_dir / "slide_scores.json")

    model_scores = {score.slide_id: score for score in scores}
    comparisons: List[Dict[str, object]] = []
    for case_id, doctor in sorted(by_case.items()):
        score = model_scores.get(case_id)
        if score is None:
            continue
        doctor_iskdc = normalize_iskdc(doctor.get("iskdc"))
        model_iskdc = score.iskdc["grade"]
        row: Dict[str, object] = {
            "case_id": case_id,
            "source_dataset": doctor.get("source_dataset", ""),
            "czi_stem": doctor.get("czi_stem", ""),
            "total_glomeruli": score.total_glomeruli,
            "doctor_iskdc_raw": doctor.get("iskdc", ""),
            "doctor_iskdc": doctor_iskdc,
            "model_iskdc": model_iskdc,
            "iskdc_match": int(doctor_iskdc == model_iskdc),
            "doctor_M": "M{}".format(doctor.get("M", "")),
            "model_M": score.oxford["M"],
            "M_match": int("M{}".format(doctor.get("M", "")) == score.oxford["M"]),
            "M_available": "1",
            "doctor_E": "E{}".format(doctor.get("E", "")),
            "model_E": score.oxford["E"],
            "E_match": int("E{}".format(doctor.get("E", "")) == score.oxford["E"]),
            "E_available": "1",
            "doctor_S": "S{}".format(doctor.get("S", "")),
            "model_S": "S_NA",
            "S_match": "",
            "S_available": "0",
            "doctor_T": "T{}".format(doctor.get("T", "")),
            "model_T": score.oxford["T"],
            "T_match": int("T{}".format(doctor.get("T", "")) == score.oxford["T"]),
            "T_available": str(int(case_id in ifta_by_case)),
            "doctor_C": "C{}".format(doctor.get("C", "")),
            "model_C": score.oxford["C"],
            "C_match": int("C{}".format(doctor.get("C", "")) == score.oxford["C"]),
            "C_available": "1",
            "model_MEST_C": "{}{}{}{}{}".format(score.oxford["M"], score.oxford["E"], "S_NA", score.oxford["T"], score.oxford["C"]),
            "mesangial_percent": "{:.4f}".format(score.percentages["mesangial"]),
            "endocapillary_percent": "{:.4f}".format(score.percentages["endocapillary"]),
            "active_crescent_percent": "{:.4f}".format(score.percentages["active_crescent"]),
            "any_crescent_percent": "{:.4f}".format(score.percentages["any_crescent"]),
            "ifta_percent": "{:.4f}".format(score.percentages["ifta"]) if "ifta" in score.percentages else "",
            "warnings": " | ".join(score.warnings),
        }
        comparisons.append(row)

    comparison_fields = list(comparisons[0].keys()) if comparisons else []
    write_csv(args.output_dir / "case_comparison.csv", comparisons, comparison_fields)

    summary_rows: List[Dict[str, object]] = []
    for field in ("M", "E", "T"):
        metrics = binary_metrics(comparisons, field)
        summary_rows.append({"endpoint": field, "metric_type": "binary", **metrics})
    c_metrics = multiclass_metrics(comparisons, "doctor_C", "model_C", ("C0", "C1", "C2"))
    summary_rows.append({"endpoint": "C", "metric_type": "three_class", **c_metrics})
    iskdc_labels = ("I", "II", "IIIa", "IIIb", "IV", "V", "VI")
    iskdc_metrics = multiclass_metrics(comparisons, "doctor_iskdc", "model_iskdc", iskdc_labels)
    summary_rows.append({"endpoint": "ISKDC", "metric_type": "collapsed_grade", **iskdc_metrics})
    summary_rows.append(
        {
            "endpoint": "M_E_T_C_exact",
            "metric_type": "exact_available_components",
            "n": len(comparisons),
            "accuracy": sum(
                all(row.get("{}_match".format(field)) == 1 for field in ("M", "E", "T", "C"))
                for row in comparisons
            ) / len(comparisons) if comparisons else "",
        }
    )
    write_csv(
        args.output_dir / "metric_summary.csv",
        summary_rows,
        (
            "endpoint", "metric_type", "n", "accuracy", "balanced_accuracy",
            "sensitivity", "specificity", "f1", "macro_f1", "weighted_f1", "kappa",
        ),
    )
    write_csv(args.output_dir / "iskdc_confusion.csv", confusion_rows(comparisons, "doctor_iskdc", "model_iskdc", iskdc_labels), ("doctor_label", "model_label", "count"))
    write_csv(args.output_dir / "c_confusion.csv", confusion_rows(comparisons, "doctor_C", "model_C", ("C0", "C1", "C2")), ("doctor_label", "model_label", "count"))
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "reference_cases": len(manifest),
                "scored_cases": len(comparisons),
                "per_glomerulus_rows": len(case_prediction_rows),
                "tubular_cases": len(ifta_by_case),
                "unmatched_prediction_scenes": dict(unmatched_scenes),
                "frozen_thresholds": {
                    "mesangial": args.mesangial_threshold,
                    "endocapillary": args.endocapillary_threshold,
                    "crescent": args.crescent_threshold,
                },
                "availability": {
                    "M": "model-derived",
                    "E": "model-derived",
                    "C": "model-derived from crescent detection and subtype classification",
                    "T": "model-derived when a tubular-atropy scene prediction is available",
                    "S": "not model-derived in the frozen inference chain; excluded from metrics",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"scored_cases": len(comparisons), "tubular_cases": len(ifta_by_case)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
