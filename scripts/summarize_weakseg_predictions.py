"""Summarize weakseg validation predictions with per-class threshold search."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=91)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def summarize_class(
    rows: List[Dict[str, str]],
    class_name: str,
    thresholds: np.ndarray,
) -> Dict[str, float]:
    true_key = f"{class_name}_true"
    prob_key = f"{class_name}_prob"
    if true_key not in rows[0] or prob_key not in rows[0]:
        raise KeyError(f"Missing columns for {class_name}: expected {true_key} and {prob_key}")

    y_true = np.array([int(float(row[true_key])) for row in rows], dtype=int)
    scores = np.array([float(row[prob_key]) for row in rows], dtype=float)
    best = max((metrics_at_threshold(y_true, scores, th) for th in thresholds), key=lambda item: item["f1"])
    best["class"] = class_name
    best["support_pos"] = float(int(y_true.sum()))
    best["support_neg"] = float(int((y_true == 0).sum()))
    return best


def format_row(row: Dict[str, float]) -> str:
    fields: List[Tuple[str, str]] = [
        ("class", str(row["class"])),
        ("f1", f"{row['f1']:.4f}"),
        ("threshold", f"{row['threshold']:.2f}"),
        ("accuracy", f"{row['accuracy']:.4f}"),
        ("precision", f"{row['precision']:.4f}"),
        ("recall", f"{row['recall']:.4f}"),
        ("specificity", f"{row['specificity']:.4f}"),
        ("pos", f"{row['support_pos']:.0f}"),
        ("neg", f"{row['support_neg']:.0f}"),
        ("tp", f"{row['tp']:.0f}"),
        ("fp", f"{row['fp']:.0f}"),
        ("fn", f"{row['fn']:.0f}"),
        ("tn", f"{row['tn']:.0f}"),
    ]
    return " ".join(f"{key}={value}" for key, value in fields)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.predictions_csv)
    if not rows:
        raise ValueError(f"No rows in {args.predictions_csv}")
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    summaries = [summarize_class(rows, class_name, thresholds) for class_name in args.classes]

    for row in summaries:
        print(format_row(row))
    print(f"macro_best_f1={np.mean([row['f1'] for row in summaries]):.4f}")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "class",
            "f1",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "support_pos",
            "support_neg",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow({key: row[key] for key in fieldnames})


if __name__ == "__main__":
    main()
