"""Summarize frozen cell experiments with paired case-bootstrap intervals."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from hzy_intrinsic_cell_upgrade import CLASSES, match_points, read_rows


def case_counts(predictions, dataset):
    counts = defaultdict(lambda: np.zeros(12, dtype=np.float64))
    for row in predictions:
        annotation = json.loads((dataset / "test_point" / (row["crop_id"] + ".json")).read_text(encoding="utf-8"))
        truth_by_class = [np.asarray(annotation[name], dtype=np.float32).reshape(-1, 2) for name in CLASSES]
        truth = np.concatenate(truth_by_class)
        pred = np.asarray(row["points"], dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(row["classes"])
        values = [match_points(pred, truth, 10), len(pred), len(truth)]
        for index, gt in enumerate(truth_by_class):
            pp = pred[labels == index]
            values += [match_points(pp, gt, 10), len(pp), len(gt)]
        counts[row["case_key"]] += values
    return counts


def endpoints(counts):
    blocks = np.asarray(counts).reshape(-1, 3)
    f1 = 2 * blocks[:, 0] / np.maximum(blocks[:, 1] + blocks[:, 2], 1)
    return np.array([f1[0], blocks[0, 0] / max(blocks[0, 2], 1), f1[2], f1[1:].mean()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--candidate-predictions", type=Path)
    args = parser.parse_args()
    reference_path = args.reference_predictions or args.results / "baseline/locked_test/test_predictions.json"
    candidate_path = args.candidate_predictions or args.results / "recall_mesangial/locked_test/test_predictions.json"
    baseline = json.loads(reference_path.read_text(encoding="utf-8"))
    upgraded = json.loads(candidate_path.read_text(encoding="utf-8"))
    a, b = case_counts(baseline, args.dataset), case_counts(upgraded, args.dataset)
    assert set(a) == set(b)
    cases = sorted(a)
    aa, bb = np.stack([a[k] for k in cases]), np.stack([b[k] for k in cases])
    rng = np.random.default_rng(20261021)
    differences = []
    for _ in range(2000):
        indexes = rng.integers(len(cases), size=len(cases))
        differences.append(endpoints(bb[indexes].sum(0)) - endpoints(aa[indexes].sum(0)))
    interval = np.percentile(differences, [2.5, 97.5], axis=0)
    delta = endpoints(bb.sum(0)) - endpoints(aa.sum(0))
    names = ["detection_f1", "detection_recall", "mesangial_f1", "macro_f1"]
    result = {"resampling_unit": "case", "case_count": len(cases), "bootstrap_samples": 2000,
              "seed": 20261021, "paired_differences": {
                  name: {"delta": float(delta[i]), "ci95": interval[:, i].tolist()} for i, name in enumerate(names)}}
    (args.results / "paired_case_bootstrap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
