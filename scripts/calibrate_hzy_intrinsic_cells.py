"""Validate two-model point fusion and class calibration without test tuning."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hzy_intrinsic_cell_upgrade import CLASSES, CellDataset, score_cached
from train_hzy_intrinsic_cell_upgrade import build_model, predict, save_report


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blend_caches(first, second, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Fusion weight must lie in [0,1]")
    if len(first) != len(second):
        raise ValueError("Model caches have different crop counts")
    result = []
    for a, b in zip(first, second):
        if a["row"]["crop_id"] != b["row"]["crop_id"] or a["scores"].shape != b["scores"].shape:
            raise ValueError("Model caches are not query-aligned")
        if not np.array_equal(a["points"], b["points"]) or not np.array_equal(a["roi"], b["roi"]):
            raise ValueError("Model caches use different ground truth or ROIs")
        confidence_a = a["scores"][:, :3].max(1) * (1 - weight)
        confidence_b = b["scores"][:, :3].max(1) * weight
        denom = np.maximum(confidence_a + confidence_b, 1e-8)
        coords = (a["coords"] * confidence_a[:, None] + b["coords"] * confidence_b[:, None]) / denom[:, None]
        result.append({**a, "coords": coords, "scores": (1 - weight) * a["scores"] + weight * b["scores"]})
    return result


def adjust_classes(cached, multipliers):
    result = []
    factors = np.array([*multipliers, 1.0], dtype=np.float32)
    if np.any(factors <= 0) or len(factors) != 4:
        raise ValueError("Three positive class multipliers are required")
    for item in cached:
        scores = item["scores"] * factors
        scores = scores / np.maximum(scores.sum(1, keepdims=True), 1e-8)
        result.append({**item, "scores": scores})
    return result


def validation_objective(metrics, reference):
    mean_truth = {name: metrics["per_class"][name]["ground_truth"] / max(metrics["crops"], 1) for name in CLASSES}
    nmae = float(np.mean([metrics["count_mae"][name] / max(mean_truth[name], 1) for name in CLASSES]))
    objective = (0.35 * metrics["detection"]["f1"] + 0.45 * metrics["macro_f1"] +
                 0.20 * metrics["per_class"]["mesangial_cell"]["f1"] - 0.05 * nmae)
    eligible = (metrics["detection"]["precision"] >= 0.60 and
                metrics["detection"]["recall"] >= reference["detection"]["recall"] - 0.02 and
                metrics["detection"]["f1"] >= reference["detection"]["f1"] - 0.005)
    return objective if eligible else -1.0, nmae


def load_or_predict(args, checkpoint, split, name):
    args.split = split
    path = args.output_dir / (name + "_" + split + "_raw.pt")
    meta_path = path.with_suffix(".json")
    identity = {"checkpoint_sha256": fingerprint(checkpoint), "split": split,
                "manifest_sha256": fingerprint(args.dataset / "manifest.csv"),
                "split_sha256": fingerprint(args.dataset / "case_splits.csv")}
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == identity:
        return torch.load(path, map_location="cpu", weights_only=False)
    args.checkpoint = checkpoint
    model, old = build_model(args)
    del old
    result = predict(model, CellDataset(args.dataset, split), args)
    del model
    torch.cuda.empty_cache()
    torch.save(result, path)
    meta_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("search", "evaluate"), default="search")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--upgraded-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=0.26)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    args.max_crops = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    if args.mode == "evaluate":
        chosen = json.loads((args.output_dir / "frozen_calibration.json").read_text())
        if chosen["source_split"] != "val":
            raise ValueError("The calibration must be selected using validation only")
        a = load_or_predict(args, args.baseline_checkpoint, "test", "baseline")
        b = load_or_predict(args, args.upgraded_checkpoint, "test", "upgraded")
        cached = adjust_classes(blend_caches(a, b, chosen["weight"]), chosen["multipliers"])
        metrics, rows, predictions = score_cached(cached, chosen["ratio"], chosen["dedup"])
        save_report(args, "test", cached, metrics, rows, predictions)
        print(json.dumps({"event": "retrospective_test_complete", "metrics": metrics}), flush=True)
        return
    a = load_or_predict(args, args.baseline_checkpoint, "val", "baseline")
    b = load_or_predict(args, args.upgraded_checkpoint, "val", "upgraded")
    reference = score_cached(b, 1.2, 10)[0]
    trials = []
    best = None

    def evaluate(weight, multipliers, ratio, dedup):
        nonlocal best
        cached = adjust_classes(blend_caches(a, b, weight), multipliers)
        metrics = score_cached(cached, ratio, dedup)[0]
        objective, nmae = validation_objective(metrics, reference)
        row = {"weight": weight, "multipliers": list(multipliers), "ratio": ratio, "dedup": dedup,
               "objective": objective, "normalized_count_mae": nmae, "metrics": metrics}
        trials.append(row)
        if best is None or row["objective"] > best["objective"]:
            best = row
            print(json.dumps({"event": "validation_best", **row}), flush=True)
        return row

    # Small predeclared grid followed by bounded coordinate descent on validation.
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        for ratio in (0.8, 1.2, 1.6):
            for dedup in (6.0, 8.0, 10.0):
                evaluate(weight, [1, 1, 1], ratio, dedup)
    seeds = sorted(trials, key=lambda r: r["objective"], reverse=True)[:2]
    for seed in seeds:
        current = copy.deepcopy(seed)
        for _ in range(2):
            for cls in range(3):
                choices = [current]
                for factor in (0.7, 1.0, 1.3):
                    factors = current["multipliers"].copy()
                    factors[cls] = float(np.clip(factors[cls] * factor, 0.49, 1.69))
                    choices.append(evaluate(current["weight"], factors, current["ratio"], current["dedup"]))
                current = max(choices, key=lambda row: row["objective"])
    chosen = {k: best[k] for k in ("weight", "multipliers", "ratio", "dedup", "objective")}
    chosen.update({"source_split": "val", "case_count": reference["cases"], "test_used_for_selection": False,
                   "selection_policy": "Detection/three-class F1 with precision/recall guards and normalized count-error penalty."})
    (args.output_dir / "frozen_calibration.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    (args.output_dir / "validation_search.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    cached = adjust_classes(blend_caches(a, b, best["weight"]), best["multipliers"])
    metrics, rows, predictions = score_cached(cached, best["ratio"], best["dedup"])
    save_report(args, "val", cached, metrics, rows, predictions)
    (args.output_dir / "validation_reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(json.dumps({"event": "validation_search_complete", "chosen": chosen, "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
