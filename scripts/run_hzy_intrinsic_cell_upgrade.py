"""Run equal-budget controls, select on validation, then evaluate locked tests."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--memory-fraction", type=float, default=0.26)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "job_status.json"
    trainer = Path(__file__).with_name("train_hzy_intrinsic_cell_upgrade.py")
    base = [sys.executable, "-u", str(trainer), "--source-root", str(args.source_root),
            "--dataset", str(args.dataset), "--workers", "2", "--memory-fraction", str(args.memory_fraction)]

    def run(phase, mode, checkpoint, output, extra=()):
        complete = output / ("val_metrics.json" if mode in ("baseline", "train") else "test_metrics.json")
        if complete.exists():
            return
        write_json(status_path, {"status": "running", "phase": phase, "pid": os.getpid(),
                                 "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        command = base + ["--mode", mode, "--checkpoint", str(checkpoint), "--output-dir", str(output)] + list(extra)
        with (args.output_dir / (phase + ".log")).open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4"})

    try:
        baseline = args.output_dir / "baseline"
        run("baseline_validation", "baseline", args.checkpoint, baseline)
        candidates = [("baseline", args.checkpoint, baseline)]
        for recipe in ("data_control", "recall_mesangial"):
            directory = args.output_dir / recipe
            run(recipe, "train", args.checkpoint, directory,
                ("--recipe", recipe, "--epochs", str(args.epochs), "--patience", str(args.patience)))
            candidates.append((recipe, directory / "best.pth", directory))
        validation = {name: json.loads((directory / "val_metrics.json").read_text())
                      for name, _, directory in candidates}
        selected = max(validation, key=lambda name: validation[name]["selection_score"])
        # This decision is persisted before any test metrics are computed.
        write_json(args.output_dir / "selected_on_validation.json", {
            "selected": selected, "validation": validation, "test_used_for_selection": False})
        summary_rows = []
        for name, checkpoint, directory in candidates:
            run(name + "_locked_test", "evaluate", checkpoint, directory / "locked_test",
                ("--split", "test", "--calibration", str(directory / "calibration.json")))
            test = json.loads((directory / "locked_test" / "test_metrics.json").read_text())
            summary_rows.append({"model": name, "selected_on_validation": name == selected,
                                 "validation_selection_score": validation[name]["selection_score"],
                                 "test_cases": test["cases"], "test_crops": test["crops"],
                                 "detection_precision": test["detection"]["precision"],
                                 "detection_recall": test["detection"]["recall"],
                                 "detection_f1": test["detection"]["f1"], "macro_f1": test["macro_f1"],
                                 **{name + "_f1": value["f1"] for name, value in test["per_class"].items()},
                                 **{name + "_count_mae": value for name, value in test["count_mae"].items()}})
        with (args.output_dir / "final_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        write_json(args.output_dir / "final_comparison.json", summary_rows)
        write_json(status_path, {"status": "complete", "selected": selected, "pid": os.getpid()})
    except Exception as error:
        write_json(status_path, {"status": "failed", "error": str(error), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
