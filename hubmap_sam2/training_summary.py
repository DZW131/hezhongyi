from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from hubmap_sam2.visualization import save_training_curves


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def merge_train_val_history(
    train_rows: List[Dict[str, object]],
    val_rows: List[Dict[str, object]],
) -> List[Dict[str, float]]:
    rows_by_epoch: Dict[int, Dict[str, float]] = {}

    for row in train_rows:
        epoch = int(row.get("Trainer/epoch", len(rows_by_epoch)))
        rows_by_epoch.setdefault(epoch, {"epoch": epoch})
        rows_by_epoch[epoch]["train_loss"] = float(row.get("Losses/train_all_loss", 0.0))

    for row in val_rows:
        epoch = int(row.get("Trainer/epoch", len(rows_by_epoch)))
        rows_by_epoch.setdefault(epoch, {"epoch": epoch})
        rows_by_epoch[epoch]["val_loss"] = float(row.get("Losses/val_all_loss", 0.0))
        rows_by_epoch[epoch]["val_dice"] = float(
            row.get("Meters_train/val_all/glomerulus_metrics/dice", row.get("Meters_train/val_all/segmentation/dice", 0.0))
        )
        rows_by_epoch[epoch]["val_iou"] = float(
            row.get("Meters_train/val_all/glomerulus_metrics/iou", row.get("Meters_train/val_all/segmentation/iou", 0.0))
        )
        rows_by_epoch[epoch]["val_precision"] = float(
            row.get("Meters_train/val_all/glomerulus_metrics/precision", row.get("Meters_train/val_all/segmentation/precision", 0.0))
        )
        rows_by_epoch[epoch]["val_recall"] = float(
            row.get("Meters_train/val_all/glomerulus_metrics/recall", row.get("Meters_train/val_all/segmentation/recall", 0.0))
        )

    return [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]


def write_history_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_training_run(run_dir: Path) -> Dict[str, object]:
    logs_dir = run_dir / "logs"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(logs_dir / "train_stats.json")
    val_rows = read_jsonl(logs_dir / "val_stats.json")
    best_rows = read_jsonl(logs_dir / "best_stats.json")

    history_rows = merge_train_val_history(train_rows, val_rows)
    write_history_csv(history_rows, analysis_dir / "history.csv")
    save_training_curves(history_rows, analysis_dir / "training_curves.png")

    best_metrics = best_rows[-1] if best_rows else {}
    with (analysis_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2)

    summary = {
        "run_dir": str(run_dir),
        "history_rows": len(history_rows),
        "best_metrics_path": str(analysis_dir / "best_metrics.json"),
    }
    with (analysis_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
