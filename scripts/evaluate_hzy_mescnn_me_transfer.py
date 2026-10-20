"""Evaluate original MESCNN M/E classifiers on HSPN fine-grained crop labels.

This is a transfer baseline for the HZY HSPN proliferation task. It keeps the
IgA-trained MESCNN heads unchanged and compares their M/E predictions against
the in-house HSPN crop manifest labels:

* mask class 1 -> mesangial hypercellularity (Oxford M)
* mask class 2 -> endocapillary hypercellularity (Oxford E)

The fine-tuning script trains a new two-output multi-label head. This script
answers the prior question: how far do the original MESCNN checkpoints transfer
before any HSPN fine-tuning?
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TARGETS = ("mesangial", "endocapillary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MESCNN direct transfer on HSPN M/E labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, help="Task root containing manifests/crops.csv")
    parser.add_argument("--m-model", required=True, help="Path to efficientnetv2-m_M_V3.pth")
    parser.add_argument("--e-model", required=True, help="Path to efficientnetv2-m_E_V3.pth")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-positive-pixels", type=int, default=8)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_class_pixels(raw: str) -> Dict[str, int]:
    if raw is None or raw == "":
        return {}
    try:
        return {str(k): int(v) for k, v in json.loads(raw).items()}
    except json.JSONDecodeError:
        return {}


class ManifestCropDataset(Dataset):
    def __init__(self, data_root: Path, split: str, min_positive_pixels: int) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        manifest = self.data_root / "manifests" / "crops.csv"
        if not manifest.exists():
            raise FileNotFoundError("Missing manifest: {}".format(manifest))

        self.rows: List[Dict[str, object]] = []
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("split") != split:
                    continue
                image_rel = row.get("image_path") or ""
                image_path = self.data_root / image_rel
                if image_path.suffix.lower() not in IMAGE_EXTS or not image_path.exists():
                    continue
                pixels = parse_class_pixels(row.get("class_pixels", ""))
                target = np.array(
                    [
                        int(pixels.get("1", 0) >= min_positive_pixels),
                        int(pixels.get("2", 0) >= min_positive_pixels),
                    ],
                    dtype=np.int32,
                )
                stored = dict(row)
                stored["image_abs_path"] = str(image_path)
                stored["target"] = target
                self.rows.append(stored)
        if not self.rows:
            raise RuntimeError("No {} rows under {}".format(split, self.data_root))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(str(row["image_abs_path"])).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(row["target"], dtype=torch.int64)
        crop_id = str(row.get("crop_id") or Path(str(row["image_abs_path"])).stem)
        return image, target, crop_id


def metric_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    return {
        "tp": int(np.logical_and(y_true == 1, y_pred == 1).sum()),
        "fp": int(np.logical_and(y_true == 0, y_pred == 1).sum()),
        "fn": int(np.logical_and(y_true == 1, y_pred == 0).sum()),
        "tn": int(np.logical_and(y_true == 0, y_pred == 0).sum()),
    }


def counts_to_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        **counts,
    }


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    scores = y_score[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[j] == scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_values(step: float) -> List[float]:
    step = max(min(float(step), 0.5), 0.01)
    return [round(float(v), 4) for v in np.arange(step, 1.0, step)]


def best_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    best = None
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(np.int32)
        metrics = counts_to_metrics(metric_counts(y_true, pred))
        metrics["threshold"] = float(threshold)
        if best is None or metrics["f1"] > best["f1"] or (
            metrics["f1"] == best["f1"] and metrics["recall"] > best["recall"]
        ):
            best = metrics
    assert best is not None
    auc = binary_auc(y_true, y_prob)
    if auc is not None:
        best["auc"] = float(auc)
    return best


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    model = torch.load(str(path), map_location=device, weights_only=False)
    model.eval().to(device)
    return model


def run_models(
    m_model: torch.nn.Module,
    e_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    targets_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    crop_ids: List[str] = []
    with torch.no_grad():
        for images, targets, names in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            m_logits = m_model(images)
            e_logits = e_model(images)
            m_prob = F.softmax(m_logits, dim=1)[:, 2]
            if e_logits.shape[1] >= 2:
                e_prob = torch.sigmoid(e_logits[:, 1])
            else:
                e_prob = torch.sigmoid(e_logits[:, 0])
            probs = torch.stack([m_prob, e_prob], dim=1)
            targets_all.append(targets.numpy())
            probs_all.append(probs.cpu().numpy())
            crop_ids.extend(list(names))
    return np.concatenate(targets_all, axis=0), np.concatenate(probs_all, axis=0), crop_ids


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ManifestCropDataset(Path(args.data_root), args.split, args.min_positive_pixels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logging.info("Loaded %d %s crops", len(dataset), args.split)
    m_model = load_model(Path(args.m_model), device)
    e_model = load_model(Path(args.e_model), device)
    y_true, y_prob, crop_ids = run_models(m_model, e_model, loader, device)

    thresholds = threshold_values(args.threshold_step)
    per_class = {
        "mesangial": best_threshold_metrics(y_true[:, 0], y_prob[:, 0], thresholds),
        "endocapillary": best_threshold_metrics(y_true[:, 1], y_prob[:, 1], thresholds),
    }
    summary = {
        "split": args.split,
        "samples": int(y_true.shape[0]),
        "positives": {
            "mesangial": int(y_true[:, 0].sum()),
            "endocapillary": int(y_true[:, 1].sum()),
            "both": int(np.logical_and(y_true[:, 0] == 1, y_true[:, 1] == 1).sum()),
        },
        "per_class": per_class,
        "macro_f1": float(np.mean([per_class[name]["f1"] for name in TARGETS])),
        "mean_auc": float(np.mean([per_class[name].get("auc", 0.0) for name in TARGETS])),
    }
    rows = []
    for idx, crop_id in enumerate(crop_ids):
        rows.append(
            {
                "crop_id": crop_id,
                "mesangial_true": int(y_true[idx, 0]),
                "endocapillary_true": int(y_true[idx, 1]),
                "mesangial_prob": float(y_prob[idx, 0]),
                "endocapillary_prob": float(y_prob[idx, 1]),
            }
        )
    save_json(summary, output_dir / "summary.json")
    save_csv(rows, output_dir / "per_crop_predictions.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
