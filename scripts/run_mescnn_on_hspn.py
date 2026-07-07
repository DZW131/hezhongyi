"""Run MESCnn M-classifier (IgA-trained) on HSPN in-house crops to test cross-disease transfer.

This loads the MESCnn EfficientNet M-classifier (trained on IgA nephropathy for
mesangial hypercellularity) and runs it on our in-house HSPN glomerulus crops.
The key question: can an IgA-trained classifier distinguish proliferation on HSPN crops?

M-label mapping (from MESCnn encoding.py):
  0 = nan_label
  1 = noM (no mesangial hypercellularity)
  2 = yesM (mesangial hypercellularity)
Binarized: yesM iff argmax == 2.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import torchvision
except ImportError as exc:
    raise SystemExit("torchvision is required: {}".format(exc))


MESC_LABELS = {0: "nan", 1: "noM", 2: "yesM"}
IMAGE_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MESCnn M-classifier on HSPN crops to test IgA->HSPN transfer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to efficientnetv2-m_M_V3.pth")
    parser.add_argument("--data-root", required=True,
                        help="Crop dataset root with <split>/images and <split>/masks")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--montage-max", type=int, default=24)
    return parser.parse_args()


class CropDataset(Dataset):
    def __init__(self, data_root: Path, split: str, transform=None):
        self.images_dir = data_root / split / "images"
        self.masks_dir = data_root / split / "masks"
        self.transform = transform
        self.image_paths = sorted(
            p for p in self.images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.image_paths:
            raise RuntimeError("No images in {}".format(self.images_dir))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        if self.transform:
            image = self.transform(image)
        # derive binary label from mask: any non-zero pixel => positive (proliferation)
        mask_path = self.masks_dir / (path.stem + ".png")
        label = -1
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            label = int(mask.max() > 0)
        return image, label, path.name


def load_mescnn_classifier(path: Path, device: torch.device):
    """Load the MESCnn M-classifier (full pickled model)."""
    model = torch.load(str(path), map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    return model


def evaluate(model, loader, device):
    """Run inference and collect per-crop predictions."""
    names_all, labels_all, preds_all, probs_all = [], [], [], []
    yesM_probs_all = []
    with torch.no_grad():
        for images, labels, names in loader:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            yesM_prob = probs[:, 2]  # class 2 = yesM
            names_all.extend(names)
            labels_all.extend(labels.tolist())
            preds_all.extend(preds.tolist())
            probs_all.extend(probs.tolist())
            yesM_probs_all.extend(yesM_prob.tolist())
    return names_all, labels_all, preds_all, probs_all, yesM_probs_all


def binary_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MESCnn uses only ToTensor() for CNN inference (no normalize), per config.py
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CropDataset(data_root, args.split, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    logging.info("Loaded %s crops from %s/%s", len(dataset), data_root, args.split)

    model = load_mescnn_classifier(Path(args.model), device)
    logging.info("Loaded MESCnn M-classifier from %s", args.model)

    names, labels, preds, probs, yesM_probs = evaluate(model, loader, device)

    # binarize: yesM (class 2) => 1, else 0
    bin_preds = [1 if p == 2 else 0 for p in preds]
    # filter out crops without mask label (label == -1)
    has_label = [(n, t, p, yp, pr) for n, t, p, yp, pr in
                 zip(names, labels, bin_preds, yesM_probs, probs) if t >= 0]
    if not has_label:
        logging.warning("No crops with mask labels found.")
        return
    y_true = [t for _, t, _, _, _ in has_label]
    y_pred = [p for _, _, p, _, _ in has_label]
    metrics = binary_metrics(y_true, y_pred)
    n_pos = sum(1 for t in y_true if t == 1)
    n_neg = sum(1 for t in y_true if t == 0)

    # also compute metrics at various yesM-prob thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    thr_metrics = {}
    for thr in thresholds:
        bp = [1 if yp >= thr else 0 for _, _, _, yp, _ in has_label]
        thr_metrics[thr] = binary_metrics(y_true, bp)

    summary = {
        "num_crops": len(dataset),
        "crops_with_label": len(has_label),
        "positive_crops": n_pos,
        "negative_crops": n_neg,
        "mescnn_argmax_metrics": metrics,
        "yesM_prob_threshold_metrics": {str(t): thr_metrics[t] for t in thresholds},
        "model": str(args.model),
        "data_root": str(data_root),
        "split": args.split,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)

    # per-crop csv
    rows = []
    for name, true_label, pred, yesM_p, prob_vec in has_label:
        rows.append({
            "name": name,
            "true_label": int(true_label),
            "mescnn_argmax": int(preds[names.index(name)]),
            "mescnn_label": MESC_LABELS[int(preds[names.index(name)])],
            "yesM_prob": round(float(yesM_p), 4),
            "prob_nan": round(float(prob_vec[0]), 4),
            "prob_noM": round(float(prob_vec[1]), 4),
            "prob_yesM": round(float(prob_vec[2]), 4),
        })
    rows.sort(key=lambda r: -r["yesM_prob"])
    with (output_dir / "per_crop.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logging.info("=== MESCnn M-classifier on HSPN crops ===")
    logging.info("Crops: %d (pos %d / neg %d)", len(has_label), n_pos, n_neg)
    logging.info("Argmax metrics: acc=%.4f prec=%.4f rec=%.4f spec=%.4f f1=%.4f",
                 metrics["accuracy"], metrics["precision"], metrics["recall"],
                 metrics["specificity"], metrics["f1"])
    for thr in thresholds:
        m = thr_metrics[thr]
        logging.info("  thr=%.2f: prec=%.4f rec=%.4f spec=%.4f f1=%.4f",
                      thr, m["precision"], m["recall"], m["specificity"], m["f1"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()