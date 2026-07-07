"""Run all MESCnn classifiers (M/E/S/C, IgA-trained) on HSPN in-house crops.

Tests cross-disease transfer for all four Oxford classification components:
  M = mesangial hypercellularity (3-class: nan/noM/yesM, argmax -> yesM iff class==2)
  E = endocapillary hypercellularity (2-class: noE/yesE, sigmoid + threshold)
  S = segmental sclerosis (3-class: GGS/NoGS/SGS, argmax -> SGS iff class==2)
  C = active crescents (2-class: noC/yesC, sigmoid + threshold)

For HSPN, M and E are the proliferation targets we care about most.
S and C are included for completeness (C crescents we already handle separately).
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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


IMAGE_SIZE = 224

# MESCnn optimized thresholds for E and C (from threshold.py, V3, CNN)
OPT_THR = {
    "E": {"efficientnetv2-m": 0.06685786802030456},
    "C": {"mobilenetv2": 0.4974670050761421},
}

# Label encoding (from encoding.py)
MESC_LABELS = {
    "M": {0: "nan", 1: "noM", 2: "yesM"},
    "E": {0: "noE", 1: "yesE"},
    "S": {0: "GGS", 1: "NoGS", 2: "SGS"},
    "C": {0: "noC", 1: "yesC"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all MESCnn classifiers on HSPN crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-dir", required=True,
                        help="Directory containing efficientnetv2-m_M_V3.pth etc.")
    parser.add_argument("--data-root", required=True,
                        help="Crop dataset root with <split>/images and <split>/masks")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
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
        mask_path = self.masks_dir / (path.stem + ".png")
        label = -1
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            label = int(mask.max() > 0)
        return image, label, path.name


def load_classifier(path: Path, device: torch.device):
    return torch.load(str(path), map_location=device, weights_only=False)


def run_classifier(model, loader, device):
    """Return (names, logits_array, probs_array)."""
    names_all = []
    logits_all = []
    with torch.no_grad():
        for images, _labels, names in loader:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            logits = model(images).cpu().numpy()
            logits_all.append(logits)
            names_all.extend(names)
    logits_arr = np.concatenate(logits_all, axis=0)
    return names_all, logits_arr


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


def binarize_lesion(lesion: str, argmax_class: int):
    """MESCnn binarize logic from oxford.py."""
    if lesion in ("E", "C"):
        return int(argmax_class)
    elif lesion == "M":
        return int(argmax_class > 1)  # class 2 = yesM
    elif lesion == "S":
        return int(argmax_class > 1)  # class 2 = SGS
    return 0


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CropDataset(data_root, args.split, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    logging.info("Loaded %s crops from %s/%s", len(dataset), data_root, args.split)

    # label vector (same for all classifiers)
    labels = [dataset[i][1] for i in range(len(dataset))]
    has_label_idx = [i for i, l in enumerate(labels) if l >= 0]
    y_true = [labels[i] for i in has_label_idx]
    n_pos = sum(1 for t in y_true if t == 1)
    n_neg = sum(1 for t in y_true if t == 0)
    logging.info("Crops with label: %d (pos %d / neg %d)", len(y_true), n_pos, n_neg)

    classifier_configs = [
        ("M", "efficientnetv2-m_M_V3.pth", "efficientnetv2-m", False),
        ("E", "efficientnetv2-m_E_V3.pth", "efficientnetv2-m", True),
        ("S", "densenet161_S_V3.pth", "densenet161", False),
        ("C", "mobilenetv2_C_V3.pth", "mobilenetv2", True),
    ]

    all_results = {}
    per_crop_rows = []

    for lesion, fname, net_name, use_sigmoid_thr in classifier_configs:
        path = models_dir / fname
        if not path.exists():
            logging.warning("%s weight not found: %s, skipping", lesion, path)
            continue
        logging.info("=== %s classifier (%s) ===", lesion, fname)
        model = load_classifier(path, device)
        model.eval()
        model.to(device)
        names, logits = run_classifier(model, loader, device)
        del model
        torch.cuda.empty_cache()

        if use_sigmoid_thr:
            # E/C: sigmoid on logit[:, 1], threshold
            prob_pos = 1.0 / (1.0 + np.exp(-logits[:, 1]))
            opt = OPT_THR[lesion][net_name]
            preds = (prob_pos > opt).astype(int)
            # also try argmax for comparison
            argmax_preds = logits.argmax(axis=1)
        else:
            # M/S: argmax
            argmax_preds = logits.argmax(axis=1)
            preds = np.array([binarize_lesion(lesion, int(p)) for p in argmax_preds])
            prob_pos = None

        # metrics vs HSPN mask-derived label
        y_pred = [int(preds[i]) for i in has_label_idx]
        metrics = binary_metrics(y_true, y_pred)

        # also compute at multiple thresholds for E/C
        thr_metrics = {}
        if prob_pos is not None:
            for thr in [0.05, 0.1, 0.2, 0.3, 0.5]:
                bp = (prob_pos[has_label_idx] > thr).astype(int)
                thr_metrics[thr] = binary_metrics(y_true, list(bp))

        all_results[lesion] = {
            "net": net_name,
            "num_classes": logits.shape[1],
            "argmax_metrics": metrics,
            "threshold_metrics": thr_metrics if thr_metrics else None,
            "optimal_threshold": opt if use_sigmoid_thr else None,
        }
        logging.info("%s argmax: acc=%.4f prec=%.4f rec=%.4f spec=%.4f f1=%.4f (tp=%d fp=%d fn=%d tn=%d)",
                     lesion, metrics["accuracy"], metrics["precision"], metrics["recall"],
                     metrics["specificity"], metrics["f1"],
                     metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])
        if thr_metrics:
            for thr, m in thr_metrics.items():
                logging.info("  %s thr=%.2f: prec=%.4f rec=%.4f spec=%.4f f1=%.4f",
                              lesion, thr, m["precision"], m["recall"], m["specificity"], m["f1"])

        # per-crop CSV for this lesion
        lesion_rows = []
        for i in has_label_idx:
            row = {
                "name": names[i],
                "true_label": int(labels[i]),
                "argmax": int(argmax_preds[i]),
                "label": MESC_LABELS[lesion].get(int(argmax_preds[i]), "?"),
                "bin_pred": int(preds[i]),
            }
            if prob_pos is not None:
                row["prob_positive"] = round(float(prob_pos[i]), 4)
            lesion_rows.append(row)
        lesion_rows.sort(key=lambda r: -r.get("prob_positive", -r["argmax"]))
        csv_path = output_dir / f"per_crop_{lesion}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(lesion_rows[0].keys()))
            writer.writeheader()
            writer.writerows(lesion_rows)

    # write summary
    with (output_dir / "summary.json").open("w", encoding="utf-8") as h:
        json.dump({
            "num_crops": len(dataset),
            "crops_with_label": len(y_true),
            "positive_crops": n_pos,
            "negative_crops": n_neg,
            "results": all_results,
            "models_dir": str(models_dir),
            "data_root": str(data_root),
            "split": args.split,
        }, h, ensure_ascii=False, indent=2)
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()