"""ROI-aware DPA-P2PNet data, loss and point-detection evaluation utilities."""

from collections import Counter
import csv
import json
from pathlib import Path
import random

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset


CLASSES = ["endothelial_cell", "mesangial_cell", "podocyte"]


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def geometric_augment(image, points, roi, horizontal=False, vertical=False, rotations=0):
    points = points.copy()
    height, width = image.shape[:2]
    if horizontal:
        image, roi = image[:, ::-1], roi[:, ::-1]
        points[:, 0] = width - 1 - points[:, 0]
    if vertical:
        image, roi = image[::-1], roi[::-1]
        points[:, 1] = height - 1 - points[:, 1]
    for _ in range(rotations % 4):
        image, roi = np.rot90(image), np.rot90(roi)
        points = np.stack((points[:, 1], width - 1 - points[:, 0]), axis=1)
        height, width = width, height
    if len(points):
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return np.ascontiguousarray(image), points, np.ascontiguousarray(roi)


class CellDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.rows = [r for r in read_rows(self.root / "manifest.csv") if r["split"] == split]
        self.mean, self.std = np.load(self.root / "mean_std.npy").astype(np.float32)
        if not self.rows:
            raise ValueError("No crops in split " + split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        stem = row["crop_id"]
        image = cv2.imread(str(self.root / (self.split + "_image") / (stem + ".jpg")))
        if image is None:
            raise FileNotFoundError(stem)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        roi = cv2.imread(str(self.root / (self.split + "_roi") / (stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if roi is None:
            raise FileNotFoundError("Missing valid annotation ROI: " + stem)
        annotation = json.loads((self.root / (self.split + "_point") / (stem + ".json")).read_text())
        if annotation["classes"] != CLASSES:
            raise ValueError("Cell label order differs: " + stem)
        arrays = [np.asarray(annotation[name], dtype=np.float32).reshape(-1, 2) for name in CLASSES]
        points = np.concatenate(arrays, axis=0)
        labels = np.concatenate([np.full(len(p), i, dtype=np.int64) for i, p in enumerate(arrays)])
        if self.augment:
            image, points, roi = geometric_augment(image, points, roi, random.random() < 0.5,
                                                  random.random() < 0.5, random.randrange(4))
            if random.random() < 0.5:
                gain = random.uniform(0.85, 1.15)
                bias = random.uniform(-10, 10)
                image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255
        tensor = (tensor - torch.from_numpy(self.mean)[:, None, None]) / torch.from_numpy(self.std)[:, None, None]
        return {"image": tensor, "points": torch.from_numpy(points), "labels": torch.from_numpy(labels),
                "roi": torch.from_numpy((roi > 0).copy()), "row": row}

    def sampling_weights(self):
        case_counts = Counter(r["case_key"] for r in self.rows)
        weights = []
        for row in self.rows:
            count = sum(int(row[name + "_points"]) for name in CLASSES)
            mesangial_fraction = int(row["mesangial_cell_points"]) / max(count, 1)
            weights.append((1.0 + 0.5 * mesangial_fraction) / case_counts[row["case_key"]])
        return weights


def valid_predictions(coords, roi):
    height, width = roi.shape
    inside = (coords[:, 0] >= 0) & (coords[:, 0] < width) & (coords[:, 1] >= 0) & (coords[:, 1] < height)
    xy = coords.detach().long()
    valid = roi[xy[:, 1].clamp(0, height - 1), xy[:, 0].clamp(0, width - 1)]
    return inside & valid


class ROIClassificationLoss(nn.Module):
    def __init__(self, matcher, focal_gamma=1.5, mesangial_weight=1.5,
                 background_weight=0.25, hard_negative_ratio=4.0, reg_weight=0.002):
        super().__init__()
        self.matcher = matcher
        self.focal_gamma = focal_gamma
        self.mesangial_weight = mesangial_weight
        self.background_weight = background_weight
        self.hard_negative_ratio = hard_negative_ratio
        self.reg_weight = reg_weight

    def forward(self, outputs, sample):
        # The existing Hungarian matcher defines the core point assignment.
        coords, logits = outputs["pnt_coords"].float(), outputs["cls_logits"].float()
        points = sample["points"].to(coords.device)
        labels = sample["labels"].to(coords.device)
        roi = sample["roi"].to(coords.device)
        targets = {"gt_points": [points], "gt_labels": [labels], "gt_nums": [len(points)]}
        indices = self.matcher({"pnt_coords": coords, "cls_logits": logits}, targets)
        src, target = (index.to(coords.device) for index in indices[0])
        classes = torch.full((logits.shape[1],), len(CLASSES), dtype=torch.long, device=coords.device)
        classes[src] = labels[target]
        valid = valid_predictions(coords[0], roi)
        # Matched positives must never disappear if a predicted point leaves the ROI.
        valid[src] = True
        ce = F.cross_entropy(logits[0], classes, reduction="none")
        probability = logits[0].softmax(-1).gather(1, classes[:, None]).squeeze(1)
        weights = torch.tensor([1.0, self.mesangial_weight, 1.0, self.background_weight], device=logits.device)
        losses = ce * (1 - probability).pow(self.focal_gamma) * weights[classes]
        positive = valid & (classes < len(CLASSES))
        negative = losses[valid & (classes == len(CLASSES))]
        if self.hard_negative_ratio > 0 and len(negative):
            k = min(len(negative), max(128, int(self.hard_negative_ratio * max(len(points), 1))))
            negative = negative.topk(k).values
        selected = torch.cat((losses[positive], negative))
        cls = selected.mean() if len(selected) else logits.sum() * 0
        reg = F.mse_loss(coords[0, src], points[target], reduction="sum") / max(len(points), 1) if len(src) else coords.sum() * 0
        return cls + self.reg_weight * reg, {"classification": float(cls.detach()), "regression": float(reg.detach()),
                                             "positive_queries": int(positive.sum()), "negative_queries": len(negative)}


def match_points(predictions, targets, distance=10.0):
    if not len(predictions) or not len(targets):
        return 0
    edges = csr_matrix(cdist(predictions, targets) < distance)
    matches = maximum_bipartite_matching(edges, perm_type="column")
    return int((matches >= 0).sum())


def select_points(coords, scores, roi, foreground_ratio=1.0, dedup_distance=6.0):
    coords = np.asarray(coords, dtype=np.float32)
    scores = np.asarray(scores)
    height, width = roi.shape
    inside = ((coords[:, 0] >= 0) & (coords[:, 0] < width) &
              (coords[:, 1] >= 0) & (coords[:, 1] < height))
    coords, scores = coords[inside], scores[inside]
    if not len(coords):
        return np.empty((0, 2)), np.empty(0, dtype=int)
    xy = coords.astype(int)
    foreground = scores[:, :len(CLASSES)].max(1)
    keep = (foreground >= foreground_ratio * scores[:, len(CLASSES)]) & roi[xy[:, 1], xy[:, 0]]
    coords, scores = coords[keep], scores[keep]
    if not len(coords):
        return np.empty((0, 2)), np.empty(0, dtype=int)
    classes = scores[:, :len(CLASSES)].argmax(1)
    order = np.argsort(-scores[:, :len(CLASSES)].max(1), kind="stable")
    selected = []
    tree = cKDTree(coords)
    suppressed = np.zeros(len(coords), dtype=bool)
    for index in order:
        if suppressed[index]:
            continue
        selected.append(index)
        if dedup_distance > 0:
            nearby = np.asarray(tree.query_ball_point(coords[index], dedup_distance), dtype=int)
            nearby = nearby[np.linalg.norm(coords[nearby] - coords[index], axis=1) < dedup_distance]
            suppressed[nearby] = True
    return coords[selected], classes[selected]


def prf(tp, pred, truth):
    precision = tp / max(pred, 1)
    recall = tp / max(truth, 1)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "matched": tp, "predicted": pred, "ground_truth": truth}


def score_cached(cached, ratio=1.0, dedup=6.0, distance=10.0):
    counts = np.zeros((4, 3), dtype=np.int64)
    rows = []
    predictions = []
    for item in cached:
        points, labels = select_points(item["coords"], item["scores"], item["roi"], ratio, dedup)
        truth, truth_labels = item["points"], item["labels"]
        row = {"crop_id": item["row"]["crop_id"], "slide_id": item["row"]["slide_id"],
               "case_key": item["row"]["case_key"], "disease": item["row"].get("disease", "unknown")}
        counts[0] += [match_points(points, truth, distance), len(points), len(truth)]
        row.update({"predicted_total": len(points), "true_total": len(truth)})
        for i, name in enumerate(CLASSES):
            pred, gt = points[labels == i], truth[truth_labels == i]
            matched = match_points(pred, gt, distance)
            counts[i + 1] += [matched, len(pred), len(gt)]
            row.update({name + "_predicted": len(pred), name + "_true": len(gt), name + "_matched": matched,
                        name + "_count_error": len(pred) - len(gt)})
        rows.append(row)
        predictions.append({**row, "points": points.round(3).tolist(), "classes": labels.tolist()})
    per_class = {name: prf(*[int(x) for x in counts[i + 1]]) for i, name in enumerate(CLASSES)}
    macro_f1 = float(np.mean([x["f1"] for x in per_class.values()]))
    metrics = {"detection": prf(*[int(x) for x in counts[0]]), "per_class": per_class, "macro_f1": macro_f1,
               "count_mae": {name: float(np.mean([abs(r[name + "_count_error"]) for r in rows])) for name in CLASSES},
               "crops": len(rows), "cases": len({r["case_key"] for r in rows}),
               "foreground_ratio": ratio, "dedup_distance": dedup, "match_distance": distance,
               "annotation_policy": "Points and predictions restricted to the same annotated ROI."}
    metrics["selection_score"] = (0.4 * metrics["detection"]["f1"] +
                                  0.4 * per_class["mesangial_cell"]["f1"] + 0.2 * macro_f1)
    return metrics, rows, predictions


def best_validation_postprocess(cached):
    choices = []
    for ratio in (0.6, 0.8, 1.0, 1.2):
        for dedup in (4.0, 6.0, 8.0, 10.0):
            metrics, _, _ = score_cached(cached, ratio, dedup)
            choices.append(metrics)
    # Precision and F1 constrain recall-oriented tuning on the validation split only.
    eligible = [m for m in choices if m["detection"]["precision"] >= 0.55]
    selected = max(eligible or choices, key=lambda m: (m["selection_score"], m["detection"]["recall"]))
    return selected, choices
