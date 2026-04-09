from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def confusion_from_binary_masks(pred_mask: np.ndarray, true_mask: np.ndarray) -> Tuple[int, int, int, int]:
    pred = np.asarray(pred_mask).astype(bool)
    target = np.asarray(true_mask).astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(target)).sum())
    return tp, fp, fn, tn


def segmentation_metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
    }


@dataclass
class BinaryMetricAccumulator:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred_mask: np.ndarray, true_mask: np.ndarray) -> None:
        tp, fp, fn, tn = confusion_from_binary_masks(pred_mask, true_mask)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def compute(self) -> Dict[str, float]:
        return segmentation_metrics_from_confusion(self.tp, self.fp, self.fn, self.tn)


def instance_map_to_binary(instance_map: np.ndarray) -> np.ndarray:
    return np.asarray(instance_map) > 0


def instance_masks_from_map(instance_map: np.ndarray) -> List[np.ndarray]:
    instance_ids = [int(instance_id) for instance_id in np.unique(instance_map) if int(instance_id) > 0]
    return [(instance_map == instance_id) for instance_id in instance_ids]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return safe_divide(intersection, union)


def greedy_match_instances(
    pred_masks: Sequence[np.ndarray],
    true_masks: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    candidates: List[Tuple[float, int, int]] = []
    for pred_index, pred_mask in enumerate(pred_masks):
        for true_index, true_mask in enumerate(true_masks):
            iou = mask_iou(pred_mask, true_mask)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, true_index))

    candidates.sort(reverse=True)
    matched_preds = set()
    matched_truth = set()
    true_positives = 0

    for _, pred_index, true_index in candidates:
        if pred_index in matched_preds or true_index in matched_truth:
            continue
        matched_preds.add(pred_index)
        matched_truth.add(true_index)
        true_positives += 1

    false_positives = len(pred_masks) - true_positives
    false_negatives = len(true_masks) - true_positives
    return true_positives, false_positives, false_negatives


def instance_level_metrics(
    pred_instance_map: np.ndarray,
    true_instance_map: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    pred_masks = instance_masks_from_map(pred_instance_map)
    true_masks = instance_masks_from_map(true_instance_map)
    tp, fp, fn = greedy_match_instances(pred_masks, true_masks, iou_threshold=iou_threshold)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "instance_tp": float(tp),
        "instance_fp": float(fp),
        "instance_fn": float(fn),
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": f1,
    }
