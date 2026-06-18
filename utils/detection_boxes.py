from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


EPSILON = 1e-6


@dataclass(frozen=True)
class Box:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    class_id: int = 1
    area: int = 0

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def box_area(self) -> int:
        return self.width * self.height

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.width, self.height

    def to_yolo(self, image_width: int, image_height: int, yolo_class_id: Optional[int] = None) -> Tuple[int, float, float, float, float]:
        class_id = self.class_id if yolo_class_id is None else yolo_class_id
        x_center = (self.x_min + self.x_max) / 2.0 / max(image_width, 1)
        y_center = (self.y_min + self.y_max) / 2.0 / max(image_height, 1)
        width = self.width / max(image_width, 1)
        height = self.height / max(image_height, 1)
        return class_id, x_center, y_center, width, height


def _clip_box(x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int, margin: int) -> Tuple[int, int, int, int]:
    return (
        max(0, int(x_min) - margin),
        max(0, int(y_min) - margin),
        min(width, int(x_max) + margin),
        min(height, int(y_max) + margin),
    )


def boxes_from_binary_mask(mask: np.ndarray, class_id: int = 1, min_area: int = 16, margin: int = 0) -> List[Box]:
    binary = np.asarray(mask).astype(bool)
    height, width = binary.shape[:2]
    labeled, count = ndimage.label(binary)
    boxes: List[Box] = []

    for label_id in range(1, count + 1):
        ys, xs = np.where(labeled == label_id)
        area = int(xs.size)
        if area < min_area:
            continue
        x_min, y_min = int(xs.min()), int(ys.min())
        x_max, y_max = int(xs.max()) + 1, int(ys.max()) + 1
        clipped = _clip_box(x_min, y_min, x_max, y_max, width=width, height=height, margin=margin)
        boxes.append(Box(*clipped, class_id=class_id, area=area))

    return boxes


def boxes_from_multiclass_mask(
    mask: np.ndarray,
    foreground_classes: Optional[Iterable[int]] = None,
    collapse_to_class: Optional[int] = None,
    min_area: int = 16,
    margin: int = 0,
) -> List[Box]:
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("Expected a 2D mask array, got shape {}".format(mask_array.shape))

    class_ids = sorted(int(value) for value in np.unique(mask_array) if int(value) != 0)
    if foreground_classes is not None:
        allowed = {int(value) for value in foreground_classes}
        class_ids = [class_id for class_id in class_ids if class_id in allowed]

    boxes: List[Box] = []
    if collapse_to_class is not None:
        foreground = np.isin(mask_array, class_ids)
        boxes.extend(boxes_from_binary_mask(foreground, class_id=int(collapse_to_class), min_area=min_area, margin=margin))
        return boxes

    for class_id in class_ids:
        boxes.extend(boxes_from_binary_mask(mask_array == class_id, class_id=class_id, min_area=min_area, margin=margin))
    return boxes


def box_iou(first: Box, second: Box) -> float:
    x_min = max(first.x_min, second.x_min)
    y_min = max(first.y_min, second.y_min)
    x_max = min(first.x_max, second.x_max)
    y_max = min(first.y_max, second.y_max)

    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    union = first.box_area + second.box_area - intersection
    return float(intersection) / float(max(union, 1))


def box_center_inside(inner: Box, outer: Box) -> bool:
    x_center = (inner.x_min + inner.x_max) / 2.0
    y_center = (inner.y_min + inner.y_max) / 2.0
    return outer.x_min <= x_center <= outer.x_max and outer.y_min <= y_center <= outer.y_max


def match_boxes(gt_boxes: Sequence[Box], pred_boxes: Sequence[Box], iou_threshold: float = 0.1, allow_center_hit: bool = True) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            if gt_box.class_id != pred_box.class_id:
                continue
            iou = box_iou(gt_box, pred_box)
            if iou >= iou_threshold or (allow_center_hit and box_center_inside(pred_box, gt_box)):
                candidates.append((iou, gt_index, pred_index))

    matches: List[Tuple[int, int, float]] = []
    used_gt = set()
    used_pred = set()
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def binary_confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }
