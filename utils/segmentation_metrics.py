from typing import Dict, Iterable, List, Optional

import torch


EPSILON = 1e-6


def logits_to_labels(logits: torch.Tensor, n_classes: int, threshold: float = 0.5) -> torch.Tensor:
    if n_classes == 1:
        return (torch.sigmoid(logits[:, 0]) > threshold).long()
    return logits.argmax(dim=1).long()


def format_metrics(metrics: Dict[str, float], keys: Optional[Iterable[str]] = None, precision: int = 4) -> str:
    metric_keys = list(keys or ("loss", "dice", "iou", "precision", "recall", "specificity", "accuracy"))
    parts = []
    for key in metric_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.{precision}f}")
    return ", ".join(parts)


class SegmentationMetricAccumulator:
    def __init__(self, n_classes: int, threshold: float = 0.5, ignore_background: bool = True) -> None:
        self.n_classes = n_classes
        self.threshold = threshold
        self.ignore_background = ignore_background

        if self.n_classes == 1:
            self.class_indices = [1]
        else:
            start_class = 1 if self.ignore_background and self.n_classes > 1 else 0
            self.class_indices = list(range(start_class, self.n_classes))

        if not self.class_indices:
            self.class_indices = [0]

        num_classes = len(self.class_indices)
        self.true_positive = torch.zeros(num_classes, dtype=torch.float64)
        self.false_positive = torch.zeros(num_classes, dtype=torch.float64)
        self.false_negative = torch.zeros(num_classes, dtype=torch.float64)
        self.true_negative = torch.zeros(num_classes, dtype=torch.float64)
        self.correct_pixels = 0.0
        self.total_pixels = 0.0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        predicted = logits_to_labels(logits.detach(), self.n_classes, self.threshold).cpu()
        truth = target.detach().cpu().long()

        if truth.ndim == 4 and truth.size(1) == 1:
            truth = truth.squeeze(1)

        self.correct_pixels += (predicted == truth).sum().item()
        self.total_pixels += truth.numel()

        for index, class_id in enumerate(self.class_indices):
            pred_mask = predicted == class_id
            true_mask = truth == class_id

            self.true_positive[index] += torch.logical_and(pred_mask, true_mask).sum().item()
            self.false_positive[index] += torch.logical_and(pred_mask, ~true_mask).sum().item()
            self.false_negative[index] += torch.logical_and(~pred_mask, true_mask).sum().item()
            self.true_negative[index] += torch.logical_and(~pred_mask, ~true_mask).sum().item()

    def compute(self) -> Dict[str, float]:
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative

        precision = (tp + EPSILON) / (tp + fp + EPSILON)
        recall = (tp + EPSILON) / (tp + fn + EPSILON)
        specificity = (tn + EPSILON) / (tn + fp + EPSILON)
        dice = (2 * tp + EPSILON) / (2 * tp + fp + fn + EPSILON)
        iou = (tp + EPSILON) / (tp + fp + fn + EPSILON)

        metrics = {
            "dice": dice.mean().item(),
            "iou": iou.mean().item(),
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "specificity": specificity.mean().item(),
            "accuracy": self.correct_pixels / max(self.total_pixels, 1.0),
        }

        if len(self.class_indices) > 1:
            for index, class_id in enumerate(self.class_indices):
                metrics["class_{}_dice".format(class_id)] = dice[index].item()
                metrics["class_{}_iou".format(class_id)] = iou[index].item()

        return metrics
