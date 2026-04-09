from __future__ import annotations

from typing import Dict

import torch
import torch.distributed as dist

from hubmap_sam2.metrics import segmentation_metrics_from_confusion


class BinarySegmentationMeter:
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.instance_count = 0.0

    def update(self, find_stages, find_metadatas=None, targets=None) -> None:
        if targets is None:
            return
        stage = find_stages[0] if isinstance(find_stages, list) else find_stages
        pred_masks = stage["pred_masks_high_res"]
        if pred_masks.dim() == 4:
            pred_masks = pred_masks[:, 0]
        target_masks = targets[0] if targets.dim() == 4 else targets

        pred_binary = pred_masks > self.threshold
        target_binary = target_masks > 0
        pred_flat = pred_binary.reshape(pred_binary.shape[0], -1)
        target_flat = target_binary.reshape(target_binary.shape[0], -1)

        tp = torch.logical_and(pred_flat, target_flat).sum(dim=1).float()
        fp = torch.logical_and(pred_flat, ~target_flat).sum(dim=1).float()
        fn = torch.logical_and(~pred_flat, target_flat).sum(dim=1).float()
        tn = torch.logical_and(~pred_flat, ~target_flat).sum(dim=1).float()

        self.tp += float(tp.sum().item())
        self.fp += float(fp.sum().item())
        self.fn += float(fn.sum().item())
        self.tn += float(tn.sum().item())

        dice = (2 * tp + 1.0) / (2 * tp + fp + fn + 1.0)
        iou = (tp + 1.0) / (tp + fp + fn + 1.0)
        self.dice_sum += float(dice.sum().item())
        self.iou_sum += float(iou.sum().item())
        self.instance_count += float(pred_flat.shape[0])

    def compute(self) -> Dict[str, float]:
        metrics = segmentation_metrics_from_confusion(int(self.tp), int(self.fp), int(self.fn), int(self.tn))
        metrics["mean_dice"] = self.dice_sum / self.instance_count if self.instance_count else 0.0
        metrics["mean_iou"] = self.iou_sum / self.instance_count if self.instance_count else 0.0
        return metrics

    def compute_synced(self) -> Dict[str, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        values = torch.tensor(
            [self.tp, self.fp, self.fn, self.tn, self.dice_sum, self.iou_sum, self.instance_count],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)

        tp, fp, fn, tn, dice_sum, iou_sum, instance_count = values.tolist()
        metrics = segmentation_metrics_from_confusion(int(tp), int(fp), int(fn), int(tn))
        metrics["mean_dice"] = dice_sum / instance_count if instance_count else 0.0
        metrics["mean_iou"] = iou_sum / instance_count if instance_count else 0.0
        return metrics

    @staticmethod
    def is_better(current: float, previous: float) -> bool:
        return current > previous
