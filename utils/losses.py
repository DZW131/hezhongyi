from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dice_score import dice_loss


AVAILABLE_LOSS_MODES = (
    'ce_dice',
    'focal_dice',
    'tversky',
    'focal_tversky',
    'generalized_dice',
    'ce_generalized_dice',
)


def _multiclass_probs_and_targets(
    logits: torch.Tensor,
    true_masks: torch.Tensor,
    n_classes: int,
    foreground_only: bool,
):
    probs = F.softmax(logits, dim=1).float()
    targets = F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float()

    if foreground_only and n_classes > 1:
        probs = probs[:, 1:]
        targets = targets[:, 1:]

    return probs, targets


def focal_cross_entropy_loss(
    logits: torch.Tensor,
    true_masks: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, true_masks, weight=class_weights, reduction='none')
    probs = F.softmax(logits, dim=1)
    pt = probs.gather(1, true_masks.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
    return (((1.0 - pt) ** gamma) * ce).mean()


def tversky_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    dims = (0, 2, 3)
    true_positive = (probs * targets).sum(dim=dims)
    false_positive = (probs * (1.0 - targets)).sum(dim=dims)
    false_negative = ((1.0 - probs) * targets).sum(dim=dims)
    score = (true_positive + epsilon) / (
        true_positive + alpha * false_positive + beta * false_negative + epsilon
    )
    return 1.0 - score.mean()


def generalized_dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    dims = (0, 2, 3)
    target_volume = targets.sum(dim=dims)
    weights = 1.0 / (target_volume.pow(2) + epsilon)
    weights = torch.where(target_volume > 0, weights, torch.zeros_like(weights))

    intersection = (probs * targets).sum(dim=dims)
    denominator = (probs + targets).sum(dim=dims)
    numerator = 2.0 * (weights * intersection).sum()
    denominator = (weights * denominator).sum()
    return 1.0 - (numerator + epsilon) / (denominator + epsilon)


def compute_segmentation_loss(
    logits: torch.Tensor,
    true_masks: torch.Tensor,
    n_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    foreground_dice_only: bool = False,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    loss_mode: str = 'ce_dice',
    focal_weight: float = 1.0,
    focal_gamma: float = 2.0,
    tversky_weight: float = 1.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    tversky_gamma: float = 1.0,
    generalized_dice_weight: float = 1.0,
) -> torch.Tensor:
    if loss_mode not in AVAILABLE_LOSS_MODES:
        raise ValueError(
            "Unsupported loss_mode '{}'. Valid modes: {}".format(
                loss_mode,
                ', '.join(AVAILABLE_LOSS_MODES),
            )
        )

    if n_classes == 1:
        bce = nn.BCEWithLogitsLoss()(logits.squeeze(1), true_masks.float())
        dice = dice_loss(torch.sigmoid(logits.squeeze(1)), true_masks.float(), multiclass=False)
        return ce_weight * bce + dice_weight * dice

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)(logits, true_masks)
    focal_loss = focal_cross_entropy_loss(
        logits=logits,
        true_masks=true_masks,
        class_weights=class_weights,
        gamma=focal_gamma,
    )
    probs, targets = _multiclass_probs_and_targets(
        logits=logits,
        true_masks=true_masks,
        n_classes=n_classes,
        foreground_only=foreground_dice_only,
    )

    dice = dice_loss(probs, targets, multiclass=True)
    tversky = tversky_loss(
        probs=probs,
        targets=targets,
        alpha=tversky_alpha,
        beta=tversky_beta,
    )
    focal_tversky = torch.pow(tversky.clamp_min(1e-6), tversky_gamma)
    generalized_dice = generalized_dice_loss(probs=probs, targets=targets)

    if loss_mode == 'ce_dice':
        return ce_weight * ce_loss + dice_weight * dice
    if loss_mode == 'focal_dice':
        return focal_weight * focal_loss + dice_weight * dice
    if loss_mode == 'tversky':
        return ce_weight * ce_loss + tversky_weight * tversky
    if loss_mode == 'focal_tversky':
        return focal_weight * focal_loss + tversky_weight * focal_tversky
    if loss_mode == 'generalized_dice':
        return generalized_dice_weight * generalized_dice
    if loss_mode == 'ce_generalized_dice':
        return ce_weight * ce_loss + generalized_dice_weight * generalized_dice

    raise AssertionError("Unhandled loss mode: {}".format(loss_mode))
