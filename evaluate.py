from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import dice_loss
from utils.segmentation_metrics import SegmentationMetricAccumulator, logits_to_labels


def _compute_validation_loss(logits: torch.Tensor, true_masks: torch.Tensor, n_classes: int) -> torch.Tensor:
    if n_classes == 1:
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(1), true_masks.float())
        loss += dice_loss(torch.sigmoid(logits.squeeze(1)), true_masks.float(), multiclass=False)
        return loss

    loss = nn.CrossEntropyLoss()(logits, true_masks)
    loss += dice_loss(
        F.softmax(logits, dim=1).float(),
        F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float(),
        multiclass=True
    )
    return loss


@torch.inference_mode()
def evaluate(
    net: torch.nn.Module,
    dataloader,
    device: torch.device,
    amp: bool,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    net.eval()
    num_val_batches = len(dataloader)
    running_loss = 0.0
    preview = None
    metrics = SegmentationMetricAccumulator(
        n_classes=net.n_classes,
        threshold=threshold,
        ignore_background=net.n_classes > 1
    )

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            logits = net(image)
            loss = _compute_validation_loss(logits, mask_true, net.n_classes)
            running_loss += loss.item()
            metrics.update(logits, mask_true)

            if preview is None and image.size(0) > 0:
                preview = {
                    'image': image[0].detach().cpu(),
                    'true_mask': mask_true[0].detach().cpu(),
                    'pred_mask': logits_to_labels(logits, net.n_classes, threshold)[0].detach().cpu(),
                }

    net.train()

    metric_values = metrics.compute()
    metric_values['loss'] = running_loss / max(num_val_batches, 1)
    return metric_values, preview
