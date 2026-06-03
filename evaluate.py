from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm

from utils.losses import compute_segmentation_loss
from utils.segmentation_metrics import SegmentationMetricAccumulator, logits_to_labels


@torch.inference_mode()
def evaluate(
    net: torch.nn.Module,
    dataloader,
    device: torch.device,
    amp: bool,
    threshold: float = 0.5,
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
) -> Tuple[Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    net.eval()
    num_val_batches = len(dataloader)
    running_loss = 0.0
    preview = None
    non_blocking = device.type == 'cuda'
    metrics = SegmentationMetricAccumulator(
        n_classes=net.n_classes,
        threshold=threshold,
        ignore_background=net.n_classes > 1
    )

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last,
                non_blocking=non_blocking,
            )
            mask_true = mask_true.to(device=device, dtype=torch.long, non_blocking=non_blocking)

            logits = net(image)
            loss = compute_segmentation_loss(
                logits,
                mask_true,
                net.n_classes,
                class_weights=class_weights,
                foreground_dice_only=foreground_dice_only,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
                loss_mode=loss_mode,
                focal_weight=focal_weight,
                focal_gamma=focal_gamma,
                tversky_weight=tversky_weight,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
                tversky_gamma=tversky_gamma,
                generalized_dice_weight=generalized_dice_weight,
            )
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
