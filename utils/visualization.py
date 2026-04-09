from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _to_numpy_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().float().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[..., np.newaxis]

    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > 1.0 or min_value < 0.0:
        scale = max(max_value - min_value, 1e-6)
        image = (image - min_value) / scale

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    return np.clip(image, 0.0, 1.0)


def _to_numpy_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    return mask.astype(np.int64)


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: Iterable[float], alpha: float = 0.35) -> np.ndarray:
    overlay = image.copy()
    mask_region = mask.astype(bool)
    if mask_region.any():
        overlay[mask_region] = (1.0 - alpha) * overlay[mask_region] + alpha * np.asarray(color)
    return overlay


def save_segmentation_preview(
    image_tensor: torch.Tensor,
    true_mask_tensor: torch.Tensor,
    pred_mask_tensor: torch.Tensor,
    output_path: Union[str, Path],
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    image = _to_numpy_image(image_tensor)
    true_mask = _to_numpy_mask(true_mask_tensor)
    pred_mask = _to_numpy_mask(pred_mask_tensor)

    true_foreground = true_mask > 0
    pred_foreground = pred_mask > 0

    error_map = np.zeros((*true_mask.shape, 3), dtype=np.float32)
    error_map[np.logical_and(pred_foreground, true_foreground)] = np.array([0.18, 0.72, 0.29], dtype=np.float32)
    error_map[np.logical_and(pred_foreground, ~true_foreground)] = np.array([0.86, 0.24, 0.23], dtype=np.float32)
    error_map[np.logical_and(~pred_foreground, true_foreground)] = np.array([0.95, 0.63, 0.17], dtype=np.float32)

    preview_path = Path(output_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(_overlay_mask(image, true_foreground, color=(0.18, 0.72, 0.29)))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(_overlay_mask(image, pred_foreground, color=(0.86, 0.24, 0.23)))
    axes[2].set_title("Prediction")
    axes[3].imshow(error_map)
    axes[3].set_title("TP / FP / FN")

    for axis in axes:
        axis.axis("off")

    if metrics:
        summary = " | ".join(
            f"{key}={metrics[key]:.4f}"
            for key in ("dice", "iou", "precision", "recall", "specificity", "accuracy")
            if key in metrics
        )
        fig.suptitle(summary, fontsize=11)

    fig.tight_layout()
    fig.savefig(preview_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history_rows: List[Dict[str, float]], output_path: Union[str, Path]) -> None:
    if not history_rows:
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history_rows]
    train_loss = [float(row.get("train_loss", 0.0)) for row in history_rows]
    val_loss = [float(row.get("val_loss", 0.0)) for row in history_rows]
    dice = [float(row.get("dice", 0.0)) for row in history_rows]
    iou = [float(row.get("iou", 0.0)) for row in history_rows]
    precision = [float(row.get("precision", 0.0)) for row in history_rows]
    recall = [float(row.get("recall", 0.0)) for row in history_rows]
    specificity = [float(row.get("specificity", 0.0)) for row in history_rows]
    accuracy = [float(row.get("accuracy", 0.0)) for row in history_rows]
    learning_rate = [float(row.get("learning_rate", 0.0)) for row in history_rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0, 0].plot(epochs, val_loss, marker="s", label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, dice, marker="o", label="dice")
    axes[0, 1].plot(epochs, iou, marker="s", label="iou")
    axes[0, 1].set_title("Overlap Metrics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, precision, marker="o", label="precision")
    axes[1, 0].plot(epochs, recall, marker="s", label="recall")
    axes[1, 0].plot(epochs, specificity, marker="^", label="specificity")
    axes[1, 0].set_title("Classification Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, accuracy, marker="o", color="#1f77b4", label="accuracy")
    axes[1, 1].set_title("Accuracy and Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].grid(alpha=0.3)

    lr_axis = axes[1, 1].twinx()
    lr_axis.plot(epochs, learning_rate, marker="s", color="#ff7f0e", label="learning_rate")
    lr_axis.set_ylabel("Learning rate")

    handles, labels = axes[1, 1].get_legend_handles_labels()
    lr_handles, lr_labels = lr_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + lr_handles, labels + lr_labels, loc="best")

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
