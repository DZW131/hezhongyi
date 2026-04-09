from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional at import time
    plt = None


def require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for visualization helpers. "
            'Install the project with `pip install -e ".[hubmap]"`.'
        )


def mask_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float], alpha: float = 0.4) -> np.ndarray:
    base = image.astype(np.float32) / 255.0
    overlay = base.copy()
    binary_mask = np.asarray(mask).astype(bool)
    overlay[binary_mask] = (1 - alpha) * overlay[binary_mask] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(overlay, 0.0, 1.0)


def save_prediction_preview(
    image: np.ndarray,
    true_instance_map: np.ndarray,
    pred_instance_map: np.ndarray,
    output_path: Path,
    metrics: Optional[Dict[str, float]] = None,
    prompt_points: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    true_binary = true_instance_map > 0
    pred_binary = pred_instance_map > 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(mask_overlay(image, true_binary, (0.1, 0.8, 0.1)))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(mask_overlay(image, pred_binary, (0.9, 0.2, 0.2)))
    axes[2].set_title("Prediction")

    if prompt_points:
        for point_x, point_y in prompt_points:
            axes[0].scatter([point_x], [point_y], c="yellow", s=16)
            axes[2].scatter([point_x], [point_y], c="yellow", s=16)

    for axis in axes:
        axis.axis("off")

    if metrics:
        title = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, (int, float)))
        fig.suptitle(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(history_rows, output_path: Path) -> None:
    require_matplotlib()
    if not history_rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history_rows if "epoch" in row]
    if not epochs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row.get("train_loss") for row in history_rows], label="train_loss")
    axes[0].plot(epochs, [row.get("val_loss") for row in history_rows], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    for key in ("val_dice", "val_iou", "val_precision", "val_recall"):
        values = [row.get(key) for row in history_rows]
        if any(value is not None for value in values):
            axes[1].plot(epochs, values, label=key)

    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

