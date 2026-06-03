import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hzy_lesion_config import resolve_lesion_tasks
from train import compute_segmentation_loss
from utils.data_loading import BasicDataset, create_train_transform


def test_crescent_binary_task_merges_all_crescent_labels():
    task = resolve_lesion_tasks(["crescent_binary"])[0]

    assert task.slug == "crescent_binary"
    assert task.num_classes == 2
    assert set(task.class_map.values()) == {1}
    assert {label.slug for label in task.labels} == {
        "cellular_crescent",
        "fibrocellular_crescent",
        "fibrous_crescent",
    }


def test_training_transform_preserves_mask_labels(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image = Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8))
    mask_array = np.zeros((16, 16), dtype=np.uint8)
    mask_array[:, :8] = 1
    mask = Image.fromarray(mask_array)
    image.save(images_dir / "sample.jpg")
    mask.save(masks_dir / "sample.png")

    dataset = BasicDataset(
        images_dir=images_dir,
        mask_dir=masks_dir,
        scale=1.0,
        transform=create_train_transform("basic", seed=7),
    )
    sample = dataset[0]

    assert sample["image"].shape == (3, 16, 16)
    assert set(torch.unique(sample["mask"]).tolist()) == {0, 1}


def test_focal_tversky_loss_is_differentiable():
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    target = torch.randint(0, 2, (2, 8, 8))

    loss = compute_segmentation_loss(
        logits=logits,
        true_masks=target,
        n_classes=2,
        loss_mode="focal_tversky",
        foreground_dice_only=True,
        focal_weight=1.0,
        tversky_weight=1.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_generalized_dice_loss_is_differentiable():
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))

    loss = compute_segmentation_loss(
        logits=logits,
        true_masks=target,
        n_classes=4,
        loss_mode="generalized_dice",
        foreground_dice_only=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
