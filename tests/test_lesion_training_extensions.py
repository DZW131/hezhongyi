import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hzy_lesion_config import resolve_lesion_tasks
from train import compute_segmentation_loss
from utils.detection_boxes import (
    Box,
    binary_confusion_metrics,
    boxes_from_multiclass_mask,
    match_boxes,
)
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


def test_proliferation_binary_task_merges_all_proliferation_labels():
    task = resolve_lesion_tasks(["proliferation_binary"])[0]

    assert task.slug == "proliferation_binary"
    assert task.num_classes == 2
    assert set(task.class_map.values()) == {1}
    assert {label.slug for label in task.labels} == {
        "mesangial_hypercellularity",
        "endocapillary_hypercellularity",
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


def test_multiclass_mask_can_be_collapsed_to_detection_boxes():
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[2:6, 3:8] = 1
    mask[12:18, 10:17] = 2

    boxes = boxes_from_multiclass_mask(mask, collapse_to_class=1, min_area=4, margin=1)

    assert len(boxes) == 2
    assert {box.class_id for box in boxes} == {1}
    assert boxes[0].x_min == 2
    assert boxes[0].y_min == 1
    assert boxes[0].x_max == 9
    assert boxes[0].y_max == 7


def test_detection_box_matching_accepts_relaxed_overlap():
    gt_boxes = [Box(10, 10, 30, 30, class_id=1)]
    pred_boxes = [Box(14, 14, 28, 28, class_id=1), Box(40, 40, 50, 50, class_id=1)]

    matches = match_boxes(gt_boxes, pred_boxes, iou_threshold=0.1)

    assert len(matches) == 1
    assert matches[0][0] == 0
    assert matches[0][1] == 0


def test_binary_confusion_metrics_report_detection_recall_and_specificity():
    metrics = binary_confusion_metrics(tp=8, fp=2, fn=1, tn=9)

    assert metrics["recall"] > 0.88
    assert metrics["specificity"] > 0.81
    assert metrics["precision"] == metrics["precision"]


def test_detection_box_flip_transforms_coordinates():
    pytest.importorskip("torchvision")
    from train_hzy_detection_boxes import YoloBoxDataset

    boxes = torch.tensor([[2.0, 3.0, 8.0, 10.0]], dtype=torch.float32)

    horizontal = YoloBoxDataset._flip_horizontal(boxes, width=20)
    vertical = YoloBoxDataset._flip_vertical(boxes, height=30)

    assert horizontal.tolist() == [[12.0, 3.0, 18.0, 10.0]]
    assert vertical.tolist() == [[2.0, 20.0, 8.0, 27.0]]
