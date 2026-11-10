import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hzy_intrinsic_cell_upgrade import (ROIClassificationLoss, geometric_augment, match_points,
                                        score_cached, select_points)
from prepare_hzy_intrinsic_cell_upgrade import assign_splits, case_key, make_roi_annotation


def test_legacy_cases_are_never_reassigned():
    old = {"train": ["20990325"], "val": ["20990332"], "test": ["20990331"]}
    slides = ["20990325", "20990332", "20990331"] + ["209904{:02d}".format(i) for i in range(19)]
    first = assign_splits(slides, old, {}, 42)
    assert first == assign_splits(slides, old, {}, 42)
    assert first["20990331"] == "test"
    assert first["20990332"] == "val"
    assert first["20990325"] == "train"
    assert case_key("SK2099515") == case_key("20990515")


def test_roi_drops_context_points_and_duplicate_marks():
    feature = {"geometry": {"coordinates": [[[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]]}}
    points = [{"x": 5, "y": 5, "class_slug": "mesangial_cell"},
              {"x": 5, "y": 5, "class_slug": "mesangial_cell"},
              {"x": 9, "y": 9, "class_slug": "mesangial_cell"}]
    annotation, mask = make_roi_annotation(feature, points, 0, 0, 10, 10)
    assert annotation["mesangial_cell"] == [[5.0, 5.0]]
    assert mask[5, 5] == 255
    assert mask[9, 9] == 0


def test_flip_and_rotation_keep_mask_point_alignment():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2, 3] = 255
    roi = image[..., 0] > 0
    transformed, points, mask = geometric_augment(image, np.array([[3., 2.]]), roi, True, True, 1)
    x, y = points[0].astype(int)
    assert mask[y, x]
    assert transformed[y, x, 0] == 255


def test_conflicting_cell_labels_are_ignored_not_arbitrarily_resolved():
    feature = {"geometry": {"coordinates": [[[0, 0], [19, 0], [19, 19], [0, 19], [0, 0]]]}}
    points = [{"x": 10, "y": 10, "class_slug": name} for name in ("mesangial_cell", "endothelial_cell")]
    annotation, mask = make_roi_annotation(feature, points, 0, 0, 20, 20)
    assert annotation["mesangial_cell"] == []
    assert annotation["endothelial_cell"] == []
    assert annotation["ambiguous_points"] == [[10, 10]]
    assert mask[10, 10] == 0


def test_small_deduplication_radius_retains_distinct_nearby_cells():
    coords = np.array([[5, 5], [11, 5]], dtype=np.float32)
    scores = np.array([[.05, .85, .05, .05], [.05, .8, .05, .1]])
    mask = np.ones((20, 20), dtype=bool)
    assert len(select_points(coords, scores, mask, dedup_distance=4)[0]) == 2
    assert len(select_points(coords, scores, mask, dedup_distance=10)[0]) == 1
    mask[5, 11] = False
    assert len(select_points(coords, scores, mask, dedup_distance=4)[0]) == 1


def test_point_matching_uses_one_to_one_maximum_cardinality():
    pred = np.array([[0., 0.], [1., 0.]])
    truth = np.array([[0.5, 0.], [-0.5, 0.]])
    assert match_points(pred, truth, 1.0) == 2
    assert match_points(np.empty((0, 2)), truth) == 0


class FixedMatcher:
    def __call__(self, outputs, targets):
        return [(torch.tensor([0]), torch.tensor([0]))]


def test_unannotated_background_has_no_loss_gradient():
    logits = torch.tensor([[[0., 2., 0., 0.], [0., 8., 0., 0.]]], requires_grad=True)
    coords = torch.tensor([[[3., 3.], [8., 8.]]], requires_grad=True)
    roi = torch.zeros((10, 10), dtype=torch.bool)
    roi[1:6, 1:6] = True
    sample = {"points": torch.tensor([[3., 3.]]), "labels": torch.tensor([1]), "roi": roi}
    loss, _ = ROIClassificationLoss(FixedMatcher())({"pnt_coords": coords, "cls_logits": logits}, sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].abs().sum() == 0


def test_matched_positive_outside_roi_still_receives_supervision():
    logits = torch.zeros((1, 1, 4), requires_grad=True)
    coords = torch.tensor([[[8., 8.]]], requires_grad=True)
    sample = {"points": torch.tensor([[3., 3.]]), "labels": torch.tensor([1]),
              "roi": torch.zeros((10, 10), dtype=torch.bool)}
    loss, _ = ROIClassificationLoss(FixedMatcher())({"pnt_coords": coords, "cls_logits": logits}, sample)
    loss.backward()
    assert coords.grad.abs().sum() > 0
    assert logits.grad.abs().sum() > 0


def test_wrong_cell_class_is_not_a_classification_true_positive():
    item = {"coords": np.array([[5., 5.]]), "scores": np.array([[.9, .03, .02, .05]]),
            "roi": np.ones((10, 10), dtype=bool), "points": np.array([[5., 5.]]),
            "labels": np.array([1]), "row": {"crop_id": "a", "slide_id": "a", "case_key": "a"}}
    metrics, _, _ = score_cached([item])
    assert metrics["detection"]["f1"] == 1
    assert metrics["per_class"]["mesangial_cell"]["f1"] == 0
