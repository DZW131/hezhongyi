from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_hzy_intrinsic_cells import adjust_classes, blend_caches, validation_objective
from refine_hzy_intrinsic_cells import ContextClassifier, cell_patches, metadata_features


def item(coords, scores):
    return {"coords": np.asarray(coords, dtype=np.float32), "scores": np.asarray(scores, dtype=np.float32),
            "points": np.array([[5, 5]], dtype=np.float32), "roi": np.ones((10, 10), dtype=bool),
            "row": {"crop_id": "a"}}


def test_fusion_endpoints_and_coordinate_weighting():
    a = [item([[2, 2]], [[.8, .1, .05, .05]])]
    b = [item([[4, 4]], [[.4, .2, .1, .3]])]
    np.testing.assert_allclose(blend_caches(a, b, 0)[0]["coords"], a[0]["coords"])
    np.testing.assert_allclose(blend_caches(a, b, 1)[0]["scores"], b[0]["scores"])
    expected = (np.array([2, 2]) * .4 + np.array([4, 4]) * .2) / .6
    np.testing.assert_allclose(blend_caches(a, b, .5)[0]["coords"][0], expected)


def test_fusion_rejects_mismatched_crop_order():
    a = [item([[2, 2]], [[.8, .1, .05, .05]])]
    b = [item([[4, 4]], [[.4, .2, .1, .3]])]
    b[0]["row"]["crop_id"] = "other"
    with pytest.raises(ValueError):
        blend_caches(a, b, .5)


def test_class_calibration_normalizes_without_mutating_input():
    a = [item([[2, 2]], [[.2, .4, .2, .2]])]
    before = a[0]["scores"].copy()
    adjusted = adjust_classes(a, [1, .5, 1])
    assert np.isclose(adjusted[0]["scores"].sum(), 1)
    assert adjusted[0]["scores"][0, 1] < before[0, 1]
    np.testing.assert_array_equal(a[0]["scores"], before)


def test_local_and_context_patch_order_and_edge_padding():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[16, 16] = [10, 20, 30]
    patches = cell_patches(image, [[16, 16], [0, 0]], sides=(16, 32), size=16)
    assert patches.shape == (2, 6, 16, 16)
    assert patches.dtype == np.uint8
    assert patches[0, :3].sum() > 0
    assert cell_patches(image, []).shape == (0, 6, 64, 64)


def test_refiner_features_do_not_include_ground_truth():
    roi = np.ones((10, 10), dtype=bool)
    features = metadata_features(np.array([[4.5, 4.5]]), np.array([[.1, .2, .3, .4]]), roi)
    assert features.shape == (1, 7)
    np.testing.assert_allclose(features[0, :4], [.1, .2, .3, .4])
    np.testing.assert_allclose(features[0, 4:], 0)


def test_refiner_cpu_forward_is_finite():
    model = ContextClassifier().eval()
    with torch.no_grad():
        output = model(torch.zeros((2, 6, 64, 64), dtype=torch.uint8), torch.zeros((2, 7)))
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
