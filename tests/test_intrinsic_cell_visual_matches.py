from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from visualize_hzy_intrinsic_cell_matches import classify_matches


def test_duplicate_prediction_cannot_be_counted_correct_twice():
    result = classify_matches([[10, 10], [11, 10]], [0, 0], [[10, 10]], [0])
    assert len(result["correct"]) == 1
    assert len(result["false_positive"]) == 1
    assert not result["missed"]


def test_class_error_is_not_marked_as_correct():
    result = classify_matches([[10, 10], [80, 80]], [1, 2], [[10, 10], [50, 50]], [0, 1])
    assert result == {"correct": [], "wrong_class": [(0, 0)], "false_positive": [1], "missed": [1]}


def test_matching_distance_is_strict_and_empty_inputs_work():
    result = classify_matches([[10, 0]], [0], [[0, 0]], [0])
    assert not result["correct"]
    assert result["false_positive"] == [0]
    assert result["missed"] == [0]
    assert classify_matches(np.empty((0, 2)), [], [[0, 0]], [0])["missed"] == [0]
