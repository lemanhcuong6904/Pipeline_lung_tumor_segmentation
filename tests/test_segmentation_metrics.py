import numpy as np
import pytest

from utils.segmentation_metrics import (
    SliceRecord,
    aggregate_case_slices,
    binary_overlap_metrics,
    binary_volume_metrics,
)


def test_overlap_scores_are_perfect_for_two_empty_masks():
    assert binary_overlap_metrics(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == {
        "dice": 1.0,
        "iou": 1.0,
    }


def test_volume_surface_distance_uses_z_spacing():
    target = np.zeros((3, 1, 1), bool)
    prediction = np.zeros_like(target)
    target[0, 0, 0] = True
    prediction[1, 0, 0] = True

    metrics = binary_volume_metrics(prediction, target, (1.0, 1.0, 3.0))

    assert metrics["assd"] == pytest.approx(3.0)
    assert metrics["hd95"] == pytest.approx(3.0)


def test_case_aggregation_rejects_missing_slice_indices():
    records = [
        SliceRecord("case", 0, np.zeros((1, 1), bool), np.zeros((1, 1), bool)),
        SliceRecord("case", 2, np.zeros((1, 1), bool), np.zeros((1, 1), bool)),
    ]

    with pytest.raises(ValueError, match="non-contiguous"):
        aggregate_case_slices(records)
