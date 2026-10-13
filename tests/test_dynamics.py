import numpy as np
import pytest

from src import FLOW_SCALE
from src.predict.dynamics import _fill_and_filter_masks, compute_masks, follow_flows
from src.prepare.flow import flow_targets, masks_to_flows


def disk(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    return (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius**2


def best_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    values = np.unique(predicted[target])
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    return max(
        np.logical_and(predicted == value, target).sum()
        / np.logical_or(predicted == value, target).sum()
        for value in values
    )


def test_empty_inputs_return_empty_outputs() -> None:
    masks = np.zeros((16, 20), dtype=np.int32)
    flows = masks_to_flows(masks)
    targets = flow_targets(masks)

    assert flows.shape == (2, 16, 20)
    assert flows.dtype == np.float32
    assert not flows.any()
    assert targets.shape == (3, 16, 20)
    assert not targets.any()
    assert follow_flows(flows, masks > 0).shape == (2, 0)

    reconstructed = compute_masks(
        flows,
        np.full(masks.shape, -1.0, dtype=np.float32),
    )
    assert reconstructed.dtype == np.uint16
    assert not reconstructed.any()


def test_invalid_inputs_are_rejected_early() -> None:
    with pytest.raises(TypeError, match="integer"):
        masks_to_flows(np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="non-negative"):
        masks_to_flows(-np.ones((8, 8), dtype=np.int32))
    with pytest.raises(ValueError, match=r"\[H, W\]"):
        masks_to_flows(np.zeros((1, 8, 8), dtype=np.int32))

    flows = np.zeros((2, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="shape must match"):
        follow_flows(flows, np.zeros((7, 8), dtype=bool))
    with pytest.raises(ValueError, match=r"\[2, H, W\]"):
        compute_masks(
            np.zeros((3, 8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="max_size_fraction"):
        compute_masks(
            flows,
            np.zeros((8, 8), dtype=np.float32),
            max_size_fraction=0.0,
        )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_dynamics_rejects_complex_fields_and_logits(dtype):
    flows = np.zeros((2, 8, 8), dtype=dtype)
    mask = np.ones((8, 8), dtype=bool)
    with pytest.raises(TypeError, match="real"):
        follow_flows(flows, mask)
    with pytest.raises(TypeError, match="real"):
        compute_masks(flows.real, np.zeros(mask.shape, dtype=dtype))


def test_flow_tracking_rejects_values_outside_float32_range():
    flows = np.full((2, 8, 8), 1e300, dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        follow_flows(flows, np.ones((8, 8), dtype=bool), niter=2)


def test_mask_flows_are_unit_normalized_and_point_inward() -> None:
    center = (16, 16)
    foreground = disk((33, 33), center, 8)
    masks = foreground.astype(np.int32)

    flows = masks_to_flows(masks)
    magnitude = np.linalg.norm(flows, axis=0)

    assert not flows[:, ~foreground].any()
    assert np.allclose(magnitude[foreground & (magnitude > 0)], 1.0, atol=1.0e-6)
    assert flows[0, center[0] - 6, center[1]] > 0.9
    assert flows[0, center[0] + 6, center[1]] < -0.9
    assert flows[1, center[0], center[1] - 6] > 0.9
    assert flows[1, center[0], center[1] + 6] < -0.9


def test_two_disks_round_trip_through_dynamics() -> None:
    masks = np.zeros((64, 64), dtype=np.int32)
    first = disk(masks.shape, (19, 19), 7)
    second = disk(masks.shape, (43, 42), 8)
    masks[first] = 4
    masks[second] = 9
    targets = flow_targets(masks)
    cell_logits = np.where(targets[0] > 0, 1.0, -1.0).astype(np.float32)

    reconstructed = compute_masks(
        targets[1:] * FLOW_SCALE,
        cell_logits,
        flow_threshold=0.4,
        min_size=15,
    )

    assert reconstructed.max() == 2
    assert best_iou(reconstructed, first) > 0.98
    assert best_iou(reconstructed, second) > 0.98


def test_small_masks_and_bad_flows_are_filtered() -> None:
    masks = np.zeros((48, 48), dtype=np.int32)
    small = disk(masks.shape, (10, 10), 2)
    large = disk(masks.shape, (31, 31), 7)
    masks[small] = 1
    masks[large] = 2
    targets = flow_targets(masks)
    cell_logits = np.where(targets[0] > 0, 1.0, -1.0).astype(np.float32)

    filtered = compute_masks(
        targets[1:] * FLOW_SCALE,
        cell_logits,
        flow_threshold=None,
        min_size=15,
    )
    assert filtered.max() == 1
    assert not filtered[small].any()
    assert best_iou(filtered, large) > 0.98

    large_targets = flow_targets(large.astype(np.int32))
    large_logits = np.where(large, 1.0, -1.0).astype(np.float32)
    without_qc = compute_masks(
        large_targets[1:],
        large_logits,
        flow_threshold=None,
        min_size=15,
    )
    with_qc = compute_masks(
        large_targets[1:],
        large_logits,
        flow_threshold=0.4,
        min_size=15,
    )
    assert without_qc.max() == 1
    assert with_qc.max() == 0


@pytest.mark.parametrize("outer, inner", [(1, 2), (2, 1)])
def test_hole_filling_preserves_enclosed_instances(outer, inner):
    masks = np.zeros((32, 32), dtype=np.int32)
    masks[3:29, 3:29] = outer
    masks[8:24, 8:24] = 0
    masks[13:19, 13:19] = inner

    filled = _fill_and_filter_masks(masks, min_size=15)

    assert set(np.unique(filled)) == {0, 1, 2}
    assert filled[4, 4] != filled[16, 16]
    assert np.all(filled[3:29, 3:29] > 0)
    assert np.all(filled[masks == outer] == filled[4, 4])
    assert np.all(filled[masks == inner] == filled[16, 16])

    targets = flow_targets(masks)
    result = compute_masks(
        targets[1:] * FLOW_SCALE,
        np.where(masks > 0, 10.0, -10.0),
        flow_threshold=None,
        max_size_fraction=1.0,
    )

    assert set(np.unique(result)) == {0, 1, 2}
    assert result[4, 4] != result[16, 16]


def test_mask_cleanup_preserves_sparse_labels_and_disconnected_regions():
    masks = np.zeros((24, 32), dtype=np.int32)
    masks[1:6, 2:8] = 9
    masks[15:21, 23:29] = 9
    masks[8:12, 10:14] = 3
    masks[2, 20] = 5
    expected = np.zeros_like(masks)
    expected[masks == 3] = 1
    expected[masks == 9] = 2

    np.testing.assert_array_equal(_fill_and_filter_masks(masks, 15), expected)
