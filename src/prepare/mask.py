# Cellpose mask helpers adapted from MouseLand/cellpose at a54cb488 (BSD-3-Clause).
# Copyright © 2025 Howard Hughes Medical Institute.

import numpy as np


def check_masks(masks: np.ndarray) -> np.ndarray:
    if not isinstance(masks, np.ndarray):
        raise TypeError("masks must be a NumPy array.")
    if masks.ndim != 2:
        raise ValueError("masks must have shape [H, W].")
    if 0 in masks.shape:
        raise ValueError("masks must have non-empty spatial dimensions.")
    if not np.issubdtype(masks.dtype, np.integer):
        raise TypeError("masks must use an integer dtype.")
    if np.any(masks < 0):
        raise ValueError("masks must contain non-negative labels.")
    return masks


def renumber_masks(masks: np.ndarray) -> np.ndarray:
    values, inverse = np.unique(masks, return_inverse=True)
    labels = np.zeros(len(values), dtype=np.int32)
    labels[values > 0] = np.arange(1, np.count_nonzero(values > 0) + 1)
    return labels[inverse].reshape(masks.shape)
