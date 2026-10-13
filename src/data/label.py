"""Temporary JSON format mapping instance IDs to class names."""

import json
from pathlib import Path

import numpy as np


def read_labels(
    path: Path, masks: np.ndarray, classes: tuple[str, ...]
) -> dict[int, int]:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate mask ID {key!r} in {path}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as file:
        labels = json.load(file, object_pairs_hook=unique_keys)
    if not isinstance(labels, dict):
        raise TypeError(f"labels must map mask IDs to class names: {path}")
    class_ids = {name: idx for idx, name in enumerate(classes)}
    result = {}
    for key, name in labels.items():
        if (
            not key.isascii()
            or not key.isdecimal()
            or int(key) <= 0
            or str(int(key)) != key
        ):
            raise ValueError(f"mask IDs must be positive integer strings: {path}")
        if not isinstance(name, str) or name not in class_ids:
            raise ValueError(f"unknown class {name!r} for mask {key} in {path}")
        result[int(key)] = class_ids[name]
    expected = set(np.unique(masks)) - {0}
    if set(result) != expected:
        raise ValueError(
            f"label IDs must match mask IDs in {path}: "
            f"missing={sorted(expected - set(result))}, extra={sorted(set(result) - expected)}"
        )
    return result


def class_target(masks: np.ndarray, labels: dict[int, int]) -> np.ndarray:
    # Background and augmentation padding never supervise an object class.
    mask_ids, inverse = np.unique(masks, return_inverse=True)
    values = np.full(len(mask_ids), -1, dtype=np.float32)
    for idx, mask_id in enumerate(mask_ids):
        if mask_id > 0:
            values[idx] = labels[int(mask_id)]
    return values[inverse].reshape(masks.shape)
