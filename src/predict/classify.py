import numpy as np
from scipy.ndimage import find_objects
from scipy.special import softmax

from ..prepare.mask import check_masks


def classify_instances(
    masks: np.ndarray, logits: np.ndarray, classes: tuple[str, ...]
) -> tuple[dict, ...]:
    masks = check_masks(masks)
    if not isinstance(logits, np.ndarray) or not np.issubdtype(
        logits.dtype, np.floating
    ):
        raise TypeError(
            "class logits must be a NumPy array with a real floating dtype."
        )
    if logits.shape != (len(classes), *masks.shape) or not classes:
        raise ValueError(
            "class logits must have shape [classes, mask height, mask width]."
        )
    if not np.isfinite(logits).all():
        raise ValueError("class logits must contain only finite values.")
    probs = softmax(logits, axis=0)
    mask_ids, inverse = np.unique(masks, return_inverse=True)
    # Compact labels keep bounding-box lookup independent of the largest ID.
    regions = find_objects(inverse.reshape(masks.shape) + 1)
    instances = []
    for mask_id, region in zip(mask_ids, regions):
        if mask_id == 0:
            continue
        region_probs = probs[(slice(None), *region)]
        scores = region_probs[:, masks[region] == mask_id].mean(axis=1)
        class_id = int(scores.argmax())
        instances.append(
            {
                "mask_id": int(mask_id),
                "class_id": class_id,
                "class_name": classes[class_id],
                "score": float(scores[class_id]),
            }
        )
    return tuple(instances)
