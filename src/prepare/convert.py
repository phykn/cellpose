import numpy as np


def normalize_image(
    image: np.ndarray,
    channel_axis: int | None = None,
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray.")
    if image.ndim not in (2, 3):
        raise ValueError(
            "image must have two spatial dimensions and up to three channels."
        )
    if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
        raise TypeError("image must contain real numeric values.")
    if image.size == 0:
        raise ValueError("image must not be empty.")

    if image.ndim == 2:
        if channel_axis is not None:
            raise ValueError(
                "channel_axis is only valid for a three-dimensional image."
            )
        channels = image[None]
    else:
        axis = get_channel_axis(channel_axis, image.ndim)
        channels = np.moveaxis(image, axis, 0)

    if not 1 <= channels.shape[0] <= 3:
        raise ValueError("image must contain between one and three channels.")
    if not np.isfinite(channels).all():
        raise ValueError("image must contain only finite values.")

    result = np.zeros((3, *channels.shape[1:]), dtype=np.float32)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for idx, channel in enumerate(channels):
                if np.issubdtype(channel.dtype, np.integer):
                    # Unsigned differences preserve even adjacent int64/uint64 values.
                    values = channel.astype(np.uint64)
                    values -= channel.min().astype(np.uint64)
                    values = values.astype(np.float64)
                else:
                    values = channel.astype(np.float64, copy=False)
                if np.abs(values).max() > np.finfo(np.float64).max / 2:
                    # Halving preserves normalization and keeps subtraction finite.
                    values = values / 2
                low, high = np.percentile(values, (1.0, 99.0))
                if high > low:
                    result[idx] = (values - low) / (high - low)
    except FloatingPointError as exc:
        raise ValueError(
            "image cannot be normalized to finite float32 values."
        ) from exc
    return result


def get_channel_axis(channel_axis: int | None, ndim: int) -> int:
    if channel_axis is None:
        return ndim - 1
    if not isinstance(channel_axis, int) or isinstance(channel_axis, bool):
        raise TypeError("channel_axis must be an integer or None.")
    axis = channel_axis + ndim if channel_axis < 0 else channel_axis
    if not 0 <= axis < ndim:
        raise ValueError(f"channel_axis must be between {-ndim} and {ndim - 1}.")
    return axis
