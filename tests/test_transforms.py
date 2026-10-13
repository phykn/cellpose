import numpy as np
import pytest

from src.prepare.convert import normalize_image


def test_grayscale_image_is_normalized_without_clipping() -> None:
    image = np.arange(100, dtype=np.float32).reshape(10, 10)

    normalized = normalize_image(image)

    low, high = np.percentile(image, (1.0, 99.0))
    expected = (image - low) / (high - low)
    assert normalized.shape == (3, 10, 10)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized[0], expected)
    assert normalized[0, 0, 0] < 0.0
    assert normalized[0, -1, -1] > 1.0
    assert not normalized[1:].any()


def test_rgb_channels_are_normalized_independently() -> None:
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    image = np.stack((values, 10.0 + 2.0 * values, 100.0 - values), axis=-1)

    normalized = normalize_image(image)

    assert normalized.shape == (3, 8, 8)
    np.testing.assert_allclose(normalized[0], normalized[1], atol=1.0e-6)
    np.testing.assert_allclose(normalized[0], normalized[2][::-1, ::-1], atol=1.0e-6)


def test_constant_and_missing_channels_are_zero() -> None:
    image = np.stack(
        (
            np.full((5, 7), 12.0, dtype=np.float32),
            np.arange(35, dtype=np.float32).reshape(5, 7),
        )
    )

    normalized = normalize_image(image, channel_axis=0)

    assert not normalized[0].any()
    assert normalized[1].min() < 0.0
    assert normalized[1].max() > 1.0
    assert not normalized[2].any()


@pytest.mark.parametrize(
    ("image", "channel_axis", "error"),
    (
        (np.zeros((4, 5, 4), dtype=np.uint8), -1, "one and three"),
        (np.zeros((4, 5, 6), dtype=np.uint8), None, "one and three"),
        (np.zeros((2, 3, 4, 5), dtype=np.uint8), None, "spatial dimensions"),
        (np.zeros((4, 5), dtype=np.uint8), 0, "only valid"),
    ),
)
def test_invalid_image_channels_are_rejected(
    image: np.ndarray,
    channel_axis: int | None,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        normalize_image(image, channel_axis=channel_axis)


@pytest.mark.parametrize("scale, offset", [(1e40, 0), (1e-50, 0), (1, 1e10)])
def test_normalization_preserves_float64_intensity_differences(scale, offset):
    image = np.arange(100, dtype=np.float64).reshape(10, 10)
    expected = normalize_image(image)

    actual = normalize_image(image * scale + offset)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "dtype, offset", [(np.uint64, 2**64 - 100), (np.int64, -(2**63))]
)
def test_normalization_preserves_large_integer_intensity_differences(dtype, offset):
    image = np.arange(100, dtype=dtype).reshape(10, 10)
    expected = normalize_image(image)

    actual = normalize_image(image + dtype(offset))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
def test_normalization_handles_full_dtype_range(dtype):
    limits = np.finfo(dtype) if np.issubdtype(dtype, np.floating) else np.iinfo(dtype)
    image = np.array([[limits.min, limits.max]], dtype=dtype)
    expected = normalize_image(np.array([[-1.0, 1.0]]))

    np.testing.assert_allclose(normalize_image(image), expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_normalization_preserves_ordinary_image_values(dtype):
    image = np.random.default_rng(31).uniform(0, 200, (32, 40)).astype(dtype)
    values = image.astype(np.float32)
    low, high = np.percentile(values, (1.0, 99.0))
    expected = ((values - low) / (high - low)).astype(np.float32)

    np.testing.assert_allclose(
        normalize_image(image)[0], expected, rtol=1e-6, atol=1e-7
    )


def test_normalization_rejects_complex_images():
    with pytest.raises(TypeError, match="real"):
        normalize_image(np.array([[0, 1 + 2j]]))


def test_normalization_rejects_unrepresentable_float32_output():
    image = np.arange(10000, dtype=np.float64).reshape(100, 100)
    image[0, 0] = 1e100
    with pytest.raises(ValueError, match="finite float32"):
        normalize_image(image)
