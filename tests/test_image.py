import numpy as np
import pytest

from src.data.image import read_array, read_mask, write_mask


def test_png_rejects_truncating_instance_labels(tmp_path):
    mask = np.array([[0, 65535, 65536]], dtype=np.uint32)
    with pytest.raises(ValueError, match="65535"):
        write_mask(tmp_path / "mask.png", mask)
    assert not (tmp_path / "mask.png").exists()
    for suffix in (".npy", ".tiff"):
        path = tmp_path / f"mask{suffix}"
        write_mask(path, mask)
        np.testing.assert_array_equal(read_array(path), mask)


def test_png_preserves_supported_labels(tmp_path):
    mask = np.array([[0, 65535]], dtype=np.uint32)
    path = tmp_path / "mask.png"
    write_mask(path, mask)
    np.testing.assert_array_equal(read_array(path), mask)


@pytest.mark.parametrize("suffix", [".npy", ".NPY", ".Npy"])
def test_numpy_mask_output_keeps_requested_filename(tmp_path, suffix):
    mask = np.array([[0, 1, 65536]], dtype=np.uint32)
    path = tmp_path / f"mask{suffix}"

    write_mask(path, mask)

    assert path.is_file()
    assert sorted(tmp_path.iterdir()) == [path]
    np.testing.assert_array_equal(read_mask(path), mask)


@pytest.mark.parametrize("suffix", [".npy", ".tiff"])
def test_mask_io_preserves_large_unsigned_ids(tmp_path, suffix):
    mask = np.array([[0, 2**63 + 7, 2**64 - 1]], dtype=np.uint64)
    path = tmp_path / f"masks{suffix}"
    write_mask(path, mask)

    restored = read_mask(path)

    assert restored.dtype == mask.dtype
    np.testing.assert_array_equal(restored, mask)


@pytest.mark.parametrize(
    "mask, error, message",
    [
        (np.zeros((2, 2, 1)), ValueError, "two-dimensional"),
        (np.zeros((0, 2)), ValueError, "non-empty"),
        (np.array([[0, -1]]), ValueError, "non-negative"),
        (np.array([[0, 1.5]]), TypeError, "integer labels"),
        (np.array([[0, np.nan]]), ValueError, "finite labels"),
        (np.array([[0, np.inf]]), ValueError, "finite labels"),
        (np.array([[0, float(2**63)]]), ValueError, "int64 range"),
        (np.array([[0, 1 + 1j]]), TypeError, "integer dtype"),
    ],
)
def test_read_mask_rejects_invalid_labels_before_conversion(
    tmp_path, mask, error, message
):
    path = tmp_path / "masks.npy"
    np.save(path, mask)
    with pytest.raises(error, match=message):
        read_mask(path)


@pytest.mark.parametrize("suffix", [".npy", ".png", ".tiff"])
@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_mask_output_rejects_empty_dimensions_before_creating_files(
    tmp_path, suffix, shape
):
    path = tmp_path / "new" / f"mask{suffix}"
    with pytest.raises(ValueError, match="non-empty"):
        write_mask(path, np.zeros(shape, dtype=np.uint16))
    assert not path.parent.exists()
