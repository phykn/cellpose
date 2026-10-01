import numpy as np
import pytest
import torch

from src.data.dataset import CellposeDataset
from src.data.image import find_pairs, read_array, write_mask


@pytest.mark.parametrize("dtype", [np.uint16, np.int64, np.float32])
def test_dataset_returns_image_and_flow_target(tmp_path, dtype) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    image = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    masks = np.zeros((32, 32), dtype=dtype)
    masks[8:24, 8:24] = 7
    np.save(image_dir / "sample.npy", image)
    np.save(mask_dir / "sample.npy", masks)

    dataset = CellposeDataset(
        image_dir,
        mask_dir,
        crop_size=32,
        augment=False,
    )
    normalized, target = dataset[0]

    assert normalized.shape == (3, 32, 32)
    assert target.shape == (3, 32, 32)
    assert normalized.dtype == torch.float32
    assert target.dtype == torch.float32
    assert torch.equal(target[0] > 0, torch.from_numpy(masks > 0))


@pytest.mark.parametrize("classified", [False, True])
def test_dataset_preserves_unsigned_instance_ids(tmp_path, classified):
    images, masks, labels = (tmp_path / name for name in ("images", "masks", "labels"))
    images.mkdir()
    masks.mkdir()
    labels.mkdir()
    mask_id = 2**63 + 7
    mask = np.zeros((16, 16), dtype=np.uint64)
    mask[4:12, 4:12] = mask_id
    np.save(images / "cell.npy", np.arange(256, dtype=np.float32).reshape(16, 16))
    np.save(masks / "cell.npy", mask)
    (labels / "cell.json").write_text(f'{{"{mask_id}": "A"}}')
    dataset = CellposeDataset(
        images,
        masks,
        crop_size=16,
        augment=False,
        label_dir=labels if classified else None,
        classes=("A",) if classified else (),
    )

    _, target = dataset[0]

    assert torch.equal(target[0] > 0, torch.from_numpy(mask > 0))
    if classified:
        assert (target[3, target[0] > 0] == 0).all()
        assert (target[3, target[0] == 0] == -1).all()


def test_find_pairs_reports_missing_stem(tmp_path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    np.save(image_dir / "image.npy", np.zeros((8, 8)))
    np.save(mask_dir / "mask.npy", np.zeros((8, 8)))

    with pytest.raises(ValueError, match="stems do not match"):
        find_pairs(image_dir, mask_dir)


def test_mask_io_round_trip(tmp_path) -> None:
    masks = np.arange(16, dtype=np.uint16).reshape(4, 4)
    path = tmp_path / "masks.tif"

    write_mask(path, masks)

    assert np.array_equal(read_array(path), masks)
