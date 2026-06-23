from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_image_mask_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at {images_dir} and {masks_dir}."
        )

    masks_by_stem = {
        path.stem: path
        for path in masks_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {dataset_dir}.")

    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_ratio: float = 0.2,
    seed: int = 42,
    max_samples: int | None = None,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)

    if max_samples is not None:
        pairs = pairs[:max_samples]

    val_count = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    return train_pairs, val_pairs


class SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        image_size: int = 128,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)
            if random.random() < 0.2:
                image = ImageOps.flip(image)
                mask = ImageOps.flip(mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor
