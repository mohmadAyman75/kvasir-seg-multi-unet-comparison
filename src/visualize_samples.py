from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import SegmentationDataset, find_image_mask_pairs, split_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize image/mask samples.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/Kvasir-SEG"))
    parser.add_argument("--output", type=Path, default=Path("runs/samples.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pairs = find_image_mask_pairs(args.data_dir)
    train_pairs, _ = split_pairs(pairs, seed=args.seed, max_samples=args.count)
    dataset = SegmentationDataset(train_pairs, image_size=args.image_size, augment=False)

    count = min(args.count, len(dataset))
    plt.figure(figsize=(count * 3, 6))

    for index in range(count):
        image, mask = dataset[index]
        image_array = image.permute(1, 2, 0).numpy()
        mask_array = np.squeeze(mask.numpy())

        plt.subplot(2, count, index + 1)
        plt.imshow(image_array)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(2, count, count + index + 1)
        plt.imshow(mask_array, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved samples to: {args.output}")


if __name__ == "__main__":
    main()
