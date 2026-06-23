from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SegmentationDataset, find_image_mask_pairs, split_pairs
from src.metrics import dice_loss, dice_score, iou_score
from src.models import build_model


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in tqdm(loader, desc="evaluate"):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)

        batch_size = images.size(0)
        loss = bce_loss(logits, masks) + dice_loss(logits, masks)
        total_loss += loss.item() * batch_size
        total_dice += dice_score(logits, masks) * batch_size
        total_iou += iou_score(logits, masks) * batch_size

    total = len(loader.dataset)
    print(f"loss={total_loss / total:.4f}")
    print(f"dice={total_dice / total:.4f}")
    print(f"iou={total_iou / total:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/Kvasir-SEG"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/unet/best.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        checkpoint["model_name"],
        base_channels=checkpoint.get("base_channels", 32),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    pairs = find_image_mask_pairs(args.data_dir)
    _, val_pairs = split_pairs(
        pairs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    dataset = SegmentationDataset(
        val_pairs,
        image_size=checkpoint.get("image_size", 128),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    evaluate_model(model, loader, device)


if __name__ == "__main__":
    main()
