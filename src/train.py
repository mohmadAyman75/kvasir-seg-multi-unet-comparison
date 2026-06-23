from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SegmentationDataset, find_image_mask_pairs, split_pairs
from src.download_data import prepare_kvasir
from src.metrics import dice_loss, dice_score, iou_score
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_loss: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = bce_loss(logits, masks) + dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    bce_loss: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for images, masks in tqdm(loader, desc="valid", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = bce_loss(logits, masks) + dice_loss(logits, masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_dice += dice_score(logits, masks) * batch_size
        running_iou += iou_score(logits, masks) * batch_size

    total = len(loader.dataset)
    return running_loss / total, running_dice / total, running_iou / total


def save_loss_curve(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    plt.plot(epochs, val_loss, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_history(history: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_dice",
                "val_iou",
                "best_val_dice",
                "epochs_without_improvement",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "all", "full"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple U-Net model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/Kvasir-SEG"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--model",
        choices=["unet", "resunet", "attention_unet", "unet_plus_plus", "transunet"],
        default="unet",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=parse_optional_int, default=160)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.download or not args.data_dir.exists():
        args.data_dir = prepare_kvasir(Path("data"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pairs = find_image_mask_pairs(args.data_dir)
    train_pairs, val_pairs = split_pairs(
        pairs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Train samples: {len(train_pairs)} | Validation samples: {len(val_pairs)}")

    train_dataset = SegmentationDataset(train_pairs, args.image_size, augment=True)
    val_dataset = SegmentationDataset(val_pairs, args.image_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()

    run_dir = args.out_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, bce_loss, device)

        improved = val_dice > best_dice + args.early_stop_min_delta
        if improved:
            best_dice = val_dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "best_val_dice": best_dice,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"dice={val_dice:.4f} | iou={val_iou:.4f} | "
            f"best_dice={best_dice:.4f}"
        )

        checkpoint = {
            "model_name": args.model,
            "base_channels": args.base_channels,
            "image_size": args.image_size,
            "max_samples": args.max_samples,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "state_dict": model.state_dict(),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if improved:
            torch.save(checkpoint, run_dir / "best.pt")
            print(f"Saved new best checkpoint: {run_dir / 'best.pt'}")

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                "Early stopping: "
                f"val_dice did not improve for {args.early_stop_patience} epochs."
            )
            break

    write_history(history, run_dir / "training_log.csv")
    save_loss_curve(history, run_dir / "loss_curve.png")
    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
