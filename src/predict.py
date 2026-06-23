from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.models import build_model


def load_image(image_path: Path, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor, original


def save_prediction_figure(
    image: Image.Image,
    probability_mask: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_resized = Image.fromarray((probability_mask * 255).astype(np.uint8)).resize(
        image.size,
        Image.BILINEAR,
    )
    mask_array = np.asarray(mask_resized, dtype=np.float32) / 255.0

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_array, cmap="gray")
    plt.title("Predicted mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(mask_array, cmap="Reds", alpha=0.45)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a mask for one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/unet/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/prediction.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = checkpoint.get("image_size", 128)

    model = build_model(
        checkpoint["model_name"],
        base_channels=checkpoint.get("base_channels", 32),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image_tensor, original_image = load_image(args.image, image_size)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probability_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()

    save_prediction_figure(original_image, probability_mask, args.output)
    print(f"Saved prediction to: {args.output}")


if __name__ == "__main__":
    main()
