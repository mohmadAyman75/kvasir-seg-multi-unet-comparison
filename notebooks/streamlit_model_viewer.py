from pathlib import Path
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model

RUNS_DIR = PROJECT_ROOT / "notebooks" / "runs"
MODEL_NAMES = ["unet", "attention_unet", "unet_plus_plus", "transunet"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Kvasir-SEG Model Viewer", layout="wide")
st.title("Kvasir-SEG Model Viewer")
st.caption("Upload one image and compare where each trained model predicts the polyp region.")

@st.cache_resource
def load_checkpoint(model_name):
    checkpoint_path = RUNS_DIR / model_name / "best.pt"
    if not checkpoint_path.exists():
        return None, checkpoint_path, None

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = build_model(
        checkpoint["model_name"],
        base_channels=checkpoint.get("base_channels", 32),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint_path, checkpoint

def preprocess(image, image_size):
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return resized, tensor

def predict(model, tensor, threshold):
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = probs > threshold
    return probs, mask

def make_overlay(image, mask, alpha):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    overlay = base.copy()
    overlay[mask] = (1.0 - alpha) * base[mask] + alpha * red[mask]
    return overlay.astype(np.uint8)

def resize_mask(mask, size):
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.resize(size, Image.NEAREST)
    return np.asarray(mask_image) > 0

def make_voting_overlay(image, masks, alpha):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    votes = np.stack(masks, axis=0).sum(axis=0).astype(np.float32)
    vote_ratio = votes / max(1, len(masks))

    heat = np.zeros_like(base)
    heat[..., 0] = 255.0
    heat[..., 1] = 190.0 * (1.0 - vote_ratio)

    strength = (vote_ratio * alpha)[..., None]
    overlay = base * (1.0 - strength) + heat * strength
    return overlay.astype(np.uint8), votes

available_models = []
missing_models = []
for model_name in MODEL_NAMES:
    checkpoint_path = RUNS_DIR / model_name / "best.pt"
    if checkpoint_path.exists():
        available_models.append(model_name)
    else:
        missing_models.append(model_name)

with st.sidebar:
    st.header("Settings")
    selected_models = st.multiselect(
        "Models",
        MODEL_NAMES,
        default=available_models,
    )
    threshold = st.slider("Mask threshold", 0.05, 0.95, 0.50, 0.05)
    alpha = st.slider("Overlay strength", 0.10, 0.90, 0.55, 0.05)
    st.write("Device:", str(DEVICE))
    st.write("Runs folder:", str(RUNS_DIR))

if missing_models:
    st.warning("Missing checkpoints: " + ", ".join(missing_models))

uploaded_file = st.file_uploader("Upload an endoscopy image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is None:
    st.info("Upload an image to see predictions from the trained models.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.subheader("Input image")
st.image(image, caption="Uploaded image", use_container_width=True)

if not selected_models:
    st.warning("Select at least one model from the sidebar.")
    st.stop()

vote_masks = []
used_model_names = []
original_size = image.size

for model_name in selected_models:
    model, checkpoint_path, checkpoint = load_checkpoint(model_name)
    if model is None:
        st.error(f"{model_name}: checkpoint not found at {checkpoint_path}")
        continue

    image_size = checkpoint.get("image_size", 256)
    resized, tensor = preprocess(image, image_size)
    probs, mask = predict(model, tensor, threshold)
    overlay = make_overlay(resized, mask, alpha)
    vote_masks.append(resize_mask(mask, original_size))
    used_model_names.append(model_name)

    st.markdown(f"### {model_name}")
    st.caption(f"Checkpoint: {checkpoint_path} | image_size={image_size} | threshold={threshold:.2f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(resized, caption="Model input", use_container_width=True)
    with col2:
        st.image(probs, caption="Probability map", clamp=True, use_container_width=True)
    with col3:
        st.image(overlay, caption="Predicted mask overlay", use_container_width=True)

    st.write(
        {
            "mean_probability": float(probs.mean()),
            "max_probability": float(probs.max()),
            "predicted_area_ratio": float(mask.mean()),
        }
    )

if vote_masks:
    st.markdown("## Voting overlay")
    st.caption(
        "Each pixel is stronger when more selected models predict it as disease. "
        "Light color means weak agreement; strong red means high agreement."
    )
    voting_overlay, vote_count = make_voting_overlay(image, vote_masks, alpha)

    col1, col2 = st.columns(2)
    with col1:
        st.image(voting_overlay, caption="Voting overlay", use_container_width=True)
    with col2:
        st.image(vote_count, caption="Vote count map", clamp=True, use_container_width=True)

    agreement = {
        f"pixels_with_{votes}_votes": float((vote_count == votes).mean())
        for votes in range(1, len(vote_masks) + 1)
    }
    st.write({"models_used": used_model_names, **agreement})
