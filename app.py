"""
Plant Disease Detection — Web Interface
========================================

Streamlit-based web application for plant disease classification
using a fine-tuned MobileNetV2 model with Grad-CAM interpretability.

Authors: Mahmut Ari, Samet Kavlan
Project: Senior Design Project II - Sakarya University
"""

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

# Ensure project root is importable from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mobilenet_model import get_mobilenet_v2
from preprocess.transform import val_transforms


# =====================================================================
# Page Configuration
# =====================================================================

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# Cached Resources
# =====================================================================

@st.cache_resource(show_spinner="Loading MobileNetV2 model...")
def load_model(checkpoint_path: str) -> torch.nn.Module:
    """
    Load trained MobileNetV2 with checkpoint weights.
    Cached by Streamlit so the model is loaded only once per session.
    """
    model = get_mobilenet_v2(num_classes=38)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def load_class_names() -> List[str]:
    """
    Return the 38 class names in alphabetical order, matching ImageFolder indexing.
    Cached so the filesystem is only read once per session.
    """
    return sorted(os.listdir("data/val"))


# =====================================================================
# Inference Helpers
# =====================================================================

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Apply val_transforms to a PIL image and add batch dimension.

    Returns:
        (1, 3, 224, 224) float tensor, normalised with ImageNet stats
    """
    return val_transforms(pil_image.convert("RGB")).unsqueeze(0)


def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Run inference and return the top-k (class_name, probability) pairs.

    Args:
        model        : trained model in eval() mode
        image_tensor : (1, 3, 224, 224) preprocessed tensor
        class_names  : list of 38 class names (alphabetical)
        top_k        : number of top predictions to return

    Returns:
        List of (class_name, probability) sorted by probability descending
    """
    with torch.no_grad():
        logits = model(image_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=top_k)
    return [
        (class_names[idx.item()], prob.item())
        for prob, idx in zip(top_probs, top_indices)
    ]


def format_class_name(raw: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato — Early Blight'  (display-friendly)"""
    parts = raw.replace("_", " ").split("   ")   # triple space after replacing ___
    if len(parts) == 2:
        return f"{parts[0].strip().title()} — {parts[1].strip().title()}"
    return raw.replace("_", " ").title()


# =====================================================================
# Grad-CAM Helpers
# =====================================================================

def compute_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    predicted_idx: int,
) -> np.ndarray:
    """
    Return a (224, 224) float32 Grad-CAM activation map for the predicted class.

    Uses a separate forward+backward pass — model.eval() stays active but
    gradients must be enabled (no torch.no_grad() here).
    """
    from analysis.gradcam import GradCAM
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)
    heatmap, _, _ = cam(image_tensor, target_class=predicted_idx)
    cam.remove_hooks()
    return heatmap


def create_overlay_visualization(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Colorize a Grad-CAM heatmap and blend it onto the source image.

    Returns:
        original_224   : (224, 224, 3) uint8 RGB
        colored_heatmap: (224, 224, 3) uint8 RGB  — JET colormap
        overlay        : (224, 224, 3) uint8 RGB  — alpha-blended composite
    """
    original      = np.array(pil_image.resize((224, 224)))
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored_bgr   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb   = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay       = (alpha * colored_rgb + (1 - alpha) * original).astype(np.uint8)
    return original, colored_rgb, overlay


# =====================================================================
# Sidebar
# =====================================================================

CHECKPOINTS = {
    "Original (PlantVillage, 97.13% val)":    "checkpoints/best_mobilenet.pth",
    "Fine-tuned (PlantDoc, 30.74% PlantDoc)": "checkpoints/best_mobilenet_finetuned.pth",
}

with st.sidebar:
    st.title("🌿 Plant Disease Detector")
    st.markdown("---")
    st.markdown("### Model")
    selected_label = st.selectbox(
        "Checkpoint",
        options=list(CHECKPOINTS.keys()),
        index=0,
    )
    selected_ckpt = CHECKPOINTS[selected_label]
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Deep learning-based plant disease classification system "
        "using MobileNetV2 trained on the New Plant Diseases Dataset "
        "(38 classes, ~87K images)."
    )
    st.markdown("---")
    st.markdown("### Model Info")
    if "Fine-tuned" in selected_label:
        st.markdown(
            "**Architecture:** MobileNetV2  \n"
            "**PlantVillage val:** 58.96%  \n"
            "**PlantDoc test:** 30.74%  \n"
            "**Web (in-dist):** 26.67%  \n"
            "**Classes:** 38  \n"
            "**Input size:** 224 × 224 px"
        )
    else:
        st.markdown(
            "**Architecture:** MobileNetV2  \n"
            "**PlantVillage val:** 97.13%  \n"
            "**PlantDoc test:** 16.02%  \n"
            "**Web (in-dist):** 6.67%  \n"
            "**Classes:** 38  \n"
            "**Input size:** 224 × 224 px"
        )
    st.markdown("---")
    st.markdown("### Project Info")
    st.markdown(
        "**Institution:** Sakarya University  \n"
        "**Department:** Software Engineering  \n"
        "**Authors:** Mahmut Arı, Samet Kavlan  \n"
        "**Year:** 2025-2026"
    )


# =====================================================================
# Load model and class names (once per session)
# =====================================================================

try:
    model       = load_model(selected_ckpt)
    class_names = load_class_names()
except Exception as e:
    st.error(
        f"**Model loading failed.**  \n"
        f"Make sure `{selected_ckpt}` and `data/val/` exist.  \n"
        f"Error: `{e}`"
    )
    st.stop()


# =====================================================================
# Main Page
# =====================================================================

st.title("Plant Disease Detection")
st.markdown(
    "Upload a leaf image to receive an automated disease diagnosis "
    "with visual explanation via Grad-CAM."
)

# ── File uploader ──────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:

    # ── Load image ─────────────────────────────────────────────────
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("❌ Could not read the uploaded file. Please upload a valid image.")
        st.stop()

    # ── Run inference (no_grad for efficiency) ─────────────────────
    try:
        with st.spinner("Analyzing image..."):
            tensor = preprocess_image(pil_image)
            top3   = predict(model, tensor, class_names, top_k=3)
    except Exception as e:
        st.error(f"❌ Prediction failed: `{e}`")
        st.stop()

    top1_raw,     top1_conf  = top3[0]
    top1_display             = format_class_name(top1_raw)
    predicted_idx            = class_names.index(top1_raw)

    # ── Two-column layout: image | results ─────────────────────────
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### Prediction Results")

        # Top-1 metric card
        st.metric(
            label="Predicted Class",
            value=top1_display,
            delta=f"{top1_conf:.2%} confidence",
        )

        st.markdown("#### Top-3 Predictions")
        for raw_name, prob in top3:
            display_name = format_class_name(raw_name)
            st.progress(prob, text=f"{display_name}: {prob:.2%}")

        # Confidence interpretation
        if top1_conf > 0.95:
            st.success("✅ High confidence prediction")
        elif top1_conf > 0.70:
            st.warning("⚠️ Moderate confidence — verify with an expert")
        else:
            st.error("❌ Low confidence — interpret with caution")

    # ── Grad-CAM section (full width, below both columns) ──────────
    st.markdown("---")
    st.subheader("🔬 Model Attention Visualization (Grad-CAM)")
    st.caption(
        "Class-discriminative localization showing which leaf regions "
        "influenced the model's prediction."
    )

    try:
        with st.spinner("Generating Grad-CAM heatmap..."):
            heatmap  = compute_gradcam(model, tensor, predicted_idx)
            original, colored_heatmap, overlay = create_overlay_visualization(
                pil_image, heatmap
            )

        gc_col1, gc_col2, gc_col3 = st.columns(3)
        with gc_col1:
            st.image(original,        caption="Original (224×224)",  use_container_width=True)
        with gc_col2:
            st.image(colored_heatmap, caption="Grad-CAM Heatmap",    use_container_width=True)
        with gc_col3:
            st.image(overlay,         caption="Overlay (α=0.4)",     use_container_width=True)

        st.info(
            "💡 **How to read this:** Red regions indicate areas the model "
            "considered most important for its prediction. Ideally, these "
            "should align with visible disease symptoms (lesions, discoloration) "
            "rather than background or healthy tissue."
        )

    except Exception as e:
        st.warning(f"⚠️ Grad-CAM unavailable for this prediction: `{e}`")

else:
    st.info(
        "👆 Upload a leaf image above to get started.  \n"
        "The model supports 38 plant disease classes from the PlantVillage dataset."
    )


# =====================================================================
# Footer
# =====================================================================

st.markdown("---")
st.caption(
    "Plant Disease Detection — Senior Design Project II  \n"
    "Powered by PyTorch + Streamlit"
)
