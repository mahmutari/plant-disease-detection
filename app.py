#!/usr/bin/env python
"""
Plant Disease Detection — Streamlit Web Interface
Phase 3.5: 3-Model Comparison Selector
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

MODELS_CONFIG = {
    "Original (Phase 2)": {
        "path": "checkpoints/best_mobilenet.pth",
        "description": "Trained on PlantVillage only (97.13% PV val accuracy)",
        "strengths": "Best for controlled laboratory imagery",
        "weaknesses": "Poor field generalization (6.67% web val)",
        "stats": {"PV": 97.13, "PD": 16.02, "Web": 6.67, "OOD_risk": 50.0},
    },
    "Fine-tuned (Phase 3)": {
        "path": "checkpoints/best_mobilenet_finetuned.pth",
        "description": "Fine-tuned on PlantDoc training set (frozen backbone)",
        "strengths": "Best for curated web imagery (26.67% web val)",
        "weaknesses": "Catastrophic forgetting on PV (58.96%)",
        "stats": {"PV": 58.96, "PD": 30.74, "Web": 26.67, "OOD_risk": 10.0},
    },
    "Hybrid V2 (Phase 3.5)": {
        "path": "checkpoints/best_mobilenet_hybrid_v2.pth",
        "description": "Trained on PlantVillage + PlantDoc with proper label mapping",
        "strengths": "Best PD accuracy (41.13%), safest OOD behavior (0% false confidence)",
        "weaknesses": "Web accuracy still limited (6.67%)",
        "stats": {"PV": 96.07, "PD": 41.13, "Web": 6.67, "OOD_risk": 0.0},
    },
}

CLASS_NAMES_DIR = "data/val"
NUM_CLASSES = 38
IMG_SIZE = 224

# ═══════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Plant Disease Detection System")
st.markdown(
    "**Senior Project — Phase 3.5**  \n"
    "MobileNetV2-based Plant Disease Classification with Multi-Model Comparison  \n"
    "*Sakarya University, Software Engineering*"
)

# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def load_model(checkpoint_path):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

@st.cache_data
def get_class_names():
    if not os.path.exists(CLASS_NAMES_DIR):
        return [f"Class_{i}" for i in range(NUM_CLASSES)]
    classes = sorted(os.listdir(CLASS_NAMES_DIR))
    return [c for c in classes if os.path.isdir(os.path.join(CLASS_NAMES_DIR, c))]

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        top3_probs, top3_indices = torch.topk(probabilities, 3)

    predictions = []
    for prob, idx in zip(top3_probs, top3_indices):
        predictions.append({
            'class': class_names[idx.item()],
            'confidence': prob.item(),
        })
    return predictions

def generate_gradcam(model, image_tensor, target_class_idx):
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    target_layer = model.features[-1]
    handle = target_layer.register_forward_hook(forward_hook)

    try:
        outputs = model(image_tensor)
        model.zero_grad()
        outputs[0, target_class_idx].backward()

        grads = gradients[0]
        acts = activations[0]

        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()
        cam = F.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.detach().numpy()
    finally:
        handle.remove()

    return cam

def overlay_heatmap(image_pil, heatmap, alpha=0.4):
    image_pil = image_pil.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image_pil)

    from scipy.ndimage import zoom
    h, w = image_array.shape[:2]
    zoom_factor = (h / heatmap.shape[0], w / heatmap.shape[1])
    heatmap_resized = zoom(heatmap, zoom_factor, order=1)

    colored_heatmap = cm.jet(heatmap_resized)[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

    overlay = (alpha * colored_heatmap + (1 - alpha) * image_array).astype(np.uint8)

    return overlay, heatmap_resized

def get_confidence_badge(confidence):
    if confidence >= 0.95:
        return ("🟢", "HIGH", "green")
    elif confidence >= 0.70:
        return ("🟡", "MODERATE", "orange")
    else:
        return ("🔴", "LOW", "red")

# ═══════════════════════════════════════════════════════════
# SIDEBAR — MODEL SELECTOR
# ═══════════════════════════════════════════════════════════

st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("### Model Selection")
selected_model_name = st.sidebar.radio(
    "Choose a model:",
    list(MODELS_CONFIG.keys()),
    index=2,
)

model_info = MODELS_CONFIG[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{selected_model_name}**")
st.sidebar.markdown(f"*{model_info['description']}*")
st.sidebar.markdown(f"✅ **Strengths:** {model_info['strengths']}")
st.sidebar.markdown(f"⚠️ **Weaknesses:** {model_info['weaknesses']}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Performance Metrics")
stats = model_info['stats']
st.sidebar.metric("PlantVillage Val", f"{stats['PV']}%")
st.sidebar.metric("PlantDoc Test", f"{stats['PD']}%")
st.sidebar.metric("Web Validation", f"{stats['Web']}%")
st.sidebar.metric("OOD False Confidence", f"{stats['OOD_risk']}%", delta_color="inverse")

st.sidebar.markdown("---")

comparison_mode = st.sidebar.checkbox(
    "🔬 Compare All Models",
    value=False,
    help="Run prediction on all 3 models simultaneously"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### 📊 Multi-Distribution Performance  \n"
    "No single model dominates across all distributions.  \n"
    "**Hybrid V2** offers the best balance.  \n"
    "**Fine-tuned** excels on curated web imagery.  \n"
    "**Original** retains highest PV accuracy."
)

# ═══════════════════════════════════════════════════════════
# MAIN AREA — IMAGE UPLOAD
# ═══════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "📤 Upload a plant leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a leaf image for disease classification"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    class_names = get_class_names()
    image_tensor = preprocess_image(image)

    # ═════════════════════════════════════════════════════
    # COMPARISON MODE — RUN ALL 3 MODELS
    # ═════════════════════════════════════════════════════
    if comparison_mode:
        st.markdown("## 🔬 Comparison Mode — All 3 Models")

        col_img, _ = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        st.markdown("### Predictions Across Models")

        cols = st.columns(3)
        for col, (model_name, info) in zip(cols, MODELS_CONFIG.items()):
            with col:
                st.markdown(f"#### {model_name}")

                if not os.path.exists(info['path']):
                    st.error(f"Model file not found: {info['path']}")
                    continue

                try:
                    model = load_model(info['path'])
                    predictions = predict(model, image_tensor, class_names)

                    top1 = predictions[0]
                    emoji, level, color = get_confidence_badge(top1['confidence'])

                    st.markdown(
                        f"**Prediction:** `{top1['class']}`  \n"
                        f"**Confidence:** {top1['confidence']*100:.1f}% {emoji}  \n"
                        f"**Level:** :{color}[{level}]"
                    )

                    st.markdown("**Top-3:**")
                    for i, pred in enumerate(predictions, 1):
                        st.markdown(
                            f"{i}. `{pred['class']}` — "
                            f"{pred['confidence']*100:.1f}%"
                        )

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.markdown("---")
        st.info(
            "💡 **Interpretation Tip:** Compare predictions across models. "
            "If all 3 agree → high confidence prediction. "
            "If they disagree → image likely out-of-distribution or ambiguous."
        )

    # ═════════════════════════════════════════════════════
    # SINGLE MODEL MODE
    # ═════════════════════════════════════════════════════
    else:
        if not os.path.exists(model_info['path']):
            st.error(
                f"Model checkpoint not found at: {model_info['path']}  \n"
                f"Please ensure the model has been trained and saved."
            )
        else:
            try:
                model = load_model(model_info['path'])

                col_image, col_predict = st.columns([1, 1])

                with col_image:
                    st.markdown("### 📷 Input Image")
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                with col_predict:
                    st.markdown("### 🎯 Prediction Result")

                    predictions = predict(model, image_tensor, class_names)
                    top1 = predictions[0]
                    emoji, level, color = get_confidence_badge(top1['confidence'])

                    st.markdown(f"### {emoji} {top1['class']}")
                    st.metric(
                        "Confidence",
                        f"{top1['confidence']*100:.2f}%",
                        delta=level
                    )

                    if top1['confidence'] >= 0.95:
                        st.success("✅ High confidence — reliable prediction")
                    elif top1['confidence'] >= 0.70:
                        st.warning("⚠️ Moderate confidence — verify result")
                    else:
                        st.error(
                            "🔴 Low confidence — image may be out-of-distribution "
                            "or ambiguous. Consider expert review."
                        )

                st.markdown("---")
                st.markdown("### 📊 Top-3 Predictions")

                for i, pred in enumerate(predictions, 1):
                    conf_pct = pred['confidence'] * 100
                    st.markdown(f"**{i}. {pred['class']}**")
                    st.progress(pred['confidence'])
                    st.caption(f"Confidence: {conf_pct:.2f}%")

                st.markdown("---")
                st.markdown("### 🔥 Grad-CAM Visualization")
                st.caption(
                    "Visual explanation showing which image regions the model "
                    "focuses on for its prediction."
                )

                with st.spinner("Generating Grad-CAM heatmap..."):
                    top1_idx = class_names.index(top1['class'])
                    cam = generate_gradcam(model, image_tensor, top1_idx)
                    overlay, heatmap = overlay_heatmap(image, cam)

                    col_orig, col_heat, col_over = st.columns(3)

                    with col_orig:
                        st.markdown("**Original**")
                        st.image(image.resize((IMG_SIZE, IMG_SIZE)), use_container_width=True)

                    with col_heat:
                        st.markdown("**Heatmap**")
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(heatmap, cmap='jet')
                        ax.axis('off')
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    with col_over:
                        st.markdown("**Overlay (α=0.4)**")
                        st.image(overlay, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)

else:
    # ═════════════════════════════════════════════════════
    # WELCOME SCREEN
    # ═════════════════════════════════════════════════════
    st.info(
        "👆 **Upload a plant leaf image to get started.**  \n"
        "Supported formats: JPG, JPEG, PNG"
    )

    st.markdown("---")
    st.markdown("### 📚 About This System")
    st.markdown(
        "This system uses MobileNetV2 deep learning models to classify "
        "plant diseases across 38 categories from the PlantVillage dataset. "
        "Three model variants are available, each with different strengths:"
    )

    cols = st.columns(3)
    for col, (model_name, info) in zip(cols, MODELS_CONFIG.items()):
        with col:
            st.markdown(f"#### {model_name}")
            st.markdown(f"*{info['description']}*")
            stats = info['stats']
            st.markdown(
                f"- PV val: **{stats['PV']}%**  \n"
                f"- PD test: **{stats['PD']}%**  \n"
                f"- Web val: **{stats['Web']}%**  \n"
                f"- OOD risk: **{stats['OOD_risk']}%**"
            )

    st.markdown("---")
    st.markdown("### 🎓 Project Information")
    st.markdown(
        "- **Authors:** Mahmut Arı, Samet Kavlan  \n"
        "- **Institution:** Sakarya University, Software Engineering  \n"
        "- **Phase 3.5:** Hybrid Training with Multi-Distribution Evaluation  \n"
    )

st.markdown("---")
st.caption(
    "Plant Disease Detection — Sakarya University Senior Project  \n"
    "MobileNetV2 | PyTorch | Streamlit | Phase 3.5"
)
