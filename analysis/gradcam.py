"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.

Usage (as module):
    from analysis.gradcam import GradCAM, get_target_layer
    from analysis.gradcam import load_image_as_tensor, overlay_heatmap_on_image

Sanity check (standalone):
    python analysis/gradcam.py
"""

import os
import sys

# Project root must be on sys.path so preprocess/ and models/ are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, Tuple

from preprocess.transform import val_transforms
from models.mobilenet_model import get_mobilenet_v2
from models.resnet_model import get_resnet50


# ---------------------------------------------------------------------------
# Target layer selection
# ---------------------------------------------------------------------------

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the final convolutional block to attach Grad-CAM hooks to.

    MobileNetV2 : model.features[-1]  — last inverted residual block
                  produces a 7x7 spatial feature map at 320 channels
    ResNet-50   : model.layer4[-1]    — last residual block
                  produces a 7x7 spatial feature map at 2048 channels
    """
    if model_name == "mobilenet":
        return model.features[-1]
    if model_name == "resnet":
        return model.layer4[-1]
    raise ValueError(f"Unknown model_name '{model_name}'. Expected 'mobilenet' or 'resnet'.")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Attaches forward and backward hooks to a target convolutional layer.
    On each call, runs a forward pass (gradients ON), then back-propagates
    only through the target-class logit to obtain gradient-weighted
    activation maps:

        alpha_k  = (1/Z) * sum_{i,j}( d y^c / d A^k_{ij} )
        L^c      = ReLU( sum_k( alpha_k * A^k ) )

    The result is resized to 224x224 and normalised to [0, 1].

    Args:
        model        : trained model in eval() mode
        target_layer : nn.Module to hook (last conv block)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # register_full_backward_hook is the stable modern API (torch >= 1.8)
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Private hooks — called automatically by PyTorch
    # ------------------------------------------------------------------

    def _save_activation(
        self,
        module: nn.Module,
        inp: Tuple,
        output: torch.Tensor,
    ) -> None:
        self._activations = output.detach()   # detach: keep values, no graph growth

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple,
        grad_output: Tuple,
    ) -> None:
        # grad_output[0]: gradient tensor w.r.t. this layer's output
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute the Grad-CAM heatmap for a single image.

        NOTE: Do NOT wrap this call in torch.no_grad() — backward pass
        requires gradient computation through the model.

        Args:
            image_tensor : (1, 3, H, W) normalised float tensor
            target_class : class index to explain; defaults to predicted class

        Returns:
            heatmap    : (224, 224) float32 ndarray in [0, 1]
            pred_idx   : predicted class index (argmax of logits)
            confidence : softmax probability of the predicted class
        """
        self.model.zero_grad()

        logits     = self.model(image_tensor)          # (1, num_classes)
        probs      = torch.softmax(logits, dim=1)
        pred_idx   = int(torch.argmax(logits, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

        if target_class is None:
            target_class = pred_idx

        # Scalar backward from single class score
        logits[0, target_class].backward()

        # alpha_k: global average pooling of gradients over spatial dims
        alpha = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted sum of activation maps + ReLU (keep only positive influence)
        cam = torch.relu(
            (alpha * self._activations).sum(dim=1)                # (1, h, w)
        ).squeeze(0)                                               # (h, w)

        # Resize to input resolution and normalise to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_np = cv2.resize(cam_np, (224, 224))
        if cam_np.max() > 0:
            cam_np /= cam_np.max()

        return cam_np.astype(np.float32), pred_idx, confidence

    def remove_hooks(self) -> None:
        """Detach hooks from the model to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def overlay_heatmap_on_image(
    image_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original image using the JET colormap.

    Args:
        image_pil : original PIL image (resized to 224x224 internally)
        heatmap   : (224, 224) float32 array in [0, 1]
        alpha     : heatmap opacity — 0 = image only, 1 = heatmap only

    Returns:
        (224, 224, 3) uint8 RGB array
    """
    img_np = np.array(image_pil.resize((224, 224)), dtype=np.uint8)  # RGB

    # cv2.applyColorMap expects uint8; outputs BGR — convert to RGB
    heatmap_u8  = (heatmap * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = (alpha       * heatmap_rgb.astype(np.float32)
               + (1 - alpha) * img_np.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_image_as_tensor(
    image_path: str,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Open an image file, apply val_transforms, and return tensor + PIL image.

    Returns:
        tensor    : (1, 3, 224, 224) float tensor, ready for model input
        image_pil : original PIL image (RGB) for overlay / display
    """
    image_pil = Image.open(image_path).convert("RGB")
    tensor    = val_transforms(image_pil).unsqueeze(0)  # add batch dim
    return tensor, image_pil


def load_model_for_gradcam(model_name: str, checkpoint_path: str) -> nn.Module:
    """
    Load a trained model from a .pth checkpoint file and set to eval mode.

    eval() disables dropout and BatchNorm running-stats updates.
    Gradients are still computed because we never call torch.no_grad() in GradCAM.

    Args:
        model_name      : 'mobilenet' or 'resnet'
        checkpoint_path : path to .pth weights file

    Returns:
        model in eval() mode on CPU
    """
    factories = {"mobilenet": get_mobilenet_v2, "resnet": get_resnet50}
    if model_name not in factories:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'mobilenet' or 'resnet'.")

    model = factories[model_name](num_classes=38).to(torch.device("cpu"))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use a clean, easy class so we expect high confidence and a clear heatmap
    TEST_CLASS   = "Apple___healthy"
    VAL_DIR      = "data/val"
    CHECKPOINT   = "checkpoints/best_mobilenet.pth"
    MODEL_NAME   = "mobilenet"
    OUT_DIR      = os.path.join("results", "gradcam_test")

    os.makedirs(OUT_DIR, exist_ok=True)

    # sorted() matches ImageFolder's alphabetical class indexing
    class_names = sorted(os.listdir(VAL_DIR))

    # Deterministic image: first file alphabetically in the test class folder
    class_dir  = os.path.join(VAL_DIR, TEST_CLASS)
    image_file = sorted(os.listdir(class_dir))[0]
    image_path = os.path.join(class_dir, image_file)
    print(f"Image      : {image_path}")

    # Load model and compute Grad-CAM
    model        = load_model_for_gradcam(MODEL_NAME, CHECKPOINT)
    target_layer = get_target_layer(model, MODEL_NAME)
    tensor, pil_image = load_image_as_tensor(image_path)

    cam = GradCAM(model, target_layer)
    heatmap, pred_idx, confidence = cam(tensor)
    cam.remove_hooks()

    pred_class = class_names[pred_idx]
    print(f"True class : {TEST_CLASS}")
    print(f"Predicted  : {pred_class}  (confidence {confidence:.4f})")
    match = "CORRECT" if pred_class == TEST_CLASS else "WRONG"
    print(f"Result     : {match}")

    # Save three-panel figure: original | raw heatmap | overlay
    overlay = overlay_heatmap_on_image(pil_image, heatmap, alpha=0.4)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(pil_image.resize((224, 224)))
    axes[0].set_title("Original", fontsize=10)
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=10)
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nPred: {pred_class}\nConf: {confidence:.3f}", fontsize=9
    )
    for ax in axes:
        ax.axis("off")

    out_path = os.path.join(OUT_DIR, "sanity_check.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved      : {out_path}")
