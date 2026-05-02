"""
Build the target image list for Grad-CAM analysis.

Reads results/confusion_data_*.npz and data/val to find representative
images across three categories, then writes results/gradcam_targets.json.

Categories
----------
A  sanity_check : correctly predicted, high-confidence images
B  confusion    : MobileNet misclassifications from top confusion pairs
C  comparison   : same image analysed by both models (challenging cases)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple

from preprocess.transform import val_transforms
from models.mobilenet_model import get_mobilenet_v2
from models.resnet_model import get_resnet50


VAL_DIR  = "data/val"
NPZ_MOB  = "results/confusion_data_mobilenet.npz"
NPZ_RES  = "results/confusion_data_resnet.npz"
OUT_JSON = "results/gradcam_targets.json"

# Extensions accepted by torchvision ImageFolder
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Image path list (must mirror ImageFolder + DataLoader(shuffle=False) order)
# ---------------------------------------------------------------------------

def build_image_path_list(
    val_dir: str, class_names: List[str]
) -> List[Tuple[str, int]]:
    """
    Return (path, class_idx) pairs in the exact order ImageFolder uses:
    classes alphabetically, files within each class alphabetically.

    Because the confusion-matrix DataLoader used shuffle=False, index i in
    y_true/y_pred maps directly to element i of this list.
    """
    pairs: List[Tuple[str, int]] = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = sorted(
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        for fname in files:
            pairs.append((os.path.join(class_dir, fname), class_idx))
    return pairs


# ---------------------------------------------------------------------------
# Confidence scoring for a list of image paths
# ---------------------------------------------------------------------------

def score_images(
    image_paths: List[str],
    model: torch.nn.Module,
    device: torch.device,
) -> List[float]:
    """
    Run a forward pass on each image and return the max-class softmax probability.
    Used to rank candidates by model confidence.
    """
    scores: List[float] = []
    with torch.no_grad():
        for path in image_paths:
            tensor = val_transforms(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            prob   = torch.softmax(model(tensor), dim=1).max().item()
            scores.append(float(prob))
    return scores


def load_model(model_name: str) -> torch.nn.Module:
    checkpoint_map = {
        "mobilenet": "checkpoints/best_mobilenet.pth",
        "resnet":    "checkpoints/best_resnet.pth",
    }
    factory_map = {
        "mobilenet": get_mobilenet_v2,
        "resnet":    get_resnet50,
    }
    model = factory_map[model_name](38)
    model.load_state_dict(torch.load(checkpoint_map[model_name], map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def find_correct(
    all_paths: List[Tuple[str, int]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_idx: int,
    max_candidates: int = 30,
) -> List[str]:
    """Return up to max_candidates paths correctly predicted for class_idx."""
    results: List[str] = []
    for i, (path, _) in enumerate(all_paths):
        if y_true[i] == class_idx and y_pred[i] == class_idx:
            results.append(path)
            if len(results) >= max_candidates:
                break
    return results


def find_confused(
    all_paths: List[Tuple[str, int]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_idx: int,
    pred_idx: int,
    n: int = 1,
) -> List[str]:
    """Return up to n paths where true label is true_idx but model predicted pred_idx."""
    results: List[str] = []
    for i, (path, _) in enumerate(all_paths):
        if y_true[i] == true_idx and y_pred[i] == pred_idx:
            results.append(path)
            if len(results) >= n:
                break
    return results


def find_wrong(
    all_paths: List[Tuple[str, int]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_idx: int,
    n: int = 1,
) -> List[str]:
    """Return up to n paths where true label is class_idx but model predicted anything else."""
    results: List[str] = []
    for i, (path, _) in enumerate(all_paths):
        if y_true[i] == class_idx and y_pred[i] != class_idx:
            results.append(path)
            if len(results) >= n:
                break
    return results


# ---------------------------------------------------------------------------
# JSON entry builder
# ---------------------------------------------------------------------------

def make_entry(
    image_path: str,
    true_class: str,
    class_names: List[str],
    category: str,
    model: str,
    note: str = "",
    confidence: Optional[float] = None,
) -> Dict:
    entry: Dict = {
        "image_path":    image_path.replace("\\", "/"),
        "true_class":    true_class,
        "true_class_idx": class_names.index(true_class),
        "category":      category,
        "model":         model,
    }
    if confidence is not None:
        entry["confidence"] = round(confidence, 4)
    if note:
        entry["note"] = note
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load NPZ ---
    print("Loading confusion data...")
    mob = np.load(NPZ_MOB, allow_pickle=True)
    res = np.load(NPZ_RES, allow_pickle=True)

    class_names: List[str]  = mob["class_names"].tolist()
    mob_y_true: np.ndarray  = mob["y_true"]
    mob_y_pred: np.ndarray  = mob["y_pred"]
    res_y_true: np.ndarray  = res["y_true"]
    res_y_pred: np.ndarray  = res["y_pred"]

    print(f"Classes: {len(class_names)} | MobileNet samples: {len(mob_y_true)} | ResNet samples: {len(res_y_true)}")

    # --- Reconstruct file path list in ImageFolder order ---
    print("Scanning data/val...")
    all_paths = build_image_path_list(VAL_DIR, class_names)
    print(f"Total images found: {len(all_paths)}")

    def idx(name: str) -> int:
        return class_names.index(name)

    # --- Load MobileNet for confidence scoring (Category A only) ---
    print("Loading MobileNetV2 for confidence scoring...")
    device     = torch.device("cpu")
    mob_model  = load_model("mobilenet")

    targets: Dict[str, List[Dict]] = {
        "sanity_check": [],
        "confusion":    [],
        "comparison":   [],
    }

    # ==================================================================
    # Category A — Sanity check
    # Correctly predicted classes; pick highest-confidence image from
    # up to 30 candidates to ensure a clean, confident example.
    # ==================================================================
    print("\n--- Category A: Sanity check ---")
    sanity_specs = [
        ("Grape___healthy",   "MobileNet top-tier class (F1=1.00)"),
        ("Apple___Black_rot", "Apple disease, high accuracy class"),
        ("Soybean___healthy", "High-support, near-perfect class"),
    ]

    for class_name, note in sanity_specs:
        candidates = find_correct(all_paths, mob_y_true, mob_y_pred, idx(class_name))
        if not candidates:
            print(f"  WARNING: No correct prediction found for {class_name}")
            continue

        # Score candidates and pick highest confidence
        scores     = score_images(candidates, mob_model, device)
        best_idx   = int(np.argmax(scores))
        best_path  = candidates[best_idx]
        best_conf  = scores[best_idx]

        print(f"  {class_name}: {len(candidates)} candidates, best conf={best_conf:.4f}")
        targets["sanity_check"].append(make_entry(
            best_path, class_name, class_names,
            category="sanity", model="mobilenet",
            note=note, confidence=best_conf,
        ))

    # ==================================================================
    # Category B — Confusion analysis
    # Images from top MobileNet confusion pairs (identified in Step 2).
    # ==================================================================
    print("\n--- Category B: Confusion analysis ---")
    confusion_specs = [
        (
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Northern_Leaf_Blight",
            "top-1 MobileNet confusion pair (103 / 410 errors)",
        ),
        (
            "Tomato___Late_blight",
            "Potato___Late_blight",
            "cross-species confusion (22 errors) — visually similar lesions",
        ),
        (
            "Tomato___Early_blight",
            "Tomato___Bacterial_spot",
            "intra-genus Tomato confusion (22 errors)",
        ),
    ]

    for true_cls, pred_cls, note in confusion_specs:
        found = find_confused(
            all_paths, mob_y_true, mob_y_pred, idx(true_cls), idx(pred_cls), n=1
        )
        if not found:
            print(f"  WARNING: No sample found for {true_cls} -> {pred_cls}")
            continue

        print(f"  Found: {true_cls} -> {pred_cls}")
        targets["confusion"].append(make_entry(
            found[0], true_cls, class_names,
            category="confusion", model="mobilenet", note=note,
        ))

    # ==================================================================
    # Category C — Architectural comparison
    # Same image analysed by both MobileNet and ResNet.
    # Prefer images that ResNet gets wrong (challenging case),
    # fall back to any correctly predicted image.
    # ==================================================================
    print("\n--- Category C: Architectural comparison ---")
    comparison_specs = [
        (
            "Tomato___Late_blight",
            "resnet_wrong",
            "ResNet worst class (F1=0.796); MobileNet handles it much better",
        ),
        (
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "resnet_wrong",
            "ResNet F1=0.931 vs MobileNet F1=0.844 — ResNet has edge here",
        ),
        (
            "Grape___healthy",
            "any_correct",
            "Both models excel (F1>=0.99) — expected clean, complete activation",
        ),
    ]

    for class_name, strategy, note in comparison_specs:
        found: Optional[str] = None

        if strategy == "resnet_wrong":
            wrong = find_wrong(all_paths, res_y_true, res_y_pred, idx(class_name), n=1)
            if wrong:
                found = wrong[0]
            else:
                # Fallback: any correctly predicted image
                print(f"  NOTE: ResNet made no errors on {class_name}, using correct prediction")
                correct = find_correct(all_paths, mob_y_true, mob_y_pred, idx(class_name))
                if correct:
                    found = correct[0]
        else:
            correct = find_correct(all_paths, mob_y_true, mob_y_pred, idx(class_name))
            if correct:
                found = correct[0]

        if found is None:
            print(f"  WARNING: Could not find a suitable image for {class_name}")
            continue

        print(f"  {class_name}: {strategy}")
        for model_name in ["mobilenet", "resnet"]:
            targets["comparison"].append(make_entry(
                found, class_name, class_names,
                category="comparison", model=model_name, note=note,
            ))

    # --- Save JSON ---
    os.makedirs("results", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_JSON}")

    # --- Terminal summary ---
    n_a = len(targets["sanity_check"])
    n_b = len(targets["confusion"])
    n_c = len(targets["comparison"])

    unique_images = {e["image_path"] for group in targets.values() for e in group}
    total_runs    = n_a + n_b + n_c   # each entry = 1 Grad-CAM run

    print("\n" + "=" * 54)
    print("  GRAD-CAM TARGET SUMMARY")
    print("=" * 54)
    print(f"  Category A  sanity_check  : {n_a:2d} entries (MobileNet)")
    print(f"  Category B  confusion     : {n_b:2d} entries (MobileNet)")
    print(f"  Category C  comparison    : {n_c:2d} entries (both models)")
    print(f"  Unique images             : {len(unique_images):2d}")
    print(f"  Total Grad-CAM runs       : {total_runs:2d}")
    print("=" * 54)


if __name__ == "__main__":
    main()
