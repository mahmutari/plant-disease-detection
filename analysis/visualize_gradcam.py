"""
Batch Grad-CAM visualization for all targets in gradcam_targets.json.

Produces a 4-panel PNG per target (Original / Heatmap / Overlay / Info)
and writes a terminal insights report on completion.

Usage:
    python analysis/visualize_gradcam.py
    python analysis/visualize_gradcam.py --targets results/gradcam_targets.json
    python analysis/visualize_gradcam.py --output_dir results/gradcam --skip_existing
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn

from analysis.gradcam import (
    GradCAM,
    get_target_layer,
    load_image_as_tensor,
    load_model_for_gradcam,
    overlay_heatmap_on_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHECKPOINT = {
    "mobilenet": "checkpoints/best_mobilenet.pth",
    "resnet":    "checkpoints/best_resnet.pth",
}

MODEL_DISPLAY = {
    "mobilenet": "MobileNetV2",
    "resnet":    "ResNet-50",
}

TARGET_LAYER_DISPLAY = {
    "mobilenet": "features[-1]  (last inverted residual)",
    "resnet":    "layer4[-1]    (last residual block)",
}

CATEGORY_TITLES = {
    "sanity":     "Sanity Check  — correct high-confidence prediction",
    "confusion":  "Confusion Analysis  — misclassified image",
    "comparison": "Architectural Comparison  — MobileNetV2 vs ResNet-50",
}

SUBDIR = {
    "sanity":     "sanity_check",
    "confusion":  "confusion",
    "comparison": "comparison",
}


def abbreviate(name: str) -> str:
    return name.replace("___", "/")


def short_filename(path: str, max_len: int = 40) -> str:
    base = os.path.basename(path)
    return base if len(base) <= max_len else "..." + base[-(max_len - 3):]


def output_filename(idx: int, entry: Dict) -> str:
    """
    Build a deterministic output filename from entry index and metadata.
    Example: 07_tomato_lateblight_mobilenet.png
    """
    cls_short = (
        entry["true_class"]
        .lower()
        .replace("(", "").replace(")", "").replace(",", "")
        .replace(" ", "_")
        .split("___")[-1]          # keep disease name only
        [:20]                       # cap length
    )
    return f"{idx:02d}_{cls_short}_{entry['model']}.png"


# ---------------------------------------------------------------------------
# 4-panel figure
# ---------------------------------------------------------------------------

def save_gradcam_figure(
    entry: Dict,
    class_names: List[str],
    heatmap: np.ndarray,
    pred_idx: int,
    confidence: float,
    output_path: str,
) -> None:
    """Build and save a 4-panel Grad-CAM figure for one entry."""
    from PIL import Image as PILImage

    image_path = entry["image_path"]
    true_class = entry["true_class"]
    model_name = entry["model"]
    category   = entry["category"]

    pil_image  = PILImage.open(image_path).convert("RGB")
    pred_class = class_names[pred_idx]
    is_correct = (pred_class == true_class)
    overlay    = overlay_heatmap_on_image(pil_image, heatmap, alpha=0.4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("#1a1a2e")

    ax_orig, ax_heat, ax_over, ax_info = axes.flatten()

    # Panel 1 — Original
    ax_orig.imshow(pil_image.resize((224, 224)))
    ax_orig.set_title("Original Image", color="white", fontsize=11, pad=6)
    ax_orig.axis("off")

    # Panel 2 — Raw heatmap
    im = ax_heat.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    ax_heat.set_title("Grad-CAM Heatmap", color="white", fontsize=11, pad=6)
    ax_heat.axis("off")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # Panel 3 — Overlay
    ax_over.imshow(overlay)
    ax_over.set_title("Overlay  (alpha=0.4)", color="white", fontsize=11, pad=6)
    ax_over.axis("off")

    # Panel 4 — Info text
    ax_info.set_facecolor("#0d0d1a")
    ax_info.axis("off")

    result_color = "#00e676" if is_correct else "#ff1744"
    result_text  = "[CORRECT]" if is_correct else "[INCORRECT]"

    info_lines = [
        ("File",        short_filename(image_path)),
        ("True class",  abbreviate(true_class)),
        ("Predicted",   abbreviate(pred_class)),
        ("Confidence",  f"{confidence:.4f}  ({confidence*100:.1f}%)"),
        ("Result",      result_text),
        ("Model",       MODEL_DISPLAY[model_name]),
        ("Layer",       TARGET_LAYER_DISPLAY[model_name]),
    ]

    y_pos = 0.92
    for label, value in info_lines:
        color = result_color if label == "Result" else "white"
        weight = "bold" if label == "Result" else "normal"
        ax_info.text(
            0.05, y_pos, f"{label}:",
            transform=ax_info.transAxes,
            color="#aaaaaa", fontsize=9, va="top",
        )
        ax_info.text(
            0.38, y_pos, value,
            transform=ax_info.transAxes,
            color=color, fontsize=9, va="top", fontweight=weight,
            wrap=True,
        )
        y_pos -= 0.13

    ax_info.set_title("Prediction Info", color="white", fontsize=11, pad=6)

    # Figure super-title
    cat_title  = CATEGORY_TITLES.get(category, category)
    suptitle   = f"{cat_title}\n{abbreviate(true_class)}  |  {MODEL_DISPLAY[model_name]}"
    fig.suptitle(suptitle, color="white", fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Grad-CAM visualization")
    parser.add_argument(
        "--targets", default="results/gradcam_targets.json",
        help="Path to gradcam_targets.json",
    )
    parser.add_argument(
        "--output_dir", default="results/gradcam",
        help="Root output directory for Grad-CAM PNGs",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip files that already exist (useful for re-runs / debugging)",
    )
    args = parser.parse_args()

    # --- Load targets ---
    with open(args.targets, encoding="utf-8") as f:
        targets_dict = json.load(f)

    # Flatten to ordered list; assign sequential index for filenames
    all_entries: List[Dict] = []
    for group in ("sanity_check", "confusion", "comparison"):
        all_entries.extend(targets_dict.get(group, []))

    total = len(all_entries)
    print(f"Loaded {total} Grad-CAM targets from {args.targets}")

    # class_names from first entry's NPZ (same for both models)
    import numpy as np
    npz = np.load("results/confusion_data_mobilenet.npz", allow_pickle=True)
    class_names: List[str] = npz["class_names"].tolist()

    # --- Load models once ---
    needed_models = {e["model"] for e in all_entries}
    print(f"Models needed: {sorted(needed_models)}")
    models: Dict[str, Tuple[nn.Module, nn.Module]] = {}
    for model_name in sorted(needed_models):
        print(f"  Loading {MODEL_DISPLAY[model_name]}...")
        model        = load_model_for_gradcam(model_name, CHECKPOINT[model_name])
        target_layer = get_target_layer(model, model_name)
        models[model_name] = (model, target_layer)
    print("Models ready.\n")

    # --- Create output subdirectories ---
    for subdir in SUBDIR.values():
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # --- Process each target ---
    t_start   = time.time()
    successes = 0
    failures: List[str] = []

    # Insights accumulators
    insights: Dict[str, List[Dict]] = {"sanity": [], "confusion": [], "comparison": []}

    for entry_idx, entry in enumerate(all_entries, start=1):
        model_name = entry["model"]
        category   = entry["category"]
        image_path = entry["image_path"]
        true_class = entry["true_class"]
        subdir     = SUBDIR[category]

        fname      = output_filename(entry_idx, entry)
        out_path   = os.path.join(args.output_dir, subdir, fname)

        print(f"  [{entry_idx:>2d}/{total}] {fname}")

        if args.skip_existing and os.path.exists(out_path):
            print(f"           Skipped (already exists)")
            successes += 1
            continue

        try:
            model, target_layer = models[model_name]
            tensor, pil_image   = load_image_as_tensor(image_path)

            cam                         = GradCAM(model, target_layer)
            heatmap, pred_idx, conf     = cam(tensor)
            cam.remove_hooks()          # release hooks immediately after use

            save_gradcam_figure(
                entry, class_names, heatmap, pred_idx, conf, out_path
            )

            pred_class = class_names[pred_idx]
            is_correct = (pred_class == true_class)
            print(f"           Pred: {abbreviate(pred_class):35s}  conf={conf:.4f}  {'OK' if is_correct else 'WRONG'}")

            insights[category].append({
                "entry":      entry,
                "pred_class": pred_class,
                "confidence": conf,
                "correct":    is_correct,
                "out_path":   out_path,
            })
            successes += 1

        except Exception as exc:
            msg = f"{fname}: {exc}"
            failures.append(msg)
            print(f"           ERROR: {exc}")

    elapsed = time.time() - t_start

    # --- Terminal insights report ---
    sep = "=" * 60
    print(f"\n{sep}")
    print("  GRAD-CAM BATCH COMPLETE")
    print(sep)
    print(f"  Processed : {successes}/{total}  ({len(failures)} errors)")
    print(f"  Elapsed   : {elapsed:.1f}s  ({elapsed/max(successes,1):.1f}s / image)")
    print(f"  Output    : {args.output_dir}/")

    # Per-folder counts
    folder_counts: Dict[str, int] = {}
    for r in [r for group in insights.values() for r in group]:
        folder = SUBDIR[r["entry"]["category"]]
        folder_counts[folder] = folder_counts.get(folder, 0) + 1
    for folder, count in sorted(folder_counts.items()):
        print(f"    {folder:<20s}: {count} files")

    # Sanity check insights
    s_records = insights["sanity"]
    if s_records:
        all_correct = all(r["correct"] for r in s_records)
        avg_conf    = sum(r["confidence"] for r in s_records) / len(s_records)
        print(f"\n  [Sanity Check]")
        print(f"    All predictions correct : {'YES' if all_correct else 'NO'}")
        print(f"    Avg confidence          : {avg_conf:.4f}")
        for r in s_records:
            status = "correct" if r["correct"] else "WRONG"
            print(f"    {abbreviate(r['entry']['true_class']):35s}  conf={r['confidence']:.4f}  {status}")

    # Confusion insights
    c_records = insights["confusion"]
    if c_records:
        print(f"\n  [Confusion Analysis]")
        for r in c_records:
            status = "misclassified as expected" if not r["correct"] else "SURPRISE: correct!"
            print(f"    True : {abbreviate(r['entry']['true_class'])}")
            print(f"    Pred : {abbreviate(r['pred_class'])}  conf={r['confidence']:.4f}  ({status})")
            print()

    # Comparison insights
    comp_records = insights["comparison"]
    if comp_records:
        print(f"  [Architectural Comparison]")
        # Group by image_path
        from collections import defaultdict
        by_image: Dict[str, List] = defaultdict(list)
        for r in comp_records:
            by_image[r["entry"]["image_path"]].append(r)
        for img_path, recs in by_image.items():
            print(f"    {abbreviate(recs[0]['entry']['true_class'])}")
            for r in recs:
                status = "correct" if r["correct"] else "WRONG"
                print(f"      {MODEL_DISPLAY[r['entry']['model']]:<15s}  pred={abbreviate(r['pred_class'])}  conf={r['confidence']:.4f}  {status}")

    if failures:
        print(f"\n  ERRORS ({len(failures)}):")
        for msg in failures:
            print(f"    {msg}")

    print(sep)


if __name__ == "__main__":
    main()
