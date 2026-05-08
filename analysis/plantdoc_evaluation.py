"""
PlantDoc cross-dataset evaluation for plant disease detection models.

Runs the trained model (PlantVillage 38-class) on the PlantDoc test set
(231 images, 27 classes) using STRICT class mapping: a prediction is
correct only when the model's top-1 PlantVillage class exactly matches
the expected PlantVillage class for the given PlantDoc ground truth.

Usage:
    python analysis/plantdoc_evaluation.py
    python analysis/plantdoc_evaluation.py --model resnet
    python analysis/plantdoc_evaluation.py --plantdoc-path data/plantdoc/PlantDoc-Dataset/test
    python analysis/plantdoc_evaluation.py --top-k 5
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import classification_report, precision_recall_fscore_support
from typing import List, Tuple

from preprocess.transform import val_transforms
from models.mobilenet_model import get_mobilenet_v2
from models.resnet_model import get_resnet50


# ---------------------------------------------------------------------------
# Class mapping (PlantDoc test set → PlantVillage training classes)
# ---------------------------------------------------------------------------

PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf":                      "Apple___Apple_scab",
    "Apple leaf":                           "Apple___healthy",
    "Apple rust leaf":                      "Apple___Cedar_apple_rust",
    "Bell_pepper leaf":                     "Pepper,_bell___healthy",
    "Bell_pepper leaf spot":                "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf":                       "Blueberry___healthy",
    "Cherry leaf":                          "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot":                  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight":                     "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf":                       "Corn_(maize)___Common_rust_",
    "Peach leaf":                           "Peach___healthy",
    "Potato leaf early blight":             "Potato___Early_blight",
    "Potato leaf late blight":              "Potato___Late_blight",
    "Raspberry leaf":                       "Raspberry___healthy",
    "Soyabean leaf":                        "Soybean___healthy",
    "Squash Powdery mildew leaf":           "Squash___Powdery_mildew",
    "Strawberry leaf":                      "Strawberry___healthy",
    "Tomato Early blight leaf":             "Tomato___Early_blight",
    "Tomato Septoria leaf spot":            "Tomato___Septoria_leaf_spot",
    "Tomato leaf":                          "Tomato___healthy",
    "Tomato leaf bacterial spot":           "Tomato___Bacterial_spot",
    "Tomato leaf late blight":              "Tomato___Late_blight",
    "Tomato leaf mosaic virus":             "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus":             "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf":                     "Tomato___Leaf_Mold",
    "grape leaf":                           "Grape___healthy",
    "grape leaf black rot":                 "Grape___Black_rot",
}

# Inverse: PV class name → PlantDoc class name (for confusion matrix columns)
_PLANTVILLAGE_TO_PLANTDOC = {v: k for k, v in PLANTDOC_TO_PLANTVILLAGE.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_plantvillage_classes(val_dir: str) -> List[str]:
    """Return the 38 PlantVillage class names in sorted order (matches training)."""
    classes = sorted(
        d for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    )
    return classes


def load_model(
    model_name: str, num_classes: int, device: torch.device
) -> torch.nn.Module:
    checkpoint_paths = {
        "mobilenet": "checkpoints/best_mobilenet.pth",
        "resnet":    "checkpoints/best_resnet.pth",
    }
    model_factories = {
        "mobilenet": get_mobilenet_v2,
        "resnet":    get_resnet50,
    }
    path = checkpoint_paths[model_name]
    model = model_factories[model_name](num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Loaded {model_name} weights from '{path}'")
    return model


def collect_image_paths(
    plantdoc_test_dir: str, pd_class_names: List[str]
) -> List[Tuple[str, int]]:
    """Return [(image_path, pd_class_idx), ...] for all readable test images."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    samples = []
    for idx, cls in enumerate(pd_class_names):
        cls_dir = os.path.join(plantdoc_test_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  Warning: class folder not found: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                samples.append((os.path.join(cls_dir, fname), idx))
    return samples


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    samples: List[Tuple[str, int]],
    pv_class_names: List[str],
    pd_class_names: List[str],
    top_k: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on all PlantDoc test images.

    Returns
    -------
    y_true    : PlantDoc class indices, shape (N,)
    y_pred_pd : Predicted PlantDoc class indices (0..n_pd-1), or n_pd for
                "Unmapped" when the model's top-1 PV prediction has no
                PlantDoc equivalent.  Shape (N,).
    y_topk    : 1 if the expected PV class appears in the top-k predictions,
                else 0.  Shape (N,).
    """
    n_pd = len(pd_class_names)
    pd_name_to_idx = {cls: i for i, cls in enumerate(pd_class_names)}

    # Pre-compute expected PV index for each PlantDoc class
    expected_pv_idx = {
        cls: pv_class_names.index(PLANTDOC_TO_PLANTVILLAGE[cls])
        for cls in pd_class_names
    }

    y_true: list[int] = []
    y_pred_pd: list[int] = []
    y_topk: list[int] = []

    total = len(samples)
    processed = 0
    batch_imgs: list[torch.Tensor] = []
    batch_labels: list[int] = []

    def _flush(imgs: list, labels: list) -> None:
        tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            logits = model(tensor)                              # (B, 38)
        topk_idx = torch.topk(logits, k=top_k, dim=1).indices  # (B, top_k)
        topk_np = topk_idx.cpu().numpy()

        for i, true_pd_idx in enumerate(labels):
            pd_cls = pd_class_names[true_pd_idx]
            exp_pv = expected_pv_idx[pd_cls]

            # Top-1 PV → PlantDoc index (n_pd if unmapped)
            pred_pv_name = pv_class_names[topk_np[i, 0]]
            pred_pd_name = _PLANTVILLAGE_TO_PLANTDOC.get(pred_pv_name)
            pred_pd_idx = pd_name_to_idx[pred_pd_name] if pred_pd_name else n_pd

            y_true.append(true_pd_idx)
            y_pred_pd.append(pred_pd_idx)
            y_topk.append(int(exp_pv in topk_np[i]))

    for path, true_pd_idx in samples:
        try:
            img = Image.open(path).convert("RGB")
            batch_imgs.append(val_transforms(img))
            batch_labels.append(true_pd_idx)
        except Exception as e:
            print(f"  Warning: skipped {path}: {e}")
            continue

        if len(batch_imgs) == batch_size:
            prev = processed
            _flush(batch_imgs, batch_labels)
            processed += len(batch_imgs)
            batch_imgs.clear()
            batch_labels.clear()
            # Print every ~20 images (crossing a multiple of 20)
            if processed // 20 > prev // 20 or processed >= total:
                print(f"  [{processed:>3d}/{total}] images processed")

    if batch_imgs:
        _flush(batch_imgs, batch_labels)
        processed += len(batch_imgs)
        print(f"  [{processed:>3d}/{total}] images processed")

    return np.array(y_true), np.array(y_pred_pd), np.array(y_topk)


# ---------------------------------------------------------------------------
# Confusion matrix (27 rows × 28 columns: 27 PlantDoc + 1 "Unmapped PV")
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    y_true: np.ndarray, y_pred_pd: np.ndarray, n_pd: int
) -> np.ndarray:
    """Build (n_pd × n_pd+1) raw-count confusion matrix."""
    cm = np.zeros((n_pd, n_pd + 1), dtype=int)
    for t, p in zip(y_true, y_pred_pd):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    pd_class_names: List[str],
    output_path: str,
) -> None:
    """Save row-normalised confusion matrix heatmap (27 rows × 28 columns)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    data = cm.astype(float) / row_sums

    col_labels = list(pd_class_names) + ["Unmapped\nPV class"]

    fig, ax = plt.subplots(figsize=(24, 16))
    im = ax.imshow(data, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(pd_class_names)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=5)
    ax.set_yticklabels(pd_class_names, fontsize=5)
    ax.set_xlabel("Predicted (PlantDoc space)", fontsize=11, labelpad=8)
    ax.set_ylabel("True (PlantDoc)", fontsize=11, labelpad=8)
    ax.set_title(
        "PlantDoc Cross-Dataset Confusion Matrix — row-normalised recall\n"
        "(last column = PV predictions that have no PlantDoc equivalent)",
        fontsize=11, pad=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Per-class CSV
# ---------------------------------------------------------------------------

def save_per_class_csv(
    pd_class_names: List[str],
    y_true: np.ndarray,
    y_pred_pd: np.ndarray,
    output_path: str,
) -> List[Tuple[str, float, int]]:
    """Compute per-class accuracy, write CSV, return rows sorted by accuracy."""
    rows = []
    for idx, cls in enumerate(pd_class_names):
        mask = y_true == idx
        n = int(mask.sum())
        correct = int(((y_pred_pd == idx) & mask).sum())
        acc = correct / n if n > 0 else 0.0
        rows.append((cls, acc, n))

    rows_sorted = sorted(rows, key=lambda x: x[1])
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "accuracy", "n_samples", "n_correct"])
        for cls, acc, n in rows_sorted:
            n_correct = int(round(acc * n))
            writer.writerow([cls, f"{acc:.4f}", n, n_correct])
    print(f"Saved: {output_path}")
    return rows


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

PLANTVILLAGE_VAL_ACCURACY = 0.9713  # baseline from training evaluation


def print_summary(
    pd_class_names: List[str],
    y_true: np.ndarray,
    y_pred_pd: np.ndarray,
    y_topk: np.ndarray,
    top_k: int,
    per_class_rows: List[Tuple[str, float, int]],
) -> None:
    n = len(y_true)
    overall_acc = (y_pred_pd == y_true).sum() / n
    topk_acc = float(y_topk.mean())

    # For sklearn metrics: treat "Unmapped" (idx=n_pd) as wrong by mapping to -1
    n_pd = len(pd_class_names)
    y_pred_sk = np.where(y_pred_pd < n_pd, y_pred_pd, -1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_sk,
        labels=list(range(n_pd)),
        average="macro",
        zero_division=0,
    )

    unmapped_count = int((y_pred_pd == n_pd).sum())

    print("\n" + "=" * 62)
    print(" PlantDoc Cross-Dataset Evaluation Results")
    print("=" * 62)
    print(f"  Total images processed     : {n}")
    print(f"  Unmapped PV predictions    : {unmapped_count} "
          f"({unmapped_count / n:.1%} of all predictions)")
    print(f"  Overall top-1 accuracy     : {overall_acc:.2%}")
    print(f"  Top-{top_k} accuracy              : {topk_acc:.2%}")
    print(f"  Macro precision            : {precision:.4f}")
    print(f"  Macro recall               : {recall:.4f}")
    print(f"  Macro F1                   : {f1:.4f}")
    print()

    sorted_rows = sorted(per_class_rows, key=lambda x: x[1])
    print(f"  Bottom 5 classes (weakest):")
    for cls, acc, n_s in sorted_rows[:5]:
        print(f"    {acc:6.1%}  {cls}  (n={n_s})")
    print()
    print(f"  Top 5 classes (strongest):")
    for cls, acc, n_s in sorted_rows[-5:]:
        print(f"    {acc:6.1%}  {cls}  (n={n_s})")

    print()
    drop_pp = (PLANTVILLAGE_VAL_ACCURACY - overall_acc) * 100
    print(f"  PlantVillage val accuracy  : {PLANTVILLAGE_VAL_ACCURACY:.2%}")
    print(f"  PlantDoc test accuracy     : {overall_acc:.2%}")
    print(f"  Performance drop           : {drop_pp:+.2f} percentage points")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation of a plant disease model on PlantDoc test set."
    )
    parser.add_argument(
        "--plantdoc-path", default="data/plantdoc/PlantDoc-Dataset/test",
        metavar="DIR", help="Path to PlantDoc test directory",
    )
    parser.add_argument(
        "--model", choices=["mobilenet", "resnet"], default="mobilenet",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, metavar="K",
        help="Compute top-K accuracy (default: 3)",
    )
    args = parser.parse_args()

    out_dir = "results/plantdoc"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")
    batch_size = 8

    print("=" * 62)
    print(" PlantDoc Cross-Dataset Evaluation")
    print(f"  Model        : {args.model}")
    print(f"  PlantDoc path: {args.plantdoc_path}")
    print(f"  Top-k        : {args.top_k}")
    print(f"  Device       : {device}  |  Batch size: {batch_size}")
    print("=" * 62)

    # 1. PlantVillage class names — must match training order exactly
    pv_class_names = load_plantvillage_classes("data/val")
    assert len(pv_class_names) == 38, (
        f"Expected 38 PV classes in data/val, found {len(pv_class_names)}"
    )

    # 2. PlantDoc class names (27, sorted alphabetically)
    pd_class_names = sorted(PLANTDOC_TO_PLANTVILLAGE.keys())

    for pd_cls, pv_cls in PLANTDOC_TO_PLANTVILLAGE.items():
        assert pv_cls in pv_class_names, (
            f"Mapping error: '{pv_cls}' not found in data/val"
        )

    # 3. Load model
    model = load_model(args.model, len(pv_class_names), device)

    # 4. Collect image paths
    samples = collect_image_paths(args.plantdoc_path, pd_class_names)
    print(f"\nFound {len(samples)} images across {len(pd_class_names)} PlantDoc classes")

    # 5. Inference
    print("\nRunning inference…")
    y_true, y_pred_pd, y_topk = run_inference(
        model, samples, pv_class_names, pd_class_names,
        args.top_k, device, batch_size,
    )

    # 6. Confusion matrix
    print("\nComputing confusion matrix…")
    cm = compute_confusion_matrix(y_true, y_pred_pd, len(pd_class_names))
    plot_confusion_matrix(
        cm, pd_class_names,
        output_path=os.path.join(out_dir, "confusion_matrix_plantdoc.png"),
    )

    # 7. Per-class accuracy CSV
    per_class_rows = save_per_class_csv(
        pd_class_names, y_true, y_pred_pd,
        output_path=os.path.join(out_dir, "per_class_results_plantdoc.csv"),
    )

    # 8. Full classification report
    n_pd = len(pd_class_names)
    y_pred_sk = np.where(y_pred_pd < n_pd, y_pred_pd, -1)
    report_str = classification_report(
        y_true, y_pred_sk,
        labels=list(range(n_pd)),
        target_names=pd_class_names,
        digits=4,
        zero_division=0,
    )
    report_path = os.path.join(out_dir, "classification_report_plantdoc.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"Saved: {report_path}")

    # 9. Raw arrays for downstream analysis
    npz_path = os.path.join(out_dir, "plantdoc_evaluation_data.npz")
    np.savez(
        npz_path,
        y_true=y_true,
        y_pred_pd=y_pred_pd,
        y_topk=y_topk,
        cm=cm,
        pd_class_names=np.array(pd_class_names),
        pv_class_names=np.array(pv_class_names),
    )
    print(f"Saved: {npz_path}")

    # 10. Placeholder for later analysis writeup
    md_path = os.path.join(out_dir, "PLANTDOC_ANALYSIS.md")
    if not os.path.exists(md_path):
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# PlantDoc Cross-Dataset Analysis\n\n_Results will be filled after evaluation._\n")
        print(f"Saved: {md_path}")

    # 11. Terminal summary
    print_summary(pd_class_names, y_true, y_pred_pd, y_topk, args.top_k, per_class_rows)

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
