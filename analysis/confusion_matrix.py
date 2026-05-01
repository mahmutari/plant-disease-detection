"""
Confusion matrix evaluation for plant disease detection models.

Usage:
    python analysis/confusion_matrix.py --model mobilenet
    python analysis/confusion_matrix.py --model resnet
    python analysis/confusion_matrix.py --model mobilenet --subset 10  # quick test
"""

import argparse
import os
import sys

# Ensure project root is on sys.path when script is run from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Optional, Tuple

from preprocess.transform import val_transforms
from models.mobilenet_model import get_mobilenet_v2
from models.resnet_model import get_resnet50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def abbreviate(name: str) -> str:
    """Convert 'Plant___Disease_name' to 'Plant/Disease_name' for axis labels."""
    return name.replace("___", "/")


def load_model(
    model_name: str, num_classes: int, device: torch.device
) -> torch.nn.Module:
    """Load a trained model and inject checkpoint weights."""
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


def build_dataloader(
    val_dir: str,
    subset: Optional[int],
    batch_size: int,
) -> Tuple[DataLoader, List[str]]:
    """
    Build a validation DataLoader from data/val.

    If subset is given, takes the first N samples per class so the caller
    can do a quick sanity-check run without waiting for all ~17 k images.
    ImageFolder sorts classes alphabetically — consistent with training.
    """
    dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    class_names: List[str] = dataset.classes  # already sorted alphabetically

    if subset is not None:
        # Collect up to `subset` indices per class label
        buckets: dict[int, list[int]] = {i: [] for i in range(len(class_names))}
        for idx, (_, label) in enumerate(dataset.samples):
            if len(buckets[label]) < subset:
                buckets[label].append(idx)
        indices = [i for bucket in buckets.values() for i in bucket]
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        print(f"Subset mode: {subset} images/class -> {len(indices)} total")
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        print(f"Full validation set: {len(dataset)} images")

    return loader, class_names


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run full-dataset inference and collect true / predicted labels.
    Prints progress every 50 batches so the user knows CPU is still working.
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    total = len(loader)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            preds = torch.argmax(model(inputs), dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  [{i + 1:>4d}/{total}] batches processed")

    return np.array(y_true), np.array(y_pred)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    output_path: str,
    normalize: bool = False,
) -> None:
    """
    Save a 38x38 confusion matrix heatmap.

    normalize=True  -> row-normalized (recall per class on diagonal)
    normalize=False -> raw counts

    Cell text annotations are omitted at this scale (38×38 = 1444 cells);
    the colorbar conveys the magnitude instead.
    """
    labels = [abbreviate(c) for c in class_names]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # guard divide-by-zero
        data = cm.astype(float) / row_sums
        vmax = 1.0
    else:
        data = cm
        vmax = float(cm.max())

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(data, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ticks = range(len(labels))
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
    ax.set_ylabel("True", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, pad=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_top_confused_pairs(
    cm: np.ndarray, class_names: List[str], top_n: int = 5
) -> None:
    """Print the N largest off-diagonal cells (most common misclassifications)."""
    n = len(class_names)
    pairs = [
        (cm[i, j], class_names[i], class_names[j])
        for i in range(n)
        for j in range(n)
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(reverse=True)

    print(f"\nTop {top_n} most confused class pairs  (True -> Predicted):")
    for count, true_cls, pred_cls in pairs[:top_n]:
        print(f"  {count:4d}  {abbreviate(true_cls):35s} -> {abbreviate(pred_cls)}")


def save_report(report: str, path: str) -> None:
    """Write sklearn classification_report to a UTF-8 text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on data/val and produce confusion matrix artefacts."
    )
    parser.add_argument(
        "--model", choices=["mobilenet", "resnet"], default="mobilenet",
        help="Model architecture to evaluate (default: mobilenet)",
    )
    parser.add_argument(
        "--subset", type=int, default=None, metavar="N",
        help="Evaluate only the first N images per class (omit for full set)",
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    device = torch.device("cpu")  # CPU-only machine
    num_classes = 38
    batch_size = 8
    model_name: str = args.model

    print("=" * 50)
    print(f" Confusion Matrix Evaluation")
    print(f" Model : {model_name}")
    print(f" Device: {device}  |  Batch size: {batch_size}")
    print("=" * 50)

    model = load_model(model_name, num_classes, device)
    loader, class_names = build_dataloader("data/val", args.subset, batch_size)

    print("\nRunning inference…")
    y_true, y_pred = run_inference(model, loader, device)

    print("\nComputing confusion matrix…")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Raw count heatmap
    plot_confusion_matrix(
        cm, class_names,
        title=f"Confusion Matrix — {model_name}  (counts)",
        output_path=f"results/confusion_matrix_counts_{model_name}.png",
        normalize=False,
    )

    # Row-normalised heatmap (diagonal = per-class recall)
    plot_confusion_matrix(
        cm, class_names,
        title=f"Confusion Matrix — {model_name}  (row-normalised recall)",
        output_path=f"results/confusion_matrix_normalized_{model_name}.png",
        normalize=True,
    )

    # Per-class precision / recall / F1
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print("\nClassification Report:\n", report)
    save_report(report, f"results/classification_report_{model_name}.txt")

    # Raw arrays saved for Grad-CAM stage (confusion_data_*.npz)
    npz_path = f"results/confusion_data_{model_name}.npz"
    np.savez(npz_path, cm=cm, class_names=np.array(class_names),
             y_true=y_true, y_pred=y_pred)
    print(f"Saved: {npz_path}")

    print_top_confused_pairs(cm, class_names, top_n=5)

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
