"""
PlantVillage Validation Set Evaluation
=======================================
Evaluates a MobileNetV2 checkpoint on the PlantVillage val set.

Usage:
    python analysis/eval_val.py
    python analysis/eval_val.py --checkpoint checkpoints/best_mobilenet_finetuned.pth
    python analysis/eval_val.py --checkpoint checkpoints/best_mobilenet_finetuned.pth --out-dir results/finetuning_evaluation
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.transform import val_transforms
from models.mobilenet_model import get_mobilenet_v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV2 on PlantVillage validation set."
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_mobilenet.pth", metavar="PATH",
        help="Checkpoint to evaluate (default: checkpoints/best_mobilenet.pth)",
    )
    parser.add_argument(
        "--val-dir", default="data/val", metavar="DIR",
        help="PlantVillage val directory (default: data/val)",
    )
    parser.add_argument(
        "--out-dir", default="results/plantvillage_val", metavar="DIR",
        help="Output directory for JSON summary",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 62)
    print(" PlantVillage Val Evaluation")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Val dir    : {args.val_dir}")
    print("=" * 62)

    dataset = ImageFolder(root=args.val_dir, transform=val_transforms)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    num_classes = len(dataset.classes)
    print(f"\nClasses : {num_classes}")
    print(f"Images  : {len(dataset)}")

    model = get_mobilenet_v2(num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded  : {args.checkpoint}")

    correct = 0
    total   = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if (i + 1) % 10 == 0:
                print(f"  [{total:>5d}/{len(dataset)}] running acc: {correct/total:.4f}")

    accuracy = correct / total
    print(f"\n  Final accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy:.2%})")

    ckpt_name = Path(args.checkpoint).stem
    summary = {
        "checkpoint":  args.checkpoint,
        "val_dir":     args.val_dir,
        "n_images":    total,
        "n_correct":   correct,
        "accuracy":    round(accuracy, 6),
    }
    json_path = out_dir / f"summary_val_{ckpt_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n=== Val evaluation complete ===")


if __name__ == "__main__":
    main()
