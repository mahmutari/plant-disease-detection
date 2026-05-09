"""
Phase 3.9.B — Web Validation Evaluation
========================================
test_images/web_validation/ veya test_images/enhanced/web_validation/
klasoründeki 25 goerseli MobileNetV2 ile degerlendirir.

Her gorsel icin: filename, subfolder, top-3 tahmin + confidence CSV'e yazilir.
OOD goerselleri de dahil edilir (PlantVillage'da karsiligi yok → model ne der?).

Kullanim:
    python analysis/web_evaluation.py --original
    python analysis/web_evaluation.py --enhanced
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from preprocess.transform import val_transforms
from models.mobilenet_model import get_mobilenet_v2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parent.parent
WEB_ORIGINAL = ROOT / "test_images" / "web_validation"
WEB_ENHANCED = ROOT / "test_images" / "enhanced" / "web_validation"
OUT_DIR      = ROOT / "results" / "web_evaluation"
CKPT         = ROOT / "checkpoints" / "best_mobilenet.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Ground truth map — dosya adi → PlantVillage sinifi (None = OOD)
# ---------------------------------------------------------------------------

GROUND_TRUTH: Dict[str, str | None] = {
    # easy
    "apple_scab_01.jpg":                    "Apple___Apple_scab",
    "apple_scab_02.jpg":                    "Apple___Apple_scab",
    "corn_common_rust_01.jpg":              "Corn_(maize)___Common_rust_",
    "potato_late_blight_01.jpg":            "Potato___Late_blight",
    "tomato_early_blight_01.jpg":           "Tomato___Early_blight",
    "tomato_septoria_leaf_01.jpg":          "Tomato___Septoria_leaf_spot",
    # medium
    "apple_scab_field_01.jpg":              "Apple___Apple_scab",
    "corn_common_rust_field_01.jpg":        "Corn_(maize)___Common_rust_",
    "pepper_bacterial_spot_field_01.jpg":   "Pepper,_bell___Bacterial_spot",
    "potato_late_blight_field_01.jpg":      "Potato___Late_blight",
    "potato_late_blight_field_02.jpg":      "Potato___Late_blight",
    "tomato_early_blight_field_01.jpg":     "Tomato___Early_blight",
    "tomato_late_blight_field_01.jpg":      "Tomato___Late_blight",
    "medium_01_tomato_misclassified_as_squash.png": None,   # belirsiz GT
    "grape_powdery_mildew_field_01.jpg":    None,           # OOD (PV'de yok)
    # hard
    "apple_fire_blight_dense_01.jpg":       None,           # OOD
    "apple_fire_blight_dense_02.jpg":       None,           # OOD
    "potato_late_blight_dense_01.jpg":      "Potato___Late_blight",
    "tomato_early_blight_heavy_01.jpg":     "Tomato___Early_blight",
    # ood
    "wheat_leaf_rust_01.jpg":               None,
    "wheat_leaf_rust_02.jpg":               None,
    "rose_black_spot_01.jpg":               None,
    "rose_black_spot_02.jpg":               None,
    "sunflower_leaf_ood_01.jpg":            None,
    "sunflower_downy_mildew_ood_01.jpg":    None,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pv_classes(val_dir: str) -> List[str]:
    return sorted(
        d for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    )


def load_model(num_classes: int, device: torch.device, checkpoint_path: str | None = None) -> torch.nn.Module:
    ckpt = Path(checkpoint_path) if checkpoint_path else CKPT
    model = get_mobilenet_v2(num_classes).to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()
    print(f"Loaded MobileNetV2 weights from '{ckpt}'")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def evaluate_images(
    src_dir: Path,
    model: torch.nn.Module,
    pv_classes: List[str],
    device: torch.device,
    top_k: int = 3,
) -> List[Dict]:
    images = sorted(
        p for p in src_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"  [WARN] No images found in {src_dir}")
        return []

    print(f"\nFound {len(images)} images in {src_dir}")
    print("-" * 62)

    rows = []
    n_correct = n_eval = 0

    for img_path in images:
        subfolder = img_path.parent.name
        gt_class  = GROUND_TRUTH.get(img_path.name)

        try:
            img    = Image.open(img_path).convert("RGB")
            tensor = val_transforms(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  [SKIP] {img_path.name}: {e}")
            continue

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)
            topk_v, topk_i = torch.topk(probs, k=top_k, dim=1)

        topk_v = topk_v[0].cpu().numpy()
        topk_i = topk_i[0].cpu().numpy()

        top1_class = pv_classes[topk_i[0]]
        correct    = (gt_class is not None) and (top1_class == gt_class)

        if gt_class is not None:
            n_eval += 1
            if correct:
                n_correct += 1

        row: Dict = {
            "filename":    img_path.name,
            "subfolder":   subfolder,
            "ground_truth": gt_class if gt_class else "OOD/UNKNOWN",
            "correct":     "YES" if correct else ("OOD" if gt_class is None else "NO"),
        }
        for i in range(top_k):
            row[f"top{i+1}_class"]      = pv_classes[topk_i[i]]
            row[f"top{i+1}_confidence"] = f"{topk_v[i]:.4f}"

        rows.append(row)

        mark = "V" if correct else ("?" if gt_class is None else "X")
        print(f"  [{mark}] {img_path.name:<50} -> {top1_class} ({topk_v[0]:.1%})")

    print("-" * 62)
    if n_eval > 0:
        print(f"  In-distribution accuracy: {n_correct}/{n_eval} ({n_correct/n_eval:.1%})")
    print(f"  Total images processed  : {len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web validation evaluation — original or enhanced images"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--original", action="store_true",
        help="Evaluate test_images/web_validation/ (originals)",
    )
    group.add_argument(
        "--enhanced", action="store_true",
        help="Evaluate test_images/enhanced/web_validation/",
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Override checkpoint path (default: checkpoints/best_mobilenet.pth)",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Override output directory (default: results/web_evaluation)",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    suffix = "enhanced" if args.enhanced else "original"
    src    = WEB_ENHANCED if args.enhanced else WEB_ORIGINAL
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    print("=" * 62)
    print(f"  Web Validation Evaluation  [{suffix.upper()}]")
    print(f"  Source : {src}")
    print("=" * 62)

    out_dir.mkdir(parents=True, exist_ok=True)

    device     = torch.device("cpu")
    pv_classes = load_pv_classes("data/val")
    assert len(pv_classes) == 38
    model      = load_model(len(pv_classes), device, args.checkpoint)

    rows = evaluate_images(src, model, pv_classes, device)

    if not rows:
        return

    # CSV
    out_csv = out_dir / f"web_results_{suffix}.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")

    # JSON summary
    n_eval    = sum(1 for r in rows if r["ground_truth"] not in ("OOD/UNKNOWN",))
    n_correct = sum(1 for r in rows if r["correct"] == "YES")
    summary   = {
        "suffix":         suffix,
        "source":         str(src),
        "total_images":   len(rows),
        "in_dist_images": n_eval,
        "correct":        n_correct,
        "accuracy":       round(n_correct / n_eval, 6) if n_eval else 0.0,
    }
    json_path = out_dir / f"summary_web_{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n=== Web evaluation complete ===")


if __name__ == "__main__":
    main()
