"""
Training Set Accuracy Check
============================
Samples ~26 images per class from data/train (~1000 total) and runs
MobileNetV2 inference to compare train accuracy vs validation accuracy.

Usage:
    python check_train_accuracy.py
"""

import os
import sys
import random
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mobilenet_model import get_mobilenet_v2
from preprocess.transform import val_transforms

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT      = "checkpoints/best_mobilenet.pth"
TRAIN_DIR       = "data/train"
SAMPLES_PER_CLASS = 26
SEED            = 42

random.seed(SEED)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading MobileNetV2...")
model = get_mobilenet_v2(num_classes=38)
state = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(state)
model.eval()
print("Model ready.\n")

# ── Build class list from val dir (canonical alphabetical order) ──────────────
class_names = sorted(os.listdir("data/val"))
class_to_idx = {c: i for i, c in enumerate(class_names)}
print(f"Classes: {len(class_names)}")

# ── Sample images ─────────────────────────────────────────────────────────────
sampled = []  # list of (path, true_idx)

for cls in class_names:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    if not os.path.isdir(cls_dir):
        print(f"  WARNING: {cls_dir} not found — skipping")
        continue
    files = [f for f in os.listdir(cls_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    chosen = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
    true_idx = class_to_idx[cls]
    for f in chosen:
        sampled.append((os.path.join(cls_dir, f), true_idx))

random.shuffle(sampled)
total = len(sampled)
print(f"Sampled {total} images ({SAMPLES_PER_CLASS} per class × {len(class_names)} classes)\n")

# ── Inference ─────────────────────────────────────────────────────────────────
correct_top1 = 0
correct_top3 = 0
conf_sum     = 0.0
t_start      = time.time()

with torch.no_grad():
    for i, (path, true_idx) in enumerate(sampled, 1):
        try:
            img    = Image.open(path).convert("RGB")
            tensor = val_transforms(img).unsqueeze(0)
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0)

            top3_vals, top3_idx = torch.topk(probs, 3)
            pred_idx  = top3_idx[0].item()
            pred_conf = top3_vals[0].item()

            if pred_idx == true_idx:
                correct_top1 += 1
                correct_top3 += 1
            elif true_idx in top3_idx.tolist():
                correct_top3 += 1

            conf_sum += pred_conf

        except Exception as e:
            print(f"  ERROR on {os.path.basename(path)}: {e}")
            total -= 1
            continue

        if i % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i:>4d}/{total}]  running top-1={correct_top1/i:.4f}  "
                  f"({elapsed:.1f}s elapsed)")

elapsed = time.time() - t_start

# ── Report ────────────────────────────────────────────────────────────────────
top1_acc  = correct_top1 / total
top3_acc  = correct_top3 / total
mean_conf = conf_sum / total

sep = "=" * 60
print(f"\n{sep}")
print("  TRAIN ACCURACY REPORT  (MobileNetV2)")
print(sep)
print(f"  Samples tested          : {total}")
print(f"  Elapsed                 : {elapsed:.1f}s")
print()
print(f"  Train  top-1 accuracy   : {top1_acc:.4f}  ({top1_acc*100:.2f}%)")
print(f"  Train  top-3 accuracy   : {top3_acc:.4f}  ({top3_acc*100:.2f}%)")
print(f"  Train  mean confidence  : {mean_conf:.4f}")
print()
print(f"  Val    top-1 accuracy   : 0.9713  (97.13%)  [from Phase 2 eval]")
print(f"  Val    mean confidence  : ~0.97")
print()

delta = top1_acc - 0.9713
if delta > 0.02:
    diagnosis = "OVERFITTING likely  (train >> val)"
elif delta < -0.02:
    diagnosis = "Underfitting or distribution mismatch  (val >> train)"
else:
    diagnosis = "Balanced  (train ≈ val) — domain gap is the likely explanation"

print(f"  Train - Val delta       : {delta:+.4f}  ({delta*100:+.2f}%)")
print(f"  Diagnosis               : {diagnosis}")
print(sep)
