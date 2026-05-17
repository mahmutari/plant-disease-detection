#!/usr/bin/env python
"""
Phase 3.5 (FIXED): Proper Hybrid Training with PD -> PV Label Mapping
Maps PlantDoc class indices to PlantVillage indices before training,
ensuring consistent label space across both datasets.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import datasets, transforms, models
from PIL import Image

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

CONFIG = {
    'pv_train_dir':    'data/train',
    'pv_val_dir':      'data/val',
    'pd_train_dir':    'data/plantdoc/PlantDoc-Dataset/train',
    'pd_test_dir':     'data/plantdoc/PlantDoc-Dataset/test',
    'output_dir':      'checkpoints',
    'results_dir':     'results/hybrid_training_v2',
    'base_checkpoint': 'checkpoints/best_mobilenet.pth',
    'num_classes':     38,
    'image_size':      224,
    'batch_size':      8,
    'epochs':          3,
    'learning_rate':   1e-3,
    'random_seed':     42,
}

torch.manual_seed(CONFIG['random_seed'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ═══════════════════════════════════════════════════════════
# PD -> PV LABEL MAPPING
# Consistent with analysis/plantdoc_evaluation.py
# "Tomato two spotted spider mites leaf" eklendi (28. sınıf)
# ═══════════════════════════════════════════════════════════

PD_TO_PV_MAPPING = {
    'Apple Scab Leaf':                      'Apple___Apple_scab',
    'Apple leaf':                           'Apple___healthy',
    'Apple rust leaf':                      'Apple___Cedar_apple_rust',
    'Bell_pepper leaf':                     'Pepper,_bell___healthy',
    'Bell_pepper leaf spot':                'Pepper,_bell___Bacterial_spot',
    'Blueberry leaf':                       'Blueberry___healthy',
    'Cherry leaf':                          'Cherry_(including_sour)___healthy',
    'Corn Gray leaf spot':                  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn leaf blight':                     'Corn_(maize)___Northern_Leaf_Blight',
    'Corn rust leaf':                       'Corn_(maize)___Common_rust_',
    'Peach leaf':                           'Peach___healthy',
    'Potato leaf early blight':             'Potato___Early_blight',
    'Potato leaf late blight':              'Potato___Late_blight',
    'Raspberry leaf':                       'Raspberry___healthy',
    'Soyabean leaf':                        'Soybean___healthy',
    'Squash Powdery mildew leaf':           'Squash___Powdery_mildew',
    'Strawberry leaf':                      'Strawberry___healthy',
    'Tomato Early blight leaf':             'Tomato___Early_blight',
    'Tomato Septoria leaf spot':            'Tomato___Septoria_leaf_spot',
    'Tomato leaf':                          'Tomato___healthy',
    'Tomato leaf bacterial spot':           'Tomato___Bacterial_spot',
    'Tomato leaf late blight':              'Tomato___Late_blight',
    'Tomato leaf mosaic virus':             'Tomato___Tomato_mosaic_virus',
    'Tomato leaf yellow virus':             'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato mold leaf':                     'Tomato___Leaf_Mold',
    'Tomato two spotted spider mites leaf': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'grape leaf':                           'Grape___healthy',
    'grape leaf black rot':                 'Grape___Black_rot',
}

# ═══════════════════════════════════════════════════════════
# CUSTOM DATASET: PlantDoc with PV label indices
# ═══════════════════════════════════════════════════════════

class PlantDocAsPV(Dataset):
    """PlantDoc dataset that returns labels remapped to PlantVillage index space."""

    def __init__(self, pd_root, pv_class_to_idx, transform=None):
        self.transform = transform
        self.samples = []

        for pd_class in sorted(os.listdir(pd_root)):
            pd_class_dir = os.path.join(pd_root, pd_class)
            if not os.path.isdir(pd_class_dir):
                continue

            pv_class = PD_TO_PV_MAPPING.get(pd_class)
            if pv_class is None:
                print(f"  [SKIP] No mapping for PD class: '{pd_class}'")
                continue

            if pv_class not in pv_class_to_idx:
                print(f"  [SKIP] PV class not in index: '{pv_class}' (from '{pd_class}')")
                continue

            pv_idx = pv_class_to_idx[pv_class]

            for fname in os.listdir(pd_class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(pd_class_dir, fname), pv_idx))

        print(f"  PlantDocAsPV: {len(self.samples)} images loaded with PV-mapped labels")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 3.5 (FIXED): HYBRID TRAINING WITH PROPER LABEL MAPPING")
print("=" * 70)

train_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

pv_train_dir = CONFIG['pv_train_dir']
if not os.path.exists(pv_train_dir):
    print(f"WARNING: {pv_train_dir} not found, using {CONFIG['pv_val_dir']}")
    pv_train_dir = CONFIG['pv_val_dir']

print(f"\nLoading PlantVillage from: {pv_train_dir}")
pv_train = datasets.ImageFolder(pv_train_dir, transform=train_transform)
print(f"  PV: {len(pv_train)} images, {len(pv_train.classes)} classes")

pv_class_to_idx = pv_train.class_to_idx

print(f"\nLoading PlantDoc from: {CONFIG['pd_train_dir']}")
pd_train = PlantDocAsPV(
    pd_root=CONFIG['pd_train_dir'],
    pv_class_to_idx=pv_class_to_idx,
    transform=train_transform,
)

combined = ConcatDataset([pv_train, pd_train])
print(f"\nCombined dataset: {len(combined)} images (CONSISTENT PV labels)")

# ═══════════════════════════════════════════════════════════
# 1:1 WEIGHTED SAMPLING
# ═══════════════════════════════════════════════════════════

pv_weight = 1.0 / len(pv_train)
pd_weight = 1.0 / len(pd_train)
sample_weights = [pv_weight] * len(pv_train) + [pd_weight] * len(pd_train)
total_samples = 2 * len(pv_train)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=total_samples,
    replacement=True,
)

print(f"\n1:1 Weighted Sampling: {total_samples} samples/epoch")
print(f"  Expected PV: ~{total_samples // 2}, PD: ~{total_samples // 2}")

train_loader = DataLoader(
    combined,
    batch_size=CONFIG['batch_size'],
    sampler=sampler,
    num_workers=0,
    pin_memory=False,
)

# ═══════════════════════════════════════════════════════════
# MODEL SETUP — Frozen backbone, trainable head
# ═══════════════════════════════════════════════════════════

print(f"\nLoading base checkpoint: {CONFIG['base_checkpoint']}")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, CONFIG['num_classes'])

checkpoint = torch.load(CONFIG['base_checkpoint'], map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

for param in model.features.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════

device = torch.device('cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=CONFIG['learning_rate'],
)

training_history = []

print(f"\n{'='*70}\nTRAINING START: {CONFIG['epochs']} epochs\n{'='*70}")
start_time = time.time()

for epoch in range(1, CONFIG['epochs'] + 1):
    print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
    print("-" * 50)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - epoch_start
            eta = elapsed / (batch_idx + 1) * (len(train_loader) - batch_idx - 1) / 60
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% | "
                  f"ETA: {eta:.1f}min")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    epoch_time = (time.time() - epoch_start) / 60

    print(f"\n  Epoch {epoch} Summary:")
    print(f"    Loss: {epoch_loss:.4f}")
    print(f"    Train Accuracy: {epoch_acc:.2f}%")
    print(f"    Time: {epoch_time:.1f} minutes")

    training_history.append({
        'epoch': epoch,
        'train_loss': epoch_loss,
        'train_accuracy': epoch_acc,
        'time_minutes': epoch_time,
    })

    checkpoint_path = os.path.join(CONFIG['output_dir'], 'best_mobilenet_hybrid_v2.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': epoch_loss,
        'train_accuracy': epoch_acc,
        'config': CONFIG,
    }, checkpoint_path)
    print(f"    Checkpoint saved: {checkpoint_path}")

    with open(os.path.join(CONFIG['results_dir'], 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

total_time = (time.time() - start_time) / 60
print(f"\n{'='*70}")
print(f"TRAINING COMPLETE: {total_time:.1f} minutes total")
print(f"{'='*70}")

# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*70}\nEVALUATION\n{'='*70}")
model.eval()


def evaluate(loader, name):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    print(f"  {name}: {correct}/{total} = {acc:.2f}%")
    return acc


pv_val = datasets.ImageFolder(CONFIG['pv_val_dir'], transform=eval_transform)
pv_val_loader = DataLoader(pv_val, batch_size=16, shuffle=False, num_workers=0)
pv_val_acc = evaluate(pv_val_loader, "PlantVillage val")

print()
pd_test = PlantDocAsPV(
    pd_root=CONFIG['pd_test_dir'],
    pv_class_to_idx=pv_class_to_idx,
    transform=eval_transform,
)
pd_test_loader = DataLoader(pd_test, batch_size=16, shuffle=False, num_workers=0)
pd_test_acc = evaluate(pd_test_loader, "PlantDoc test (mapped labels)")

# ═══════════════════════════════════════════════════════════
# SAVE & COMPARE
# ═══════════════════════════════════════════════════════════

comparison = {
    'PlantVillage_val': {
        'Phase2_Original':    97.13,
        'Phase3_FineTuned':   58.96,
        'Phase35_HybridV1':   96.09,
        'Phase35_HybridV2':   round(pv_val_acc, 4),
    },
    'PlantDoc_test': {
        'Phase2_Original':    16.02,
        'Phase3_FineTuned':   30.74,
        'Phase35_HybridV1':   31.17,
        'Phase35_HybridV2':   round(pd_test_acc, 4),
    },
}

with open(os.path.join(CONFIG['results_dir'], 'final_results.json'), 'w') as f:
    json.dump({
        'PlantVillage_val': pv_val_acc,
        'PlantDoc_test': pd_test_acc,
        'training_history': training_history,
    }, f, indent=2)

with open(os.path.join(CONFIG['results_dir'], 'comparison_v2.json'), 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"\n{'Model':<22} {'PV val':>8} {'PD test':>8}")
print("-" * 40)
for model_name in ['Phase2_Original', 'Phase3_FineTuned', 'Phase35_HybridV1', 'Phase35_HybridV2']:
    pv = comparison['PlantVillage_val'][model_name]
    pd = comparison['PlantDoc_test'][model_name]
    print(f"  {model_name:<20} {pv:>7.2f}%  {pd:>7.2f}%")

print(f"\nResults saved to: {CONFIG['results_dir']}/")
print(f"Checkpoint: checkpoints/best_mobilenet_hybrid_v2.pth")
print(f"\n{'='*70}")
print("PHASE 3.5 V2 COMPLETE — LABEL MAPPING FIXED")
print(f"{'='*70}")
