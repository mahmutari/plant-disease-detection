#!/usr/bin/env python
"""
Phase 3.5: Hybrid Training with Frozen Backbone + Weighted Sampling
Trains MobileNetV2 classifier head on combined PlantVillage + PlantDoc
data using 1:1 weighted sampling to balance the two distributions.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import datasets, transforms, models
from collections import Counter

# ═══════════════════════════════════════════════════════════
# CONFIGURATION (Phase 2 ile tutarlı hyperparameter'lar)
# ═══════════════════════════════════════════════════════════

CONFIG = {
    'pv_train_dir':      'data/train',
    'pd_train_dir':      'data/plantdoc/PlantDoc-Dataset/train',
    'pv_val_dir':        'data/val',
    'pd_test_dir':       'data/plantdoc/PlantDoc-Dataset/test',
    'web_dir':           'test_images/web_validation',
    'output_dir':        'checkpoints',
    'results_dir':       'results/hybrid_training',
    'log_dir':           'logs',

    'num_classes':       38,
    'image_size':        224,
    'batch_size':        8,
    'epochs':            3,
    'learning_rate':     1e-3,
    'num_workers':       0,
    'random_seed':       42,

    'base_checkpoint':   'checkpoints/best_mobilenet.pth',
}

torch.manual_seed(CONFIG['random_seed'])
torch.backends.cudnn.deterministic = True

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs(CONFIG['log_dir'], exist_ok=True)

# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 3.5: HYBRID TRAINING")
print("=" * 70)
print(f"Configuration: {CONFIG}")
print()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

eval_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    normalize,
])

pv_train_dir = CONFIG['pv_train_dir']
if not os.path.exists(pv_train_dir):
    print(f"WARNING: {pv_train_dir} not found, using val dir as PV train")
    pv_train_dir = CONFIG['pv_val_dir']

print("Loading PlantVillage training data...")
pv_train = datasets.ImageFolder(pv_train_dir, transform=train_transform)
print(f"  PV train: {len(pv_train)} images, {len(pv_train.classes)} classes")

print("Loading PlantDoc training data...")
pd_train = datasets.ImageFolder(CONFIG['pd_train_dir'], transform=train_transform)
print(f"  PD train: {len(pd_train)} images, {len(pd_train.classes)} classes")

combined = ConcatDataset([pv_train, pd_train])
print(f"  Combined: {len(combined)} images")

# ═══════════════════════════════════════════════════════════
# WEIGHTED SAMPLING (1:1 PV:PD ratio)
# ═══════════════════════════════════════════════════════════

pv_weight = 1.0 / len(pv_train)
pd_weight = 1.0 / len(pd_train)

sample_weights = (
    [pv_weight] * len(pv_train) +
    [pd_weight] * len(pd_train)
)

total_samples = 2 * len(pv_train)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=total_samples,
    replacement=True,
)

print(f"\n1:1 Weighted Sampling Setup:")
print(f"  PV samples per epoch (expected): ~{total_samples // 2}")
print(f"  PD samples per epoch (expected): ~{total_samples // 2}")
print(f"  Total samples per epoch: {total_samples}")

train_loader = DataLoader(
    combined,
    batch_size=CONFIG['batch_size'],
    sampler=sampler,
    num_workers=CONFIG['num_workers'],
    pin_memory=False,
)

# ═══════════════════════════════════════════════════════════
# MODEL SETUP (Frozen Backbone, Trainable Head)
# ═══════════════════════════════════════════════════════════

print(f"\nModel Setup:")
print(f"  Loading base checkpoint: {CONFIG['base_checkpoint']}")

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
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
print(f"  Frozen parameters: {total_params - trainable_params:,}")

# ═══════════════════════════════════════════════════════════
# TRAINING SETUP
# ═══════════════════════════════════════════════════════════

device = torch.device('cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=CONFIG['learning_rate']
)

# ═══════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════

training_history = []

print(f"\n" + "=" * 70)
print(f"TRAINING START: {CONFIG['epochs']} epochs")
print("=" * 70)

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

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - epoch_start
            progress = (batch_idx + 1) / len(train_loader) * 100
            eta = elapsed / (batch_idx + 1) * (len(train_loader) - batch_idx - 1) / 60
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} "
                  f"({progress:.1f}%) | Loss: {loss.item():.4f} | "
                  f"Acc: {100. * correct / total:.2f}% | ETA: {eta:.1f}min")

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

    checkpoint_path = os.path.join(CONFIG['output_dir'], 'best_mobilenet_hybrid.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_loss,
        'train_accuracy': epoch_acc,
        'config': CONFIG,
    }, checkpoint_path)
    print(f"    Checkpoint saved: {checkpoint_path}")

    with open(os.path.join(CONFIG['results_dir'], 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

total_time = (time.time() - start_time) / 60
print(f"\n" + "=" * 70)
print(f"TRAINING COMPLETE")
print(f"Total time: {total_time:.1f} minutes")
print(f"Final train accuracy: {training_history[-1]['train_accuracy']:.2f}%")
print("=" * 70)

with open(os.path.join(CONFIG['results_dir'], 'training_history.json'), 'w') as f:
    json.dump(training_history, f, indent=2)

print(f"\nResults saved to: {CONFIG['results_dir']}/")
print(f"Checkpoint saved to: {CONFIG['output_dir']}/best_mobilenet_hybrid.pth")

# ═══════════════════════════════════════════════════════════
# EVALUATION ON ALL 3 DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════

print(f"\n" + "=" * 70)
print(f"EVALUATION PHASE")
print("=" * 70)

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


print(f"\nEvaluating on PlantVillage validation...")
pv_val = datasets.ImageFolder(CONFIG['pv_val_dir'], transform=eval_transform)
pv_val_loader = DataLoader(pv_val, batch_size=16, shuffle=False, num_workers=0)
pv_val_acc = evaluate(pv_val_loader, "PlantVillage val")

print(f"\nEvaluating on PlantDoc test...")
pd_test = datasets.ImageFolder(CONFIG['pd_test_dir'], transform=eval_transform)
pd_test_loader = DataLoader(pd_test, batch_size=16, shuffle=False, num_workers=0)
pd_test_acc = evaluate(pd_test_loader, "PlantDoc test")

print(f"\n" + "=" * 70)
print(f"FINAL RESULTS — HYBRID MODEL")
print("=" * 70)
print(f"  PlantVillage val: {pv_val_acc:.2f}%")
print(f"  PlantDoc test:    {pd_test_acc:.2f}%")
print(f"  (Web validation requires manual mapping, do separately)")

comparison = {
    'PlantVillage_val': {
        'Phase2_Original':  97.13,
        'Phase3_FineTuned': 58.96,
        'Phase35_Hybrid':   pv_val_acc,
    },
    'PlantDoc_test': {
        'Phase2_Original':  16.02,
        'Phase3_FineTuned': 30.74,
        'Phase35_Hybrid':   pd_test_acc,
    },
}

with open(os.path.join(CONFIG['results_dir'], 'final_comparison.json'), 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\nFinal comparison saved to: {CONFIG['results_dir']}/final_comparison.json")
print(f"\n" + "=" * 70)
print(f"PHASE 3.5 COMPLETE - HYBRID TRAINING SUCCESSFUL")
print("=" * 70)
