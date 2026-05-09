"""
Frozen Backbone Fine-Tuning on PlantDoc Train Set
==================================================

Loads the PlantVillage-trained MobileNetV2 checkpoint, freezes the
convolutional backbone, and fine-tunes only the classifier head
on the PlantDoc training set.

Strategy rationale:
- Backbone preserves PlantVillage feature representations
- Classifier head adapts to mixed-domain output mapping
- 38-class output preserved (no class structure changes)
- Conservative: minimal risk of catastrophic forgetting

Usage:
    python analysis/finetune_plantdoc.py

Outputs:
    checkpoints/best_mobilenet_finetuned.pth
    results/finetuning/training_history.json
    results/finetuning/training_curves.png
"""

import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.mobilenet_model import get_mobilenet_v2
from preprocess.transform import train_transforms, val_transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANTDOC_TRAIN = PROJECT_ROOT / "data" / "plantdoc" / "PlantDoc-Dataset" / "train"
CHECKPOINT_IN = PROJECT_ROOT / "checkpoints" / "best_mobilenet.pth"
CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints" / "best_mobilenet_finetuned.pth"
RESULTS_DIR = PROJECT_ROOT / "results" / "finetuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # Windows
DEVICE = torch.device("cpu")

# PlantDoc -> PlantVillage class mapping
# Modelimiz 38 PlantVillage sınıfı tanır
# PlantDoc 28 sınıfı bunlara map ediliyor
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


class PlantDocDataset(Dataset):
    """
    Custom dataset that maps PlantDoc class folders to PlantVillage indices.
    Modelimiz val klasöründen sorted() ile aldığı indekslere göre tahmin yapıyor.
    """
    def __init__(self, plantdoc_root, plantvillage_classes, transform=None):
        self.transform = transform
        self.samples = []

        # PlantVillage class isimlerini index'e map et (sorted!)
        self.pv_class_to_idx = {cls: idx for idx, cls in enumerate(plantvillage_classes)}

        # PlantDoc klasörlerini gez
        for plantdoc_class in os.listdir(plantdoc_root):
            class_dir = os.path.join(plantdoc_root, plantdoc_class)
            if not os.path.isdir(class_dir):
                continue

            # PlantDoc class -> PlantVillage class -> index
            pv_class = PLANTDOC_TO_PLANTVILLAGE.get(plantdoc_class)
            if pv_class is None:
                print(f"  [SKIP] {plantdoc_class} not in mapping")
                continue

            pv_idx = self.pv_class_to_idx.get(pv_class)
            if pv_idx is None:
                print(f"  [SKIP] {pv_class} not in PlantVillage classes")
                continue

            # Tüm görselleri ekle
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, pv_idx))

        print(f"  Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    print("=" * 70)
    print(" Phase 3.10 — Frozen Backbone Fine-Tuning on PlantDoc")
    print("=" * 70)

    # 1. PlantVillage class isimlerini al (sorted!)
    pv_val_dir = PROJECT_ROOT / "data" / "val"
    plantvillage_classes = sorted(os.listdir(pv_val_dir))
    print(f"\n[1] PlantVillage classes: {len(plantvillage_classes)}")
    assert len(plantvillage_classes) == 38, f"Expected 38, got {len(plantvillage_classes)}"

    # 2. Dataset hazırla
    print(f"\n[2] Loading PlantDoc TRAIN dataset...")
    train_dataset = PlantDocDataset(
        PLANTDOC_TRAIN,
        plantvillage_classes,
        transform=train_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # 3. Model yükle
    print(f"\n[3] Loading pretrained model from {CHECKPOINT_IN.name}...")
    model = get_mobilenet_v2(num_classes=38)
    model.load_state_dict(torch.load(CHECKPOINT_IN, map_location=DEVICE))
    model = model.to(DEVICE)

    # 4. Backbone'u dondur, sadece classifier'ı eğit
    print(f"\n[4] Freezing backbone...")
    for param in model.features.parameters():
        param.requires_grad = False

    # Sadece classifier parametreleri trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")
    print(f"  Frozen params:    {total_params - trainable_params:,}")

    # 5. Optimizer (sadece trainable parametreler için)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    # 6. Eğitim döngüsü
    print(f"\n[5] Starting fine-tuning ({EPOCHS} epochs)...")
    history = {"epoch": [], "train_loss": [], "train_acc": []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Progress (her 30 batch'te)
            if (batch_idx + 1) % 30 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} - Batch {batch_idx+1}/{len(train_loader)} "
                      f"- Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        print(f"\n  >>> Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

    # 7. Modeli kaydet
    print(f"\n[6] Saving fine-tuned model to {CHECKPOINT_OUT.name}...")
    torch.save(model.state_dict(), CHECKPOINT_OUT)

    # 8. Eğitim geçmişini kaydet
    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 9. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history["epoch"], history["train_loss"], "o-", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Fine-tuning Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], history["train_acc"], "o-", color="tab:green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Accuracy (%)")
    ax2.set_title("Fine-tuning Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[7] Training complete!")
    print(f"  Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  Model saved to:       {CHECKPOINT_OUT}")
    print(f"  History saved to:     {RESULTS_DIR / 'training_history.json'}")
    print(f"  Curves saved to:      {RESULTS_DIR / 'training_curves.png'}")
    print("\n" + "=" * 70)
    print(" Next: Run evaluation on PlantVillage val + PlantDoc test + Web")
    print("=" * 70)


if __name__ == "__main__":
    main()
