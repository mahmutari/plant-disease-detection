import os
from torchvision import datasets
from torch.utils.data import DataLoader
from preprocess.transform import train_transforms, val_transforms

# 1. Klasör yollarını tanımlayalım
train_dir = 'data/train'
val_dir = 'data/val'

# 2. Dataset nesnelerini oluşturalım (Klasör yapısını sınıflara dönüştürür)
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

# 3. DataLoader'ları oluşturalım
# Batch size: Her adımda kaç fotoğraf işlenecek (Genelde 32 veya 64 seçilir)
BATCH_SIZE = 32

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"Eğitim batch sayısı: {len(train_loader)}")
print(f"Doğrulama batch sayısı: {len(val_loader)}")