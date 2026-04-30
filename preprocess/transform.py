import torchvision.transforms as transforms

# Modellerin (ResNet/MobileNet) eğitildiği standart ImageNet değerleri
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Eğitim verisi için: Hem boyutlandırma hem de çeşitlilik (Augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),      # Görüntüyü 224x224 yap
    transforms.RandomHorizontalFlip(),  # Sağa-sola çevir (Çeşitlilik)
    transforms.RandomRotation(10),      # 10 derece döndür (Çeşitlilik)
    transforms.ToTensor(),              # Tensör formatına çevir
    transforms.Normalize(mean, std)     # Normalize et
])

# Doğrulama/Test verisi için: Sadece standartlaştırma
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])