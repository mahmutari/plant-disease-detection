import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from preprocess.transform import val_transforms
from models.resnet_model import get_resnet50
from models.mobilenet_model import get_mobilenet_v2
import os
from PIL import Image

# 1. Ayarlar ve Cihaz Seçimi (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")

# 2. Modelleri Yükle ve Hazırla
num_classes = 38
resnet = get_resnet50(num_classes).to(device)
mobilenet = get_mobilenet_v2(num_classes).to(device)

# Modelleri 'eval' moduna al (Eğitim bitti, test ediyoruz)
resnet.eval()
mobilenet.eval()

# 3. Rastgele Bir Fotoğraf Seç (Test Verisinden)
test_dir = 'data/val'
classes = os.listdir(test_dir)
random_class = random.choice(classes)
class_path = os.path.join(test_dir, random_class)
random_image = random.choice(os.listdir(class_path))
image_path = os.path.join(class_path, random_image)

# 4. Görüntüyü Oku ve Preprocessing Uygula
image = Image.open(image_path).convert('RGB')
input_tensor = val_transforms(image).unsqueeze(0).to(device) # Batch boyutu ekle

# 5. Modellerden Geçir (Inference)
with torch.no_grad(): # Gradyan hesaplama (bellek tasarrufu)
    res_output = resnet(input_tensor)
    mob_output = mobilenet(input_tensor)

# 6. Tahminleri Al (Henüz eğitilmedikleri için tahminler rastgele olacak)
# En yüksek olasılığa sahip sınıfın indeksini al
res_pred = torch.argmax(res_output, dim=1).item()
mob_pred = torch.argmax(mob_output, dim=1).item()

# 7. Sonuçları Görselleştir
plt.figure(figsize=(10, 5))
plt.imshow(image)
plt.title(f"Gerçek Sınıf: {random_class}\n"
          f"ResNet-50 Tahmini: Sınıf {res_pred}\n"
          f"MobileNetV2 Tahmini: Sınıf {mob_pred}")
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"\n--- Son Testinference Başarılı! ---")