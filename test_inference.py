import torch
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
from preprocess.transform import val_transforms
from models.resnet_model import get_resnet50
from models.mobilenet_model import get_mobilenet_v2

# 1. Ayarlar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 38

# 2. Modelleri Yükle ve Ağırlıkları Enjekte Et
resnet = get_resnet50(num_classes).to(device)
resnet.load_state_dict(torch.load('checkpoints/best_resnet.pth', map_location=device))

mobilenet = get_mobilenet_v2(num_classes).to(device)
mobilenet.load_state_dict(torch.load('checkpoints/best_mobilenet.pth', map_location=device))

resnet.eval()
mobilenet.eval()

# 3. Rastgele Bir Test Fotoğrafı Seç
test_dir = 'data/val'
# sorted() ensures index alignment with ImageFolder, which also sorts classes alphabetically
all_classes = sorted(os.listdir(test_dir))
random_class = random.choice(all_classes)
class_path = os.path.join(test_dir, random_class)
random_image = random.choice(os.listdir(class_path))
image_path = os.path.join(class_path, random_image)

# 4. Görüntüyü İşle
image = Image.open(image_path).convert('RGB')
input_tensor = val_transforms(image).unsqueeze(0).to(device)

# 5. Tahmin Yap
with torch.no_grad():
    res_out = resnet(input_tensor)
    mob_out = mobilenet(input_tensor)
    
    res_pred = torch.argmax(res_out, dim=1).item()
    mob_pred = torch.argmax(mob_out, dim=1).item()

# 6. Sonuçları Görselleştir
plt.figure(figsize=(12, 6))
plt.imshow(image)
# Sınıf isimlerini düzgün göstermek için (Klasör ismini kullanıyoruz)
plt.title(f"GERÇEK: {random_class}\n"
          f"ResNet-50 Tahmini: {all_classes[res_pred]}\n"
          f"MobileNetV2 Tahmini: {all_classes[mob_pred]}")
plt.axis('off')
plt.show()

print(f"Test Tamamlandı! Gerçek Sınıf: {random_class}")