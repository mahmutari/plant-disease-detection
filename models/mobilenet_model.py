import torch.nn as nn
from torchvision import models

def get_mobilenet_v2(num_classes):
    # 1. Önceden eğitilmiş MobileNetV2'yi yükle
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    
    # 2. Classifier (Sınıflandırıcı) kısmını değiştirelim
    # MobileNetV2'de son katman 'classifier' adındaki bir dizinin 1. elemanıdır.
    # Giriş özellikleri (in_features) genellikle 1280'dir.
    in_features = model.classifier[1].in_features
    
    # Yeni çıkış katmanını tanımlıyoruz
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

# Test Amaçlı
if __name__ == "__main__":
    num_classes = 38
    my_model = get_mobilenet_v2(num_classes)
    print(f"MobileNetV2 Başarıyla Oluşturuldu!")
    print(f"Çıkış Katmanı Özellikleri: {my_model.classifier[1]}")