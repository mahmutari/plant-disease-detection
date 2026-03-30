import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes):
    # 1. Önceden eğitilmiş (Pre-trained) ResNet-50 modelini yükle
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # 2. Modelin son katmanını (Classifier) kendi sınıf sayımıza göre değiştirelim
    # ResNet-50'nin son katmanındaki giriş sayısı (in_features) 2048'dir.
    in_features = model.fc.in_features
    
    # Kendi sınıf sayımıza (38) uygun yeni bir çıkış katmanı ekliyoruz
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

# Test amaçlı: Modelin doğru kurulup kurulmadığını kontrol edelim
if __name__ == "__main__":
    num_classes = 38
    my_model = get_resnet50(num_classes)
    print(f"ResNet-50 Başarıyla Oluşturuldu!")
    print(f"Çıkış Katmanı Özellikleri: {my_model.fc}")