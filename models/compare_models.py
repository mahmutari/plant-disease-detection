from resnet_model import get_resnet50
from mobilenet_model import get_mobilenet_v2

def count_parameters(model):
    # Modeldeki eğitilebilir parametre sayısını hesaplar
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_classes = 38
resnet = get_resnet50(num_classes)
mobilenet = get_mobilenet_v2(num_classes)

print("--- Model Kıyaslama Analizi ---")
print(f"ResNet-50 Parametre Sayısı  : {count_parameters(resnet):,}")
print(f"MobileNetV2 Parametre Sayısı: {count_parameters(mobilenet):,}")

# Küçük bir mühendislik çıkarımı
diff = count_parameters(resnet) / count_parameters(mobilenet)
print(f"\nResNet-50, MobileNetV2'den yaklaşık {diff:.1f} kat daha büyük bir model.")