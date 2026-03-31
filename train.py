import torch
import torch.nn as nn
import torch.optim as optim
from data_setup import train_loader, val_loader
from models.resnet_model import get_resnet50 
from models.mobilenet_model import get_mobilenet_v2 # Burayı kontrol et
import time
import os # Yeni eklendi: Klasör yönetimi için

# 1. Kayıt Klasörünü Oluştur (Sistemsel ilerleme için şart)
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# ... cihaz ve model yükleme kısımları aynı kalıyor ...

# 4. Gelişmiş Eğitim Fonksiyonu (Checkpoint Destekli)
def train_model(model, criterion, optimizer, num_epochs=3):
    since = time.time() # time burada kullanıldı, uyarı gidecektir

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader if phase == 'train' else val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset if phase == 'train' else val_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset if phase == 'train' else val_loader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # EĞER EN İYİ MODELSE KAYDET (Sistemsel Güvence)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'checkpoints/best_resnet.pth')
                print(">>> En iyi model checkpoints klasörüne kaydedildi!")

    time_elapsed = time.time() - since
    print(f'\nEğitim tamamlandı: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model

# --- EĞİTİMİ BAŞLATAN ANA KISIM ---
if __name__ == "__main__":
    # 1. Cihaz Ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # 2. Modeli Yükle
    num_classes = 38
    print("ResNet-50 Modeli Hazırlanıyor...")
    model = get_resnet50(num_classes).to(device)

    # 3. Kayıp Fonksiyonu ve Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Fonksiyonu Çağır (Eğitimi Başlat)
    print("Eğitim Döngüsü Başlatılıyor...")
    trained_model = train_model(model, criterion, optimizer, num_epochs=3)