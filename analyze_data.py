import os

# Veri yolunu tanımlayalım
train_path = 'data/train'

# Sınıf listesini alalım (Klasör isimleri)
classes = os.listdir(train_path)
print(f"Toplam Sınıf Sayısı: {len(classes)}")

# Her sınıfta kaç görsel olduğunu sayalım
print("\n--- Sınıf Dağılımı ---")
for c in classes:
    class_path = os.path.join(train_path, c)
    num_images = len(os.listdir(class_path))
    print(f"{c}: {num_images} fotoğraf")