import os
import random
import matplotlib.pyplot as plt
import cv2

def show_random_samples(base_path, num_samples=5):
    classes = os.listdir(base_path)
    plt.figure(figsize=(15, 10))
    
    for i in range(num_samples):
        # Rastgele bir sınıf seç
        random_class = random.choice(classes)
        class_path = os.path.join(base_path, random_class)
        
        # O sınıftan rastgele bir fotoğraf seç
        random_image = random.choice(os.listdir(class_path))
        image_path = os.path.join(class_path, random_image)
        
        # Görüntüyü oku ve renklerini düzelt (OpenCV BGR okur, Matplotlib RGB bekler)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ekrana bas
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(random_class.split('___')[-1], fontsize=8) # Sadece hastalık ismini yaz
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Fonksiyonu çalıştıralım
show_random_samples('data/train')