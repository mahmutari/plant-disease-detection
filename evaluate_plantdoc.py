"""
PlantDoc Benchmark Evaluation
==============================
PlantDoc test seti üzerinde MobileNetV2 ve ResNet50 modellerini değerlendirir.
PlantDoc sınıf isimleri → PlantVillage indekslerine manuel eşleme kullanılır.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from collections import defaultdict

from models.mobilenet_model import get_mobilenet_v2
from models.resnet_model import get_resnet50
from preprocess.transform import val_transforms

# ── Sabitler ──────────────────────────────────────────────────────────────────

PLANTDOC_TEST_DIR = "data/plantdoc/PlantDoc-Dataset/test"
CHECKPOINT_DIR    = "checkpoints"
RESULTS_DIR       = "results/plantdoc"
NUM_CLASSES       = 38

# PlantVillage sınıfları (ImageFolder gibi alfabetik sıralı → indeks)
PLANTVILLAGE_CLASSES = sorted([
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
])
PV_IDX = {cls: i for i, cls in enumerate(PLANTVILLAGE_CLASSES)}

# PlantDoc sınıf → PlantVillage sınıfı eşlemesi
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf":                      "Apple___Apple_scab",
    "Apple leaf":                           "Apple___healthy",
    "Apple rust leaf":                      "Apple___Cedar_apple_rust",
    "Bell_pepper leaf":                     "Pepper,_bell___healthy",
    "Bell_pepper leaf spot":                "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf":                       "Blueberry___healthy",
    "Cherry leaf":                          "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot":                  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight":                     "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf":                       "Corn_(maize)___Common_rust_",
    "Peach leaf":                           "Peach___healthy",
    "Potato leaf early blight":             "Potato___Early_blight",
    "Potato leaf late blight":              "Potato___Late_blight",
    "Raspberry leaf":                       "Raspberry___healthy",
    "Soyabean leaf":                        "Soybean___healthy",
    "Squash Powdery mildew leaf":           "Squash___Powdery_mildew",
    "Strawberry leaf":                      "Strawberry___healthy",
    "Tomato Early blight leaf":             "Tomato___Early_blight",
    "Tomato Septoria leaf spot":            "Tomato___Septoria_leaf_spot",
    "Tomato leaf":                          "Tomato___healthy",
    "Tomato leaf bacterial spot":           "Tomato___Bacterial_spot",
    "Tomato leaf late blight":             "Tomato___Late_blight",
    "Tomato leaf mosaic virus":             "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus":             "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf":                     "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf":                           "Grape___healthy",
    "grape leaf black rot":                 "Grape___Black_rot",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# ── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────────

def load_model(model_fn, checkpoint_path, device):
    model = model_fn(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def collect_test_samples(test_dir):
    """(image_path, plantvillage_class_idx) çiftlerini toplar."""
    samples = []
    skipped_classes = []

    for cls_name in sorted(os.listdir(test_dir)):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        pv_cls = PLANTDOC_TO_PLANTVILLAGE.get(cls_name)
        if pv_cls is None:
            skipped_classes.append(cls_name)
            continue

        pv_idx = PV_IDX[pv_cls]
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1] in IMAGE_EXTENSIONS:
                samples.append((os.path.join(cls_dir, fname), pv_idx, cls_name))

    if skipped_classes:
        print(f"[UYARI] Eşleme bulunamayan sınıflar atlandı: {skipped_classes}")

    return samples


@torch.no_grad()
def evaluate(model, samples, device):
    correct = 0
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)
    errors = []

    for img_path, true_idx, cls_name in samples:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        tensor = val_transforms(img).unsqueeze(0).to(device)
        logits = model(tensor)
        pred_idx = torch.argmax(logits, dim=1).item()

        per_class_total[cls_name] += 1
        if pred_idx == true_idx:
            correct += 1
            per_class_correct[cls_name] += 1
        else:
            errors.append({
                "image": os.path.basename(img_path),
                "class": cls_name,
                "true": PLANTVILLAGE_CLASSES[true_idx],
                "pred": PLANTVILLAGE_CLASSES[pred_idx],
            })

    overall_acc = correct / len(samples) if samples else 0.0
    per_class_acc = {
        cls: per_class_correct[cls] / per_class_total[cls]
        for cls in per_class_total
    }
    return overall_acc, per_class_acc, errors


def print_results(model_name, overall_acc, per_class_acc):
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Genel Doğruluk : {overall_acc*100:.2f}%")
    print(f"\n  Sınıf Bazında Doğruluk:")
    for cls, acc in sorted(per_class_acc.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 20)
        print(f"    {cls:<45} {acc*100:5.1f}%  {bar}")


# ── Ana Akış ──────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    print("\n[1/4] Test örnekleri toplanıyor...")
    samples = collect_test_samples(PLANTDOC_TEST_DIR)
    print(f"      {len(samples)} görüntü bulundu.")

    if not samples:
        print("HATA: Hiç test görüntüsü bulunamadı. PLANTDOC_TEST_DIR yolunu kontrol et.")
        return

    print("[2/4] Modeller yükleniyor...")
    mobilenet = load_model(get_mobilenet_v2, f"{CHECKPOINT_DIR}/best_mobilenet.pth", device)
    resnet    = load_model(get_resnet50,    f"{CHECKPOINT_DIR}/best_resnet.pth",    device)

    print("[3/4] MobileNetV2 değerlendiriliyor...")
    mob_acc, mob_per_class, mob_errors = evaluate(mobilenet, samples, device)

    print("[4/4] ResNet50 değerlendiriliyor...")
    res_acc, res_per_class, res_errors = evaluate(resnet, samples, device)

    # ── Çıktı ────────────────────────────────────────────────────────────────
    print_results("MobileNetV2  —  PlantDoc Test Seti", mob_acc, mob_per_class)
    print_results("ResNet50     —  PlantDoc Test Seti", res_acc, res_per_class)

    print(f"\n{'='*60}")
    print(f"  ÖZET KARŞILAŞTIRMA")
    print(f"{'='*60}")
    print(f"  MobileNetV2  : {mob_acc*100:.2f}%")
    print(f"  ResNet50     : {res_acc*100:.2f}%")
    print(f"  Fark         : {abs(mob_acc - res_acc)*100:.2f}pp")
    print(f"  Test seti    : {len(samples)} görüntü, {len(mob_per_class)} sınıf")

    # ── Kayıt ────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "dataset": "PlantDoc",
        "test_images": len(samples),
        "num_classes": len(mob_per_class),
        "mobilenet": {
            "overall_accuracy": round(mob_acc, 4),
            "per_class_accuracy": {k: round(v, 4) for k, v in mob_per_class.items()},
        },
        "resnet50": {
            "overall_accuracy": round(res_acc, 4),
            "per_class_accuracy": {k: round(v, 4) for k, v in res_per_class.items()},
        },
    }
    out_path = os.path.join(RESULTS_DIR, "plantdoc_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Sonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
