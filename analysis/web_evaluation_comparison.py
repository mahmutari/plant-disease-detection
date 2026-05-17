#!/usr/bin/env python
"""
Phase 3.5 Web Validation: Compare 3 Models
Evaluates Original, Fine-tuned, and Hybrid MobileNetV2 models
on the 24 web validation images.
"""

import os
import json
import csv
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

CONFIG = {
    'web_dir':           'test_images/web_validation',
    'output_dir':        'results/web_evaluation',
    'num_classes':       38,
    'image_size':        224,
    'models': {
        'Original':    'checkpoints/best_mobilenet.pth',
        'Fine-tuned':  'checkpoints/best_mobilenet_finetuned.pth',
        'Hybrid_V1':   'checkpoints/best_mobilenet_hybrid.pth',
        'Hybrid_V2':   'checkpoints/best_mobilenet_hybrid_v2.pth',
    },
    'class_names_source': 'data/val',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ═══════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════

def load_model(checkpoint_path, num_classes=38):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

def get_class_names(source_dir='data/val'):
    classes = sorted(os.listdir(source_dir))
    return [c for c in classes if os.path.isdir(os.path.join(source_dir, c))]

def predict_image(model, image_path, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        top3_probs, top3_indices = torch.topk(probabilities, 3)

    predictions = []
    for prob, idx in zip(top3_probs, top3_indices):
        predictions.append({
            'class': class_names[idx.item()],
            'confidence': prob.item()
        })
    return predictions

# ═══════════════════════════════════════════════════════════
# GROUND TRUTH MAPPING
# ═══════════════════════════════════════════════════════════

GROUND_TRUTH = {
    # Easy (6 images, all in-distribution)
    'easy/apple_scab_01.jpg':              ('Apple___Apple_scab', True),
    'easy/apple_scab_02.jpg':              ('Apple___Apple_scab', True),
    'easy/corn_common_rust_01.jpg':        ('Corn_(maize)___Common_rust_', True),
    'easy/potato_late_blight_01.jpg':      ('Potato___Late_blight', True),
    'easy/tomato_early_blight_01.jpg':     ('Tomato___Early_blight', True),
    'easy/tomato_septoria_leaf_01.jpg':    ('Tomato___Septoria_leaf_spot', True),

    # Medium (9 images, 7 in-dist + 2 OOD)
    'medium/apple_scab_field_01.jpg':                ('Apple___Apple_scab', True),
    'medium/corn_common_rust_field_01.jpg':          ('Corn_(maize)___Common_rust_', True),
    'medium/grape_powdery_mildew_field_01.jpg':      ('OOD', False),
    'medium/medium_01_tomato_misclassified_as_squash.png': ('AMBIGUOUS', False),
    'medium/pepper_bacterial_spot_field_01.jpg':     ('Pepper,_bell___Bacterial_spot', True),
    'medium/potato_late_blight_field_01.jpg':        ('Potato___Late_blight', True),
    'medium/potato_late_blight_field_02.jpg':        ('Potato___Late_blight', True),
    'medium/tomato_early_blight_field_01.jpg':       ('Tomato___Early_blight', True),
    'medium/tomato_late_blight_field_01.jpg':        ('Tomato___Late_blight', True),

    # Hard (4 images, 2 in-dist + 2 OOD)
    'hard/apple_fire_blight_dense_01.jpg':           ('OOD', False),
    'hard/apple_fire_blight_dense_02.jpg':           ('OOD', False),
    'hard/potato_late_blight_dense_01.jpg':          ('Potato___Late_blight', True),
    'hard/tomato_early_blight_heavy_01.jpg':         ('Tomato___Early_blight', True),

    # OOD (5 images, all OOD)
    'ood/rose_black_spot_01.jpg':       ('OOD', False),
    'ood/rose_black_spot_02.jpg':       ('OOD', False),
    'ood/sunflower_downy_01.jpg':       ('OOD', False),
    'ood/sunflower_leaf_ood_01.jpg':    ('OOD', False),
    'ood/wheat_leaf_rust_01.jpg':       ('OOD', False),
    'ood/wheat_leaf_rust_02.jpg':       ('OOD', False),
}

# ═══════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════

print("="*70)
print("PHASE 3.5 - WEB VALIDATION COMPARISON")
print("="*70)

print("\nLoading class names...")
class_names = get_class_names(CONFIG['class_names_source'])
print(f"  Found {len(class_names)} classes")

transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

print("\nLoading 3 models...")
loaded_models = {}
for name, path in CONFIG['models'].items():
    if os.path.exists(path):
        print(f"  Loading {name}: {path}")
        loaded_models[name] = load_model(path, CONFIG['num_classes'])
    else:
        print(f"  WARNING: {path} not found, skipping {name}")

print(f"\nScanning web validation directory: {CONFIG['web_dir']}")
web_images = []
for subdir in ['easy', 'medium', 'hard', 'ood']:
    subdir_path = os.path.join(CONFIG['web_dir'], subdir)
    if not os.path.exists(subdir_path):
        print(f"  WARNING: {subdir_path} not found")
        continue
    for fname in sorted(os.listdir(subdir_path)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            rel_path = f"{subdir}/{fname}"
            web_images.append({
                'subdir': subdir,
                'filename': fname,
                'full_path': os.path.join(subdir_path, fname),
                'rel_path': rel_path,
            })

print(f"  Found {len(web_images)} images")

print(f"\nEvaluating {len(web_images)} images with {len(loaded_models)} models...")
print("-" * 70)

all_results = []

for idx, img_info in enumerate(web_images):
    rel_path = img_info['rel_path']
    gt_info = GROUND_TRUTH.get(rel_path, ('UNKNOWN', False))
    true_class, is_in_dist = gt_info

    row = {
        'index': idx + 1,
        'filename': img_info['filename'],
        'category': img_info['subdir'],
        'true_class': true_class,
        'in_distribution': is_in_dist,
    }

    for model_name, model in loaded_models.items():
        try:
            predictions = predict_image(
                model, img_info['full_path'], transform, class_names
            )
            top1 = predictions[0]
            is_correct = is_in_dist and top1['class'] == true_class

            row[f'{model_name}_pred'] = top1['class']
            row[f'{model_name}_conf'] = round(top1['confidence'], 4)
            row[f'{model_name}_correct'] = is_correct if is_in_dist else None
            row[f'{model_name}_top3'] = [p['class'] for p in predictions]
        except Exception as e:
            print(f"  Error on {rel_path} with {model_name}: {e}")
            row[f'{model_name}_pred'] = 'ERROR'
            row[f'{model_name}_conf'] = 0.0
            row[f'{model_name}_correct'] = None

    all_results.append(row)

    if (idx + 1) % 5 == 0:
        print(f"  Processed {idx+1}/{len(web_images)} images...")

print(f"\nProcessing complete!")

# ═══════════════════════════════════════════════════════════
# CALCULATE AGGREGATE METRICS
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print("AGGREGATE METRICS")
print("="*70)

summary = {
    'total_images': len(all_results),
    'in_distribution_count': sum(1 for r in all_results if r['in_distribution']),
    'ood_count': sum(1 for r in all_results if not r['in_distribution']),
    'models': {}
}

for model_name in loaded_models.keys():
    in_dist_correct = sum(
        1 for r in all_results
        if r['in_distribution'] and r.get(f'{model_name}_correct') == True
    )
    in_dist_total = sum(1 for r in all_results if r['in_distribution'])

    in_dist_confs = [
        r[f'{model_name}_conf'] for r in all_results
        if r['in_distribution']
    ]

    ood_high_conf = sum(
        1 for r in all_results
        if not r['in_distribution'] and r[f'{model_name}_conf'] > 0.90
    )
    ood_total = sum(1 for r in all_results if not r['in_distribution'])

    accuracy = (in_dist_correct / in_dist_total * 100) if in_dist_total > 0 else 0
    mean_conf = sum(in_dist_confs) / len(in_dist_confs) if in_dist_confs else 0
    ood_high_conf_rate = (ood_high_conf / ood_total * 100) if ood_total > 0 else 0

    summary['models'][model_name] = {
        'accuracy_in_dist': round(accuracy, 2),
        'correct': in_dist_correct,
        'total_in_dist': in_dist_total,
        'mean_confidence': round(mean_conf, 4),
        'ood_high_conf_count': ood_high_conf,
        'ood_high_conf_rate': round(ood_high_conf_rate, 2),
    }

    print(f"\n{model_name}:")
    print(f"  In-distribution accuracy: {accuracy:.2f}% ({in_dist_correct}/{in_dist_total})")
    print(f"  Mean confidence (in-dist): {mean_conf:.4f}")
    print(f"  OOD high-conf rate (>0.90): {ood_high_conf_rate:.2f}% ({ood_high_conf}/{ood_total})")

# ═══════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════

results_json_path = os.path.join(CONFIG['output_dir'], 'comparison_3models_detailed.json')
with open(results_json_path, 'w') as f:
    json.dump({
        'summary': summary,
        'detailed_results': all_results,
    }, f, indent=2)
print(f"\nDetailed results saved: {results_json_path}")

summary_json_path = os.path.join(CONFIG['output_dir'], 'comparison_3models_summary.json')
with open(summary_json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_json_path}")

csv_path = os.path.join(CONFIG['output_dir'], 'comparison_3models.csv')
if all_results:
    keys = ['index', 'filename', 'category', 'true_class', 'in_distribution']
    for model_name in loaded_models.keys():
        keys.extend([
            f'{model_name}_pred',
            f'{model_name}_conf',
            f'{model_name}_correct',
        ])

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV saved: {csv_path}")

# ═══════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FINAL COMPARISON TABLE")
print("="*70)
print(f"\n{'Model':<15} {'Accuracy':<12} {'Mean Conf':<12} {'OOD >0.90':<12}")
print("-" * 51)

for model_name, metrics in summary['models'].items():
    print(f"{model_name:<15} "
          f"{metrics['accuracy_in_dist']:>6.2f}%     "
          f"{metrics['mean_confidence']:.4f}      "
          f"{metrics['ood_high_conf_rate']:>5.2f}%")

print("\n" + "="*70)
print("PHASE 3.5 WEB VALIDATION COMPARISON COMPLETE")
print("="*70)
