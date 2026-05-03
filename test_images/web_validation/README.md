# Web Interface — External Validation Test Protocol

**Project:** Plant Disease Detection — Senior Design Project II  
**Institution:** Sakarya University, Software Engineering  
**Authors:** Mahmut Arı, Samet Kavlan  
**Phase:** 3 — Step 4C (Streamlit + Grad-CAM UI Validation)  
**Date:** May 2026

---

## 1. Purpose

This directory holds images used to evaluate the deployed Streamlit web application
(`app.py`) under realistic, non-training conditions. The goal is to assess:

1. **Prediction correctness** — does the model generalise beyond the PlantVillage dataset?
2. **Confidence calibration** — is high confidence correlated with correct predictions?
3. **Grad-CAM spatial validity** — does the heatmap attend to disease-relevant leaf regions?
4. **UI robustness** — does the interface degrade gracefully on unexpected input?

All images in this directory are **external** — they must not appear in
`data/train/` or `data/val/`. Results obtained here are suitable for inclusion
in the Final Report as evidence of real-world generalisability.

---

## 2. Directory Structure

```
test_images/web_validation/
├── easy/       # Lab-quality: single leaf, white/uniform background
├── medium/     # Field-quality: leaf in foreground, natural background
├── hard/       # Challenging: multiple leaves, occlusion, complex background
├── ood/        # Out-of-distribution: unsupported species or non-plant objects
└── README.md   # This file
```

---

## 3. Category Specifications

### 3.1 `easy/` — Lab-Condition Images

**Description:**  
Images that closely resemble PlantVillage dataset conditions: single isolated
leaf against a white or uniform background, well-lit, centred, no occlusion.

**Target species/diseases (representative, not exhaustive):**

| Class | Notes |
|---|---|
| `Tomato___Early_blight` | Concentric ring lesions clearly visible |
| `Potato___Late_blight` | Brown necrotic patches, white mycelium edge |
| `Corn_(maize)___Common_rust_` | Orange pustule clusters on leaf surface |
| `Apple___Apple_scab` | Olive-green to brown lesion spots |
| `Grape___healthy` | Full leaf blade, no visible symptoms |
| `Tomato___healthy` | Green, intact leaf, uniform colour |

**Success criteria:**
- Top-1 accuracy ≥ 90% on easy images
- Mean top-1 confidence ≥ 0.85
- Grad-CAM activation concentrated on leaf tissue (not background)
- No prediction should fall below 0.60 confidence on clear single-leaf images

**Grad-CAM expectation:**  
Tight, lesion-focused heatmap. Peak activation should coincide with the
visible symptom region (e.g., necrotic spots, rust pustules). Background
pixels should receive near-zero activation.

---

### 3.2 `medium/` — Field-Condition Images

**Description:**  
Images captured in natural agricultural settings. The target leaf is
in the foreground but may have a green/brown field background, partial
shadow, or mild occlusion by adjacent leaves. Representative of
smartphone photos taken by a farmer.

**Target scenarios:**

| Scenario | Expected behaviour |
|---|---|
| Single symptomatic leaf, green background | Correct top-1, conf > 0.70 |
| Leaf with dew or moisture on surface | Slight confidence drop acceptable |
| Partial leaf (>60% visible) | Correct top-1, lower confidence |
| Mixed lighting (partial shade) | Correct top-1 or correct within top-3 |
| Multiple same-species leaves | Correct top-1 if dominant leaf is symptomatic |

**Success criteria:**
- Top-1 accuracy ≥ 75% on medium images
- Top-3 accuracy ≥ 90% (correct class appears in at least one of the three predictions)
- Confidence badge correctly reflects uncertainty:
  - Predictions with conf < 0.70 should receive the amber "Moderate confidence" warning
- Grad-CAM should still prefer leaf tissue over background, though diffusion
  is acceptable when multiple leaves are present

**Grad-CAM expectation:**  
Broader activation across the leaf area than in `easy/`. Some background
activation is expected and acceptable. The primary concern is that the
heatmap does *not* concentrate exclusively on non-leaf regions (sky, soil,
stems without symptoms).

---

### 3.3 `hard/` — Challenging Conditions

**Description:**  
Images where correct prediction is genuinely difficult: dense canopy shots,
multiple overlapping leaves from different species, strong shadows, motion
blur, or severe image compression. These cases probe the limits of the
current single-leaf patch classification paradigm.

**Target scenarios:**

| Scenario | Expected behaviour |
|---|---|
| Two leaves from different species in frame | Unpredictable — document which leaf dominates |
| Late-stage disease (necrosis > 70% of leaf) | Possible misclassification — acceptable |
| Image taken at steep angle (>45° tilt) | Confidence drop expected |
| Heavy shadow over lesion area | Lower confidence, possible wrong prediction |
| JPEG artefacts (quality < 40%) | Document if artefacts cause hallucinated class |
| Small leaf region (<30% of frame) | Model may attend to background; document |

**Success criteria (relaxed):**
- Top-3 accuracy ≥ 60%
- When top-1 is wrong, confidence should ideally be < 0.70 (model uncertainty
  should reflect prediction difficulty — see Grad-CAM Analysis, Section 5, Finding 5)
- Grad-CAM diffusion is expected; document any cases where activation is
  entirely on background (potential shortcut learning indicator)

**Grad-CAM expectation:**  
Diffuse maps are acceptable and expected. The key diagnostic question is:
does the model attend to *some* leaf region, even if not precisely to the
lesion? Fully background-concentrated activation on a hard image is a
noteworthy finding worth reporting.

---

### 3.4 `ood/` — Out-of-Distribution

**Description:**  
Images that fall outside the 38-class PlantVillage taxonomy. The model
will always predict one of the 38 classes (softmax does not produce a
"none of the above" output). These images test confidence calibration:
ideally, OOD images should receive *lower* confidence than valid in-distribution
images.

**Sub-categories and expected predictions:**

| Sub-category | Example | Expected softmax behaviour |
|---|---|---|
| Unsupported plant species | Sunflower leaf, wheat, rice | Arbitrary class, usually conf < 0.70 |
| Healthy leaf of unsupported species | Basil, mint, fern | Predicts nearest healthy class |
| Non-leaf plant part | Fruit, stem, root | Conf typically < 0.50 |
| Non-plant object | Soil sample, tree bark, fabric | Arbitrary, often low conf |
| Human-made artefact | Paper, painted surface, tile pattern | Conf 0.30–0.90 (unpredictable) |
| Adversarial-like image | Solid colour image, noise | May produce spuriously high conf |

**Success criteria:**
- Document the predicted class and confidence for *every* OOD image
- Flag any OOD image that receives conf > 0.90 as a **calibration failure**
  (model is overconfident on OOD input)
- The UI red badge ("Low confidence") should appear for the majority of OOD images
- Grad-CAM for OOD inputs is informative: if a noise image produces a
  structured heatmap, it indicates the model is responding to low-level
  texture artefacts

**Known limitation to document:**  
The current architecture is a closed-set classifier. It cannot abstain.
Adding an entropy-based reject option (e.g., reject if `H(p) > threshold`)
is listed as a future work item and can be referenced from these OOD results.

---

## 4. Test Execution Protocol

### 4.1 Procedure

1. Start the Streamlit server:
   ```
   streamlit run app.py
   ```
2. Open `http://localhost:8501` in a browser.
3. For each test image, record the following in the results table (Section 5):
   - Filename
   - Category (`easy` / `medium` / `hard` / `ood`)
   - True class (or `OOD` if not in the 38-class taxonomy)
   - Predicted class (top-1)
   - Confidence (from the metric card)
   - Top-3 contains true class? (Y/N)
   - UI badge shown (`High` / `Moderate` / `Low`)
   - Grad-CAM quality (subjective: `Focused` / `Diffuse` / `Background-only`)
   - Notes (unusual behaviour, interesting heatmap pattern, etc.)

### 4.2 Minimum Image Counts

| Category | Minimum | Recommended |
|---|---|---|
| `easy/` | 5 | 10 |
| `medium/` | 5 | 10 |
| `hard/` | 3 | 6 |
| `ood/` | 4 | 8 |
| **Total** | **17** | **34** |

---

## 5. Results Template

Copy this table into your test session notes or a spreadsheet:

| # | File | Category | True Class | Predicted (Top-1) | Conf | Top-3 Hit | Badge | Grad-CAM | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | |
| 2 | | | | | | | | | |
| … | | | | | | | | | |

**Aggregate metrics to compute after testing:**

```
Easy     top-1 accuracy  = correct_easy / total_easy
Medium   top-1 accuracy  = correct_medium / total_medium
Medium   top-3 accuracy  = top3_hit_medium / total_medium
Hard     top-3 accuracy  = top3_hit_hard / total_hard
OOD      high-conf rate  = (conf > 0.90 count) / total_ood
Overall  mean confidence (in-distribution only)
```

---

## 6. Grad-CAM Quality Rubric

Use the following rubric when assigning a Grad-CAM quality score:

| Score | Label | Description |
|---|---|---|
| 3 | **Focused** | Peak activation on lesion or diagnostic region; background suppressed |
| 2 | **Diffuse** | Broad activation across leaf blade; background partially activated |
| 1 | **Background-only** | Activation concentrated outside leaf tissue |
| 0 | **Uninformative** | Near-uniform activation across entire image |

A score of 3 or 2 is considered a passing result for report purposes.
Score 1 or 0 on an easy image constitutes a noteworthy finding and should
be discussed in the Final Report Limitations section.

---

## 7. Connections to Final Report

| Finding | Report section | Evidence source |
|---|---|---|
| Easy-image top-1 accuracy | Section 4 (Results) | `easy/` aggregate |
| Field-condition generalisation | Section 5.1 (Discussion) | `medium/` aggregate |
| Failure modes on hard images | Section 5.2 (Limitations) | `hard/` notes |
| OOD calibration failure cases | Section 5.2 (Limitations) | `ood/` high-conf entries |
| Grad-CAM quality on real images | Section 4 (Interpretability) | Quality rubric scores |
| Softmax closed-set limitation | Section 5.2 (Limitations) | `ood/` all entries |

---

*Protocol version: 1.0 — Phase 3, Step 4C*  
*Model evaluated: MobileNetV2 — `checkpoints/best_mobilenet.pth`*  
*Validation set baseline: 97.13% accuracy (17,572 images, PlantVillage)*

---

## 8. Validation Test Results

**Test date:** 2026-05-03  
**Images tested:** 25 (6 easy · 9 medium · 4 hard · 6 ood)  
**In-distribution images:** 16  
**Model:** MobileNetV2 — `checkpoints/best_mobilenet.pth`

### 8.1 Full Results Table

| # | File | Cat | True Class | Predicted (Top-1) | Conf | Top-3 | Badge | Grad-CAM | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | apple_scab_01.jpg | easy | Apple — Apple Scab | Strawberry — Leaf Scorch | 0.999 | N | High | Uninformative | High-conf wrong; texture confused with Strawberry scorch |
| 2 | apple_scab_02.jpg | easy | Apple — Apple Scab | Corn — Northern Leaf Blight | 0.703 | N | Moderate | Background-only | Low resolution; lesion pattern ambiguous |
| 3 | corn_common_rust_01.jpg | easy | Corn — Common Rust | Tomato — Healthy | 0.992 | N | High | Uninformative | Orange pustules misread as healthy tomato; domain shift |
| 4 | potato_late_blight_01.jpg | easy | Potato — Late Blight | Tomato — Late Blight | 0.498 | N | Low | Background-only | Cross-host confusion (Phytophthora); known ambiguity |
| 5 | tomato_early_blight_01.jpg | easy | Tomato — Early Blight | Peach — Bacterial Spot | 0.857 | N | High | Uninformative | Concentric rings missed; attends to background |
| 6 | tomato_septoria_leaf_01.jpg | easy | Tomato — Septoria Leaf Spot | Peach — Healthy | 0.976 | N | High | Background-only | Small white spots not recognized; overfit to lab images |
| 7 | apple_scab_field_01.jpg | medium | Apple — Apple Scab | Strawberry — Leaf Scorch | 0.651 | N | Moderate | Background-only | Field background confuses model; same wrong class as #1 |
| 8 | corn_common_rust_field_01.jpg | medium | Corn — Common Rust | Pepper — Bacterial Spot | 0.525 | N | Moderate | Background-only | Low confidence reflects genuine uncertainty |
| 9 | grape_powdery_mildew_field_01.jpg | medium | OOD (Grape Powdery Mildew) | Strawberry — Leaf Scorch | 0.987 | N/A | High | Background-only | OOD disease; high-conf calibration failure |
| 10 | medium_01_tomato_misclassified_as_squash.png | medium | Tomato — Early Blight | Squash — Powdery Mildew | 0.871 | N | High | Uninformative | Confirmed misclassification; white lesion texture misleads |
| 11 | pepper_bacterial_spot_field_01.jpg | medium | Pepper — Bacterial Spot | Pepper — Bacterial Spot | 0.989 | Y | High | Background-only | **Only correct prediction** in external test set |
| 12 | potato_late_blight_field_01.jpg | medium | Potato — Late Blight | Corn — Cercospora Leaf Spot | 0.582 | N | Moderate | Uninformative | Low confidence; complex background |
| 13 | potato_late_blight_field_02.jpg | medium | Potato — Late Blight | Orange — Haunglongbing | 0.241 | N | Low | Background-only | Lowest confidence in set; model uncertain (correct signal) |
| 14 | tomato_early_blight_field_01.jpg | medium | Tomato — Early Blight | Corn — Healthy | 0.994 | N | High | Uninformative | Very high conf on wrong class; severe domain shift |
| 15 | tomato_late_blight_field_01.jpg | medium | Tomato — Late Blight | Corn — Healthy | 0.709 | N | Moderate | Uninformative | Field photo; lesion area small relative to background |
| 16 | apple_fire_blight_dense_01.jpg | hard | OOD (Apple Fire Blight) | Pepper — Bacterial Spot | 0.695 | N/A | Moderate | Background-only | Dense branch scene; OOD disease |
| 17 | apple_fire_blight_dense_02.jpg | hard | OOD (Apple Fire Blight) | Grape — Esca (Black Measles) | 1.000 | N/A | High | Background-only | Perfect-confidence OOD; severe calibration failure |
| 18 | potato_late_blight_dense_01.jpg | hard | Potato — Late Blight | Corn — Cercospora Leaf Spot | 0.499 | N | Low | Uninformative | Multiple leaves; low conf partially reflects difficulty |
| 19 | tomato_early_blight_heavy_01.jpg | hard | Tomato — Early Blight | Pepper — Healthy | 0.776 | N | Moderate | Uninformative | Late-stage necrosis; original class features destroyed |
| 20 | rose_black_spot_01.jpg | ood | OOD (Rose) | Strawberry — Leaf Scorch | 1.000 | N/A | High | Uninformative | Perfect-conf OOD; leaf shape similar to Strawberry |
| 21 | rose_black_spot_02.jpg | ood | OOD (Rose) | Apple — Cedar Apple Rust | 0.669 | N/A | Moderate | Uninformative | Lower conf; rounded dark spots similar to cedar rust |
| 22 | sunflower_downy_mildew_ood_01.jpg | ood | OOD (Sunflower) | Tomato — Early Blight | 0.984 | N/A | High | Background-only | Mildew yellowing confused with early blight lesions |
| 23 | sunflower_leaf_ood_01.jpg | ood | OOD (Sunflower) | Corn — Northern Leaf Blight | 0.986 | N/A | High | Background-only | Long leaf texture triggers NLB class |
| 24 | wheat_leaf_rust_01.jpg | ood | OOD (Wheat) | Strawberry — Leaf Scorch | 0.986 | N/A | High | Uninformative | Rust pustules visually similar to Strawberry scorch |
| 25 | wheat_leaf_rust_02.jpg | ood | OOD (Wheat) | Peach — Healthy | 0.792 | N/A | Moderate | Uninformative | Thin grass-like blade; no matching class |

### 8.2 Aggregate Metrics

| Metric | Value | Target | Pass? |
|---|---|---|---|
| Easy top-1 accuracy | 0.0% (0/6) | ≥ 90% | ❌ |
| Easy mean confidence | 0.838 | ≥ 0.85 | ❌ |
| Medium top-1 accuracy | 12.5% (1/8) | ≥ 75% | ❌ |
| Medium top-3 accuracy | 12.5% (1/8) | ≥ 90% | ❌ |
| Medium mean confidence | 0.695 | — | — |
| Hard top-3 accuracy | 50.0% (1/2) | ≥ 60% | ❌ |
| OOD high-conf rate (>0.90) | 66.7% (4/6) | Low (calibration) | ❌ |
| Overall in-dist top-1 | 6.2% (1/16) | — | — |
| Overall in-dist top-3 | 12.5% (2/16) | — | — |

### 8.3 Grad-CAM Quality Summary

| Category | Focused (3) | Diffuse (2) | Background-only (1) | Uninformative (0) |
|---|---|---|---|---|
| easy (6) | 0 | 0 | 2 | 4 |
| medium (9) | 0 | 0 | 5 | 4 |
| hard (4) | 0 | 0 | 2 | 2 |
| ood (6) | 0 | 0 | 2 | 4 |
| **Total (25)** | **0** | **0** | **11** | **14** |

No image received a Focused or Diffuse score. All Grad-CAM activations were either spatially diffuse across the image or concentrated on non-discriminative regions. This contrasts sharply with the PlantVillage validation results (focused lesion activation in all sanity-check cases) and is a direct consequence of the domain shift discussed in Section 8.4.

### 8.4 Key Findings

#### Finding 1: Severe Domain Gap — PlantVillage vs. Real-World Images

The model achieves **97.13% accuracy on the PlantVillage validation set** but only **6.2% top-1 accuracy on external images**. This is the most significant finding of the external validation test and aligns with the well-known limitation of PlantVillage-trained models reported in the literature (Mohanty et al., 2016: 99.35% on dataset, ~31% on field images).

**Root cause:** PlantVillage was created under controlled laboratory conditions — isolated leaves against white backgrounds, consistent lighting, single-leaf framing. The model has learned to exploit these dataset-level features (leaf shape against uniform background, consistent scale, centered framing) rather than purely disease-specific visual patterns. When these conditions change, the learned features no longer apply.

#### Finding 2: Overconfident Wrong Predictions (Calibration Failure)

Mean confidence on external in-distribution images is **0.741**, while top-1 accuracy is only **6.2%**. The model is confidently wrong. Examples:
- `corn_common_rust_01.jpg`: 99.2% confidence → wrong class
- `tomato_early_blight_field_01.jpg`: 99.4% confidence → wrong class
- `rose_black_spot_01.jpg` (OOD): 100.0% confidence → arbitrary class

This indicates that the confidence values produced by this model are **not calibrated** for out-of-distribution inputs and should not be interpreted as meaningful uncertainty estimates in real-world deployment.

#### Finding 3: Single Correct Prediction — Pepper Bacterial Spot

The one correct prediction (`pepper_bacterial_spot_field_01.jpg`, conf=0.989) is noteworthy: Pepper Bacterial Spot produces small, water-soaked angular lesions that are visually distinctive enough to survive the domain shift. The fact that this class — with its characteristic inter-veinal lesion pattern — generalises better than others (e.g., Tomato Early Blight, which also has distinctive concentric rings) suggests that lesion distinctiveness alone is not sufficient for generalisation; the leaf shape and background context are also critical factors the model relies on.

#### Finding 4: OOD Calibration Failures

**4 out of 6 OOD images** received confidence > 0.90, including two cases of near-perfect confidence (1.000 for rose, 1.000 for apple fire blight). A well-calibrated model should assign lower probability to OOD inputs. The absence of an entropy-based reject mechanism means the Streamlit UI will display a "High confidence" green badge on these misclassified OOD images, which could mislead a real user.

#### Finding 5: Grad-CAM Collapse on External Images

All 25 external images received a Grad-CAM quality score of 0 (Uninformative) or 1 (Background-only). This is in stark contrast to the PlantVillage validation results where all sanity-check images received focused, lesion-centred heatmaps. The collapse of Grad-CAM spatial coherence on external images confirms that the model's internal feature representations are not transferring: it is responding to global image statistics (colour distribution, background texture) rather than localised disease features.

### 8.5 Implications for Final Report

| Section | Evidence from this test |
|---|---|
| Section 5.2 (Limitations) | Domain gap: 97.13% → 6.2% on external images. Cite Mohanty et al. (2016) as precedent. |
| Section 5.2 (Limitations) | Calibration failure: model is overconfident on wrong and OOD predictions. |
| Section 5.2 (Limitations) | Grad-CAM collapse confirms feature non-transferability to field images. |
| Section 6 (Future Work) | Add temperature scaling or entropy-based rejection; fine-tune on field images; collect real-world training data. |
| Section 4 (Results) | Note that 97.13% validation accuracy was obtained under matched lab conditions — a necessary caveat when reporting model performance. |

---

*Test executed: 2026-05-03 — Phase 3, Step 4C*  
*25 images tested across 4 categories (easy/medium/hard/ood)*  
*Script: automated inference via `app.py` prediction pipeline + GradCAM from `analysis/gradcam.py`*
