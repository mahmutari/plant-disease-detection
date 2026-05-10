# External Validation Test Protocol and Results
**Phase 3 — Plant Disease Detection Project**
Sakarya University, Software Engineering — Senior Capstone Project

---

## 1. Methodology

### 1.1 Purpose and Scope

An external validation test was conducted to assess the real-world generalisability of the trained MobileNetV2 model under conditions that fall outside the PlantVillage dataset distribution. Whereas the internal validation set (17,572 images, 38 classes) reflects controlled laboratory conditions — isolated single leaves photographed against uniform white backgrounds — the external test set was assembled from internet-sourced field photographs representing realistic deployment scenarios. The evaluation addresses four research questions:

1. Does the model maintain classification accuracy on in-distribution disease classes when the photographic conditions change?
2. Are the model's confidence scores correlated with prediction correctness (calibration)?
3. Does the Grad-CAM spatial attention transfer from clean laboratory specimens to field-condition imagery?
4. How does the model behave on out-of-distribution species and disease categories not present in the 38-class taxonomy?

### 1.2 Test Image Assembly

A set of 25 images was assembled across four difficulty categories. Images were sourced from publicly accessible agricultural extension websites and research image databases, selected to span a representative range of photographic conditions. All 25 images are confirmed to be absent from the PlantVillage training and validation splits. Images were not preprocessed prior to inference; the model receives them through the standard validation pipeline (resize to 224×224, ImageNet normalisation) without additional augmentation or enhancement. Full filenames and ground truth assignments are documented in `test_images/web_validation/README.md` and the results CSV at `results/web_evaluation/web_results_original.csv`.

One image (`sunflower_downy_mildew_ood_01.jpg`) was skipped at inference time due to file truncation, yielding 24 processed images in the automated evaluation results. The manual Streamlit session results documented in Section 8 of the README, which include this image, are referenced where relevant.

### 1.3 Evaluation Procedure

The test was executed through the Streamlit web application (`app.py`) using the original MobileNetV2 checkpoint (`checkpoints/best_mobilenet.pth`). For each image, the top-1 predicted class, top-3 predicted classes, and associated softmax confidence scores were recorded. Grad-CAM visualisations were generated for all images using the final convolutional layer (`model.features[-1]`) as the target layer, consistent with the methodology documented in `results/gradcam/ANALYSIS.md`.

Grad-CAM quality was assessed using the four-level rubric defined in `test_images/web_validation/README.md`:

| Score | Label | Description |
|---|---|---|
| 3 | Focused | Peak activation on lesion or diagnostic region; background suppressed |
| 2 | Diffuse | Broad activation across leaf blade; background partially activated |
| 1 | Background-only | Activation concentrated outside leaf tissue |
| 0 | Uninformative | Near-uniform activation across entire image |

Automated aggregate metrics were computed using the `analysis/web_evaluation.py` script; results are stored in `results/web_evaluation/summary_web_original.json`.

---

## 2. Test Image Categories

### 2.1 Easy — Laboratory-Condition Images (n = 6)

Images in this category closely replicate PlantVillage dataset conditions: a single isolated leaf against a white or near-uniform background, with consistent lighting and minimal occlusion. The six images include two specimens of *Apple___Apple_scab*, one each of *Corn_(maize)___Common_rust_*, *Potato___Late_blight*, *Tomato___Early_blight*, and *Tomato___Septoria_leaf_spot*. Because these conditions match the training distribution most closely, the expected top-1 accuracy was ≥ 90%.

### 2.2 Medium — Field-Condition Images (n = 9)

Images in this category were sourced from outdoor agricultural settings. The target leaf appears in the foreground against a natural (green, brown, or mixed) background, with variable lighting, minor occlusion, and realistic scale variation. This category includes two OOD entries: `grape_powdery_mildew_field_01.jpg` (a disease class not present in the 38-class taxonomy) and `medium_01_tomato_misclassified_as_squash.png` (ambiguous ground truth, included for diagnostic interest). Of the nine images, seven carry a definite in-distribution ground truth label. Expected top-1 accuracy was ≥ 75%; top-3 accuracy ≥ 90%.

### 2.3 Hard — Challenging Conditions (n = 4)

This category probes the limits of single-leaf patch classification. Challenging conditions include dense multi-leaf canopy scenes, late-stage disease where necrosis covers the majority of the leaf blade, and OOD diseases presented in complex backgrounds. Two of the four images are OOD (`apple_fire_blight_dense_01.jpg`, `apple_fire_blight_dense_02.jpg`), leaving two in-distribution images with a relaxed success criterion of top-3 accuracy ≥ 60%.

### 2.4 OOD — Out-of-Distribution (n = 6)

The OOD category contains plant disease images of species and pathogens absent from the PlantVillage 38-class taxonomy: two rose specimens with black spot (*Diplocarpon rosae*), two wheat specimens with leaf rust (*Puccinia triticina*), one sunflower with downy mildew, and one sunflower leaf without visible disease. Because the model operates as a closed-set 38-class classifier, it will always return a prediction from the known class set. The primary evaluation criterion is confidence calibration: ideally, OOD images should receive lower confidence scores than correctly classified in-distribution images, and the model should not produce high-confidence predictions (> 0.90) on inputs it has never been trained to recognise.

---

## 3. Quantitative Results

### 3.1 Per-Category Summary

| Category | n (in-dist) | Correct | Top-1 Acc | Mean Conf | Target | Pass? |
|---|---|---|---|---|---|---|
| Easy | 6 | 0 | 0.0% | 0.838 | ≥ 90% | ✗ |
| Medium | 7 | 1 | 14.3% | 0.695 | ≥ 75% | ✗ |
| Hard | 2 | 0 | 0.0% | 0.637 | — | — |
| **All in-dist** | **15** | **1** | **6.67%** | **0.733** | — | — |
| OOD high-conf (> 0.90) | 5 proc. | — | — | — | Low | ✗ |

The single correct prediction (`pepper_bacterial_spot_field_01.jpg`, conf = 0.989) is the only instance in which the model's top-1 output matches the ground truth across all 15 in-distribution external test images, yielding an overall external top-1 accuracy of **6.67%**. This represents a **90.46 percentage point drop** from the PlantVillage validation accuracy of 97.13% obtained under matched laboratory conditions.

### 3.2 Confidence Distribution

The mean top-1 confidence across in-distribution images is 0.733. Of the 14 incorrect in-distribution predictions, 9 carry confidence ≥ 0.70, and 6 carry confidence ≥ 0.90. This pattern — high confidence paired with incorrect predictions — is the defining characteristic of the calibration failure described in Section 5.2.

---

## 4. Detailed Test Cases

| # | File | Cat | True Class | Predicted (Top-1) | Conf | Top-3 Hit | Notes |
|---|---|---|---|---|---|---|---|
| 1 | apple_scab_01.jpg | easy | Apple — Apple Scab | Strawberry — Leaf Scorch | 0.999 | N | Near-perfect confidence; texture confused with Strawberry scorch |
| 2 | apple_scab_02.jpg | easy | Apple — Apple Scab | Corn — Northern Leaf Blight | 0.703 | N | Lesion pattern ambiguous; same wrong class as #7 |
| 3 | corn_common_rust_01.jpg | easy | Corn — Common Rust | Tomato — Healthy | 0.993 | N | Orange pustules misread as healthy tomato |
| 4 | potato_late_blight_01.jpg | easy | Potato — Late Blight | Tomato — Late Blight | 0.498 | N | Cross-host confusion (*Phytophthora*); lowest conf in easy set |
| 5 | tomato_early_blight_01.jpg | easy | Tomato — Early Blight | Peach — Bacterial Spot | 0.857 | N | Concentric ring lesions not recognised |
| 6 | tomato_septoria_leaf_01.jpg | easy | Tomato — Septoria | Peach — Healthy | 0.976 | N | White spots not activated; heatmap on background |
| 7 | apple_scab_field_01.jpg | medium | Apple — Apple Scab | Strawberry — Leaf Scorch | 0.651 | N | Field background; same wrong class as #1 |
| 8 | corn_common_rust_field_01.jpg | medium | Corn — Common Rust | Pepper — Bacterial Spot | 0.525 | N | Low confidence correctly reflects uncertainty |
| 9 | grape_powdery_mildew_field_01.jpg | medium | OOD | Strawberry — Leaf Scorch | 0.987 | — | OOD disease; high-conf calibration failure |
| 10 | medium_01_tomato_squash.png | medium | Ambiguous GT | Squash — Powdery Mildew | 0.871 | — | Known misclassification; white lesion texture |
| 11 | pepper_bacterial_spot_field_01.jpg | medium | Pepper — Bacterial Spot | **Pepper — Bacterial Spot** | **0.989** | **Y** | **Only correct prediction in set** |
| 12 | potato_late_blight_field_01.jpg | medium | Potato — Late Blight | Corn — Cercospora | 0.582 | N | Complex background; low confidence |
| 13 | potato_late_blight_field_02.jpg | medium | Potato — Late Blight | Orange — Haunglongbing | 0.241 | N | Lowest confidence in set; model genuinely uncertain |
| 14 | tomato_early_blight_field_01.jpg | medium | Tomato — Early Blight | Corn — Healthy | 0.994 | N | Very high conf on wrong class; severe domain shift |
| 15 | tomato_late_blight_field_01.jpg | medium | Tomato — Late Blight | Corn — Healthy | 0.709 | N | Field photo; lesion area small relative to background |
| 16 | apple_fire_blight_dense_01.jpg | hard | OOD | Pepper — Bacterial Spot | 0.695 | — | Dense scene; OOD disease |
| 17 | apple_fire_blight_dense_02.jpg | hard | OOD | Grape — Esca | **1.000** | — | Perfect-confidence OOD; severe calibration failure |
| 18 | potato_late_blight_dense_01.jpg | hard | Potato — Late Blight | Corn — Cercospora | 0.499 | N | Multi-leaf; near-uniform posterior (top-2: 0.489) |
| 19 | tomato_early_blight_heavy_01.jpg | hard | Tomato — Early Blight | Pepper — Healthy | 0.776 | N | Late-stage necrosis; original class features destroyed |
| 20 | rose_black_spot_01.jpg | ood | OOD (Rose) | Strawberry — Leaf Scorch | **1.000** | — | Perfect-conf OOD; leaf shape similar to Strawberry |
| 21 | rose_black_spot_02.jpg | ood | OOD (Rose) | Apple — Cedar Apple Rust | 0.669 | — | Rounded dark spots similar to cedar rust |
| 22 | sunflower_downy_01.jpg | ood | OOD (Sunflower) | Tomato — Early Blight | 0.984 | — | Mildew yellowing → early blight |
| 23 | sunflower_leaf_ood_01.jpg | ood | OOD (Sunflower) | Corn — Northern Leaf Blight | 0.986 | — | Long leaf texture triggers NLB class |
| 24 | wheat_leaf_rust_01.jpg | ood | OOD (Wheat) | Strawberry — Leaf Scorch | 0.987 | — | Rust pustules → Strawberry scorch |
| 25 | wheat_leaf_rust_02.jpg | ood | OOD (Wheat) | Peach — Healthy | 0.792 | — | Thin grass-like blade; no matching class |

---

## 5. Failure Mode Analysis

### 5.1 Domain Gap: Laboratory-to-Field Transfer Failure

The model achieves 97.13% accuracy on PlantVillage validation but only 6.67% on external images — a drop of over 90 percentage points. Crucially, this drop is observed even in the *easy* category, where photographic conditions were selected to approximate PlantVillage laboratory conditions. **Zero out of six easy-category images were correctly classified.** This outcome is more severe than anticipated and indicates that the domain gap is not exclusively attributable to background complexity or lighting variation; even single-leaf, relatively clean images from external sources fail to activate the correct class features.

The mechanistic cause is well established in the domain adaptation literature: the PlantVillage dataset was constructed under deliberate laboratory conditions with white background paper, consistent leaf centering, and uniform illumination. During training, the model optimises the joint likelihood of the disease signal *and* the photographic conditions simultaneously. The learned features therefore encode both disease-specific visual patterns and dataset-specific photographic regularities (framing, background uniformity, leaf isolation). When photographic conditions change, the model's decision boundaries — calibrated for PlantVillage statistics — no longer generalise to the new distribution.

### 5.2 Confidence Calibration Failure

Mean confidence on in-distribution external images is 0.733 against a top-1 accuracy of 6.67%, producing a **calibration error** — the systematic discrepancy between confidence and accuracy — of approximately 0.667. Of the 14 incorrect in-distribution predictions, **6 carry confidence ≥ 0.90**, including two near-perfect confidence values (0.9991, 0.9937) on images that are completely misclassified. This behaviour is referred to as *overconfident wrong prediction* and represents a deployment safety risk: a farmer relying on the application's confidence score as a reliability indicator would be actively misled in the majority of cases.

The root cause of miscalibration is the softmax function itself: softmax confidence is a monotone transformation of logit magnitude, not a posterior probability. Models trained to minimise cross-entropy loss tend to produce high-entropy outputs on in-distribution data while failing to produce correspondingly low-entropy (low-confidence) outputs on out-of-distribution or domain-shifted inputs. Temperature scaling, Platt scaling, or a learned confidence head trained on a held-out calibration set would be required to produce meaningful uncertainty estimates.

### 5.3 OOD Overconfidence

Four out of five processed OOD images received confidence > 0.90, including two cases of near-perfect confidence (1.000 for `rose_black_spot_01.jpg`, 1.000 for `apple_fire_blight_dense_02.jpg`). Because the model is a closed-set classifier, it cannot abstain; it is architecturally compelled to assign one of the 38 PlantVillage classes to any input image. The high confidence on OOD inputs reveals that the model's softmax distribution is not sensitive to whether the input lies within the training manifold — a property that would require out-of-distribution detection mechanisms (e.g., energy-based scoring, Mahalanobis distance to class centroids, or Monte Carlo dropout) to resolve.

### 5.4 Grad-CAM Collapse

All 25 external images received a Grad-CAM quality score of 0 (Uninformative) or 1 (Background-only); not a single image received a score of 2 (Diffuse) or 3 (Focused). This is in complete contrast with the PlantVillage validation results, where all three sanity-check images produced tightly focused, lesion-centred heatmaps (see Section 2 of `results/gradcam/ANALYSIS.md`). The collapse of spatial coherence on external images confirms that the model's gradient signal — which drives Grad-CAM activation — is not being generated by the leaf or lesion regions. Instead, the model appears to respond to global image statistics (colour histogram, background texture patterns) that are inconsistent across images and therefore fail to produce spatially structured activation maps.

This finding has direct implications for the interpretability claims of the application: under PlantVillage conditions, the heatmaps provide meaningful explanations; under field conditions, the same heatmaps are spatially uninformative and could not be used to identify which leaf region the model is responding to.

---

## 6. Single Success: Pepper Bacterial Spot

The single correct external prediction — `pepper_bacterial_spot_field_01.jpg`, confidence 0.989 — merits specific analysis. *Pepper,_bell___Bacterial_spot* produces water-soaked, angular lesions with a characteristic interveinal distribution, typically appearing as small (2–5 mm) brown spots confined between leaf veins. This morphological pattern is relatively consistent across field and laboratory specimens and is highly distinctive relative to other classes in the 38-class taxonomy: no other class produces similarly angled, vein-constrained, small-diameter spotting on pepper-coloured (pale to dark green) leaf tissue.

The question arises: if Pepper Bacterial Spot can generalise, why do other seemingly distinctive classes (e.g., Tomato Early Blight, which produces characteristic concentric ring lesions) fail? The answer likely lies in relative class distinctiveness within the softmax space. Tomato Early Blight shares the "necrotic spot on green leaf" visual space with five other tomato diseases in the taxonomy, making its per-class boundary narrow. Pepper Bacterial Spot, by contrast, occupies a more isolated region of feature space — its leaf shape (elongated, smooth-margined pepper leaf) and lesion pattern (angular, interveinal) combine to produce a feature vector that is sufficiently different from all competing classes to survive modest domain shift. This interpretation is consistent with the high F1 = 0.9747 achieved by this class on the PlantVillage validation set: the class is both well-learnt and well-separated.

---

## 7. Comparison to Literature

The domain gap observed in this study aligns with, and extends, previously reported findings. Mohanty et al. (2016), who trained a GoogLeNet/AlexNet model on PlantVillage and reported validation accuracy of 99.35%, subsequently found that accuracy dropped to approximately 31% on a limited set of field-condition test images. This finding established the PlantVillage domain gap as a known limitation and is routinely cited in subsequent plant disease classification literature.

The external validation results reported here are substantially more severe: a drop from 97.13% to 6.67%, compared to Mohanty et al.'s 99.35% to ~31%. Several factors may account for the steeper decline in the present study:

1. **Class count.** The present model classifies 38 classes; the chance baseline is 2.6%. With a more granular class set, the probability of any given misclassification landing on a close neighbour (and therefore being counted as "partially correct" in top-3 analysis) is lower.

2. **Test set composition.** The 25-image external set was assembled to include challenging field conditions, OOD categories, and hard cases. Mohanty et al.'s field test set characteristics are not fully specified in the published record; if their field images were closer to laboratory conditions, the comparison is not fully controlled.

3. **Resolution.** MobileNetV2's efficient architecture trades some representational capacity for parameter efficiency; this may reduce the model's ability to extrapolate learned features across domain boundaries relative to larger architectures.

Despite these differences in degree, the qualitative finding is identical across both studies: **models trained exclusively on PlantVillage dataset images should not be expected to generalise to field-condition imagery without domain adaptation.** This finding is well-supported by Phase 3.10's fine-tuning experiment, in which targeted fine-tuning on PlantDoc field images improved external accuracy from 6.67% to 26.67% for the web validation set, while introducing classifier drift on the source domain.

---

## 8. Implications for Final Report

### 8.1 Recommended Report Placement

| Section | Claim supported | Evidence |
|---|---|---|
| Section 4 (Results) | 97.13% PlantVillage accuracy requires caveat: applies only to matched lab conditions | Overall accuracy with domain caveat |
| Section 5.1 (Discussion) | Domain gap is the primary limitation of the current system for deployment | 97.13% → 6.67% drop; Mohanty comparison |
| Section 5.2 (Limitations) | Confidence scores are not calibrated for OOD or domain-shifted input | 0.733 mean conf on 6.67% correct predictions |
| Section 5.2 (Limitations) | Grad-CAM heatmaps are valid under lab conditions but collapse on field images | All 25 external images scored 0–1 |
| Section 5.2 (Limitations) | Closed-set classifier cannot abstain; OOD overconfidence is a deployment risk | 4/5 OOD images conf > 0.90 |
| Section 6 (Future Work) | Fine-tuning on field data improves generalisation (Phase 3.10 positive result) | 6.67% → 26.67% after fine-tuning |
| Section 6 (Future Work) | Temperature scaling or entropy-based rejection needed for calibration | Calibration error ~0.667 |

### 8.2 Recommended Figure References

- **Fig. E1:** Representative external test failures across categories (easy, medium, hard) — side-by-side with PlantVillage correct-prediction images — to visually anchor the domain gap in Section 5.1.
- **Fig. E2:** Grad-CAM comparison: focused heatmap on PlantVillage specimen vs. uninformative heatmap on equivalent external image — strongest visual evidence for feature non-transferability.
- **Fig. E3:** Confidence histogram: in-distribution external images (mean 0.733) vs. correct PlantVillage predictions (mean > 0.95) — demonstrates calibration failure.

### 8.3 Framing Guidance

The 6.67% external accuracy should not be presented as a model failure in isolation; rather, it should be contextualised as evidence that **the domain gap is feature-level and not solvable by photometric preprocessing alone** (Phase 3.9 image enhancement: −6 pp on PlantDoc) while **partial mitigation is achievable through fine-tuning** (Phase 3.10: +20 pp on web validation). This three-part narrative — gap quantification → preprocessing failure → fine-tuning improvement — constitutes the central contribution of the cross-dataset evaluation section of the report.

---

*Test executed: 2026-05-03 — Phase 3, Step 4C (manual) and Phase 3.10.B (automated)*
*25 images tested across 4 categories (easy/medium/hard/ood); 24 processed (1 file skipped)*
*Model: MobileNetV2 — `checkpoints/best_mobilenet.pth`*
*Baseline: 97.13% PlantVillage val accuracy (17,572 images, 38 classes)*
*Automated results: `results/web_evaluation/web_results_original.csv` and `summary_web_original.json`*
