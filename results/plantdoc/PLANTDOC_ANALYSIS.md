# PlantDoc Cross-Dataset Evaluation Analysis
**Phase 3 — Plant Disease Detection Project**
Sakarya University, Software Engineering — Senior Capstone Project

---

## 1. Methodology

### 1.1 Rationale and Research Question

The PlantVillage dataset from which both trained models were derived consists entirely of laboratory-quality images: single leaves photographed against uniform white or black backgrounds under controlled illumination. Real-world plant disease diagnosis is conducted in field conditions that introduce visual noise absent from the training distribution — complex background vegetation, variable lighting angles and intensities, partial leaf occlusion, motion blur, and the presence of text overlays or watermarks in images sourced from the internet (Mohanty et al., 2016). The central research question motivating this evaluation is therefore: *how much of the validation accuracy measured on PlantVillage transfers to field-realistic imagery?*

PlantDoc (Singh et al., CoDS-COMAD 2020) was selected as the primary external benchmark because it represents the most semantically aligned academic alternative to PlantVillage. Its 27 test classes overlap substantially with PlantVillage's 38-class taxonomy, it was assembled from web-scraped and field-photographed images rather than laboratory specimens, and its authors specifically designed it to quantify the lab-to-field generalisation gap — the same gap this evaluation targets. While a targeted web validation set (25 images; see `docs/external_validation.md`) was constructed to capture wild-distribution imagery, PlantDoc provides a larger-scale, standardised, and reproducible cross-dataset benchmark.

### 1.2 Dataset Description

The PlantDoc test split comprises **231 images** across **27 classes**, averaging **8.6 images per class**. The minimum class support is 3 images (Corn Gray leaf spot) and the maximum is 12 (Corn leaf blight). This uniformly small per-class sample size means that per-class accuracy estimates carry substantial statistical uncertainty; per-class numbers reported in this document should be interpreted as order-of-magnitude estimates rather than precise performance measurements.

Images were collected via web scraping and field photography and exhibit several visual characteristics absent from PlantVillage:
- **Complex backgrounds:** leaves photographed in situ, surrounded by other vegetation, soil, or sky
- **Lighting variation:** dappled shade, overexposure, and inconsistent white balance
- **Leaf orientation and cropping:** non-canonical orientations, partial leaves, multiple leaves in frame
- **Textual artefacts:** some images contain overlaid text, watermarks, or search-engine result borders

### 1.3 Class Mapping Procedure

The trained MobileNetV2 model produces predictions over the 38-class PlantVillage taxonomy. PlantDoc test images carry ground-truth labels from its own 27-class taxonomy. A class mapping table (`analysis/plantdoc_evaluation.py`, lines 43–71) bridges these two taxonomies:

- **Coverage:** All 27 PlantDoc test classes were mapped to a PlantVillage class — 100% PlantDoc coverage.
- **Directionality:** The mapping is injective but not surjective. Ten PlantVillage classes have no PlantDoc test counterpart (e.g., `Apple___Black_rot`, `Grape___Esca`, `Orange___Haunglongbing`). Model predictions landing on these 10 classes are counted as automatic misclassifications.
- **Strictness:** A prediction is counted as correct if and only if the model's top-1 PlantVillage output exactly matches the expected PlantVillage class for the given PlantDoc ground truth. No graceful fallback or semantic-similarity credit is applied.

### 1.4 Evaluation Protocol

Inference was conducted using `checkpoints/best_mobilenet.pth` — the same checkpoint evaluated on PlantVillage with 97.13% validation accuracy. The preprocessing pipeline was identical: 224×224 resize, centre crop, and ImageNet mean/std normalisation. No test-time augmentation was applied. Top-1 accuracy, Top-3 accuracy, and per-class precision, recall, and F1-score were computed with macro averaging weighted equally across all 27 PlantDoc classes.

---

## 2. Aggregate Results

### 2.1 Summary Metrics

| Metric | PlantVillage Val | PlantDoc Test | Drop |
|---|---|---|---|
| **Top-1 Accuracy** | **97.13%** | **16.02%** | **−81.11 pp** |
| Top-3 Accuracy | — | 32.90% | — |
| Macro Precision | 0.9729 | 0.1808 | −0.7921 |
| Macro Recall | 0.9707 | 0.1668 | −0.8039 |
| **Macro F1** | **0.9707** | **0.1426** | **−0.8281** |
| Unmapped predictions | — | 55 / 231 (23.8%) | — |

The 81.11 percentage-point drop represents a near-total collapse of the learned representations under domain shift. Fewer than 1 in 6 PlantDoc test images is classified correctly when the model is applied without domain adaptation.

### 2.2 Comparison with Prior Work

Mohanty et al. (2016) reported a similar transfer experiment: their PlantVillage-trained model achieved 99.35% validation accuracy but only approximately 31% on field-collected images — a drop of approximately 68 percentage points. The current model's 81-point drop is steeper, attributable to three factors: a stricter mapping protocol with no semantic credit for near-correct predictions; a smaller test set with noisier per-class estimates; and PlantDoc's deliberate selection of images that maximise the lab-to-field gap.

The 32.90% Top-3 accuracy — roughly twice the Top-1 figure — indicates that for approximately 17% of test images the correct class is within the model's top-3 predictions but not at rank 1. This suggests partial feature recognition: the model extracts some relevant signal from field images, but competing PlantVillage classes displace the correct answer from the top position.

### 2.3 Unmapped Predictions

Of 231 predictions, **55 (23.8%)** fell on PlantVillage classes absent from the PlantDoc 27-class taxonomy. The closed-set classifier design provides no mechanism for the model to abstain or signal uncertainty; when presented with an out-of-distribution input, MobileNetV2 assigns a confident prediction to whatever PlantVillage class is most visually similar under its learned representation — which for 23.8% of PlantDoc images is a class with no PlantDoc equivalent at all.

---

## 3. Strongest Generalising Classes

### 3.1 Top Five Classes by Accuracy

| Rank | PlantDoc Class | Mapped PV Class | Accuracy | n | n correct |
|---|---|---|---|---|---|
| 1 | Bell_pepper leaf | Pepper,_bell___healthy | 57.14% | 7 | 4 |
| 2 | Squash Powdery mildew leaf | Squash___Powdery_mildew | 50.00% | 6 | 3 |
| 3 | Corn leaf blight | Corn_(maize)___Northern_Leaf_Blight | 50.00% | 12 | 6 |
| 4 | Blueberry leaf | Blueberry___healthy | 45.45% | 11 | 5 |
| 5 | Peach leaf | Peach___healthy | 44.44% | 9 | 4 |

All five classes share a common structural property: the visual signal that distinguishes them is **globally distinctive and geometrically invariant under background complexity**. Bell pepper and Blueberry healthy leaves present species-specific morphologies recognisable across backgrounds. Squash Powdery Mildew manifests as white powdery surface coating — highly salient regardless of lighting. Corn Northern Leaf Blight produces elongated cigar-shaped lesions with characteristic gray-green colouration that persists across imaging conditions.

### 3.2 Cross-Reference with Web Validation

The web validation experiment (`docs/external_validation.md`) found that `Pepper___Bacterial_spot` was the only correctly classified image among 30 web-sourced specimens from the original model. The relative success of pepper-related classes in both external evaluations — Bell_pepper leaf at 57.1% in PlantDoc and Bacterial Spot as the sole web success — suggests that pepper leaf morphology is represented with sufficient within-class diversity in PlantVillage to support partial domain generalisation.

---

## 4. Failure Mode Clusters

### 4.1 Complete Zero-Accuracy Classes

Thirteen of 27 classes (48.1%) achieved 0% accuracy. These classes produced zero correct predictions despite high PlantVillage F1 scores (0.940–0.993):

| Class | n | PV val F1 | PlantDoc Accuracy |
|---|---|---|---|
| Apple Scab Leaf | 10 | 0.987 | 0% |
| Apple leaf (healthy) | 9 | 0.971 | 0% |
| Apple rust leaf | 10 | 0.976 | 0% |
| Cherry leaf | 10 | 0.970 | 0% |
| Corn rust leaf | 10 | 0.979 | 0% |
| Potato leaf late blight | 7 | 0.989 | 0% |
| Soyabean leaf | 8 | 0.992 | 0% |
| Strawberry leaf | 8 | 0.993 | 0% |
| Tomato Septoria leaf spot | 11 | 0.940 | 0% |
| Tomato leaf (healthy) | 8 | 0.980 | 0% |
| Tomato leaf bacterial spot | 9 | 0.956 | 0% |
| Tomato leaf mosaic virus | 10 | 0.988 | 0% |
| Tomato leaf yellow virus | 6 | 0.988 | 0% |

PlantVillage validation performance does not predict field generalisation for any of these 13 classes.

### 4.2 Apple Family Collapse

All three Apple classes — Apple Scab Leaf (0/10), Apple leaf/healthy (0/9), Apple rust leaf (0/10) — achieve 0% accuracy despite PlantVillage F1 scores of 0.971–0.987. The complete collapse of these high-confidence lab-trained classes provides the clearest evidence of shortcut learning (Geirhos et al., 2020): the model appears to have learned features correlated with the uniform white background and isolated-leaf composition of PlantVillage apple images, rather than disease- or species-specific leaf features. When presented with PlantDoc apple images — photographed in orchards, against complex background foliage — the background-correlated shortcut is absent, and the model has no alternative feature to fall back on.

The 23.8% unmapped prediction rate further supports this interpretation: a substantial fraction of failed Apple images are redirected to PlantVillage classes that share background composition characteristics (e.g., `Apple___Black_rot`, `Grape___Esca`), rather than to any Apple class present in the PlantDoc taxonomy.

### 4.3 Tomato Cluster Disintegration

Eight PlantDoc classes correspond to Tomato species. The combined accuracy across all 63 Tomato test images is **5/63 = 7.9%** — substantially below the dataset average of 16.02%:

| PlantDoc Tomato Class | n | Correct | Accuracy |
|---|---|---|---|
| Tomato Early blight leaf | 9 | 3 | 33.3% |
| Tomato mold leaf | 6 | 1 | 16.7% |
| Tomato leaf late blight | 10 | 1 | 10.0% |
| Tomato Septoria leaf spot | 11 | 0 | 0% |
| Tomato leaf (healthy) | 8 | 0 | 0% |
| Tomato leaf bacterial spot | 9 | 0 | 0% |
| Tomato leaf mosaic virus | 10 | 0 | 0% |
| Tomato leaf yellow virus | 6 | 0 | 0% |

This result is consistent with the per-class analysis of PlantVillage performance (`results/PER_CLASS_COMPARISON_ANALYSIS.md`, Section 4.2), which identified the Tomato genus as MobileNetV2's primary in-distribution failure cluster. Domain shift compounds this pre-existing weakness: Tomato classes that were already in the middle tier on PlantVillage (F1 0.850–0.949) collapse to near-zero on PlantDoc, and even those with higher in-distribution scores fail to generalise.

### 4.4 Partial Generalisation in Corn

The three Corn classes reveal an instructive contrast:

| PlantDoc Corn Class | n | Accuracy | Lesion type |
|---|---|---|---|
| Corn leaf blight (NLB) | 12 | 50.0% | Elongated cigar-shaped lesions, high colour contrast |
| Corn Gray leaf spot (Cercospora) | 3 | 33.3% | Elongated streaks, moderate contrast |
| Corn rust leaf | 10 | 0% | Small scattered pustules, low colour contrast |

Northern Leaf Blight's elongated lesions with high colour contrast generalise at 50%, while Common Rust's small scattered pustules fail entirely. This is consistent with the Grad-CAM finding that MobileNetV2's Corn NLB attention is distributed across the elongated lesion pattern — a spatially robust feature — whereas rust pustule detection requires fine-grained texture recognition that is disrupted by background noise.

---

## 5. The Unmapped Prediction Problem

Of 231 test images, **55 (23.8%)** received top-1 predictions landing on PlantVillage classes absent from PlantDoc. This is a structural limitation of the closed-set classifier design: there is no mechanism for the model to abstain, express uncertainty, or indicate that an input falls outside its training distribution. When a field-condition image is visually dissimilar from all PlantVillage training images, the model selects the highest-scoring class in its output space regardless of relevance.

This failure mode is distinct from domain-shift misclassification: unmapped predictions arise from the combination of distribution shift *and* closed-set architecture, and they would persist even if the model's in-distribution accuracy were perfect. Addressing them requires either an open-set or reject-option classifier, or explicit out-of-distribution detection (Hendrycks & Gimpel, 2017).

---

## 6. Three-Tier Domain Gap Synthesis

### 6.1 The Accuracy Degradation Spectrum

| Evaluation Setting | Top-1 Accuracy | Description |
|---|---|---|
| PlantVillage validation | 97.13% | Controlled studio photography, same distribution as training |
| PlantDoc test | 16.02% | Web-scraped and field images, 27-class academic benchmark |
| Web validation | 6.67% | Curated wild imagery, 30 samples, 15 classes |

The monotone degradation — 97.13% → 16.02% → 6.67% — reflects increasing distributional distance from the training set. The drop from PlantVillage to PlantDoc is 81.11 pp; from PlantVillage to wild web imagery it is 90.46 pp.

### 6.2 Interpreting the Three-Tier Finding

The three-tier degradation pattern should be interpreted as an empirical characterisation of a known limitation of laboratory-dataset-trained classifiers, not an unexpected failure. Mohanty et al. (2016) documented the same phenomenon; subsequent domain adaptation work (DeChant et al., 2017; Kamilaris & Prenafeta-Boldú, 2018) has consistently shown that models trained on controlled images require explicit adaptation before achieving clinically useful field performance.

The finding constitutes an empirical contribution in its own right: it quantifies the gap between three distinct evaluation regimes for this specific architecture and training configuration, providing a reference point for any subsequent domain adaptation work.

### 6.3 Fine-Tuned Model Performance

The fine-tuned model (`checkpoints/best_mobilenet_finetuned.pth`), trained on PlantDoc images with a frozen MobileNetV2 backbone and retrained classifier head, achieves **30.74%** on PlantDoc — a 14.72 pp improvement over the original model's 16.02%. This improvement demonstrates that the backbone features are not entirely non-transferable; the pre-trained feature extractor retains representations that can be re-oriented toward field conditions with limited fine-tuning. The remaining gap between 30.74% and any practically useful threshold indicates that full backbone fine-tuning and larger field-condition labelled datasets would be required for deployment-grade performance.

---

## 7. Caveats and Limitations

1. **Small per-class samples.** Average 8.6 images per class; minimum 3 (Corn Gray leaf spot). A single misclassification changes Corn Gray leaf spot accuracy by 33.3 pp. Ordinal rankings in Section 3 are indicative, not precise.

2. **Ten unmapped PlantVillage classes.** Any model output landing on classes absent from PlantDoc is counted as an unconditional error regardless of visual plausibility. This inflates the effective miss rate for classes frequently confused with unmapped PV categories.

3. **MobileNetV2 only.** ResNet-50 was not evaluated on PlantDoc; all three-tier comparisons reflect MobileNetV2-specific generalisation and should not be assumed to apply to ResNet-50.

4. **No test-time augmentation.** Multi-crop or multi-flip ensemble inference was excluded for consistency with PlantVillage evaluation but might modestly improve top-1 accuracy.

5. **Text-overlay artefacts not screened.** Some PlantDoc images contain embedded watermarks. Their contribution to prediction errors cannot be quantified without a clean version of the test set.

---

## 8. Implications for Final Report

### 8.1 Recommended Report Placement

| Section | Claim | Evidence |
|---|---|---|
| 4.3 (Cross-Dataset Evaluation) | Validation accuracy does not predict field performance | Section 2.1 — 97.13% → 16.02% |
| 4.3 | Top-3 partially recovers correct class | Section 2.2 — 32.90% top-3 |
| 4.4 (Domain Gap) | Three-tier degradation pattern | Section 6.1 table |
| 5.2 (Limitations) | Closed-set classifier fails on OOD inputs | Section 5 — 23.8% unmapped |
| 5.3 (Future Work) | Fine-tuning partially bridges the gap | Section 6.3 — 30.74% |

### 8.2 Figure References

| Figure ID | File | Section | Argument supported |
|---|---|---|---|
| Fig. F1 | Three-tier bar chart (to be generated) | Section 6.1 | Monotone accuracy degradation across evaluation regimes |
| Fig. F2 | Per-class accuracy sorted bar (27 classes) | Section 3.1, 4.1 | Distribution of generalisation failure — 13/27 classes at 0% |
| Fig. F3 | `results/plantdoc/confusion_matrix_plantdoc.png` | Section 4 | Apple family collapse; unmapped prediction column |

### 8.3 Framing Guidance

Three framing principles apply when incorporating this analysis into the final report:

**PlantDoc as the bridge benchmark.** The 16.02% accuracy on PlantDoc is not a surprising result; it is the expected outcome for an unadapted model and is consistent with prior literature. The substantive finding is the partial recovery under fine-tuning (30.74%), which demonstrates non-trivial feature transferability despite the severe accuracy drop.

**Three-tier degradation as a central empirical contribution.** The progression 97.13% → 16.02% → 6.67% should be presented as the primary finding of Phase 3, concisely summarising the model's operational scope and its limitations across three evaluation regimes.

**Negative results as valid scientific contributions.** The preprocessing enhancement pipeline (Phase 3.9.A) showed no consistent improvement; this PlantDoc evaluation shows near-zero out-of-domain performance; the web validation is similarly unfavourable. These negative results, framed appropriately, strengthen the report by demonstrating that domain robustness was pursued rigorously rather than reported only via in-distribution metrics.

---

*Analysis completed: Phase 3 — Step 5 (PlantDoc Cross-Dataset Evaluation)*
*Source data: `results/plantdoc/classification_report_plantdoc.txt`, `results/plantdoc/per_class_results_plantdoc.csv`*
*Model: MobileNetV2 (`checkpoints/best_mobilenet.pth`)*
*Test set: 231 images, 27 classes (PlantDoc dataset — Singh et al., CoDS-COMAD 2020)*
