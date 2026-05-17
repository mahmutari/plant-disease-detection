# Hybrid Training Methodology and Results
**Phase 3.5 — Plant Disease Detection Project**
Sakarya University, Software Engineering — Senior Capstone Project

---

## 1. Background

The Phase 3 evaluation revealed a substantial laboratory-to-field generalisation gap for the trained MobileNetV2 model: validation accuracy on PlantVillage dropped from 97.13% to 16.02% on the PlantDoc benchmark and to 6.67% on a curated web validation set. Two adaptation strategies were initially tested in Phase 3 — classical computer-vision preprocessing (which was rejected with a -6 pp regression on PlantDoc) and frozen-backbone fine-tuning on PlantDoc training data (which produced +14.72 pp on PlantDoc but a -38.17 pp catastrophic forgetting on PlantVillage).

The fine-tuning result demonstrated that the convolutional backbone retains transferable features but established that classifier-only adaptation on a single target domain is insufficient to preserve source-domain performance. This motivated the design of a hybrid training approach in Phase 3.5: training on a combined PlantVillage + PlantDoc dataset to test whether catastrophic forgetting could be mitigated while retaining cross-domain improvement.

## 2. Initial Hybrid Approach (V1) and Discovered Defect

### 2.1 V1 Methodology

The initial hybrid training implementation combined PlantVillage and PlantDoc training data using `torch.utils.data.ConcatDataset` with 1:1 weighted random sampling. The MobileNetV2 backbone was frozen (consistent with the Phase 3 fine-tuning methodology); only the classifier head was trained. Hyperparameters were preserved from Phase 2: Adam optimization, learning rate 1e-3, batch size 8, 3 epochs, CPU-only execution.

### 2.2 V1 Results

| Test Distribution | Accuracy |
|---|---|
| PlantVillage validation | 96.09% |
| PlantDoc test           | 31.17% |
| Web validation          | 0.00% |

The PlantVillage and PlantDoc results were promising — catastrophic forgetting appeared to have been resolved (PV dropped only -1.04 pp from Phase 2). However, the web validation accuracy of 0.00% revealed a critical defect in the training procedure.

### 2.3 Root Cause Analysis

Investigation determined that `ConcatDataset` preserved the original class indices of each constituent dataset without remapping. PlantVillage and PlantDoc use different sorted-alphabetical class orderings, resulting in inconsistent label semantics for the same numerical index:

| Class Index | PlantVillage Class | PlantDoc Class |
|---|---|---|
| 0 | Apple___Apple_scab | Apple Scab Leaf |
| 1 | Apple___Black_rot | Apple leaf |
| 2 | Apple___Cedar_apple_rust | Apple rust leaf |
| 3 | Apple___healthy | Bell_pepper leaf |

Only index 0 corresponded to semantically equivalent classes across both datasets. The remaining 26 indices presented the classifier head with contradictory training signals — the same output neuron was being trained to recognize both "Apple Black Rot" (when receiving a PlantVillage image) and "Apple healthy" (when receiving a PlantDoc image). This is a classic *label space inconsistency* problem in multi-source training.

The defect was not detected during PlantVillage validation because the validation distribution matched the training distribution — the model produced outputs consistent with its training, even though those outputs were internally contradictory. The defect became visible only on out-of-distribution web validation imagery, where the corrupted classifier head failed entirely (0.00% accuracy).

## 3. Corrected Hybrid Approach (V2)

### 3.1 Solution Design

The defect was corrected by implementing a custom `PlantDocAsPV` dataset class that remaps PlantDoc class names to PlantVillage class indices before training. The remapping table was derived from the existing PlantDoc evaluation pipeline (`analysis/plantdoc_evaluation.py`) and covers all 27 PlantDoc classes that have a PlantVillage counterpart.

### 3.2 V2 Methodology

The corrected hybrid training procedure is implemented in `analysis/hybrid_training_v2.py`. Key design decisions:

- **Frozen backbone:** The MobileNetV2 convolutional feature extractor is frozen; only the classifier head (48,678 parameters, 2.1% of total) is trained. This preserves apples-to-apples comparability with Phase 3 fine-tuning.

- **Unified label space:** All training images (both PV and PD) are labeled in PlantVillage's 38-class index space. PlantDoc images are mapped to their semantic PV equivalent (e.g., PlantDoc "Apple leaf" → PlantVillage "Apple___healthy").

- **1:1 Weighted sampling:** Sample weights are set inversely proportional to dataset size (PV weight = 1/N_pv, PD weight = 1/N_pd). This ensures balanced exposure to both distributions despite their 31× size disparity (70,000 PV vs 2,246 PD images).

- **Hyperparameters:** Adam optimization, learning rate 1e-3, batch size 8, 3 epochs (consistent with Phase 2 and Phase 3 training budgets).

### 3.3 V2 Training Trajectory

| Epoch | Train Loss | Train Accuracy | Duration |
|---|---|---|---|
| 1 | 1.4416 | 62.44% | 67.4 min |
| 2 | 1.1957 | 65.33% | 54.1 min |
| 3 | 1.1624 | 66.13% | 52.3 min |
| **Total** | | | **173.8 min (~2.9 h)** |

The training trajectory shows monotonic loss reduction and accuracy improvement with diminishing returns by epoch 3, suggesting that 3 epochs is an appropriate budget for this configuration.

## 4. Multi-Distribution Evaluation Results

### 4.1 Final Comparison Across All Models

| Model | PlantVillage val | PlantDoc test | Web val | OOD risk |
|---|---|---|---|---|
| Phase 2 Original    | 97.13% | 16.02% | 6.67%  | 50.00% |
| Phase 3 Fine-tuned  | 58.96% | 30.74% | 26.67% | 10.00% |
| Phase 3.5 Hybrid V1 (defective) | 96.09% | 31.17% | 0.00% | 10.00% |
| **Phase 3.5 Hybrid V2 (fixed)** | **96.07%** | **41.13%** | 6.67%  | **0.00%** |

OOD risk is measured as the rate of high-confidence (>0.90) predictions on the 9 out-of-distribution test images (rose, sunflower, wheat — all absent from PlantVillage's 38-class taxonomy).

### 4.2 Key Findings

**Finding 1 — V2 successfully resolves catastrophic forgetting.**
V2 retains 96.07% accuracy on PlantVillage validation — only 1.06 percentage points below Phase 2's 97.13%. This contrasts sharply with Phase 3 fine-tuning's 38.17 pp regression and demonstrates that hybrid training with proper label mapping eliminates catastrophic forgetting as a barrier to cross-domain adaptation.

**Finding 2 — V2 achieves the best PlantDoc performance across all models.**
V2's 41.13% PlantDoc accuracy is +10.39 pp higher than Phase 3's fine-tuned model (30.74%) and +25.11 pp higher than Phase 2's original (16.02%). The corrected label mapping appears to enable more effective transfer learning than single-domain fine-tuning.

**Finding 3 — V2 is the safest model on out-of-distribution inputs.**
V2 produces zero high-confidence predictions on OOD inputs, compared to 50% for the original model and 10% for the fine-tuned model. This is a critical property for production deployment: a model that refuses to be confident on out-of-distribution data is safer than one that confidently misclassifies novel inputs.

**Finding 4 — No single model dominates across all test distributions.**
The Phase 2 original retains the highest PlantVillage accuracy. The Phase 3 fine-tuned model retains the highest curated web validation accuracy (26.67%). The Phase 3.5 V2 hybrid achieves the highest PlantDoc accuracy and the safest OOD behavior. This pattern is consistent with the "no free lunch" principle in machine learning: domain-specific models are optimal for their respective domains, and a multi-model deployment strategy provides flexibility that no single model can match.

### 4.3 Web Validation Considerations

Hybrid V2's web validation accuracy (6.67%) does not exceed the Phase 3 fine-tuned model's 26.67%. Three factors contribute to this asymmetric outcome:

1. **Curated web imagery is a distinct distribution.** The web validation set consists of carefully selected real-world images that differ from both PlantVillage (laboratory) and PlantDoc (web-scraped academic benchmark) in lighting, framing, and image quality.

2. **Fine-tuned model's exclusive focus.** The Phase 3 fine-tuned model overfits more aggressively to field-style imagery because it sees only field data during training. While this overfitting harms PlantVillage performance, it provides an edge on curated web imagery that shares characteristics with the field training data.

3. **Hybrid training's balancing constraint.** V2's hybrid training data forces the classifier to maintain compatibility with both lab and field distributions, which may sacrifice peak performance on curated wild imagery in exchange for broader robustness.

This finding supports the multi-model deployment strategy implemented in the Streamlit interface: users facing controlled-laboratory imagery should select the Original model, users facing curated field imagery should select the Fine-tuned model, and users requiring balanced performance with safe OOD behavior should select Hybrid V2.

## 5. Implementation Notes

### 5.1 File Locations

- Training script: `analysis/hybrid_training_v2.py`
- V2 checkpoint:   `checkpoints/best_mobilenet_hybrid_v2.pth` (8.9 MB)
- V1 checkpoint:   `checkpoints/best_mobilenet_hybrid.pth` (8.9 MB, retained for comparison)
- Training history: `results/hybrid_training_v2/training_history.json`
- Final results:    `results/hybrid_training_v2/final_results.json`
- Comparison data:  `results/hybrid_training_v2/comparison_v2.json`

### 5.2 Reproduction

To reproduce the V2 hybrid training:

```bash
python analysis/hybrid_training_v2.py
```

The script automatically:
1. Loads PlantVillage and PlantDoc training data
2. Applies the PD→PV label remapping
3. Initializes 1:1 weighted random sampling
4. Loads the Phase 2 pretrained MobileNetV2 checkpoint
5. Freezes the backbone, trains the classifier head
6. Saves checkpoints after each epoch
7. Evaluates on PlantVillage validation and PlantDoc test
8. Outputs comparison JSON with all four models' results

### 5.3 Streamlit Interface Integration

The corrected hybrid model is integrated into the Streamlit web application (`app.py`) as one of three selectable models. Users can:

- Select any of the three models via radio button
- View performance metrics for the selected model in the sidebar
- Receive top-1 and top-3 predictions with confidence-based color coding
- Inspect Grad-CAM heatmaps for the selected model's decision
- Enable "Compare All Models" mode to run all three models in parallel on a single input

## 6. Methodological Lessons

This phase produced two methodological insights that may inform subsequent work:

**Lesson 1 — Validate cross-domain test performance during multi-source training development.**
The V1 defect produced acceptable in-domain accuracy (96.09% PV, 31.17% PD) but completely failed on web validation (0.00%). Single-distribution validation is insufficient to detect label-space inconsistencies; cross-domain test cases must be included in the development feedback loop.

**Lesson 2 — Document and preserve failed attempts.**
The V1 defect, when properly characterized, became a methodological contribution rather than an embarrassment. The hybrid training documentation now includes both the failed attempt and its correction, providing reproducible evidence of the importance of label-space consistency in multi-source training.

## 7. Implications for Final Report

The Phase 3.5 hybrid training results extend the Phase 3 cross-distribution evaluation in three substantive ways:

1. The three-tier degradation pattern (PV 97% → PD 16% → Web 7%) documented in Phase 3 can now be partially reversed for two of the three test distributions through hybrid training, while the third (curated web imagery) remains better served by a single-domain fine-tuned model.

2. The "catastrophic forgetting trade-off" identified in Phase 3 fine-tuning is no longer fundamental: hybrid training with proper label mapping demonstrates that source-domain accuracy can be preserved (96.07%) while still achieving substantial cross-domain improvement (+25 pp on PlantDoc).

3. The OOD safety analysis introduces a new evaluation axis beyond top-1 accuracy: V2's 0% OOD high-confidence rate represents a deployment-relevant property that is not visible in standard accuracy metrics and that may be of greater practical importance than marginal accuracy differences for safety-critical applications.

These findings will be integrated into the Final Report as Section 7 ("Phase 3.5: Hybrid Training and Multi-Distribution Evaluation"), with the multi-model deployment strategy presented as the principal practical contribution of the project.

---

*Document version: 1.0*
*Last updated: 14 May 2026*
*Authors: Mahmut Arı, Samet Kavlan*
*Project: github.com/mahmutari/plant-disease-detection*
