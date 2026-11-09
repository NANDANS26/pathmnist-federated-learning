# Empirical Evaluation of Federated Averaging and Federated Proximal Optimization under Non-IID Partitioning and Symmetric Label Noise on PathMNIST

**Author:** Nandan S  
**Academic Supervision:** Dr. Sachin D N, Assistant Professor, Department of Computer Science & Engineering, IIIT Bhopal  
**Submission Track:** IndiaAI Fellowship 2026–2027 Screening  
**Repository:** [github.com/NANDANS26/pathmnist-federated-learning](https://github.com/NANDANS26/pathmnist-federated-learning)  
**Date of Completion:** November 9, 2026  

---

## Abstract

Federated learning (FL) enables decentralized participants to collaboratively train a shared predictive model without pooling raw private observations. In digital histopathology applications, FL frequently confronts two simultaneous challenges: inter-client class distribution skew (non-IID data) and local annotation noise arising from subjective grading or tissue preparation artifacts. While Federated Proximal Optimization (FedProx) is hypothesized to mitigate local client drift by penalizing parameter deviation from the global model, empirical validation under paired non-IID conditions coupled with severe label noise on histopathological imaging remains limited.

In this work, we conduct a seed-controlled paired benchmark comparing Federated Averaging (**FedAvg**) against Federated Proximal Optimization (**FedProx**, proximal parameter $\mu = 0.01$) across five canonical random seeds (42, 123, 2026, 3407, 7777). We evaluate on PathMNIST, a public 9-class colorectal histopathology dataset ($89,996$ training, $10,004$ validation, and $7,180$ test images), partitioned into five simulated federated clients using a Dirichlet distribution ($\alpha = 0.3$). To assess resilience against annotation errors, Client 5 is subjected to exactly $40.00\%$ symmetric label noise. Both algorithms were trained across 30 communication rounds using an identical 2-stage convolutional network (**CompactPathCNN**, 36,425 trainable parameters), identical seed-specific partitions, and identical initialization weights, totaling 300 executed communication rounds on an AMD Ryzen 5 5600H CPU execution platform.

Over the five seeds, FedAvg achieved a mean held-out test accuracy of $68.880\% \pm 9.266\text{ pp}$ (mean macro-F1: $0.6008 \pm 0.0948$), while FedProx achieved $68.279\% \pm 8.551\text{ pp}$ (mean macro-F1: $0.5934 \pm 0.0960$), yielding a paired difference of $-0.602 \pm 1.685\text{ pp}$ in test accuracy ($t(4) = -0.7983$, two-sided $p = 0.46938$, 95% CI: $[-2.694\text{ pp}, +1.491\text{ pp}]$) and $-0.0075 \pm 0.0105$ in macro-F1 ($t(4) = -1.5908$, $p = 0.18686$, 95% CI: $[-0.0205, +0.0056]$). Under the evaluated configuration, **no statistically significant difference in test performance was detected between FedAvg and FedProx across the five random seeds**. Because five seeds provide limited statistical power and no formal equivalence test with predefined equivalence margins was conducted, this non-significant result does not establish statistical equivalence. Mechanistically, the proximal penalty restricts the Euclidean step size of local parameter updates but does not alter the corrupted gradient directions induced by noisy annotations. Consequently, generic proximal drift penalties alone are insufficient to neutralize severe client-level label corruption in simulated histopathological federated learning without explicit noise filtering or adaptive client reweighting.

---

## 1. Introduction & Research Scope

Federated learning allows clinical and research centers to collaboratively train shared diagnostic models while keeping digitized histology images localized. However, decentralized histopathology introduces significant heterogeneity:
1. **Label Distribution Skew (Non-IID Data):** Specialized clinics observe disparate distributions of tissue pathologies (e.g., academic cancer centers encounter disproportionate adenocarcinoma relative to normal mucosa).
2. **Annotation Noise (Symmetric Label Noise):** Staining variations, tissue folding, and inter-observer diagnostic disagreement generate mislabeled patches.

### Research Scope & Operational Boundaries
- **Simulated Federation:** This investigation utilizes the public PathMNIST dataset partitioned synthetically into five client shards. No real-world multi-hospital network was deployed.
- **Privacy Controls:** Differential privacy (DP) and secure multiparty computation (SMPC) mechanisms were not implemented or evaluated.
- **Algorithmic Scope:** Conclusions regarding FedProx are strictly bounded to the evaluated configuration ($\mu = 0.01$, CompactPathCNN architecture, PathMNIST dataset, Dirichlet $\alpha = 0.3$, single-client $40\%$ symmetric noise, and 30 rounds). We do not assert generic failure of FedProx across all medical FL settings.

### Research Questions
- **RQ1:** Does FedProx with proximal regularization $\mu = 0.01$ mitigate client drift and improve global test generalization compared to FedAvg under Dirichlet distribution skew ($\alpha = 0.3$) when one client experiences $40\%$ symmetric label noise?
- **RQ2:** How does proximal regularization affect validation convergence trajectories, peak validation performance, and per-class diagnostic sensitivity across minority versus majority histological tissue classes?
- **RQ3:** What is the empirical wall-clock computational overhead of evaluating the proximal gradient term during local client optimization on standard CPU hardware?

---

## 2. Experimental Setup & Methodology

### 2.1 Benchmark Dataset: PathMNIST
PathMNIST is derived from the NCT-CRC-HE-100K colorectal cancer histology dataset (MedMNIST v2 benchmark suite). It contains $107,180$ patches ($28 \times 28 \times 3$ RGB) across 9 tissue classes:
- Class 0: Adipose tissue (ADI)
- Class 1: Background (BACK)
- Class 2: Debris (DEB)
- Class 3: Lymphocytes (LYM)
- Class 4: Mucus (MUC)
- Class 5: Smooth muscle (MUS)
- Class 6: Normal colon mucosa (NORM)
- Class 7: Cancer-associated stroma (STR)
- Class 8: Colorectal adenocarcinoma epithelium (TUM)

Splits comprise $89,996$ training samples, $10,004$ validation samples, and $7,180$ test samples. Test data were quarantined and loaded strictly post-training.

### 2.2 Client Partitioning & Label Corruption
- **Non-IID Partitioning:** A Dirichlet distribution with concentration parameter $\alpha = 0.3$ was sampled independently for each random seed, apportioning the $89,996$ training samples among 5 federated clients.
- **Client Validation Shards:** The $10,004$ global validation samples were partitioned into 5 shards apportioned to mirror local client class distributions for local tracking, while global validation evaluated all $10,004$ samples.
- **Symmetric Label Noise:** Client 5 (Target Client Index 4) was subjected to exactly $40.00\%$ symmetric label noise. A deterministic pseudo-random mask selected $40\%$ of Client 5's training samples, replacing each true label $y$ with a corrupt label $(y + \Delta y) \pmod 9$, where $\Delta y \in \{1, \dots, 8\}$.

### 2.3 Model Architecture: CompactPathCNN
To enable computationally efficient federated optimization, we instantiated **CompactPathCNN**, a 2-stage convolutional network with exactly **36,425 trainable parameters**:
- **Layer 1 (Conv1):** Conv2D($3 \to 32$, kernel $3 \times 3$, padding 1), ReLU, MaxPool2D($2 \times 2$, stride 2). Output: $(32, 14, 14)$.
- **Layer 2 (Conv2):** Conv2D($32 \to 64$, kernel $3 \times 3$, padding 1), ReLU, MaxPool2D($2 \times 2$, stride 2). Output: $(64, 7, 7)$.
- **Pooling:** AdaptiveAvgPool2d($2 \times 2$), Flatten to 256 dimensions.
- **Fully Connected (FC1):** Linear($256 \to 64$), ReLU.
- **Classification Head (FC2):** Linear($64 \to 9$), producing 9 unnormalized class logits.

### 2.4 Federated Optimization Algorithms
- **Federated Averaging (FedAvg):** At communication round $t$, the server broadcasts global weights $\mathbf{w}^t$. Each client $k$ minimizes local empirical cross-entropy loss $\mathcal{L}_k(\mathbf{w})$ over $E = 2$ epochs using SGD with batch size $B = 32$ and learning rate $\eta = 0.01$. The server aggregates local weights via sample weighting: $\mathbf{w}^{t+1} = \sum_{k=1}^K \frac{n_k}{N} \mathbf{w}_k^{t, E}$.
- **Federated Proximal Optimization (FedProx):** Adds a proximal regularization term to the local objective:
  $$\min_{\mathbf{w}} h_k(\mathbf{w}; \mathbf{w}^t) = \mathcal{L}_k(\mathbf{w}) + \frac{\mu}{2} \|\mathbf{w} - \mathbf{w}^t\|^2$$
  Here $\mu = 0.01$. The proximal term penalizes deviation from the global model during local optimization, pulling local updates toward $\mathbf{w}^t$. Importantly, this term does not impose a hard bound on total parameter drift, but continuously scales the effective restoring force with Euclidean distance.

---

## 3. Five-Seed Experimental Benchmark Results

Experiments were executed across five canonical seeds: **42, 123, 2026, 3407, 7777**. All five paired runs (30 rounds FedAvg + 30 rounds FedProx) executed to completion without data leaks or restarts.

### 3.1 Primary Performance Results Table

| Seed | Dirichlet Client Counts | Client 5 Noisy Samples | FedAvg Test Acc (%) | FedProx Test Acc (%) | Paired Delta (pp) | FedAvg Macro-F1 | FedProx Macro-F1 | Delta F1 | FedAvg Bal Acc (%) | FedProx Bal Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `[24650, 30566, 14388, 12497, 7895]` | 3,158 / 7,895 (40.0%) | 79.624 | 79.638 | **+0.014** | 0.7403 | 0.7413 | +0.0010 | 74.357 | 75.428 |
| **123** | `[21522, 19760, 14721, 13041, 20952]` | 8,381 / 20,952 (40.0%) | 70.738 | 70.292 | **-0.446** | 0.6181 | 0.6136 | -0.0045 | 64.794 | 64.316 |
| **2026** | `[6502, 19450, 28146, 15764, 20134]` | 8,054 / 20,134 (40.0%) | 57.730 | 57.981 | **+0.251** | 0.5186 | 0.5133 | -0.0053 | 56.294 | 56.307 |
| **3407** | `[20279, 14263, 11886, 30337, 13231]` | 5,292 / 13,231 (40.0%) | 75.195 | 71.671 | **-3.524** | 0.6214 | 0.5957 | -0.0257 | 65.631 | 63.183 |
| **7777** | `[11406, 6714, 20804, 30392, 20680]` | 8,272 / 20,680 (40.0%) | 61.114 | 61.811 | **+0.697** | 0.5057 | 0.5030 | -0.0027 | 54.802 | 55.278 |
| **Mean** | — | — | **68.880** | **68.279** | **-0.602** | **0.6008** | **0.5934** | **-0.0075** | **63.176** | **62.902** |
| **Std** | — | — | $\pm 9.266$ | $\pm 8.551$ | $\pm 1.685$ | $\pm 0.0948$ | $\pm 0.0960$ | $\pm 0.0105$ | $\pm 7.924$ | $\pm 8.072$ |

*Note: Differences are reported in percentage points (pp) for accuracy metrics, defined as $\text{Metric}_{\text{FedProx}} - \text{Metric}_{\text{FedAvg}}$. Sample standard deviations are calculated with $ddof = 1$.*

### 3.2 Formal Statistical Inference ($df = 4$)

We evaluated the null hypothesis $H_0: \mu_{\text{FedProx}} - \mu_{\text{FedAvg}} = 0$ against the two-sided alternative $H_1: \mu_{\text{FedProx}} - \mu_{\text{FedAvg}} \neq 0$:

1. **Test Accuracy:**
   - Mean Paired Difference: $\bar{d} = -0.602\text{ pp}$ ($s_d = 1.685\text{ pp}$, $\text{SE} = 0.754\text{ pp}$)
   - Paired Student-$t$ Statistic: $t(4) = -0.7983$, $p = 0.46938$ (two-sided, non-significant at $\alpha = 0.05$)
   - $95\%$ Confidence Interval: $[-2.694\text{ pp}, +1.491\text{ pp}]$
   - Non-parametric Wilcoxon Signed-Rank Test: $W = 7.0, p = 1.000$

2. **Test Macro-F1:**
   - Mean Paired Difference: $\bar{d} = -0.0075$ ($s_d = 0.0105$, $\text{SE} = 0.0047$)
   - Paired Student-$t$ Statistic: $t(4) = -1.5908$, $p = 0.18686$ (two-sided, non-significant)
   - $95\%$ Confidence Interval: $[-0.0205, +0.0056]$
   - Wilcoxon Signed-Rank Test: $W = 1.0, p = 0.125$

3. **Test Balanced Accuracy:**
   - Mean Paired Difference: $\bar{d} = -0.273\text{ pp}$ ($s_d = 1.344\text{ pp}$, $\text{SE} = 0.601\text{ pp}$)
   - Paired Student-$t$ Statistic: $t(4) = -0.4546$, $p = 0.67294$ (two-sided, non-significant)
   - $95\%$ Confidence Interval: $[-1.942\text{ pp}, +1.395\text{ pp}]$
   - Wilcoxon Signed-Rank Test: $W = 7.0, p = 1.000$

4. **Best Validation Accuracy:**
   - Mean Paired Difference: $\bar{d} = -0.520\text{ pp}$ ($s_d = 0.166\text{ pp}$, $\text{SE} = 0.074\text{ pp}$)
   - Paired Student-$t$ Statistic: $t(4) = -7.0096$, $p = 0.00218$
   - $95\%$ Confidence Interval: $[-0.726\text{ pp}, -0.314\text{ pp}]$
   - *Interpretation:* Peak global validation accuracy was consistently slightly higher for FedAvg across all five seeds ($0.27$ to $0.68\text{ pp}$ advantage), reflecting the slight regularizing drag imposed by the proximal term $\mu = 0.01$ on clean client optimization.

5. **Test Loss:**
   - FedAvg Mean = $0.9082 \pm 0.1897$; FedProx Mean = $0.9239 \pm 0.1877$
   - Mean Paired Difference = $+0.0156 \pm 0.0407$ ($t(4) = 0.8566$, $p = 0.43994$, 95% CI: $[-0.0350, +0.0662]$)

---

## 4. Per-Class Histopathological Generalization Analysis

Per-class F1 metrics were calculated as the unweighted arithmetic mean across the five independent seed-level evaluations (rather than from pooled predictions across seeds):

| Class Index | Histopathological Tissue Type | Abbr | FedAvg Mean F1 | FedProx Mean F1 | Delta F1 | Sample Support (Test) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | Adipose Tissue | ADI | 0.863 | 0.869 | +0.006 | 1,338 |
| **1** | Background | BACK | 0.796 | 0.803 | +0.007 | 847 |
| **2** | Debris | DEB | 0.281 | 0.279 | -0.002 | 339 |
| **3** | Lymphocytes | LYM | 0.693 | 0.700 | +0.007 | 634 |
| **4** | Mucus | MUC | 0.546 | 0.510 | -0.036 | 1,035 |
| **5** | Smooth Muscle | MUS | 0.447 | 0.431 | -0.016 | 592 |
| **6** | Normal Colon Mucosa | NORM | 0.589 | 0.565 | -0.024 | 741 |
| **7** | Cancer-Associated Stroma | STR | 0.385 | 0.381 | -0.004 | 421 |
| **8** | Colorectal Adenocarcinoma Epithelium | TUM | 0.782 | 0.770 | -0.012 | 1,233 |

- Morphologically distinct tissues with substantial support (Adipose $F1 \approx 0.87$, Adenocarcinoma $F1 \approx 0.78$, Background $F1 \approx 0.80$) remained stable across both algorithms.
- Structurally heterogeneous minority classes (Debris $F1 \approx 0.28$, Stroma $F1 \approx 0.38$) suffered severe degradation under non-IID partitioning and label corruption, with proximal regularization providing no measurable protection.

---

## 5. Visual Deliverables & Graphical Analysis

### Figure 1: Test Accuracy Comparison Across All 5 Seeds
![FIG-P7-01 Test Accuracy](figures/FIG-P7-01_test_accuracy_comparison.png)  
*Figure 1: Grouped bar chart comparing held-out test accuracy across all five random seeds, with paired difference callouts in percentage points.*

### Figure 2: Test Macro-F1 Comparison Across All 5 Seeds
![FIG-P7-02 Macro-F1](figures/FIG-P7-02_macro_f1_comparison.png)  
*Figure 2: Grouped bar chart of test Macro-F1 scores across all five seeds.*

### Figure 3: Test Balanced Accuracy Comparison Across All 5 Seeds
![FIG-P7-03 Balanced Accuracy](figures/FIG-P7-03_balanced_accuracy_comparison.png)  
*Figure 3: Grouped bar chart of test Balanced Accuracy across all five seeds.*

### Figure 4: Validation Learning Curves Across 30 Communication Rounds
![FIG-P7-04 Learning Curves](figures/FIG-P7-04_validation_learning_curves.png)  
*Figure 4: Global validation accuracy convergence trajectories for all five seeds and the 5-seed aggregate mean curve with $\pm 1$ sample standard deviation bands.*

### Figure 5: Aggregate Benchmark Metrics (Mean $\pm$ 1 Sample Standard Deviation)
![FIG-P7-05 Aggregate Metrics](figures/FIG-P7-05_aggregate_metrics_mean_std.png)  
*Figure 5: Benchmark summary comparing FedAvg and FedProx across key performance indicators with exact sample standard deviation error bars (ddof=1).*

### Figure 6: Per-Class Generalization Comparison Across 9 Tissue Classes
![FIG-P7-06 Per-Class F1](figures/FIG-P7-06_per_class_f1_comparison.png)  
*Figure 6: Per-class F1 score comparison across the 9 PathMNIST histopathology tissue types.*

---

## 6. Scientific Discussion & Update Dynamics

### 6.1 Why FedProx Does Not Neutralize Label Noise
FedProx was originally designed to bound local model drift under statistical heterogeneity by penalizing $\frac{\mu}{2} \|\mathbf{w} - \mathbf{w}^t\|^2$. However:
1. **Directional vs. Step-Size Error:** Under $40\%$ symmetric label noise, Client 5 computes cross-entropy gradients with respect to incorrect target classes. The proximal term penalizes the Euclidean distance $\|\mathbf{w}_5 - \mathbf{w}^t\|$, which moderates the magnitude of parameter movement, but does **not correct the erroneous directional orientation** of the gradient updates.
2. **Aggregated Impact:** When aggregated via sample weighting, Client 5's corrupted model delta is integrated into the global model proportionally to its sample count ($n_5 / N$), contaminating the global update regardless of the proximal penalty.

### 6.2 Update Cosine Similarities: Label Noise and Class Skew
Across the round logs, negative pairwise cosine similarities were observed between Client 5 and clean clients (ranging from $-0.04$ to $+0.27$, compared to clean-clean similarities of $+0.29$ to $+0.62$). Importantly, **label noise alone cannot be claimed as the sole cause of negative cosine similarities**: Dirichlet label skew ($\alpha = 0.3$) creates severe client-level class imbalance, which independently produces orthogonal or divergent gradient vectors across clients. Both factors jointly contribute to update discordance.

---

## 7. Computational Runtime & Execution Analysis

Experiments were conducted on an AMD Ryzen 5 5600H processor configured for dual-thread PyTorch execution. CUDA availability and GPU capabilities were verified during environment pre-flight diagnostics, but **the federated training pipeline was executed entirely on the CPU** in compliance with project configuration:

| Seed | FedAvg Wall-Clock Time | FedProx Wall-Clock Time | Total Seed Wall-Clock |
| :---: | :---: | :---: | :---: |
| **42** | 54.41 min | 58.30 min | 112.76 min |
| **123** | 54.66 min | 43.38 min | 98.09 min |
| **2026** | 26.19 min | 89.46 min | 115.72 min |
| **3407** | 54.70 min | 41.78 min | 96.52 min |
| **7777** | 24.34 min | 24.49 min | 48.87 min |
| **Total** | **214.30 min** | **257.41 min** | **471.71 min (~7.86 hrs)** |
| **Mean** | **42.86 $\pm$ 16.03 min** | **51.48 $\pm$ 24.38 min** | **94.34 $\pm$ 26.54 min** |

The algorithmic overhead of calculating proximal gradients added approximately $0.5\% - 7.1\%$ compute time during stable CPU power states. Variations between seeds reflect operating system power management and core frequency throttling across multi-hour background runs.

---

## 8. Limitations & Scope of Findings

1. **Statistical Power ($N = 5$):** The sample size ($N = 5$ seeds) provides limited power to detect differences smaller than $1.5\text{ pp}$. The $95\%$ confidence interval on accuracy difference ($[-2.694\text{ pp}, +1.491\text{ pp}]$) bounds the plausible effect size, but does not prove statistical equivalence.
2. **Fixed Hyperparameters:** The benchmark evaluated a single proximal coefficient ($\mu = 0.01$), single learning rate ($\eta = 0.01$), and constant batch size ($B = 32$). Results should not be extrapolated to larger values of $\mu$ (e.g., $\mu = 0.1$ or $1.0$) without empirical verification.
3. **Synthetic Federation Scope:** Findings reflect single-client $40\%$ symmetric label noise under Dirichlet $\alpha = 0.3$ on PathMNIST; multi-hospital deployments with complex multi-client noise or non-IID feature shift may exhibit different dynamics.

---

## 9. Conclusion & Research Recommendations

Across 300 communication rounds covering five random seeds on PathMNIST, **no statistically significant performance advantage was detected for FedProx ($\mu = 0.01$) over standard FedAvg** in the presence of Dirichlet non-IID partitioning and single-client symmetric label noise ($p = 0.469$).

For digital histopathology federated learning deployments facing label noise, algorithmic design must move beyond generic proximal drift penalties toward explicit noise-aware paradigms:
1. **Noise-Robust Loss Functions:** Integrating Generalized Cross-Entropy (GCE) or Symmetric Cross-Entropy (SCE) into local client objectives.
2. **Adaptive Server Reweighting:** Applying cosine similarity filtering or client loss variance tracking to dynamically down-weight suspicious client updates during aggregation.
3. **Loss-Based Sample Pruning:** Implementing small-loss sample selection algorithms locally prior to backward propagation.

---

## References

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *Artificial Intelligence and Statistics (AISTATS)*, PMLR 54:1273-1282.
2. Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). Federated Optimization in Heterogeneous Networks. *Proceedings of Machine Learning and Systems (MLSys)*, 2:429-450.
3. Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10:41.
4. Kairouz, P., et al. (2021). Advances and Open Problems in Federated Learning. *Foundations and Trends in Machine Learning*, 14(1–2):1–210.
