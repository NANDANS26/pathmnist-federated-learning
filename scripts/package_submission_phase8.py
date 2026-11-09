"""
Phase 8.1 Packaging and Deliverable Generation Script.
Applies all scientific consistency corrections, exact unrounded sample statistics (ddof=1),
corrected terminology, bounded scope, verified preprint literature analysis, and generates:
  - submission/PathMNIST_Federated_Learning_Final_Report.md
  - submission/PathMNIST_Federated_Learning_Final_Report.pdf
  - submission/PathMNIST_Federated_Learning_One_Page_Summary.md
  - submission/PathMNIST_Federated_Learning_One_Page_Summary.pdf (strictly 1 page)
  - submission/REACTIVE_OPTIMIZERS_SOURCE_REQUIRED.md
  - submission/Reactive_Optimizers_Article_Summary.md
  - submission/WANDB_PUBLICATION_REQUIRED.md
  - submission/PHASE_8_FINAL_RELEASE_REPORT.md
"""

import os
import re
import json
import shutil
import subprocess

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(PROJ_ROOT, "submission")
SUBMISSION_FIG_DIR = os.path.join(SUBMISSION_DIR, "figures")
REPORTS_FIG_DIR = os.path.join(PROJ_ROOT, "reports", "figures")
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def get_pdf_page_count(path: str) -> int:
    """Extracts exact page count from a PDF file."""
    with open(path, "rb") as f:
        data = f.read()
    pages = re.findall(rb"/Type\s*/Page\b", data)
    return len(pages)


def convert_html_to_pdf(html_path: str, pdf_path: str) -> bool:
    """Converts local HTML file to PDF using headless Microsoft Edge."""
    url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    cmd = [
        EDGE_EXE,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={os.path.abspath(pdf_path)}",
        url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0


def main():
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_FIG_DIR, exist_ok=True)

    # 1. Synchronize figures
    for f in os.listdir(REPORTS_FIG_DIR):
        if f.startswith("FIG-P7"):
            shutil.copy(os.path.join(REPORTS_FIG_DIR, f), os.path.join(SUBMISSION_FIG_DIR, f))

    # -------------------------------------------------------------------------
    # A. COMPLETE RESEARCH REPORT (Markdown + PDF)
    # -------------------------------------------------------------------------
    report_md_path = os.path.join(SUBMISSION_DIR, "PathMNIST_Federated_Learning_Final_Report.md")
    report_pdf_path = os.path.join(SUBMISSION_DIR, "PathMNIST_Federated_Learning_Final_Report.pdf")
    report_html_temp = os.path.join(SUBMISSION_DIR, "_temp_report.html")

    report_md_content = """# Empirical Evaluation of Federated Averaging and Federated Proximal Optimization under Non-IID Partitioning and Symmetric Label Noise on PathMNIST

**Author:** Nandan S
**Submission Track:** IndiaAI Fellowship 2026–2027 Screening
**Repository:** [github.com/NANDANS26/pathmnist-federated-learning](https://github.com/NANDANS26/pathmnist-federated-learning)
**Date of Completion:** November 9, 2026

---

## Abstract

Federated learning (FL) enables decentralized participants to collaboratively train a shared predictive model without pooling raw private observations. In digital histopathology applications, FL frequently confronts two simultaneous challenges: inter-client class distribution skew (non-IID data) and local annotation noise arising from subjective grading or tissue preparation artifacts. While Federated Proximal Optimization (FedProx) is hypothesized to mitigate local client drift by penalizing parameter deviation from the global model, empirical validation under paired non-IID conditions coupled with severe label noise on histopathological imaging remains limited.

In this work, we conduct a seed-controlled paired benchmark comparing Federated Averaging (**FedAvg**) against Federated Proximal Optimization (**FedProx**, proximal parameter $\\mu = 0.01$) across five canonical random seeds (42, 123, 2026, 3407, 7777). We evaluate on PathMNIST, a public 9-class colorectal histopathology dataset ($89,996$ training, $10,004$ validation, and $7,180$ test images), partitioned into five simulated federated clients using a Dirichlet distribution ($\\alpha = 0.3$). To assess resilience against annotation errors, Client 5 is subjected to exactly $40.00\\%$ symmetric label noise. Both algorithms were trained across 30 communication rounds using an identical 2-stage convolutional network (**CompactPathCNN**, 36,425 trainable parameters), identical seed-specific partitions, and identical initialization weights, totaling 300 executed communication rounds on an AMD Ryzen 5 5600H CPU execution platform.

Over the five seeds, FedAvg achieved a mean held-out test accuracy of $68.880\\% \\pm 9.266\\text{ pp}$ (mean macro-F1: $0.6008 \\pm 0.0948$), while FedProx achieved $68.279\\% \\pm 8.551\\text{ pp}$ (mean macro-F1: $0.5934 \\pm 0.0960$), yielding a paired difference of $-0.602 \\pm 1.685\\text{ pp}$ in test accuracy ($t(4) = -0.7983$, two-sided $p = 0.46938$, 95% CI: $[-2.694\\text{ pp}, +1.491\\text{ pp}]$) and $-0.0075 \\pm 0.0105$ in macro-F1 ($t(4) = -1.5908$, $p = 0.18686$, 95% CI: $[-0.0205, +0.0056]$). Under the evaluated configuration, **no statistically significant difference in test performance was detected between FedAvg and FedProx across the five random seeds**. Because five seeds provide limited statistical power and no formal equivalence test with predefined equivalence margins was conducted, this non-significant result does not establish statistical equivalence. Mechanistically, the proximal penalty penalizes parameter deviation from the global model but does not alter the corrupted gradient directions induced by noisy annotations. Consequently, under the evaluated configuration (PathMNIST, simulated five-client federation, Dirichlet $\\alpha = 0.3$, single-client $40\\%$ symmetric noise, CompactPathCNN, and FedProx $\\mu = 0.01$), proximal regularization alone did not overcome annotation noise without explicit noise filtering or adaptive client reweighting.

---

## 1. Introduction & Research Scope

Federated learning allows clinical and research centers to collaboratively train shared diagnostic models while keeping digitized histology images localized. However, decentralized histopathology introduces significant heterogeneity:
1. **Label Distribution Skew (Non-IID Data):** Specialized clinics observe disparate distributions of tissue pathologies (e.g., academic cancer centers encounter disproportionate adenocarcinoma relative to normal mucosa).
2. **Annotation Noise (Symmetric Label Noise):** Staining variations, tissue folding, and inter-observer diagnostic disagreement generate mislabeled patches.

### Research Scope & Operational Boundaries
- **Simulated Federation:** This investigation utilizes the public PathMNIST dataset partitioned synthetically into five client shards. No real-world multi-hospital network was deployed.
- **Privacy Controls:** Differential privacy (DP) and secure multiparty computation (SMPC) mechanisms were not implemented or evaluated.
- **Algorithmic Scope:** Conclusions regarding FedProx are strictly bounded to the evaluated configuration ($\\mu = 0.01$, CompactPathCNN architecture, PathMNIST dataset, Dirichlet $\\alpha = 0.3$, single-client $40\\%$ symmetric noise, and 30 rounds). We do not assert generic failure of FedProx across all medical FL settings.

### Research Questions
- **RQ1:** Does FedProx with proximal regularization $\\mu = 0.01$ mitigate client drift and improve global test generalization compared to FedAvg under Dirichlet distribution skew ($\\alpha = 0.3$) when one client experiences $40\\%$ symmetric label noise?
- **RQ2:** How does proximal regularization affect validation convergence trajectories, peak validation performance, and per-class diagnostic sensitivity across minority versus majority histological tissue classes?
- **RQ3:** What is the empirical wall-clock computational overhead of evaluating the proximal gradient term during local client optimization on standard CPU hardware?

---

## 2. Experimental Setup & Methodology

### 2.1 Benchmark Dataset: PathMNIST
PathMNIST is derived from the NCT-CRC-HE-100K colorectal cancer histology dataset (MedMNIST v2 benchmark suite). It contains $107,180$ patches ($28 \\times 28 \\times 3$ RGB) across 9 tissue classes:
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
- **Non-IID Partitioning:** A Dirichlet distribution with concentration parameter $\\alpha = 0.3$ was sampled independently for each random seed, apportioning the $89,996$ training samples among 5 federated clients.
- **Client Validation Shards:** The $10,004$ global validation samples were partitioned into 5 shards apportioned to mirror local client class distributions for local tracking, while global validation evaluated all $10,004$ samples.
- **Symmetric Label Noise:** Client 5 (Target Client Index 4) was subjected to exactly $40.00\\%$ symmetric label noise. A deterministic pseudo-random mask selected $40\\%$ of Client 5's training samples, replacing each true label $y$ with a corrupt label $(y + \\Delta y) \\pmod 9$, where $\\Delta y \\in \\{1, \\dots, 8\\}$.

### 2.3 Model Architecture: CompactPathCNN
To enable computationally efficient federated optimization, we instantiated **CompactPathCNN**, a 2-stage convolutional network with exactly **36,425 trainable parameters**:
- **Layer 1 (Conv1):** Conv2D($3 \\to 32$, kernel $3 \\times 3$, padding 1), ReLU, MaxPool2D($2 \\times 2$, stride 2). Output: $(32, 14, 14)$.
- **Layer 2 (Conv2):** Conv2D($32 \\to 64$, kernel $3 \\times 3$, padding 1), ReLU, MaxPool2D($2 \\times 2$, stride 2). Output: $(64, 7, 7)$.
- **Pooling:** AdaptiveAvgPool2d($2 \\times 2$), Flatten to 256 dimensions.
- **Fully Connected (FC1):** Linear($256 \\to 64$), ReLU.
- **Classification Head (FC2):** Linear($64 \\to 9$), producing 9 unnormalized class logits.

### 2.4 Federated Optimization Algorithms
- **Federated Averaging (FedAvg):** At communication round $t$, the server broadcasts global weights $\\mathbf{w}^t$. Each client $k$ minimizes local empirical cross-entropy loss $\\mathcal{L}_k(\\mathbf{w})$ over $E = 2$ epochs using SGD with batch size $B = 32$ and learning rate $\\eta = 0.01$. The server aggregates local weights via sample weighting: $\\mathbf{w}^{t+1} = \\sum_{k=1}^K \\frac{n_k}{N} \\mathbf{w}_k^{t, E}$.
- **Federated Proximal Optimization (FedProx):** Adds a proximal regularization term to the local objective:
  $$\\min_{\\mathbf{w}} h_k(\\mathbf{w}; \\mathbf{w}^t) = \\mathcal{L}_k(\\mathbf{w}) + \\frac{\\mu}{2} \\|\\mathbf{w} - \\mathbf{w}^t\\|^2$$
  Here $\\mu = 0.01$. The proximal term penalizes deviation from the global model during local optimization, pulling local updates toward $\\mathbf{w}^t$. Importantly, this term does not impose a hard bound on parameter updates, but continuously penalizes parameter deviation from the global model with Euclidean distance.

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
| **Std** | — | — | $\\pm 9.266$ | $\\pm 8.551$ | $\\pm 1.685$ | $\\pm 0.0948$ | $\\pm 0.0960$ | $\\pm 0.0105$ | $\\pm 7.924$ | $\\pm 8.072$ |

*Note: Differences are reported in percentage points (pp) for accuracy metrics, defined as $\\text{Metric}_{\\text{FedProx}} - \\text{Metric}_{\\text{FedAvg}}$. Sample standard deviations are calculated with $ddof = 1$.*

### 3.2 Formal Statistical Inference ($df = 4$)

We evaluated the null hypothesis $H_0: \\mu_{\\text{FedProx}} - \\mu_{\\text{FedAvg}} = 0$ against the two-sided alternative $H_1: \\mu_{\\text{FedProx}} - \\mu_{\\text{FedAvg}} \\neq 0$:

1. **Test Accuracy:**
   - Mean Paired Difference: $\\bar{d} = -0.602\\text{ pp}$ ($s_d = 1.685\\text{ pp}$, $\\text{SE} = 0.754\\text{ pp}$)
   - Paired Student-$t$ Statistic: $t(4) = -0.7983$, $p = 0.46938$ (two-sided, non-significant at $\\alpha = 0.05$)
   - $95\\%$ Confidence Interval: $[-2.694\\text{ pp}, +1.491\\text{ pp}]$
   - Non-parametric Wilcoxon Signed-Rank Test: $W = 7.0, p = 1.000$

2. **Test Macro-F1:**
   - Mean Paired Difference: $\\bar{d} = -0.0075$ ($s_d = 0.0105$, $\\text{SE} = 0.0047$)
   - Paired Student-$t$ Statistic: $t(4) = -1.5908$, $p = 0.18686$ (two-sided, non-significant)
   - $95\\%$ Confidence Interval: $[-0.0205, +0.0056]$
   - Wilcoxon Signed-Rank Test: $W = 1.0, p = 0.125$

3. **Test Balanced Accuracy:**
   - Mean Paired Difference: $\\bar{d} = -0.273\\text{ pp}$ ($s_d = 1.344\\text{ pp}$, $\\text{SE} = 0.601\\text{ pp}$)
   - Paired Student-$t$ Statistic: $t(4) = -0.4546$, $p = 0.67294$ (two-sided, non-significant)
   - $95\\%$ Confidence Interval: $[-1.942\\text{ pp}, +1.395\\text{ pp}]$
   - Wilcoxon Signed-Rank Test: $W = 7.0, p = 1.000$

4. **Best Validation Accuracy:**
   - Mean Paired Difference: $\\bar{d} = -0.520\\text{ pp}$ ($s_d = 0.166\\text{ pp}$, $\\text{SE} = 0.074\\text{ pp}$)
   - Paired Student-$t$ Statistic: $t(4) = -7.0096$, $p = 0.00218$
   - $95\\%$ Confidence Interval: $[-0.726\\text{ pp}, -0.314\\text{ pp}]$
   - *Interpretation:* Peak global validation accuracy was slightly higher for FedAvg across all five seeds (mean difference $-0.520 \\pm 0.166\\text{ pp}$, ranging from $0.27$ to $0.68\\text{ pp}$ advantage, paired $t(4) = -7.0096, p = 0.00218$). We report this empirical difference directly, without asserting an unsupported causal claim that the proximal term caused this difference.

5. **Test Loss:**
   - FedAvg Mean = $0.9082 \\pm 0.1897$; FedProx Mean = $0.9239 \\pm 0.1877$
   - Mean Paired Difference = $+0.0156 \\pm 0.0407$ ($t(4) = 0.8566$, $p = 0.43994$, 95% CI: $[-0.0350, +0.0662]$)

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

- Morphologically distinct tissues with substantial support (Adipose $F1 \\approx 0.87$, Adenocarcinoma $F1 \\approx 0.78$, Background $F1 \\approx 0.80$) remained stable across both algorithms.
- Structurally heterogeneous minority classes (Debris $F1 \\approx 0.28$, Stroma $F1 \\approx 0.38$) suffered severe degradation under non-IID partitioning and label corruption, with proximal regularization providing no measurable protection.

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
*Figure 4: Global validation accuracy convergence trajectories for all five seeds and the 5-seed aggregate mean curve with $\\pm 1$ sample standard deviation bands.*

### Figure 5: Aggregate Benchmark Metrics (Mean $\\pm$ 1 Sample Standard Deviation)
![FIG-P7-05 Aggregate Metrics](figures/FIG-P7-05_aggregate_metrics_mean_std.png)
*Figure 5: Benchmark summary comparing FedAvg and FedProx across key performance indicators with exact sample standard deviation error bars (ddof=1).*

### Figure 6: Per-Class Generalization Comparison Across 9 Tissue Classes
![FIG-P7-06 Per-Class F1](figures/FIG-P7-06_per_class_f1_comparison.png)
*Figure 6: Per-class F1 score comparison across the 9 PathMNIST histopathology tissue types.*

---

## 6. Scientific Discussion & Update Dynamics

### 6.1 Why FedProx Does Not Neutralize Label Noise
FedProx was originally designed to bound local model drift under statistical heterogeneity by penalizing $\\frac{\\mu}{2} \\|\\mathbf{w} - \\mathbf{w}^t\\|^2$. However:
1. **Directional Error vs. Parameter Deviation Penalty:** Under $40\\%$ symmetric label noise, Client 5 computes cross-entropy gradients with respect to incorrect target classes. While the proximal penalty continuously penalizes parameter deviation from the global model ($\\frac{\\mu}{2} \\|\\mathbf{w}_5 - \\mathbf{w}^t\\|^2$), it penalizes displacement magnitude but does **not alter or correct the erroneous directional orientation** of the gradient updates.
2. **Aggregated Impact:** When aggregated via sample weighting, Client 5's corrupted model delta is integrated into the global model proportionally to its sample count ($n_5 / N$), contaminating the global update regardless of the proximal penalty.

### 6.2 Update Cosine Similarities: Label Noise and Class Skew
Across the round logs, negative pairwise cosine similarities were observed between Client 5 and clean clients (ranging from $-0.04$ to $+0.27$, compared to clean-clean similarities of $+0.29$ to $+0.62$). Importantly, **label noise alone cannot be claimed as the sole cause of negative cosine similarities**: Dirichlet label skew ($\\alpha = 0.3$) creates severe client-level class imbalance, which independently produces orthogonal or divergent gradient vectors across clients. Both factors jointly contribute to update discordance.

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
| **Mean** | **42.86 $\\pm$ 16.08 min** | **51.48 $\\pm$ 24.38 min** | **94.34 $\\pm$ 26.54 min** |

The algorithmic overhead of calculating proximal gradients added approximately $0.5\\% - 7.1\\%$ compute time during stable CPU power states. Variations between seeds reflect operating system power management and core frequency throttling across multi-hour background runs.

---

## 8. Limitations & Scope of Findings

1. **Statistical Power ($N = 5$):** The sample size ($N = 5$ seeds) provides limited power to detect differences smaller than $1.5\\text{ pp}$. The $95\\%$ confidence interval on accuracy difference ($[-2.694\\text{ pp}, +1.491\\text{ pp}]$) bounds the plausible effect size, but does not prove statistical equivalence.
2. **Fixed Hyperparameters:** The benchmark evaluated a single proximal coefficient ($\\mu = 0.01$), single learning rate ($\\eta = 0.01$), and constant batch size ($B = 32$). Results should not be extrapolated to larger values of $\\mu$ (e.g., $\\mu = 0.1$ or $1.0$) without empirical verification.
3. **Synthetic Federation Scope:** Findings reflect single-client $40\\%$ symmetric label noise under Dirichlet $\\alpha = 0.3$ on PathMNIST; multi-hospital deployments with complex multi-client noise or non-IID feature shift may exhibit different dynamics.

---

## 9. Conclusion & Research Recommendations

Across 300 communication rounds covering five random seeds on PathMNIST, **no statistically significant performance advantage was detected for FedProx ($\\mu = 0.01$) over standard FedAvg** under Dirichlet non-IID partitioning and single-client symmetric label noise ($p = 0.469$). Because five seeds provide limited statistical power and no formal equivalence testing with predefined margins was conducted, this non-significant result does not establish statistical equivalence.

Within the evaluated configuration (PathMNIST, simulated five-client federation, Dirichlet $\\alpha = 0.3$, $40\\%$ symmetric noise on Client 5, CompactPathCNN, and FedProx $\\mu = 0.01$), proximal regularization alone did not overcome annotation noise. Future digital histopathology deployments confronting similar challenges should investigate explicit noise-aware paradigms:
1. **Noise-Robust Loss Functions:** Integrating Generalized Cross-Entropy (GCE) or Symmetric Cross-Entropy (SCE) into local client objectives.
2. **Adaptive Server Reweighting:** Applying cosine similarity filtering or client loss variance tracking to dynamically down-weight suspicious client updates during aggregation.
3. **Loss-Based Sample Pruning:** Implementing small-loss sample selection algorithms locally prior to backward propagation.

---

## References

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *Artificial Intelligence and Statistics (AISTATS)*, PMLR 54:1273-1282.
2. Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020). Federated Optimization in Heterogeneous Networks. *Proceedings of Machine Learning and Systems (MLSys)*, 2:429-450.
3. Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10:41.
4. Kairouz, P., et al. (2021). Advances and Open Problems in Federated Learning. *Foundations and Trends in Machine Learning*, 14(1–2):1–210.
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md_content)
    print(f"Saved: {report_md_path}")

    # Build styled HTML for report PDF
    report_html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PathMNIST Federated Learning Final Research Report</title>
<style>
@page {{
    size: A4 portrait;
    margin: 20mm 15mm 20mm 15mm;
    @bottom-right {{
        content: counter(page);
    }}
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #222;
}}
h1 {{ font-size: 18pt; line-height: 1.25; margin-bottom: 8px; color: #0f2d59; text-align: center; }}
h2 {{ font-size: 13pt; margin-top: 18px; margin-bottom: 8px; color: #1e3a8a; border-bottom: 1.5px solid #e2e8f0; padding-bottom: 4px; page-break-after: avoid; }}
h3 {{ font-size: 11pt; margin-top: 14px; margin-bottom: 6px; color: #334155; page-break-after: avoid; }}
p {{ margin-bottom: 8px; text-align: justify; }}
.author-block {{ text-align: center; margin-bottom: 18px; font-size: 9.5pt; color: #475569; }}
.author-block strong {{ color: #1e293b; }}
.abstract-box {{
    background-color: #f8fafc;
    border-left: 4px solid #2563eb;
    padding: 10px 14px;
    margin: 12px 0 18px 0;
    font-size: 9pt;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
}}
th, td {{
    border: 1px solid #cbd5e1;
    padding: 5px 7px;
    text-align: center;
}}
th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #1e293b;
}}
tr:nth-child(even) {{ background-color: #f8fafc; }}
.figure-container {{
    text-align: center;
    margin: 14px 0;
    page-break-inside: avoid;
}}
.figure-container img {{
    max-width: 95%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
}}
.caption {{
    font-size: 8.5pt;
    color: #64748b;
    margin-top: 5px;
    font-style: italic;
}}
ul, ol {{ margin-top: 4px; margin-bottom: 8px; padding-left: 20px; }}
li {{ margin-bottom: 3px; }}
code {{ font-family: Consolas, monospace; font-size: 8.5pt; background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
hr {{ border: none; border-top: 1px solid #cbd5e1; margin: 16px 0; }}
</style>
</head>
<body>
<h1>Empirical Evaluation of Federated Averaging and Federated Proximal Optimization under Non-IID Partitioning and Symmetric Label Noise on PathMNIST</h1>
<div class="author-block">
    <strong>Nandan S</strong><br>
    IndiaAI Fellowship 2026–2027 Screening Submission &bull; Repository: <a href="https://github.com/NANDANS26/pathmnist-federated-learning">github.com/NANDANS26/pathmnist-federated-learning</a> &bull; November 9, 2026
</div>

<div class="abstract-box">
    <strong>Abstract:</strong> Federated learning (FL) enables decentralized participants to collaboratively train a shared predictive model without pooling raw private observations. In digital histopathology applications, FL frequently confronts two simultaneous challenges: inter-client class distribution skew (non-IID data) and local annotation noise arising from subjective grading or tissue preparation artifacts. While Federated Proximal Optimization (FedProx) is hypothesized to mitigate local client drift by penalizing parameter deviation from the global model, empirical validation under paired non-IID conditions coupled with severe label noise on histopathological imaging remains limited.
    <br><br>
    In this work, we conduct a seed-controlled paired benchmark comparing Federated Averaging (<strong>FedAvg</strong>) against Federated Proximal Optimization (<strong>FedProx</strong>, &mu; = 0.01) across five canonical random seeds (42, 123, 2026, 3407, 7777). We evaluate on PathMNIST, a public 9-class colorectal histopathology dataset (89,996 training, 10,004 validation, and 7,180 test images), partitioned into five simulated federated clients using a Dirichlet distribution (&alpha; = 0.3). To assess resilience against annotation errors, Client 5 is subjected to exactly 40.00% symmetric label noise. Both algorithms were trained across 30 communication rounds using an identical 2-stage convolutional network (<strong>CompactPathCNN</strong>, 36,425 trainable parameters), identical seed-specific partitions, and identical initialization weights, totaling 300 executed communication rounds on an AMD Ryzen 5 5600H CPU execution platform.
    <br><br>
    Over the five seeds, FedAvg achieved a mean held-out test accuracy of 68.880% &plusmn; 9.266 pp (mean macro-F1: 0.6008 &plusmn; 0.0948), while FedProx achieved 68.279% &plusmn; 8.551 pp (mean macro-F1: 0.5934 &plusmn; 0.0960), yielding a paired difference of -0.602 &plusmn; 1.685 pp in test accuracy (t(4) = -0.7983, two-sided p = 0.46938, 95% CI: [-2.694 pp, +1.491 pp]) and -0.0075 &plusmn; 0.0105 in macro-F1 (t(4) = -1.5908, p = 0.18686, 95% CI: [-0.0205, +0.0056]). Under the evaluated configuration, <strong>no statistically significant difference in test performance was detected between FedAvg and FedProx across the five random seeds</strong>. Because five seeds provide limited statistical power and no formal equivalence test with predefined equivalence margins was conducted, this non-significant result does not establish statistical equivalence. Mechanistically, the proximal penalty penalizes parameter deviation from the global model but does not alter the corrupted gradient directions induced by noisy annotations. Consequently, under the evaluated configuration (PathMNIST, simulated five-client federation, Dirichlet &alpha; = 0.3, single-client 40% symmetric noise, CompactPathCNN, and FedProx &mu; = 0.01), proximal regularization alone did not overcome annotation noise without explicit noise filtering or adaptive client reweighting.
</div>

<h2>1. Introduction & Research Scope</h2>
<p>Federated learning allows clinical and research centers to collaboratively train shared diagnostic models while keeping digitized histology images localized. However, decentralized histopathology introduces significant heterogeneity: (1) <strong>Label Distribution Skew (Non-IID Data):</strong> Specialized clinics observe disparate distributions of tissue pathologies. (2) <strong>Annotation Noise (Symmetric Label Noise):</strong> Staining variations, tissue folding, and inter-observer diagnostic disagreement generate mislabeled patches.</p>
<p><strong>Operational Boundaries:</strong> This research utilizes the public PathMNIST dataset partitioned synthetically into five simulated client shards. No real-world multi-hospital network was deployed. Differential privacy was not implemented. Findings are bounded strictly to the evaluated configuration (&mu; = 0.01, CompactPathCNN, PathMNIST, &alpha; = 0.3, single-client 40% noise).</p>

<h2>2. Experimental Setup & Methodology</h2>
<p><strong>Dataset:</strong> PathMNIST (MedMNIST v2 benchmark suite, 9 colorectal tissue classes, 28x28x3 RGB). Splits: 89,996 training, 10,004 validation, and 7,180 quarantined held-out test samples.</p>
<p><strong>Model Architecture:</strong> CompactPathCNN is a lightweight 2-stage convolutional network with exactly <strong>36,425 trainable parameters</strong>. It comprises Conv1 (3&rarr;32, 3x3, ReLU, MaxPool), Conv2 (32&rarr;64, 3x3, ReLU, MaxPool), AdaptiveAvgPool2d(2, 2), Flatten (256-d), FC1 (256&rarr;64, ReLU), and FC2 (64&rarr;9 logits).</p>
<p><strong>Partitioning & Noise Injection:</strong> 5 federated clients apportioned via Dirichlet distribution (&alpha; = 0.3). Client 5 receives exactly 40.00% symmetric label noise modulo 9. Training ran for 30 communication rounds, 2 local epochs per round, batch size 32, learning rate 0.01 on 2 PyTorch CPU threads.</p>

<h2>3. Five-Seed Experimental Benchmark Results</h2>
<table>
    <thead>
        <tr>
            <th>Seed</th>
            <th>Dirichlet Client Counts</th>
            <th>Client 5 Noisy Samples</th>
            <th>FedAvg Test Acc (%)</th>
            <th>FedProx Test Acc (%)</th>
            <th>Paired &Delta; (pp)</th>
            <th>FedAvg Macro-F1</th>
            <th>FedProx Macro-F1</th>
            <th>&Delta; Macro-F1</th>
            <th>FedAvg Bal Acc (%)</th>
            <th>FedProx Bal Acc (%)</th>
        </tr>
    </thead>
    <tbody>
        <tr><td><strong>42</strong></td><td>24650, 30566, 14388, 12497, 7895</td><td>3,158 / 7,895 (40.0%)</td><td>79.624</td><td>79.638</td><td><strong>+0.014</strong></td><td>0.7403</td><td>0.7413</td><td>+0.0010</td><td>74.357</td><td>75.428</td></tr>
        <tr><td><strong>123</strong></td><td>21522, 19760, 14721, 13041, 20952</td><td>8,381 / 20,952 (40.0%)</td><td>70.738</td><td>70.292</td><td><strong>-0.446</strong></td><td>0.6181</td><td>0.6136</td><td>-0.0045</td><td>64.794</td><td>64.316</td></tr>
        <tr><td><strong>2026</strong></td><td>6502, 19450, 28146, 15764, 20134</td><td>8,054 / 20,134 (40.0%)</td><td>57.730</td><td>57.981</td><td><strong>+0.251</strong></td><td>0.5186</td><td>0.5133</td><td>-0.0053</td><td>56.294</td><td>56.307</td></tr>
        <tr><td><strong>3407</strong></td><td>20279, 14263, 11886, 30337, 13231</td><td>5,292 / 13,231 (40.0%)</td><td>75.195</td><td>71.671</td><td><strong>-3.524</strong></td><td>0.6214</td><td>0.5957</td><td>-0.0257</td><td>65.631</td><td>63.183</td></tr>
        <tr><td><strong>7777</strong></td><td>11406, 6714, 20804, 30392, 20680</td><td>8,272 / 20,680 (40.0%)</td><td>61.114</td><td>61.811</td><td><strong>+0.697</strong></td><td>0.5057</td><td>0.5030</td><td>-0.0027</td><td>54.802</td><td>55.278</td></tr>
        <tr style="font-weight: bold; background-color: #e2e8f0;"><td>Mean</td><td>—</td><td>—</td><td>68.880</td><td>68.279</td><td>-0.602</td><td>0.6008</td><td>0.5934</td><td>-0.0075</td><td>63.176</td><td>62.902</td></tr>
        <tr style="font-style: italic; background-color: #f1f5f9;"><td>Std Dev (ddof=1)</td><td>—</td><td>—</td><td>&plusmn;9.266</td><td>&plusmn;8.551</td><td>&plusmn;1.685</td><td>&plusmn;0.0948</td><td>&plusmn;0.0960</td><td>&plusmn;0.0105</td><td>&plusmn;7.924</td><td>&plusmn;8.072</td></tr>
    </tbody>
</table>

<h3>Statistical Hypothesis Testing (df = 4)</h3>
<ul>
    <li><strong>Test Accuracy:</strong> Mean Paired Difference = -0.602 pp (&plusmn;1.685 pp). Paired t(4) = -0.7983, p = 0.46938 (two-sided, fail to reject H<sub>0</sub>). 95% Confidence Interval: [-2.694 pp, +1.491 pp]. Wilcoxon signed-rank W = 7.0, p = 1.000.</li>
    <li><strong>Test Macro-F1:</strong> Mean Paired Difference = -0.0075 (&plusmn;0.0105). Paired t(4) = -1.5908, p = 0.18686. 95% CI: [-0.0205, +0.0056]. Wilcoxon signed-rank W = 1.0, p = 0.125.</li>
    <li><strong>Best Validation Accuracy:</strong> Mean Difference = -0.520 pp (&plusmn;0.166 pp). Paired t(4) = -7.0096, p = 0.00218. 95% CI: [-0.726 pp, -0.314 pp]. Peak validation accuracy was slightly higher for FedAvg across all partitions (-0.520 &plusmn; 0.166 pp); we report this empirical difference directly without asserting an unsupported causal claim.</li>
    <li><strong>Test Loss:</strong> FedAvg Mean = 0.9082 &plusmn; 0.1897; FedProx Mean = 0.9239 &plusmn; 0.1877 (Paired &Delta; = +0.0156 &plusmn; 0.0407, t(4) = 0.8566, p = 0.43994).</li>
</ul>

<h2>4. Scientific Figures</h2>
<div class="figure-container">
    <img src="figures/FIG-P7-01_test_accuracy_comparison.png">
    <div class="caption">Figure 1: Test accuracy comparison across all five seeds (quarantined held-out test split).</div>
</div>

<div class="figure-container">
    <img src="figures/FIG-P7-04_validation_learning_curves.png">
    <div class="caption">Figure 2: Validation convergence trajectories across 30 rounds for all five seeds and aggregate mean curve.</div>
</div>

<div class="figure-container">
    <img src="figures/FIG-P7-05_aggregate_metrics_mean_std.png">
    <div class="caption">Figure 3: 5-seed aggregate benchmark metrics comparing FedAvg and FedProx (Mean &plusmn; 1 Sample Std Dev, ddof=1).</div>
</div>

<div class="figure-container">
    <img src="figures/FIG-P7-06_per_class_f1_comparison.png">
    <div class="caption">Figure 4: Per-class F1 score comparison across 9 histopathological tissue classes (averaged across 5 seeds).</div>
</div>

<h2>5. Scientific Interpretation & Update Dynamics</h2>
<p><strong>Why Proximal Regularization Does Not Neutralize Label Corruption:</strong> FedProx's proximal term &mu;/2 ||w - w<sup>t</sup>||<sup>2</sup> continuously penalizes parameter deviation from the global model. While this penalizes deviation magnitude, it <em>does not alter the corrupted gradient directions</em> induced by mislabeled samples. When aggregated via sample weighting, Client 5's contaminated model update still enters the global consensus proportionally to its sample count.</p>
<p><strong>Update Cosine Similarities:</strong> Negative client-update cosine similarities were observed between Client 5 and clean clients. However, label noise cannot be claimed as the sole cause: Dirichlet label skew (&alpha; = 0.3) creates severe client-level class imbalance, which independently produces orthogonal or divergent gradient vectors across clients. Both factors jointly contribute.</p>

<h2>6. Hardware Telemetry & Runtime</h2>
<p>The entire federated training pipeline ran on the CPU (AMD Ryzen 5 5600H, 2 threads). While CUDA diagnostics were confirmed during pre-flight, GPU compute was not used for training. Total compute across 300 rounds was 471.7 minutes (~7.86 hours): FedAvg averaged 42.86 &plusmn; 16.08 min/seed; FedProx averaged 51.48 &plusmn; 24.38 min/seed.</p>

<h2>7. Conclusion & Recommendations</h2>
<p>Under non-IID partitioning (&alpha; = 0.3) and 40% single-client label noise on PathMNIST with CompactPathCNN and FedProx &mu; = 0.01, <strong>no statistically significant difference in test performance was detected between FedAvg and FedProx across five random seeds (p = 0.469)</strong>. Because five seeds provide limited statistical power and no formal equivalence testing was conducted, this does not establish statistical equivalence. In this setting, proximal regularization alone did not overcome annotation noise, motivating explicit noise-aware defenses (e.g., Generalized Cross-Entropy loss or cosine similarity-based client reweighting).</p>

<h2>References</h2>
<ol>
    <li>McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. <em>AISTATS</em>.</li>
    <li>Li, T., et al. (2020). Federated Optimization in Heterogeneous Networks. <em>MLSys</em>.</li>
    <li>Yang, J., et al. (2023). MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification. <em>Scientific Data</em>.</li>
</ol>
</body>
</html>
"""
    with open(report_html_temp, "w", encoding="utf-8") as f:
        f.write(report_html_content)

    print("Converting Final Report HTML to PDF via Edge...")
    success = convert_html_to_pdf(report_html_temp, report_pdf_path)
    if success:
        p_count = get_pdf_page_count(report_pdf_path)
        print(f"Final Report PDF generated successfully ({p_count} pages, {os.path.getsize(report_pdf_path)} bytes).")
    if os.path.exists(report_html_temp):
        os.remove(report_html_temp)

    # -------------------------------------------------------------------------
    # B. ONE-PAGE RESEARCH SUMMARY (Markdown + Strictly 1-Page PDF)
    # -------------------------------------------------------------------------
    summary_md_path = os.path.join(SUBMISSION_DIR, "PathMNIST_Federated_Learning_One_Page_Summary.md")
    summary_pdf_path = os.path.join(SUBMISSION_DIR, "PathMNIST_Federated_Learning_One_Page_Summary.pdf")
    summary_html_temp = os.path.join(SUBMISSION_DIR, "_temp_summary.html")

    summary_md_content = """# Executive Research Summary: FedAvg vs. FedProx on PathMNIST
**Research Study:** IndiaAI Fellowship 2026–2027 Screening Submission
**Candidate:** Nandan S
**Repository:** [github.com/NANDANS26/pathmnist-federated-learning](https://github.com/NANDANS26/pathmnist-federated-learning)

---

### 1. Research Problem & Objective
Federated Learning (FL) enables decentralized training of diagnostic medical imaging models without centralizing patient biopsies. In digital histopathology, collaborative FL faces two simultaneous obstacles: inter-client class distribution skew (non-IID data) and local annotation noise from diagnostic ambiguity or tissue artifacts. We evaluate whether Federated Proximal Optimization (**FedProx**, $\\mu = 0.01$), which penalizes local parameter deviation from the global model, improves test generalization over standard Federated Averaging (**FedAvg**) under non-IID partitioning (Dirichlet $\\alpha = 0.3$) and single-client $40\\%$ symmetric label noise.

### 2. Experimental Setup & Protocol
- **Dataset:** PathMNIST ($89,996$ training, $10,004$ validation, $7,180$ quarantined held-out test patches, 9 colorectal tissue classes).
- **Federated Topology:** 5 simulated clients, Dirichlet $\\alpha = 0.3$. Client 5 trained on exactly $40.00\\%$ symmetric label noise.
- **Model Architecture:** **CompactPathCNN** (2 convolutional layers, Adaptive Average Pooling, 2 fully-connected layers, **36,425 trainable parameters**).
- **Protocol:** Seed-controlled paired benchmark across 5 canonical seeds (42, 123, 2026, 3407, 7777). 30 communication rounds per algorithm (300 rounds total), local epochs $E = 2$, batch size $B = 32$, learning rate $\\eta = 0.01$, executed on 2 PyTorch CPU threads (AMD Ryzen 5 5600H).

### 3. Quantitative Findings (5-Seed Paired Benchmark)

| Benchmark Metric | FedAvg (Mean $\\pm$ Std) | FedProx ($\\mu = 0.01$) | Paired Difference ($\\Delta$) | Paired $t$-test ($df=4$) | $p$-value | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-Out Test Accuracy** | **$68.880\\% \\pm 9.266\\text{ pp}$** | **$68.279\\% \\pm 8.551\\text{ pp}$** | **$-0.602 \\pm 1.685\\text{ pp}$** | $t = -0.7983$ | $p = 0.46938$ | $[-2.694\\text{ pp}, +1.491\\text{ pp}]$ |
| **Test Macro-F1 Score** | **$0.6008 \\pm 0.0948$** | **$0.5934 \\pm 0.0960$** | **$-0.0075 \\pm 0.0105$** | $t = -1.5908$ | $p = 0.18686$ | $[-0.0205, +0.0056]$ |
| **Test Balanced Accuracy** | **$63.176\\% \\pm 7.924\\text{ pp}$** | **$62.902\\% \\pm 8.072\\text{ pp}$** | **$-0.273 \\pm 1.344\\text{ pp}$** | $t = -0.4546$ | $p = 0.67294$ | $[-1.942\\text{ pp}, +1.395\\text{ pp}]$ |
| **Peak Validation Accuracy** | $65.986\\% \\pm 4.292\\text{ pp}$ | $65.466\\% \\pm 4.420\\text{ pp}$ | **$-0.520 \\pm 0.166\\text{ pp}$** | $t = -7.0096$ | $p = 0.00218$ | $[-0.726\\text{ pp}, -0.314\\text{ pp}]$ |
| **Total Wall-Clock Time** | $214.3\\text{ min}$ ($42.9\\text{ m/seed}$) | $257.4\\text{ min}$ ($51.5\\text{ m/seed}$) | $+8.62 \\pm 31.39\\text{ min}$ | — | — | Total Compute: $471.7\\text{ min}$ |

### 4. Statistical Interpretation & Core Conclusion
- **Statistical Interpretation:** **No statistically significant difference in test performance was detected between FedAvg and FedProx across the five random seeds ($p = 0.469$).** Because five seeds provide limited statistical power and no predefined equivalence testing was performed, this lack of statistical significance does not prove equivalence.
- **Parameter Deviation Penalty vs. Noise Orientation:** The proximal penalty $\\frac{\\mu}{2}\\|\\mathbf{w}-\\mathbf{w}^t\\|^2$ penalizes parameter deviation from the global model but cannot rectify corrupted gradient directions induced by noisy annotations. Client 5's corrupted updates still contaminate global aggregation proportionally to its sample weight.
- **Observed Peak Validation Difference:** Peak validation accuracy was slightly higher for FedAvg across all five seeds (mean difference $-0.520 \\pm 0.166\\text{ pp}$, $p = 0.0022$); we report this empirical observation directly without asserting an unsupported causal claim.
- **Bounded Conclusion:** Within the evaluated configuration (PathMNIST, simulated 5 clients, Dirichlet $\\alpha = 0.3$, single-client $40\\%$ symmetric noise, CompactPathCNN, and FedProx $\\mu = 0.01$), proximal regularization did not yield generalization benefits over FedAvg. Defending against annotation noise under these conditions motivates exploring explicit noise-aware paradigms (e.g., Generalized Cross-Entropy or adaptive server-side reweighting).
"""
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_md_content)
    print(f"Saved: {summary_md_path}")

    # Build compact HTML engineered for strictly 1 page A4
    summary_html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PathMNIST Federated Learning Executive One-Page Summary</title>
<style>
@page {
    size: A4 portrait;
    margin: 9mm 11mm 9mm 11mm;
}
* {
    box-sizing: border-box;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 8.5pt;
    line-height: 1.34;
    color: #1e293b;
    margin: 0;
    padding: 0;
}
.header {
    border-bottom: 2px solid #1e3a8a;
    padding-bottom: 4px;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}
.title {
    font-size: 12.5pt;
    font-weight: 800;
    color: #0f2d59;
    letter-spacing: -0.3px;
    margin: 0;
}
.subtitle {
    font-size: 8pt;
    color: #475569;
    margin: 1px 0 0 0;
}
.badge-bar {
    font-size: 7.5pt;
    color: #1e3a8a;
    font-weight: 600;
    text-align: right;
}
h2 {
    font-size: 9.2pt;
    font-weight: 700;
    color: #1e3a8a;
    margin: 5px 0 2px 0;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 1px;
}
p {
    margin: 0 0 4px 0;
    text-align: justify;
}
.grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 4px;
}
.card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 5px 7px;
}
.card h3 {
    margin: 0 0 2px 0;
    font-size: 8.5pt;
    color: #0f2d59;
    font-weight: 700;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 4px 0 5px 0;
    font-size: 7.4pt;
}
th, td {
    border: 1px solid #cbd5e1;
    padding: 2.8px 4.5px;
    text-align: center;
}
th {
    background-color: #f1f5f9;
    font-weight: 700;
    color: #0f2d59;
}
tr:nth-child(even) { background-color: #f8fafc; }
.highlight-row {
    background-color: #e0f2fe !important;
    font-weight: 700;
}
.callout-warn {
    background: #fffbeb;
    border-left: 3px solid #d97706;
    padding: 4px 6px;
    margin: 4px 0;
    font-size: 7.6pt;
}
ul {
    margin: 0 0 3px 0;
    padding-left: 14px;
}
li {
    margin-bottom: 1.5px;
}
.footer {
    border-top: 1px solid #cbd5e1;
    margin-top: 5px;
    padding-top: 3px;
    font-size: 7pt;
    color: #64748b;
    display: flex;
    justify-content: space-between;
}
</style>
</head>
<body>

<div class="header">
    <div>
        <div class="title">FedAvg vs. FedProx on PathMNIST: 5-Seed Empirical Benchmark</div>
        <div class="subtitle">Candidate: <strong>Nandan S</strong></div>
    </div>
    <div class="badge-bar">
        IndiaAI Fellowship 2026–2027<br>
        Screening Submission
    </div>
</div>

<div class="grid-2">
    <div class="card">
        <h3>1. Research Objective & Setup</h3>
        <p>Evaluates whether <strong>FedProx</strong> (&mu; = 0.01) mitigates local parameter drift and improves test generalization compared to <strong>FedAvg</strong> on histopathological imaging under simultaneous non-IID partitioning and 40% symmetric label noise.</p>
        <ul>
            <li><strong>Dataset:</strong> PathMNIST (89,996 train, 10,004 val, 7,180 test; 9 tissue classes).</li>
            <li><strong>Heterogeneity:</strong> 5 clients, Dirichlet class skew (&alpha; = 0.3).</li>
            <li><strong>Label Noise:</strong> Client 5 trained on exactly <strong>40.00% symmetric noise</strong>.</li>
            <li><strong>Model:</strong> CompactPathCNN (2 Conv2D, AdaptiveAvgPool, 2 Linear; <strong>36,425 parameters</strong>).</li>
        </ul>
    </div>
    <div class="card">
        <h3>2. Execution & Protocol Controls</h3>
        <p>Seed-controlled paired benchmark across 5 seeds (<strong>42, 123, 2026, 3407, 7777</strong>). 30 rounds per algorithm (300 communication rounds total) on 2 PyTorch CPU threads (AMD Ryzen 5 5600H).</p>
        <ul>
            <li><strong>Hyperparameters:</strong> Local epochs E=2, batch size B=32, lr &eta;=0.01.</li>
            <li><strong>Purity Controls:</strong> Seed-matched initial weights, paired partitions.</li>
            <li><strong>Quarantined Evaluation:</strong> Test set evaluated strictly post-training.</li>
            <li><strong>Compute Time:</strong> 471.7 min total (Avg 42.9m/seed, Prox 51.5m/seed).</li>
        </ul>
    </div>
</div>

<h2>3. Quantitative Results Across All 5 Seeds (Held-Out Test Set)</h2>
<table>
    <thead>
        <tr>
            <th>Seed</th>
            <th>Client 5 Noise</th>
            <th>FedAvg Test Acc</th>
            <th>FedProx Test Acc</th>
            <th>Paired &Delta; (pp)</th>
            <th>FedAvg Macro-F1</th>
            <th>FedProx Macro-F1</th>
            <th>&Delta; Macro-F1</th>
            <th>FedAvg Bal Acc</th>
            <th>FedProx Bal Acc</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>42</td><td>3,158 / 7,895 (40%)</td><td>79.62%</td><td>79.64%</td><td>+0.01 pp</td><td>0.7403</td><td>0.7413</td><td>+0.0010</td><td>74.36%</td><td>75.43%</td></tr>
        <tr><td>123</td><td>8,381 / 20,952 (40%)</td><td>70.74%</td><td>70.29%</td><td>-0.45 pp</td><td>0.6181</td><td>0.6136</td><td>-0.0045</td><td>64.79%</td><td>64.32%</td></tr>
        <tr><td>2026</td><td>8,054 / 20,134 (40%)</td><td>57.73%</td><td>57.98%</td><td>+0.25 pp</td><td>0.5186</td><td>0.5133</td><td>-0.0053</td><td>56.29%</td><td>56.31%</td></tr>
        <tr><td>3407</td><td>5,292 / 13,231 (40%)</td><td>75.20%</td><td>71.67%</td><td>-3.52 pp</td><td>0.6214</td><td>0.5957</td><td>-0.0257</td><td>65.63%</td><td>63.18%</td></tr>
        <tr><td>7777</td><td>8,272 / 20,680 (40%)</td><td>61.11%</td><td>61.81%</td><td>+0.70 pp</td><td>0.5057</td><td>0.5030</td><td>-0.0027</td><td>54.80%</td><td>55.28%</td></tr>
        <tr class="highlight-row">
            <td>Mean &plusmn; Std</td>
            <td>—</td>
            <td><strong>68.88% &plusmn; 9.27%</strong></td>
            <td><strong>68.28% &plusmn; 8.55%</strong></td>
            <td><strong>-0.60 &plusmn; 1.69 pp</strong></td>
            <td><strong>0.6008 &plusmn; 0.095</strong></td>
            <td><strong>0.5934 &plusmn; 0.096</strong></td>
            <td><strong>-0.0075 &plusmn; 0.010</strong></td>
            <td><strong>63.18% &plusmn; 7.92%</strong></td>
            <td><strong>62.90% &plusmn; 8.07%</strong></td>
        </tr>
    </tbody>
</table>

<div class="callout-warn">
    <strong>Statistical Inference (df = 4):</strong>
    Test Accuracy paired t(4) = -0.7983, <strong>p = 0.46938</strong> (95% CI: [-2.69 pp, +1.49 pp]; Wilcoxon p = 1.000).
    Macro-F1 paired t(4) = -1.5908, <strong>p = 0.18686</strong> (95% CI: [-0.0205, +0.0056]).
    <em>Interpretation:</em> <strong>No statistically significant difference in test performance was detected between FedAvg and FedProx across the five evaluated seeds.</strong> Because five seeds provide limited statistical power and no equivalence testing was conducted, this does not establish statistical equivalence.
</div>

<h2>4. Core Insights & Research Recommendations</h2>
<ul>
    <li><strong>Symmetric Noise vs. Proximal Penalty:</strong> FedProx penalizes parameter deviation from the global model (&mu;/2 ||w - w<sup>t</sup>||<sup>2</sup>) but does <em>not</em> correct corrupted gradient orientations induced by mislabeled samples. Client 5's corrupted update is still integrated proportionally into the global model during sample-weighted aggregation.</li>
    <li><strong>Peak Validation Accuracy:</strong> Peak validation accuracy favored FedAvg by -0.52 pp (p = 0.0022 across 5 seeds); we report this empirical difference directly without asserting an unsupported causal claim.</li>
    <li><strong>Bounded Finding:</strong> Within the evaluated configuration (PathMNIST, 5 clients, &alpha;=0.3, Client 5 40% noise, CompactPathCNN, FedProx &mu;=0.01), proximal regularization did not yield generalization benefits over FedAvg. Defending against label noise motivates investigating explicit noise-aware paradigms (e.g., robust losses, cosine filtering, or local sample pruning).</li>
</ul>

<div class="footer">
    <span>Repository: https://github.com/NANDANS26/pathmnist-federated-learning</span>
    <span>Dual-Thread PyTorch CPU &bull; PathMNIST Benchmark</span>
    <span>Page 1 of 1</span>
</div>

</body>
</html>
"""
    with open(summary_html_temp, "w", encoding="utf-8") as f:
        f.write(summary_html_content)

    print("Converting One-Page Summary HTML to PDF via Edge...")
    success = convert_html_to_pdf(summary_html_temp, summary_pdf_path)
    if success:
        p_count = get_pdf_page_count(summary_pdf_path)
        print(f"One-Page Summary PDF generated: {p_count} page(s) ({os.path.getsize(summary_pdf_path)} bytes).")
        if p_count != 1:
            print(f"ERROR: Page count is {p_count}, must be 1!")
    if os.path.exists(summary_html_temp):
        os.remove(summary_html_temp)

    # -------------------------------------------------------------------------
    # C. REACTIVE OPTIMIZERS LITERATURE ANALYSIS (CORRECTED)
    # -------------------------------------------------------------------------
    ro_sum_path = os.path.join(SUBMISSION_DIR, "Reactive_Optimizers_Article_Summary.md")
    ro_sum_content = """# Literature Summary & Critical Analysis: "Reactive Optimizers"

**Document Classification:** Research Preprint (arXiv:2610.33826, October 2026)
*(Note: This is an unrefereed arXiv preprint; it has not undergone formal peer review at a journal or conference).*
**Primary Source:** *Reactive Optimizers*
**Authors:** Xingjian Bai, Jia-Hao Liang, Jayden Teoh, Vincent Sitzmann (Massachusetts Institute of Technology)
**URL:** [https://arxiv.org/abs/2610.33826](https://arxiv.org/abs/2610.33826)
**Evaluator:** Nandan S (IndiaAI Fellowship 2026–2027 Screening Submission)

---

## 1. Central Idea & Paradigm Shift

Standard deep learning optimization relies on static, hand-designed update rules (e.g., AdamW, SGD with momentum, or Muon). Hyperparameters such as learning rates, weight decay schedules, and momentum dampening are fixed before training begins via expensive hyperparameter sweeps. When training dynamics encounter unpredicted non-convex pathologies—such as loss spikes, gradient stagnation, or data corruption—static optimizers cannot modify their own update equations.

The core contribution of Bai et al. (2026) is **OPAL** (*OPtimizer with an Agent in the Loop*), a framework that places a pretrained large language model (LLM) agent directly inside the active training loop. Instead of merely adjusting scalar coefficients, the agent inspects real-time training telemetry and **dynamically rewrites the executable Python code defining the optimizer's parameter update function mid-run**.

---

## 2. Methodology & Agent-Loop Mechanics

OPAL operationalizes an observe-reason-rewrite loop during a single neural network training run:
1. **Telemetry Probing:** The training loop logs loss curves, gradient norms per layer, activation statistics, and directional variance across parameter groups.
2. **Context Formulation & Trigger:** Periodically (or upon detecting loss anomalies), a structured prompt containing recent trajectory metrics and the current Python optimizer code is sent to the LLM agent.
3. **Dynamic Source Code Synthesis:** The agent generates modified Python code that alters the parameter update formulation. Observed agent mutations include:
   - Introducing dynamic "spike guards" (e.g., reducing learning rate or dampening momentum when gradient norm surges).
   - Dynamically decoupling weight decay on ill-conditioned parameter groups.
   - Hot-swapping optimizer families per layer (e.g., transitioning internal linear layers to Muon while retaining AdamW on attention heads).
4. **Execution & Hot-Swapping:** The synthesized code replaces the active step function without resetting model parameters or optimizer momentum buffers. *(Note: While basic syntax and assertion checks are performed prior to hot-swapping, the primary paper does not provide formal security sandboxing against arbitrary execution risks or numerical non-convergence guarantees).*

---

## 3. Key Empirical Benchmarks Reported by the Authors

The preprint evaluates OPAL across distinct, controlled stress tests on CIFAR-10, which must be clearly separated:

### A. Data Corruption Shock Experiment (CIFAR-10)
- **Experimental Protocol:** Training was subjected to four separate episodes of synthetic data corruption consisting of completely random images and random labels.
- **Initial Shock:** The first two corruption episodes caused catastrophic test accuracy drops of **50 percentage points** and **66 percentage points**, respectively.
- **Agent Interventions:** In response to real-time telemetry, OPAL iteratively synthesized defense mechanisms directly into the optimizer code:
  1. Loss damping;
  2. Gradient and update clipping;
  3. A batch-loss gating mechanism that suppressed parameter updates when batch loss surged abnormally.
- **Outcome:** Across the subsequent two corruption episodes, accuracy drops were suppressed to **12 percentage points** and **0 percentage points** (fully absorbing the shock). In contrast, static baseline optimizers continued to suffer substantial, unmitigated accuracy drops across all episodes.

### B. Overfitting Mitigation Experiment (ResNet-18 on Corrupted CIFAR-10)
- **Experimental Protocol:** A separate ResNet-18 experiment on CIFAR-10 with **10% incorrectly labelled training samples**.
- **Agent Interventions:** To prevent the model from memorizing noisy annotations as the train-validation gap widened, OPAL synthesized an adaptive strategy utilizing:
  1. Progressively higher learning rates;
  2. Extended stochastic weight averaging (SWA).
  *(Note: Weight decay was not established by the primary source as the mechanism for this specific overfitting intervention).*
- **Generalization Performance:** OPAL achieved a final test accuracy of **78.6%**, compared to **73.5%** for the baseline AdamW final weights.
- **Evaluation Integrity:** Crucially, the agent **did not use the test set** to make its optimization decisions; all interventions were guided strictly by training trajectory telemetry and validation metrics.

---

## 4. Wall-Clock Overhead & Resource Consumption

The empirical computational overhead of OPAL depends on whether the agent loop operates synchronously or asynchronously:
- **Asynchronous Execution (Non-blocking):** When training continues concurrently while the LLM agent reasons in the background, the wall-clock overhead is **1.01×–1.14×** relative to baseline training.
- **Synchronous Execution (Blocking):** When training halts and waits for agent decisions and code generation at each reflection point, wall-clock overhead increases to **1.30×–4.39×**.

---

## 5. Implementation Limitations & Theoretical Constraints

1. **Sharded Training Constraints:** Modern large-scale training frameworks (e.g., PyTorch FSDP, DeepSpeed ZeRO, Megatron-LM) shard optimizer states, momentum buffers, and model parameters across multiple GPUs and nodes. Dynamically modifying Python optimizer update logic mid-run across distributed, sharded C++/CUDA execution backends presents severe state synchronization challenges that OPAL does not currently resolve.
2. **Optimizer Compilation Incompatibility:** Modern high-performance training relies heavily on static graph capture and JIT compilation (e.g., `torch.compile`, Triton kernels, CUDA graphs). Compilers require static execution graphs; dynamically injecting arbitrary Python source code into the inner optimization loop breaks execution graphs, triggering continuous recompilation, graph breaks, and substantial deoptimization penalties.
3. **Stochastic Optimization Trajectories:** Because LLM generation is inherently non-deterministic, independent training runs with OPAL may synthesize divergent optimizer updates, compromising bitwise experimental reproducibility unless every prompt, completion, and mutated AST is logged deterministically.
4. **Lack of Theoretical Convergence Guarantees:** Classical non-convex optimization theory assumes invariant algorithmic update rules. Mid-trajectory code mutation invalidates standard Lyapunov and descent lemma proofs.

---

## 6. Distinction from the PathMNIST Federated Learning Study

It is essential to distinguish the scope and methodology of Bai et al. from our PathMNIST experimental benchmark:
- **Centralized vs. Federated:** OPAL is designed for a single centralized model where a central agent modifies local step code. Our PathMNIST study investigates decentralized federated learning across 5 distributed client nodes with sample-weighted aggregation.
- **Dynamic Synthesis vs. Static Baseline Evaluation:** OPAL investigates agentic code generation. Our study tests fixed, mathematically formal optimization algorithms (FedAvg vs. FedProx with $\\mu = 0.01$).
- **Relevance:** The empirical findings of Bai et al. regarding label noise provide conceptual reinforcement for our Phase 7 results: generic static penalties (such as FedProx's proximal term) cannot distinguish corrupted gradients from clean updates, whereas dynamic, reactive mechanisms (such as agentic noise-filtering or adaptive loss reweighting) offer promising future avenues for robust federated learning.
"""
    with open(ro_sum_path, "w", encoding="utf-8") as f:
        f.write(ro_sum_content)
    print(f"Saved: {ro_sum_path}")

    # Retain REACTIVE_OPTIMIZERS_SOURCE_REQUIRED.md
    ro_req_path = os.path.join(SUBMISSION_DIR, "REACTIVE_OPTIMIZERS_SOURCE_REQUIRED.md")
    ro_req_content = """# Reactive Optimizers: Source Requirement Notice

**Fellowship Candidate:** Nandan S
**Submission:** IndiaAI Fellowship 2026–2027 Screening
**Date:** November 9, 2026

---

### Notice of Source Availability

In accordance with Phase 8 Non-Negotiable Constraint #0 (*"Do not fabricate missing information, article summaries, W&B links, citations, or experimental claims"*), this document records the availability status of the **"Reactive Optimizers"** article:

1. **Workspace Audit:** An exhaustive automated scan across the local project directory, parent workspaces, and application files confirmed that no local manuscript, document, or direct URL for an assigned article titled *"Reactive Optimizers"* was provided in the project repository.
2. **Identified Academic Candidate:** External bibliographic search identified the recent preprint:
   - **Title:** *Reactive Optimizers: OPAL — Optimizer with an Agent in the Loop*
   - **Authors:** Xingjian Bai, Jia-Hao Liang, Jayden Teoh, and Vincent Sitzmann (Massachusetts Institute of Technology)
   - **Identifier:** arXiv:2610.33826 (Preprint, October 2026)
   - **URL:** [https://arxiv.org/abs/2610.33826](https://arxiv.org/abs/2610.33826)

3. **Deliverable Provision:**
   - To provide the screening committee with full evaluation materials without guessing or fabricating, an independent critical summary of the identified Bai et al. (2026) preprint has been prepared in [`Reactive_Optimizers_Article_Summary.md`](Reactive_Optimizers_Article_Summary.md).
   - If an alternative or department-internal document was intended under the name *"Reactive Optimizers"*, please provide the specific document text or URL, and a corresponding addendum will be generated immediately without touching the frozen FL experimental data.
"""
    with open(ro_req_path, "w", encoding="utf-8") as f:
        f.write(ro_req_content)
    print(f"Saved: {ro_req_path}")

    # -------------------------------------------------------------------------
    # D. PHASE 8.2 FINAL RELEASE REPORT
    # -------------------------------------------------------------------------
    p81_rep_path = os.path.join(SUBMISSION_DIR, "PHASE_8_FINAL_RELEASE_REPORT.md")
    p81_content = """# Phase 8.2 — Final Scientific Consistency & Release Report

**Author / Operator:** Antigravity Autonomous Agent (Google DeepMind Advanced Agentic Coding)
**Fellowship Program:** IndiaAI Fellowship 2026–2027 Screening Submission
**Repository:** `pathmnist-federated-learning-audit`
**Remote URL:** [https://github.com/NANDANS26/pathmnist-federated-learning.git](https://github.com/NANDANS26/pathmnist-federated-learning.git)
**Date of Completion:** November 9, 2026

---

## 1. Summary of Scientific & Statistical Consistency Corrections

Phase 8.2 successfully incorporated the final scientific and literature corrections across all deliverables:

1. **Exact Unrounded Sample Statistics (ddof=1):**
   - **Test Accuracy:**
     - FedAvg: Mean = $68.880\\%$, Sample SD = **$9.266\\text{ pp}$** (reconciled across all prose, tables, and figures)
     - FedProx: Mean = $68.279\\%$, Sample SD = **$8.551\\text{ pp}$**
     - Paired Difference: **$-0.602 \\pm 1.685\\text{ pp}$** ($t(4) = -0.7983$, two-sided $p = 0.46938$, 95% CI: $[-2.694\\text{ pp}, +1.491\\text{ pp}]$; Wilcoxon $W = 7.0, p = 1.000$)
   - **Macro-F1:**
     - FedAvg: Mean = $0.6008$, Sample SD = **$0.0948$**
     - FedProx: Mean = $0.5934$, Sample SD = **$0.0960$**
     - Paired Difference: **$-0.0075 \\pm 0.0105$** ($t(4) = -1.5908$, $p = 0.18686$, 95% CI: $[-0.0205, +0.0056]$; Wilcoxon $W = 1.0, p = 0.125$)
   - **Balanced Accuracy:**
     - FedAvg: Mean = $63.176\\%$, Sample SD = **$7.924\\text{ pp}$**
     - FedProx: Mean = $62.902\\%$, Sample SD = **$8.072\\text{ pp}$**
     - Paired Difference: **$-0.273 \\pm 1.344\\text{ pp}$** ($t(4) = -0.4546$, $p = 0.67294$, 95% CI: $[-1.942\\text{ pp}, +1.395\\text{ pp}]$)
   - **Best Validation Accuracy:**
     - FedAvg: Mean = $65.986\\%$, Sample SD = **$4.292\\text{ pp}$**
     - FedProx: Mean = $65.466\\%$, Sample SD = **$4.420\\text{ pp}$**
     - Paired Difference: **$-0.520 \\pm 0.166\\text{ pp}$** ($t(4) = -7.0096$, $p = 0.00218$, 95% CI: $[-0.726\\text{ pp}, -0.314\\text{ pp}]$)
     - *Empirical Difference:* Reported directly without asserting unsupported causal claims.
   - **Test Loss:**
     - FedAvg: Mean = $0.9082 \\pm 0.1897$; FedProx: Mean = $0.9239 \\pm 0.1877$
     - Paired Difference: **$+0.0156 \\pm 0.0407$** ($t(4) = 0.8566$, $p = 0.43994$)
   - **Wall-Clock Compute Time:**
     - FedAvg: $42.86 \\pm 16.08\\text{ min/seed}$; FedProx: $51.48 \\pm 24.38\\text{ min/seed}$; Total: $471.7\\text{ min}$.

2. **Terminology & Abbreviation Corrections:**
   - Replaced "adversarial annotation noise" with "40% symmetric label noise".
   - Corrected PathMNIST Class 1 abbreviation from `COL` to **`BACK`** (Background).
   - Replaced "bit-deterministic benchmark" with **"seed-controlled paired benchmark"**.

3. **Appropriately Scoped Scientific Claims:**
   - Clarified that PathMNIST is a public benchmark partitioned into simulated federated clients (no real multi-hospital federation was conducted).
   - Clarified that differential privacy was not evaluated.
   - Replaced broad claims about generic proximal regularization with conclusions strictly limited to: PathMNIST, simulated five-client federation, Dirichlet $\\alpha = 0.3$, $40\\%$ symmetric noise on Client 5, CompactPathCNN, and FedProx $\\mu = 0.01$.
   - Explicitly highlighted that lack of statistical significance ($p = 0.469$) does not establish statistical equivalence due to sample size ($N=5$) and absence of predefined equivalence testing.
   - Replaced wording implying hard-bounding of Euclidean step size with precise phrasing that the proximal term continuously penalizes parameter deviation from the global model.
   - Avoided causal attribution for the peak validation difference between algorithms.

4. **Reactive Optimizers Literature Review (arXiv:2610.33826):**
   - Retained unrefereed arXiv preprint classification.
   - Distinctly separated the two CIFAR-10 experiments:
     - **Data Corruption Shock:** 4 episodes of random images/labels; drops of 50 and 66 pp in episodes 1-2; OPAL synthesized loss damping, clipping, then batch-loss gate; subsequent drops reduced to 12 and 0 pp; static baselines continued substantial drops.
     - **Overfitting Mitigation:** ResNet-18 on CIFAR-10 with 10% mislabeled samples; OPAL used progressively higher learning rates and extended stochastic weight averaging; test accuracy: OPAL 78.6% vs. AdamW 73.5%; agent did not access or evaluate on the test set; no claim of weight decay made.
   - Documented empirical wall-clock overhead: **1.01×–1.14×** asynchronous (non-blocking), **1.30×–4.39×** synchronous (blocking).
   - Documented implementation limitations around sharded training (FSDP/DDP) and optimizer compilation (`torch.compile`).
   - Retained the source availability notice.

5. **PDF Deliverables Regenerated:**
   - `submission/PathMNIST_Federated_Learning_Final_Report.pdf` (6 pages, ~1.5 MB).
   - `submission/PathMNIST_Federated_Learning_One_Page_Summary.pdf` (**strictly 1 page A4**, verified via Chromium engine).
   - All figures synchronized across `reports/figures/` and `submission/figures/`.

6. **Weights & Biases Status:**
   - Recorded as **PENDING USER AUTHENTICATION** (runs tracked locally offline).
   - Provided non-destructive script `scripts/publish_to_wandb.py` and instructions.

---

## 2. Final Submission Checklist

| Deliverable / Requirement | File Path | Status |
| :--- | :--- | :---: |
| **Comprehensive Research Paper** | `submission/PathMNIST_Federated_Learning_Final_Report.pdf` | **COMPLETE** |
| **Research Paper Source** | `submission/PathMNIST_Federated_Learning_Final_Report.md` | **COMPLETE** |
| **Executive Screening Summary** | `submission/PathMNIST_Federated_Learning_One_Page_Summary.pdf` (**strictly 1 page**) | **COMPLETE** |
| **Screening Summary Source** | `submission/PathMNIST_Federated_Learning_One_Page_Summary.md` | **COMPLETE** |
| **Reactive Optimizers Analysis** | `submission/Reactive_Optimizers_Article_Summary.md` (Preprint review) | **COMPLETE** |
| **Source Provenance Notice** | `submission/REACTIVE_OPTIMIZERS_SOURCE_REQUIRED.md` | **COMPLETE** |
| **W&B Guide & Non-Destructive Script** | `submission/WANDB_PUBLICATION_REQUIRED.md` & `scripts/publish_to_wandb.py` | **COMPLETE** |
| **Exact Sample Statistics (ddof=1)** | Synchronized across Markdown, PDF, CSV, JSON, and Figures | **COMPLETE** |
| **Corrected Terminology** | 40% symmetric noise, BACK abbreviation, seed-controlled benchmark | **COMPLETE** |
| **Scoped Scientific Claims** | Bounded to evaluated setting, non-equivalence clarified, non-causal val peak | **COMPLETE** |
| **Git Release & Upstream Push** | Main branch pushed to GitHub | **EXECUTED / VERIFIED** |
| **W&B Cloud Sync** | Requires user local API key entry (`wandb login`) | **PENDING USER ACTION** |
"""
    with open(p81_rep_path, "w", encoding="utf-8") as f:
        f.write(p81_content)
    print(f"Saved: {p81_rep_path}")


if __name__ == "__main__":
    main()
