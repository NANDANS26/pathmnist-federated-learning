# PHASE 1 FINAL TECHNICAL CORRECTION & RESEARCH VALIDATION REPORT

**Project:** Comparative Experimental Analysis of Federated Averaging (FedAvg) and Federated Proximal (FedProx) on Non-IID PathMNIST with 40% Client-Specific Label Noise  
**Supervision:** Dr. Sachin D N, Assistant Professor, Department of Computer Science and Engineering, IIIT Bhopal  
**Fellowship Target:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Author:** Senior Federated Learning Research Scientist & PyTorch Systems Architect  
**Review Status:** Formal Technical Review & Correction Completed  
**Gate Status:** **PHASE 1 COMPLETE — APPROVAL GATE LOCKED PENDING USER AUTHORIZATION FOR PHASE 2**  

---

## 1. Executive Summary

This report documents the comprehensive technical audit, mathematical recalculations, and methodological corrections applied across all Phase 0 and Phase 1 research documentation for the comparative evaluation of **FedAvg** (McMahan et al., 2017) and **FedProx** (Li et al., 2020) on PathMNIST under non-IID Dirichlet distribution ($\alpha = 0.3$) and 40% client-specific label noise.

Prior to entering Phase 2 (Implementation), an independent technical verification was performed on the existing documentation to eliminate mathematical inaccuracies, unsupported empirical claims, ambiguous definitions, and overstatements inherited from earlier drafts. Every document in the repository was cross-checked, synchronized, and aligned with standard federated learning literature and the strict physical constraints of the host hardware environment (**Intel Pentium G3240 dual-core CPU @ 3.10 GHz, 4 GB physical RAM, no GPU**).

All corrections have been integrated into the repository markdown reports, configuration files, and architectural specifications without modifying upstream `third_party/PFLlib`.

---

## 2. Inventory of Issues Identified, Root Causes & Applied Corrections

| # | Technical Area | Original Inaccurate Claim / Defect | Root Cause Analysis | Corrected Formulation & Resolution `[STATUS]` |
|---|---|---|---|---|
| 1 | **CNN Computational Complexity** | Listed ~1.1 MFLOPs / 35 MFLOPs without formal spatial convolution derivations. | Neglected output spatial resolution multipliers ($H_{\text{out}} \times W_{\text{out}}$) and conflated MACs with FLOPs. | Formally derived spatial MACs for `CompactPathCNN` ($4,307,008$ MACs/image, $137,824,256$ MACs/batch-32). Applied standard scientific convention $1 \text{ MAC} = 2 \text{ FLOPs}$ yielding $8,614,016$ FLOPs/image ($8.61$ MFLOPs) and $275,648,512$ FLOPs/batch ($275.65$ MFLOPs). Preserved exact parameter count: **36,425**. `[VERIFIED]` |
| 2 | **Input Batch Memory** | Claimed 98 KB per batch of 32 images. | Calculated memory for single-channel grayscale ($32 \times 1 \times 28 \times 28 \times 4\text{ B} = 100,352\text{ B} \approx 98\text{ KiB}$) rather than 3-channel RGB. | Corrected to 3-channel RGB float32: $32 \times 3 \times 28 \times 28 \times 4\text{ B} = 301,056\text{ B} = 294.0\text{ KiB}$. Clarified that this represents strictly the raw input batch tensor $X$. `[VERIFIED]` |
| 3 | **Activation Tensor Memory** | Total per-batch forward activations cited loosely as ~3.7 MB, and later miscalculated as 1,668,096 elements / 6.07 MiB. | Arithmetically omitted activation stages in intermediate sum and conflated static dimensions with peak memory. | Derived exact dimensions for all 11 forward tensor stages ($802,816 + 802,816 + 200,704 + 401,408 + 401,408 + 100,352 + 8,192 + 8,192 + 2,048 + 2,048 + 288 = 2,730,272$ elements $\approx 10.42\text{ MiB} \approx 10.92\text{ MB}$ if materialized). Explained that `Flatten` is a view and ReLUs are `inplace=False`. Clarified that static forward dimensions do not establish peak training RSS, which will be measured empirically via `psutil` in Phase 2. `[INFERRED / PROPOSED]` |
| 4 | **Round Calibration Policy** | Contradictory conditional ordering where $T \le 60\text{s} \to 50\text{r}$ preceded $T \le 30\text{s} \to 100\text{r}$, rendering 100 rounds unreachable. | Non-exhaustive and overlapping interval logic in preliminary markdown draft. | Replaced with strict ascending, mutually exclusive piecewise intervals: $<35\text{s} \to 100\text{r}$, $35\text{--}75\text{s} \to 50\text{r}$, $75\text{--}120\text{s} \to 30\text{r}$, $>120\text{s} \to$ Halt/Review. Added emergency stop if free system RAM falls below 150 MB. `[PROPOSED]` |
| 5 | **Validation Split Allocation** | Partitioning validation set (10,004 samples) by non-IID proportions risked fractional counts and unassigned samples. | Simple floating-point rounding $\text{round}(q_{c,k} \cdot \|V^{(k)}\|)$ can sum to $\ne \|V^{(k)}\|$. | Adopted **Hamilton-Hare Largest Remainder Method** for integer quota allocation. Mathematically guarantees $\sum_c \|V_c\| = 10,004$ and $\bigcap_c V_c = \emptyset$. `[PROPOSED]` |
| 6 | **Statistical Inference Claims** | Early drafts suggested computing inferential p-values across communication rounds to claim statistical superiority. | Violation of statistical independence: rounds within a single training run are time-series trajectories, and single-seed (`seed=42`) runs cannot yield inferential population variance. | Reframed entire study as an **exploratory empirical comparison**. Removed all claims of hypothesis testing on single-run round series. Hypotheses reformulated as directional exploratory predictions. `[PROPOSED]` |
| 7 | **Alignment Terminology** | Used the term "gradient alignment" to describe server-side telemetry. | The server never receives client gradient vectors $\nabla F_k$; it receives aggregated parameter updates $\Delta_{k,t} = w_{k,t} - w_t$ resulting from multiple local SGD steps. | Standardized nomenclature to **Client Update-Direction Alignment** throughout all reports and logging schemas. `[VERIFIED]` |
| 8 | **Clean-to-Noisy Alignment Metric** | Mentioned measuring "clean vs noisy alignment" without a concrete mathematical formula. | Omitted formal aggregation operator over the 4 clean client updates. | Defined formal metric: $\cos_{\text{clean-noisy}}(t) = \frac{1}{4} \sum_{k=1}^4 \cos(\Delta_{k,t}, \Delta_{5,t})$, measuring the average cosine similarity between clean client updates and the corrupted client update. `[PROPOSED]` |
| 9 | **Checkpoint Selection Policy** | Ambiguous rule for selecting the "best" model checkpoint for final test benchmark evaluation. | Unspecified metric, timing, and tie-breaking criteria. | Formalized policy: Select global checkpoint $w_{t^*}$ maximizing global validation accuracy $\text{Acc}_{\text{val}}(w_t)$; break ties by minimum validation loss $\mathcal{L}_{\text{val}}(w_t)$. Final test split (7,180 samples) evaluated exactly once post-training on $w_{t^*}$. `[PROPOSED]` |
| 10 | **Multiclass AUC Specification** | Omitted multi-class averaging strategy for ROC-AUC. | Binary AUC formulas are undefined for 9-class classification without averaging convention. | Specified macro-averaged One-vs-Rest (OvR) ROC-AUC using predicted softmax probabilities across all 9 classes. Classified as optional diagnostic metric. `[PROPOSED]` |
| 11 | **Reproducibility Manifests** | Ambiguous specification of deterministic data loader seeds and model state hashing. | Python standard hashing is randomized across processes; PyTorch data loaders need explicit epoch-dependent seeding. | Specified deterministic seed generator: $\text{seed}(t, c) = 42 + 1000 \cdot t + c$. Specified canonical state dict hashing: sorted keys, dtypes, shapes, contiguous byte buffers hashed via SHA-256. `[PROPOSED]` |
| 12 | **Upstream Leakage Scope** | Conflated upstream MNIST data leakage with PathMNIST. | PFLlib's `generate_MNIST.py` pooled train and test splits, but MedMNIST provides native isolated splits. | Clarified that PFLlib's MNIST split pooling proved non-preservation in upstream scripts, but split isolation in our pipeline is a scientific precaution guaranteeing zero test contamination for PathMNIST. `[VERIFIED]` |

---

## 3. Detailed Mathematical Derivations

### 3.1 CompactPathCNN Computational Complexity

`CompactPathCNN` processes input tensors of shape $(B, 3, 28, 28)$ where $B = 32$.

#### Layer-by-Layer MACs Derivation:
1. **Conv1:**
   - Input: $(3, 28, 28)$
   - Output: $(32, 28, 28)$ (Kernel $3 \times 3$, padding 1, stride 1)
   - Operations: Each output activation requires computing dot product of kernel ($3 \times 3 \times 3 = 27$ multiplies and 26 adds, 1 bias add):
     $$\text{MACs}_{\text{Conv1}} = H_{\text{out}} \times W_{\text{out}} \times C_{\text{out}} \times (C_{\text{in}} \times K_h \times K_w)$$
     $$\text{MACs}_{\text{Conv1}} = 28 \times 28 \times 32 \times (3 \times 3 \times 3) = 25,088 \times 27 = 677,376 \text{ MACs}$$

2. **MaxPool1:** Output $(32, 14, 14)$ (Kernel $2 \times 2$, stride 2). Max-pooling involves comparisons, not multiply-accumulates ($0 \text{ MACs}$).

3. **Conv2:**
   - Input: $(32, 14, 14)$
   - Output: $(64, 14, 14)$ (Kernel $3 \times 3$, padding 1, stride 1)
     $$\text{MACs}_{\text{Conv2}} = H_{\text{out}} \times W_{\text{out}} \times C_{\text{out}} \times (C_{\text{in}} \times K_h \times K_w)$$
     $$\text{MACs}_{\text{Conv2}} = 14 \times 14 \times 64 \times (32 \times 3 \times 3) = 12,544 \times 288 = 3,612,672 \text{ MACs}$$

4. **MaxPool2:** Output $(64, 7, 7)$ ($0 \text{ MACs}$).

5. **AdaptiveAvgPool2d((2, 2)):** Output $(64, 2, 2)$. Spatial average pooling per channel ($0 \text{ MACs}$).

6. **FC1 (Linear):**
   - Flattened Input: $64 \times 2 \times 2 = 256$
   - Output: $64$
     $$\text{MACs}_{\text{FC1}} = N_{\text{in}} \times N_{\text{out}} = 256 \times 64 = 16,384 \text{ MACs}$$

7. **FC2 (Linear):**
   - Input: $64$
   - Output: $9$
     $$\text{MACs}_{\text{FC2}} = N_{\text{in}} \times N_{\text{out}} = 64 \times 9 = 576 \text{ MACs}$$

#### Total Multiply-Accumulate Operations (MACs):
$$\text{MACs}_{\text{total}} = 677,376 + 3,612,672 + 16,384 + 576 = 4,307,008 \text{ MACs per image} \approx 4.31 \text{ MMACs}$$

#### Floating-Point Operations (FLOPs):
Following standard scientific literature (where $1 \text{ Multiply-Accumulate} = 1 \text{ multiplication} + 1 \text{ addition} = 2 \text{ FLOPs}$):
$$\text{FLOPs}_{\text{total}} = 2 \times 4,307,008 = 8,614,016 \text{ FLOPs per image} \approx 8.61 \text{ MFLOPs}$$

#### Batch-Level Complexity ($B = 32$):
- **Batch MACs:** $32 \times 4,307,008 = 137,824,256 \text{ MACs} \approx 137.82 \text{ MMACs}$
- **Batch FLOPs:** $32 \times 8,614,016 = 275,648,512 \text{ FLOPs} \approx 275.65 \text{ MFLOPs}$

#### Trainable Parameter Verification:
- Conv1 Weights: $32 \times 3 \times 3 \times 3 = 864$; Biases: $32 \implies 896$
- Conv2 Weights: $64 \times 32 \times 3 \times 3 = 18,432$; Biases: $64 \implies 18,496$
- FC1 Weights: $64 \times 256 = 16,384$; Biases: $64 \implies 16,448$
- FC2 Weights: $9 \times 64 = 576$; Biases: $9 \implies 585$
- **Total Trainable Parameters:** $896 + 18,496 + 16,448 + 585 = \mathbf{36,425}$ `[VERIFIED]`

---

### 3.2 Tensor Memory Sizing

#### Input Batch Tensor Memory:
For batch size $B = 32$, channels $C = 3$, height $H = 28$, width $W = 28$, data type float32 (4 bytes/element):
$$\text{Memory}_{\text{input}} = 32 \times 3 \times 28 \times 28 \times 4 \text{ bytes} = 301,056 \text{ bytes} = \mathbf{294.0 \text{ KiB}} \approx 0.301 \text{ MB}$$

*(Note: Earlier draft references to 98 KB corresponded to $32 \times 1 \times 28 \times 28 \times 4 = 100,352 \text{ B} \approx 98 \text{ KiB}$, which mistakenly calculated memory for single-channel grayscale).*

#### Forward Activation Tensors Memory:
During the forward pass of batch size 32, the intermediate feature maps across the network are:
1. `Conv1` output: $32 \times 32 \times 28 \times 28 = 802,816$ elements ($3,211,264$ bytes $\approx 3.06$ MiB)
2. `ReLU1` output (`inplace=False`): $32 \times 32 \times 28 \times 28 = 802,816$ elements ($3,211,264$ bytes $\approx 3.06$ MiB)
3. `MaxPool1` output: $32 \times 32 \times 14 \times 14 = 200,704$ elements ($802,816$ bytes $\approx 784$ KiB)
4. `Conv2` output: $32 \times 64 \times 14 \times 14 = 401,408$ elements ($1,605,632$ bytes $\approx 1.53$ MiB)
5. `ReLU2` output (`inplace=False`): $32 \times 64 \times 14 \times 14 = 401,408$ elements ($1,605,632$ bytes $\approx 1.53$ MiB)
6. `MaxPool2` output: $32 \times 64 \times 7 \times 7 = 100,352$ elements ($401,408$ bytes $\approx 392$ KiB)
7. `AdaptiveAvgPool2d` output: $32 \times 64 \times 2 \times 2 = 8,192$ elements ($32,768$ bytes $= 32$ KiB)
8. `Flatten` output: $32 \times 256 = 8,192$ elements *(view of pooled tensor; zero separate buffer allocation)*
9. `FC1` output: $32 \times 64 = 2,048$ elements ($8,192$ bytes $= 8$ KiB)
10. `ReLU3` output (`inplace=False`): $32 \times 64 = 2,048$ elements ($8,192$ bytes $= 8$ KiB)
11. `FC2` (Logits) output: $32 \times 9 = 288$ elements ($1,152$ bytes $\approx 1.1$ KiB)

$$\text{Sum of All 11 Forward Shapes} = \mathbf{2,730,272 \text{ float32 elements}} \quad (10,921,088 \text{ bytes} \approx \mathbf{10.42 \text{ MiB}} \approx \mathbf{10.92 \text{ MB}})$$

**Important Implementation Details & Memory Principles:**
- **Flatten View:** `torch.flatten(x, 1)` operates as a zero-copy memory view and does not allocate a separate 8,192-element buffer.
- **ReLU In-Place Policy:** ReLUs are specified out-of-place (`inplace=False`) for autograd computational graph clarity. Autograd retains only those tensors required for derivative computation during the backward pass.
- **Static Dimensions vs. Peak RSS:** The sum of static forward tensor dimensions ($2,730,272$ elements) does NOT establish peak training process memory. Peak process RSS is governed by backward autograd graph buffers, parameter gradients, optimizer states (or cloned proximal targets), Python/NumPy runtime overhead, and OS page allocations.
- **Telemetry Verification:** Actual peak process RSS will be monitored and measured empirically via `psutil` during the Phase 2 CPU pilot run.

---

### 3.3 Complete Dataset Storage Calculations `[VERIFIED]`
PathMNIST images are uncompressed $28 \times 28 \times 3$ `uint8` arrays ($2,352$ bytes/image):
- **Training Images (89,996 samples):** $89,996 \times 2,352\text{ B} = 211,670,592\text{ B} = \mathbf{211.67\text{ MB}} \approx \mathbf{201.86\text{ MiB}}$.
- **Validation Images (10,004 samples):** $10,004 \times 2,352\text{ B} = 23,529,408\text{ B} = \mathbf{23.53\text{ MB}} \approx \mathbf{22.44\text{ MiB}}$.
- **Test Images (7,180 samples):** $7,180 \times 2,352\text{ B} = 16,887,360\text{ B} = \mathbf{16.89\text{ MB}} \approx \mathbf{16.11\text{ MiB}}$.
- **All Three Image Splits Stored Simultaneously (107,180 samples):** $107,180 \times 2,352\text{ B} = 252,087,360\text{ B} = \mathbf{252.09\text{ MB}} \approx \mathbf{240.41\text{ MiB}}$.
- **Labels (107,180 uint8 scalars):** $107,180\text{ bytes} \approx 104.67\text{ KiB}$.
- **In-Memory Training Footprint:** During the federated training loop, holding the Training and Validation image arrays in memory requires **$224.30\text{ MiB}$** ($235.20\text{ MB}$), well within physical RAM limits. Test data ($16.11\text{ MiB}$) may remain resident or be loaded on demand post-training.

---

### 3.4 Image Preprocessing Pipeline `[PROPOSED]`
Both FedAvg and FedProx will employ an identical, deterministic mini-batch transformation:
```python
x = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
y = torch.from_numpy(batch_labels).squeeze(1).to(torch.int64)
```
- Input: shape $(B, 3, 28, 28)$, dtype `float32`, pixel range $[0.0, 1.0]$.
- Target: shape $(B,)$, dtype `int64`, class indices $0\text{--}8$.
- Identical across train, val, and test splits. No augmentation or channel-wise mean/std normalization.

---

### 3.5 Validation Split Allocation (Hamilton-Hare Largest Remainder)

To evaluate local client validation accuracy without introducing distribution shift, the official PathMNIST validation set ($10,004$ samples) is distributed to clients proportionally to each client's training label distribution.

Let:
- $V^{(k)}$ be the set of validation samples belonging to class $k \in \{0, \dots, 8\}$.
- $q_{c,k} = \frac{N_{c,k}}{\sum_{i=1}^5 N_{i,k}}$ be client $c$'s proportion of class $k$ in the training set.

To guarantee that exactly $|V^{(k)}|$ samples are allocated without duplicates or unassigned samples, we apply the **Hamilton-Hare Largest Remainder Method**:
1. Compute exact floating-point quota: $F_{c,k} = q_{c,k} \times |V^{(k)}|$.
2. Assign base integer quota: $n_{c,k} = \lfloor F_{c,k} \rfloor$.
3. Compute fractional remainder: $r_{c,k} = F_{c,k} - n_{c,k}$.
4. Compute deficit: $D_k = |V^{(k)}| - \sum_{c=1}^5 n_{c,k}$.
5. Sort clients descending by remainder $r_{c,k}$ and assign $+1$ additional sample to the top $D_k$ clients.

This guarantees:
$$\sum_{c=1}^5 |V_c| = 10,004 \quad \text{and} \quad V_a \cap V_b = \emptyset \quad \forall a \ne b$$

---

## 4. Corrected Experimental & Algorithmic Policies

### 4.1 Mutually Exclusive Round Calibration Policy
To protect against runaway execution times on the 2-core Pentium G3240 while maintaining scientific rigor, communication horizons are determined by an ordered, mutually exclusive calibration policy evaluated during the Phase 2 2-round pilot:

```
IF (T_round < 35 seconds):
    Target Communication Horizon R = 100 rounds (~58 min maximum runtime)
ELSE IF (35 seconds <= T_round <= 75 seconds):
    Target Communication Horizon R = 50 rounds (29 to 62.5 min runtime)
ELSE IF (75 seconds < T_round <= 120 seconds):
    Target Communication Horizon R = 30 rounds (37.5 to 60 min runtime)
ELSE (T_round > 120 seconds):
    HALT execution immediately and consult research mentor Dr. Sachin D N
```
**Emergency Safety Threshold:** If system available physical memory drops below **150 MB** at any point, the experiment terminates immediately with an emergency state dump to prevent OS kernel paging lockup.

### 4.2 Scientific Framing: Exploratory Single-Seed Study
- **Seeding:** The study evaluates master `seed = 42`.
- **Epistemic Framing:** Because single-seed federated training produces a single realization of a stochastic process, and communication rounds within a run are time-dependent, round-by-round accuracy differences **cannot be subjected to inferential hypothesis testing** (e.g., paired t-tests or ANOVA across rounds).
- **Correct Formulation:** The study is framed as an **exploratory empirical comparison**. All hypotheses (H1, H2, H3) are directional exploratory predictions. Findings will report observable empirical trajectories, terminal test set performance, and diagnostic update behaviors.

### 4.3 Client Update Diagnostics Formulation
- **Update Vector:** $\Delta_{k,t} = w_{k,t} - w_t$ (difference between local model after $E=2$ epochs and starting global model).
- **Update $L_2$ Norm:** $\|\Delta_{k,t}\|_2 = \sqrt{\sum_j (\Delta_{k,t}[j])^2}$.
- **Pairwise Cosine Similarity:**
  $$\cos(\Delta_{a,t}, \Delta_{b,t}) = \frac{\langle \Delta_{a,t}, \Delta_{b,t} \rangle}{\|\Delta_{a,t}\|_2 \|\Delta_{b,t}\|_2} \quad \text{for all 10 distinct pairs } 1 \le a < b \le 5$$
- **Similarity to Mean Update:**
  $$\bar{\Delta}_t = \sum_{k=1}^5 \frac{n_k}{N} \Delta_{k,t}, \quad \cos(\Delta_{k,t}, \bar{\Delta}_t) = \frac{\langle \Delta_{k,t}, \bar{\Delta}_t \rangle}{\|\Delta_{k,t}\|_2 \|\bar{\Delta}_t\|_2}$$
- **Clean-to-Noisy Alignment:**
  $$\cos_{\text{clean-noisy}}(t) = \frac{1}{4} \sum_{k=1}^4 \cos(\Delta_{k,t}, \Delta_{5,t})$$

### 4.4 Checkpoint Selection & Held-Out Test Evaluation
- **Validation Evaluation:** Evaluated at every round $t$ on clean validation partitions ($10,004$ samples total) for both global model and all 5 clients.
- **Checkpoint Selection Rule:** Identify round $t^*$ that achieves maximum global validation accuracy:
  $$t^* = \arg\max_t \text{Acc}_{\text{val}}(w_t)$$
  *(Tie-breaking rule: choose round with lower validation loss $\mathcal{L}_{\text{val}}(w_t)$).*
- **Held-Out Test Benchmark:** The official PathMNIST test set ($7,180$ samples) is evaluated **exactly once** using $w_{t^*}$ post-training. Reporting includes Test Accuracy, Test Loss, Macro-F1, Multi-class Macro-AUC (OvR), and a $9 \times 9$ Confusion Matrix.

---

## 5. Comprehensive Inventory of Modified Documents

The following documents have been updated and synchronized in the repository:

1. `reports/PHASE_1_RESEARCH_PROTOCOL.md`
   - Replaced "gradient alignment" with "Client Update-Direction Alignment".
   - Reframed statistical significance testing to exploratory empirical comparison under single seed.
   - Refined hypotheses H1, H2, H3 as directional exploratory predictions.
2. `reports/PHASE_1_EXPERIMENT_DESIGN.md`
   - Incorporated exact `CompactPathCNN` complexity ($4,307,008$ MACs, $8,614,016$ FLOPs per image).
   - Corrected RGB float32 batch size to 294 KiB ($301,056$ bytes).
   - Formulated Hamilton-Hare Largest Remainder validation split allocation.
   - Integrated mutually exclusive round calibration policy ($<35\text{s} \to 100\text{r}$, $35\text{--}75\text{s} \to 50\text{r}$, $75\text{--}120\text{s} \to 30\text{r}$, $>120\text{s} \to \text{Halt}$).
3. `reports/PHASE_1_METRIC_SPECIFICATION.md`
   - Watermarked illustrative JSON schema with explicit disclaimer: `ILLUSTRATIVE SCHEMA ONLY — NOT EXPERIMENTAL RESULTS`.
   - Defined formal clean-to-noisy alignment metric $\frac{1}{4} \sum_{k=1}^4 \cos(\Delta_{k}, \Delta_{5})$.
   - Formalized checkpoint selection policy and multi-class One-vs-Rest Macro-AUC.
4. `reports/PHASE_1_REPRODUCIBILITY_PLAN.md`
   - Specified deterministic data loader seeding formula $\text{seed}(t, c) = 42 + 1000 \cdot t + c$.
   - Defined canonical model state hashing (ordered parameter keys, dtypes, shapes, contiguous raw byte buffers hashed via SHA-256) and `torch.equal()` tensor verification.
5. `reports/PHASE_1_ARCHITECTURE_SPECIFICATION.md`
   - Synchronized architecture diagrams and memory flow specifications with corrected tensor dimensions and zero-waste uint8 dataset caching.
6. `reports/PHASE_1_ACCEPTANCE_CRITERIA.md`
   - Added system memory monitoring criteria, canonical hash verification, and explicit demarcation of optional secondary metrics.
7. `reports/PHASE_1_DECISIONS_AND_OPEN_QUESTIONS.md`
   - Formally documented the resolution of the round calibration policy, validation allocation mechanism, and exploratory statistical framing.
8. `reports/PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md`
   - Updated audit to clarify that upstream `generate_MNIST.py` pooled train and test splits, while MedMNIST PathMNIST provides native isolated splits.
9. `reports/PHASE_0_RESOURCE_FEASIBILITY_REPORT.md`
   - Updated with corrected 294 KiB input batch tensor, 2,730,272 forward activation elements sum, and provisional 5.6–14.1 s/epoch runtime projections.
10. `reports/PHASE_0_RISK_REGISTER.md`
    - Corrected RSK-01 (split pooling scope) and RSK-02 (memory monitoring with 150 MB emergency stop).
11. `reports/PHASE_0_FINAL_REVIEW_REPORT.md`
    - Synchronized line-item memory breakdown to 294 KiB raw RGB input tensor.
12. `configs/experiment_protocol.yaml` & `experiments/configs/experiment_protocol.yaml`
    - Added exact MACs, FLOPs, RGB batch bytes, and updated round calibration intervals.
13. `README.md`
    - Synchronized Phase 1 completion status, hardware constraints, and report references.
14. `reports/PHASE_1_FINAL_TECHNICAL_CORRECTION_REPORT.md`
    - Created this formal final review report.

---

## 6. Scientific Status & Epistemic Demarcation

To preserve academic integrity, all statements throughout the project are tagged according to their epistemic status:

- **`[VERIFIED]` Established Facts:**
  - Upstream PFLlib branch is `master`, commit `0169ba7e412c9856a08bb3faefab1e35f538a3c1`.
  - Host hardware: Windows 10 Pro 64-bit, Intel Pentium G3240 @ 3.10 GHz, 2 cores, 2 threads, no AVX2 (SSE4.2 only), 4 GB physical RAM.
  - PathMNIST native splits: 89,996 train, 10,004 val, 7,180 test ($28 \times 28 \times 3$ RGB, 9 classes).
  - `CompactPathCNN` parameters: Exactly 36,425.
  - `CompactPathCNN` complexity: Exactly 4,307,008 MACs and 8,614,016 FLOPs per image; 137.82 MMACs and 275.65 MFLOPs per batch of 32.
  - Raw float32 RGB input batch size: Exactly 301,056 bytes (294.0 KiB).
  - PFLlib's `clientprox.py:86` contains an unsquared $L_2$ norm in its reported scalar training loss.
  - Upstream `generate_MNIST.py` pools original train and test arrays before Dirichlet partitioning.

- **`[PROPOSED]` Methodological & Engineering Designs:**
  - Zero-waste NumPy uint8 in-memory dataset storage (training images: 211.67 MB / 201.86 MiB; validation images: 23.53 MB / 22.44 MiB; total 3 splits: 252.09 MB / 240.41 MiB).
  - Hamilton-Hare largest remainder allocation for validation split.
  - Symmetric multiclass label noise on Client 5 ($\tilde{y} = (y + r) \pmod 9$).
  - Clean-to-noisy update alignment metric $\frac{1}{4} \sum_{k=1}^4 \cos(\Delta_k, \Delta_5)$.
  - Mutually exclusive round calibration logic ($<35\text{s} \to 100\text{r}$, $35\text{--}75\text{s} \to 50\text{r}$, $75\text{--}120\text{s} \to 30\text{r}$).
  - Best-validation-accuracy checkpoint selection with tie-breaking by validation loss.
  - Canonical model state dict hashing (keys, dtypes, shapes, contiguous raw byte buffer).

- **`[UNVERIFIED]` Empirical Values Requiring Phase 2 Measurement:**
  - Actual per-batch CPU forward/backward latency on Pentium G3240 (provisional range: 10–25 ms/batch).
  - Actual per-epoch training time (provisional range: 5.6–14.1 s/epoch).
  - Actual round latency (provisional range: 56–141 s/round for $E=2$).
  - Actual peak process RSS during training (provisional operational target: $< 1,200 \text{ MB}$, warning threshold: $> 2,000 \text{ MB}$).

---

## 7. Phase 1 Final Formal Verification Checklist

| # | Verification Criterion | Status | Evidence / Location |
|---|---|---|---|
| 1 | Upstream `third_party/PFLlib` is 100% untouched and clean | **PASS** | `git status` clean on `master` commit `0169ba7` |
| 2 | No model training code implemented prior to Phase 2 | **PASS** | Verified; `research/` directory contains no training scripts |
| 3 | No dataset files downloaded prior to Phase 2 | **PASS** | Verified; no PathMNIST `.npz` files present on disk |
| 4 | No Python packages installed prior to Phase 2 | **PASS** | Verified; host Python environments untouched |
| 5 | Dual-core Pentium G3240 constraints respected | **PASS** | `torch.set_num_threads(2)`, sequential client loop specified |
| 6 | 4 GB physical RAM constraints respected | **PASS** | uint8 lazy pipeline designed, 150 MB emergency stop floor |
| 7 | `CompactPathCNN` parameter count is exactly 36,425 | **PASS** | Formally verified via layer-by-layer parameter derivation |
| 8 | `CompactPathCNN` MACs per image is exactly 4,307,008 | **PASS** | Derived in Section 3.1 ($677,376 + 3,612,672 + 16,384 + 576$) |
| 9 | `CompactPathCNN` FLOPs per image is exactly 8,614,016 | **PASS** | Derived in Section 3.1 using standard $1 \text{ MAC} = 2 \text{ FLOPs}$ |
| 10 | Batch-32 complexity is 137.82 MMACs and 275.65 MFLOPs | **PASS** | Derived in Section 3.1 ($32 \times 4,307,008$ and $32 \times 8,614,016$) |
| 11 | Raw input batch size for 32 RGB images is 294.0 KiB | **PASS** | Derived in Section 3.2 ($32 \times 3 \times 28 \times 28 \times 4\text{ B} = 301,056\text{ B}$) |
| 12 | Forward activation shapes sum is 2,730,272 elements (~10.42 MiB) | **PASS** | Derived in Section 3.2; Flatten is a view, ReLUs out-of-place, peak RSS to be measured |
| 13 | 1.2 GB peak RSS marked as unverified target, not measured ceiling | **PASS** | Formally categorized as `[UNVERIFIED]` operational target |
| 14 | Validation split allocation uses Hamilton-Hare largest remainder | **PASS** | Formally specified in Section 3.3; guarantees exact sample sums |
| 15 | Dirichlet partitioning ($\alpha=0.3$) has no artificial floor | **PASS** | Retains pure Dirichlet draw with seed 42 and distribution audit |
| 16 | Noise injection on Client 5 is 40% symmetric multiclass | **PASS** | Formulated as $\tilde{y} = (y + r) \pmod 9$ on $40\%$ of local training set |
| 17 | Round calibration intervals are mutually exclusive and ordered | **PASS** | Formally specified: $<35\text{s} \to 100$, $35\text{--}75\text{s} \to 50$, $75\text{--}120\text{s} \to 30$ |
| 18 | Emergency memory stop floor defined (< 150 MB available RAM) | **PASS** | Formally specified in protocol and configuration files |
| 19 | Metric logging uses local durable JSONL + optional W&B | **PASS** | Local JSONL is primary; W&B is supplementary with offline mode |
| 20 | Telemetry termed "Client Update-Direction Alignment" | **PASS** | Replaced all references to "gradient alignment" |
| 21 | Clean-to-noisy update alignment metric formally defined | **PASS** | Defined as $\frac{1}{4} \sum_{k=1}^4 \cos(\Delta_k, \Delta_5)$ |
| 22 | Checkpoint selection policy formally specified | **PASS** | Max validation accuracy with min validation loss tie-breaker |
| 23 | Final test evaluation is once per algorithm on selected best-validation checkpoint | **PASS** | Evaluated once for FedAvg and once for FedProx post-training; zero test use in training |
| 24 | Multiclass AUC specified as macro-averaged One-vs-Rest (OvR) | **PASS** | Formally documented as optional post-training metric |
| 25 | Illustrative JSON metrics schemas marked with warning disclaimer | **PASS** | Disclaimers added across all metric documentation |
| 26 | Data loader seeding specified deterministically per round and client | **PASS** | Specified as $\text{seed}(t, c) = 42 + 1000 \cdot t + c$ |
| 27 | Model state hashing specified canonically via SHA-256 | **PASS** | Specified over sorted keys, dtypes, shapes, contiguous raw bytes |
| 28 | Single-seed study framed as exploratory empirical comparison | **PASS** | Removed invalid inferential claims; hypotheses framed directionally |

---

## 8. Conclusion & Phase 2 Gate Status

All mathematical inconsistencies, dimensional discrepancies, and methodological ambiguities have been resolved and documented. The experimental protocol is mathematically rigorous, reproducible, and ready for software implementation.

**APPROVAL GATE:**  
Phase 1 is complete. In strict adherence to project instructions, **work is now stopped**. No Phase 2 implementation, dataset downloading, environment configuration, or training runs will occur until explicit authorization is provided by the research user.
