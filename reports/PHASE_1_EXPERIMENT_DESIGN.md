# PHASE 1: EXPERIMENT DESIGN (CORRECTED)
## Comprehensive Specifications for Data Partitioning, Noise Injection, Model Architecture, and Execution Protocols

**Project:** Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Client-Specific Label Noise  
**Research Supervision:** Dr. Sachin D N, Assistant Professor, Department of Computer Science and Engineering, IIIT Bhopal  
**Fellowship Program:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Author:** Senior Federated Learning Research Scientist & PyTorch Systems Architect  
**Document Status:** Formal Experiment Design (`Phase 1`)  
**Date:** November 7, 2026  

---

## 1. Dataset Specification: PathMNIST

PathMNIST is part of the standardized MedMNIST v2 benchmark suite, constructed from colorectal cancer histology slides (NCT-CRC-HE-100K).

### 1.1 Dataset Properties `[VERIFIED]`
- **Domain:** Colorectal histology tissue classification (multi-class image recognition).
- **Image Resolution:** $28 \times 28$ pixels.
- **Color Channels:** 3 (RGB).
- **Number of Classes:** 9 distinct histological categories:
  1. `0: adipose (ADI)`
  2. `1: background (BACK)`
  3. `2: debris (DEB)`
  4. `3: lymphocytes (LYM)`
  5. `4: mucus (MUC)`
  6. `5: smooth muscle (MUS)`
  7. `6: normal colon mucosa (NORM)`
  8. `7: cancer-associated stroma (STR)`
  9. `8: colorectal adenocarcinoma epithelium (TUM)`
- **Official Splits:**
  - Training Set: **89,996 images**, labels shape `(89996, 1)`
  - Validation Set: **10,004 images**, labels shape `(10004, 1)`
  - Test Set: **7,180 images**, labels shape `(7180, 1)`
  - Total Sample Count: **107,180 images**
- **Data Representation:** Distributed as NumPy `.npz` arrays storing images as `uint8` ($0 \dots 255$) and labels as `uint8` ($0 \dots 8$).

### 1.2 Split Allocation Protocol `[PROPOSED]`
1. **Training Split (89,996 samples):** Distributed across the 5 federated clients using Dirichlet non-IID partitioning. All 89,996 images are utilized.
2. **Validation Split (10,004 samples):** Partitioned into 5 disjoint, clean simulated client-specific validation shards for communication round evaluation. Also evaluated as a single pooled 10,004-sample dataset for global validation metrics.
3. **Test Split (7,180 samples):** Maintained strictly isolated from training, tuning, and round validation. Each algorithm's selected best-validation checkpoint is evaluated once on the official PathMNIST test set after training has finished.

### 1.3 Exact In-Memory Dataset Storage Calculations `[VERIFIED]`
Each PathMNIST image is stored as an uncompressed $28 \times 28 \times 3$ `uint8` array ($2,352$ bytes per image).

- **Training Image Storage (89,996 samples):**
  $$89,996 \times 2,352 \text{ bytes} = 211,670,592 \text{ bytes} = \mathbf{211.67 \text{ MB}} \approx \mathbf{201.86 \text{ MiB}}$$
- **Validation Image Storage (10,004 samples):**
  $$10,004 \times 2,352 \text{ bytes} = 23,529,408 \text{ bytes} = \mathbf{23.53 \text{ MB}} \approx \mathbf{22.44 \text{ MiB}}$$
- **Test Image Storage (7,180 samples):**
  $$7,180 \times 2,352 \text{ bytes} = 16,887,360 \text{ bytes} = \mathbf{16.89 \text{ MB}} \approx \mathbf{16.11 \text{ MiB}}$$
- **All Three Image Splits Stored Simultaneously (107,180 samples):**
  $$107,180 \times 2,352 \text{ bytes} = 252,087,360 \text{ bytes} = \mathbf{252.09 \text{ MB}} \approx \mathbf{240.41 \text{ MiB}}$$
- **Label Storage:** Labels in official `.npz` files are stored as 1-byte uint8 scalars ($89,996\text{ B} \approx 87.89\text{ KiB}$ train; $10,004\text{ B} \approx 9.77\text{ KiB}$ val; $7,180\text{ B} \approx 7.01\text{ KiB}$ test; total labels $107,180\text{ B} \approx 104.67\text{ KiB}$).

**In-Memory Lifecycle & Footprint Distinction:**
1. *Active Training Loop:* The federated training loop holds the Training images ($201.86\text{ MiB}$) and Validation images ($22.44\text{ MiB}$) in memory as contiguous `uint8` NumPy arrays (totaling **$224.30\text{ MiB}$** of raw image arrays). This enables zero-disk-I/O client iterations without memory fragmentation.
2. *Post-Training Benchmark:* The Test split ($16.11\text{ MiB}$) may either remain in memory or be loaded on demand post-training. Even if all three splits are co-resident, total image array storage is **$240.41\text{ MiB}$**.
3. *Total Process RSS:* The total process RSS will include Python runtime, NumPy metadata, PyTorch dynamically loaded libraries, model parameters, gradients, and optimizer states. The actual peak process RSS is not established by dataset tensor dimensions alone and will be measured empirically during the Phase 2 pilot test using `psutil`.

### 1.4 Formal Image Preprocessing Pipeline `[PROPOSED]`
The official PathMNIST arrays are stored in NHWC format (`(N, 28, 28, 3)`, dtype `uint8`). PyTorch CNN models expect NCHW batches (`(B, 3, 28, 28)`, dtype `float32`).

Both FedAvg and FedProx will employ an identical, deterministic preprocessing pipeline executed per mini-batch:
```python
# Conceptual batch preprocessing pipeline (identical for train, val, test)
x = torch.from_numpy(batch_uint8)     # (B, 28, 28, 3), uint8
x = x.permute(0, 3, 1, 2)              # Convert NHWC -> NCHW: (B, 3, 28, 28)
x = x.to(torch.float32)               # Convert uint8 -> float32
x = x.div_(255.0)                     # In-place scaling to [0.0, 1.0]

# Label preprocessing
y = torch.from_numpy(batch_labels)    # (B, 1), uint8
y = y.squeeze(1).to(torch.int64)      # Convert to 1D int64 tensor: (B,)
```

**Pipeline Invariants:**
- **Identical Processing:** Training, validation, and test splits use identical preprocessing.
- **No Augmentation:** No random cropping, horizontal flipping, or data augmentation is applied in the controlled primary experiment.
- **No Channel Normalization:** Pixel values are scaled strictly by $1/255.0$ into the range $[0.0, 1.0]$. No channel-wise mean/std standardization is applied, preserving the pure image distribution.
- **Tensor Specifications:**
  - Input tensor shape: $(B, 3, 28, 28)$
  - Input tensor dtype: `torch.float32`
  - Input pixel range: $[0.0, 1.0]$
  - Target label shape: $(B,)$
  - Target label dtype: `torch.int64`
  - Target class range: $\{0, 1, \dots, 8\}$

---

## 2. Dirichlet Non-IID Partitioning Methodology

To simulate clinical non-IID conditions where hospitals encounter skewed histological distributions, we partition the 89,996 training samples using a Dirichlet distribution $\text{Dir}(\alpha)$ with concentration parameter $\alpha = 0.3$.

### 2.1 Upstream PFLlib Dirichlet Adaptation `[VERIFIED / PROPOSED]`
In `third_party/PFLlib/dataset/utils/dataset_utils.py:87-112`, PFLlib allocates data on a per-class basis:
1. For each class $k \in \{0, 1, \dots, 8\}$, extract all sample indices $I_k = \{i \mid y_i = k\}$.
2. Shuffle $I_k$ with a deterministic random generator seeded by `seed=42`.
3. Sample a 5-dimensional client proportion vector:
   $$p_k \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_5), \quad \sum_{c=1}^5 p_{k,c} = 1.0$$
4. Allocate samples of class $k$ to the 5 clients proportionally according to $p_k$:
   $$\text{splits}_k = \text{cumsum}(p_k) \times |I_k|$$
5. Handle integer rounding by assigning any remaining samples of class $k$ to the last client, ensuring exact sample conservation:
   $$\sum_{c=1}^5 |I_{k,c}| = |I_k|$$

### 2.2 Removal of Arbitrary Client Floors & Empty-Client Handling Policy `[PROPOSED]`
- As established in the Phase 0 Final Review, **no arbitrary rejection sampling floor (e.g., 1,000 samples)** will be enforced. Rejection sampling truncates the tails of $\text{Dir}(\alpha=0.3)$, artificially dampening client heterogeneity.
- **Empty Class vs. Empty Client Distinction:**
  - A client having zero samples of one or more individual classes ($N_{c,k} = 0$) is entirely possible and expected under non-IID Dirichlet partitioning ($\alpha = 0.3$). This natural class sparsity must be faithfully recorded in the per-client class distribution manifest.
  - A client receiving zero *total* training samples ($\sum_{k=0}^8 N_{c,k} = 0$) across all 9 classes is extremely unlikely given 89,996 total training samples distributed across 5 clients, but it is not mathematically impossible under unconstrained Dirichlet draws.
- **Empty-Client Acceptance Assertion:**
  - The partitioning implementation must explicitly assert that each client has a non-empty total training partition:
    $$\sum_{k=0}^8 |I_{c,k}| > 0 \quad \forall c \in \{1, 2, 3, 4, 5\}$$
  - If any client receives zero total samples, partition acceptance must immediately halt with an informative assertion error, rather than proceeding with a degenerative client.
  - The implementation must **never silently transfer samples** between clients to "fix" an allocation, as this invalidates the Dirichlet distribution.
  - Missing class support within a client must be documented in class histograms; we do not claim that micro-averaging or any metric heuristic resolves absent class support.
- Exact per-client sample counts and complete class histograms will be recorded directly after partitioning in Phase 2.

### 2.3 Partition Invariance and Manifest `[PROPOSED]`
- **Mutual Exclusivity:** $\bigcap_{c=1}^5 I_c = \emptyset$.
- **Complete Conservation:** $\sum_{c=1}^5 |I_c| = 89,996$.
- **Static Artifact:** Partitioning is performed once and stored to disk as `experiments/configs/partition_manifest.json` recording client indices, sample counts, and class distributions. Both FedAvg and FedProx consume the identical partition manifest.

---

## 3. Formalized Validation Shard Integer Allocation

The objective is to divide the official 10,004 validation samples into 5 disjoint clean evaluation shards $V_1 \dots V_5$ such that each client's validation shard reflects its local training class distribution.

### 3.1 Largest-Remainder (Hamilton-Hare) Apportionment Procedure `[PROPOSED]`
To convert class-wise fractional allocations into exact integer counts without loss or duplication:

1. **Class-Wise Grouping:**
   Group official validation indices by ground-truth class $k \in \{0, \dots, 8\}$:
   $$V^{(k)} = \{i \in [0, 10003] \mid y_i^{\text{val}} = k\}, \quad \sum_{k=0}^8 |V^{(k)}| = 10,004$$
2. **Class-Wise Proportions from Training:**
   Let $D_{c,k}^{\text{train}}$ be the number of training samples of class $k$ allocated to client $c \in \{1, \dots, 5\}$.  
   The target proportion of validation class $k$ assigned to client $c$ is:
   $$q_{c,k} = \frac{D_{c,k}^{\text{train}}}{\sum_{m=1}^5 D_{m,k}^{\text{train}}}$$
   *(If $\sum_{m=1}^5 D_{m,k}^{\text{train}} = 0$, set $q_{c,k} = 0.20$ uniformly).*
3. **Fractional Quotas:**
   The exact fractional allocation of class $k$ to client $c$ is:
   $$F_{c,k} = q_{c,k} \times |V^{(k)}|$$
4. **Base Integer Allocations:**
   $$n_{c,k} = \lfloor F_{c,k} \rfloor$$
5. **Remainder Calculation:**
   $$R_{c,k} = F_{c,k} - n_{c,k}$$
   The count of unallocated remainder samples for class $k$ is:
   $$L_k = |V^{(k)}| - \sum_{c=1}^5 n_{c,k}, \quad \text{where } 0 \le L_k < 5$$
6. **Deterministic Apportionment:**
   Rank the 5 clients by remainder $R_{c,k}$ in descending order. Break ties deterministically using the client index $c \in \{1, 2, 3, 4, 5\}$. Allocate $+1$ additional sample to each of the top $L_k$ clients:
   $$n_{c,k}^* = \begin{cases} n_{c,k} + 1 & \text{if client } c \text{ is in top } L_k \text{ remainders} \\ n_{c,k} & \text{otherwise} \end{cases}$$
   This guarantees exact conservation:
   $$\sum_{c=1}^5 n_{c,k}^* = |V^{(k)}|$$
7. **Deterministic Shard Slicing:**
   Shuffle $V^{(k)}$ using a dedicated NumPy generator seeded by `validation_seed=42`. Slice $V^{(k)}$ sequentially according to $[n_{1,k}^*, n_{2,k}^*, \dots, n_{5,k}^*]$ and append to client shards $V_c$.
8. **Sample Conservation and Disjointness Invariants:**
   $$\sum_{c=1}^5 |V_c| = 10,004 \quad \text{and} \quad \bigcap_{c=1}^5 V_c = \emptyset$$
9. **Scientific Interpretation `[PROPOSED]`:**
   These shards represent **simulated client-specific clean validation sets** derived from the synthetic training class distribution. They do **not** claim to replicate genuine clinical hospital-specific distribution shift.
10. **Validation Labels are 100% Clean:** No noise is injected into validation data.

---

## 4. Label Noise Experiment Design

### 4.1 Noisy Client Assignment
- **Designated Noisy Client:** Human-readable **Client 5** (zero-based index 4).
- **Clean Clients:** Clients 1, 2, 3, and 4 (zero-based indices 0, 1, 2, 3) undergo **0% noise**.

### 4.2 Symmetric Multiclass Noise Model `[PROPOSED]`
Let $M_5$ be the total training samples assigned to Client 5.
1. **Sample Selection:**
   Using a dedicated deterministic noise seed (`noise_seed=42`), select exactly $M_{\text{noisy}}$ distinct indices without replacement from Client 5's training set:
   $$M_{\text{noisy}} = \text{round}(0.40 \times M_5)$$
2. **Symmetric Label Flipping:**
   For each selected sample index $i \in I_{\text{noisy}}$ with original clean ground-truth label $y_i \in \{0, 1, \dots, 8\}$:
   $$\tilde{y}_i = (y_i + r_i) \pmod 9, \quad \text{where } r_i \sim \text{Uniform}(\{1, 2, 3, 4, 5, 6, 7, 8\})$$
   Because $r_i \in [1, 8]$, $\tilde{y}_i \neq y_i$ with probability 1.0 (no sample flips to its true class).
3. **Data Integrity and Audit Trail:**
   - Client 5's image arrays $X$ are **never modified**.
   - Original clean labels $y$ are preserved in an audit array `y_clean`.
   - A boolean array `noise_mask` of length $M_5$ is stored:
     $$\text{noise\_mask}[i] = \begin{cases} \text{True} & \text{if } i \in I_{\text{noisy}} \\ \text{False} & \text{if } i \notin I_{\text{noisy}} \end{cases}$$
   - Both FedAvg and FedProx train on the identical noise-corrupted dataset loaded from the persisted manifest.

---

## 5. Model Architecture Selection & Exact Computational Complexity

Four candidate architectures were compared theoretically based on parameter count, tensor shapes, and estimated computational cost. No empirical throughput benchmarks were conducted.

### 5.1 Selected Architecture: `CompactPathCNN` `[PROPOSED]`
A custom, lightweight 2-stage ConvNet designed to minimize memory and computational footprint:

```
Input: (3, 28, 28) RGB Histology
  │
  ├── Conv2d(3 -> 32, kernel=3, padding=1, stride=1) --> (32, 28, 28)
  ├── ReLU(inplace=True)
  ├── MaxPool2d(kernel=2, stride=2)                  --> (32, 14, 14)
  │
  ├── Conv2d(32 -> 64, kernel=3, padding=1, stride=1) --> (64, 14, 14)
  ├── ReLU(inplace=True)
  ├── MaxPool2d(kernel=2, stride=2)                  --> (64, 7, 7)
  │
  ├── AdaptiveAvgPool2d((2, 2))                      --> (64, 2, 2)
  ├── Flatten()                                      --> (256)
  │
  ├── Linear(256 -> 64)                              --> (64)
  ├── ReLU(inplace=True)
  └── Linear(64 -> 9)                                --> (9 logits)
```

### 5.2 Exact Trainable Parameter Count `[VERIFIED]`
- **Conv1 Layer:** $3 \times 32 \times (3 \times 3) + 32 \text{ (bias)} = 864 + 32 = \mathbf{896}$
- **Conv2 Layer:** $32 \times 64 \times (3 \times 3) + 64 \text{ (bias)} = 18,432 + 64 = \mathbf{18,496}$
- **FC1 Layer:** $256 \times 64 + 64 \text{ (bias)} = 16,384 + 64 = \mathbf{16,448}$
- **FC2 Layer:** $64 \times 9 + 9 \text{ (bias)} = 576 + 9 = \mathbf{585}$
- **Total Trainable Parameters:** $896 + 18,496 + 16,448 + 585 = \mathbf{36,425 \text{ parameters}}$
- **State Vector Size (`float32`):** $36,425 \times 4 \text{ bytes} \approx \mathbf{145.7 \text{ KB}}$

### 5.3 Exact Computational Complexity (MACs and FLOPs) `[INFERRED]`
We distinguish strictly between Multiply-Accumulate operations (MACs) and Floating-Point Operations (FLOPs), adopting the standard scientific convention that **1 MAC = 2 FLOPs** (one multiplication and one addition):

1. **Conv1 Layer MACs:**
   $$\text{MACs}_{\text{Conv1}} = 28 \times 28 \times 32 \times (3 \times 3 \times 3) = 784 \times 32 \times 27 = \mathbf{677,376 \text{ MACs}}$$
2. **Conv2 Layer MACs:**
   $$\text{MACs}_{\text{Conv2}} = 14 \times 14 \times 64 \times (32 \times 3 \times 3) = 196 \times 64 \times 288 = \mathbf{3,612,672 \text{ MACs}}$$
3. **FC1 Layer MACs:**
   $$\text{MACs}_{\text{FC1}} = 256 \times 64 = \mathbf{16,384 \text{ MACs}}$$
4. **FC2 Layer MACs:**
   $$\text{MACs}_{\text{FC2}} = 64 \times 9 = \mathbf{576 \text{ MACs}}$$

#### Complexity Totals:
- **Total MACs Per Image:**
  $$\text{MACs}_{\text{image}} = 677,376 + 3,612,672 + 16,384 + 576 = \mathbf{4,307,008 \text{ MACs}} \approx \mathbf{4.31 \text{ MMACs}}$$
- **Total FLOPs Per Image (1 MAC = 2 FLOPs):**
  $$\text{FLOPs}_{\text{image}} = 2 \times 4,307,008 = \mathbf{8,614,016 \text{ FLOPs}} \approx \mathbf{8.61 \text{ MFLOPs}}$$
- **Total MACs Per Batch (Batch Size = 32):**
  $$\text{MACs}_{\text{batch}} = 32 \times 4,307,008 = \mathbf{137,824,256 \text{ MACs}} \approx \mathbf{137.82 \text{ MMACs}}$$
- **Total FLOPs Per Batch (Batch Size = 32):**
  $$\text{FLOPs}_{\text{batch}} = 32 \times 8,614,016 = \mathbf{275,648,512 \text{ FLOPs}} \approx \mathbf{275.65 \text{ MFLOPs}}$$

### 5.4 Exact Forward Activation Tensor Sizing `[INFERRED]`
For a mini-batch of 32 samples (float32, 4 bytes per element), the individual forward tensor dimensions and element counts across the network are:

1. **Conv1 Feature Maps:** Shape $(32, 32, 28, 28) \implies 802,816 \text{ elements}$ ($3,211,264 \text{ bytes} \approx \mathbf{3.06 \text{ MiB}}$)
2. **ReLU1 Activations (`inplace=False`):** Shape $(32, 32, 28, 28) \implies 802,816 \text{ elements}$ ($3,211,264 \text{ bytes} \approx \mathbf{3.06 \text{ MiB}}$)
3. **MaxPool1 Feature Maps:** Shape $(32, 32, 14, 14) \implies 200,704 \text{ elements}$ ($802,816 \text{ bytes} \approx \mathbf{784 \text{ KiB}}$)
4. **Conv2 Feature Maps:** Shape $(32, 64, 14, 14) \implies 401,408 \text{ elements}$ ($1,605,632 \text{ bytes} \approx \mathbf{1.53 \text{ MiB}}$)
5. **ReLU2 Activations (`inplace=False`):** Shape $(32, 64, 14, 14) \implies 401,408 \text{ elements}$ ($1,605,632 \text{ bytes} \approx \mathbf{1.53 \text{ MiB}}$)
6. **MaxPool2 Feature Maps:** Shape $(32, 64, 7, 7) \implies 100,352 \text{ elements}$ ($401,408 \text{ bytes} \approx \mathbf{392 \text{ KiB}}$)
7. **AdaptiveAvgPool2d Maps:** Shape $(32, 64, 2, 2) \implies 8,192 \text{ elements}$ ($32,768 \text{ bytes} = \mathbf{32 \text{ KiB}}$)
8. **Flatten Layer Output:** Shape $(32, 256) \implies 8,192 \text{ elements}$. *(Implementation Note: In PyTorch, `torch.flatten(x, 1)` normally returns a memory view of the pooled tensor and does not allocate a new underlying buffer).*
9. **FC1 Activations:** Shape $(32, 64) \implies 2,048 \text{ elements}$ ($8,192 \text{ bytes} = \mathbf{8 \text{ KiB}}$)
10. **ReLU3 Activations (`inplace=False`):** Shape $(32, 64) \implies 2,048 \text{ elements}$ ($8,192 \text{ bytes} = \mathbf{8 \text{ KiB}}$)
11. **FC2 Output Logits:** Shape $(32, 9) \implies 288 \text{ elements}$ ($1,152 \text{ bytes} \approx \mathbf{1.1 \text{ KiB}}$)

- **Sum of All 11 Individual Forward Tensor Shapes:**
  $$802,816 + 802,816 + 200,704 + 401,408 + 401,408 + 100,352 + 8,192 + 8,192 + 2,048 + 2,048 + 288 = \mathbf{2,730,272 \text{ elements}}$$
  *(Equivalent to $10,921,088\text{ bytes} \approx \mathbf{10.42\text{ MiB}} \approx \mathbf{10.92\text{ MB}}$ if every intermediate tensor were independently materialized).*

**Crucial Implementation & Memory Caveats:**
- **Flatten View:** `Flatten` operates as a zero-copy tensor view, avoiding redundant allocation.
- **Activation In-Place Policy:** The architecture specifies standard out-of-place ReLUs (`nn.ReLU(inplace=False)`) for numerical stability and backward graph clarity. During the backward pass, PyTorch autograd saves only the input tensors necessary for derivative calculations.
- **No Unsupported Peak Claim:** Static forward tensor dimensions do not determine actual peak training memory. Peak process RSS comprises dynamic backward autograd graph buffers, parameter gradients, optimizer states, dataset arrays, runtime libraries, and operating system paging.
- **Empirical Measurement:** Actual peak training process RSS will be empirically monitored and recorded in Phase 2 using system and process telemetry (`psutil`).

- **Raw Input Batch Tensor ($32 \times 3 \times 28 \times 28$):**
  $$32 \times 3 \times 28 \times 28 \times 4 \text{ bytes} = 301,056 \text{ bytes} = \mathbf{294.0 \text{ KiB}} \approx \mathbf{0.301 \text{ MB}}$$
  *(Crucial Note: 294 KiB is strictly the raw RGB float32 input batch tensor size, correcting earlier 98 KB grayscale references).*

---

## 6. FedAvg and FedProx Algorithmic Configurations

Both algorithms will be benchmarked under strictly fair, identical hyperparameters:

| Hyperparameter | Value `[PROPOSED]` | Justification |
|---|---|---|
| **Base Model Architecture** | `CompactPathCNN` (36,425 parameters) | Uniform across both algorithms. |
| **Initial Parameter Seed** | `init_seed = 42` | Generates identical initial model weights $w_0$. |
| **Client Selection Ratio ($C$)** | 1.0 (Full participation: all 5 clients every round) | Eliminates client subsampling variance. |
| **Local Optimizer** | SGD (momentum = 0.0, weight_decay = 0.0) | Standard unbiased federated baseline. |
| **Local Learning Rate ($\eta$)** | 0.01 (Proposed starting value) | Proposed default pending pilot stability check. |
| **Local Epochs ($E$)** | 2 epochs per client per round | Produces realistic client drift without immediate divergence. |
| **Batch Size ($B$)** | 32 (Input batch tensor: ~294 KiB) | Memory-friendly with stable gradient updates. |
| **Aggregation Weights** | Normalized training samples: $p_k = n_{k,\text{train}} / 89,996$ | Theoretical standard for FedAvg and FedProx. |
| **FedProx Proximal $\mu$** | $\mu = 0.01$ (Core baseline) | Standard literature default (Li et al., 2020). |
| **Communication Rounds ($R$)** | Provisional horizon (Calibrated via Pilot Policy) | Calibrated via mutually exclusive pilot policy. |

---

## 7. Primary and Optional Experiment Matrix

### 7.1 Primary Mandatory Comparison
- **Run 1 (`fedavg_alpha0.3_noise40_seed42`):**
  FedAvg baseline on non-IID PathMNIST ($\alpha = 0.3$), Client 5 corrupted with 40% label noise.
- **Run 2 (`fedprox_mu0.01_alpha0.3_noise40_seed42`):**
  FedProx with $\mu = 0.01$, identical data partition, identical Client 5 noise, identical $w_0$.

### 7.2 Optional Sensitivity / Ablation Matrix (Post-Mandatory)
If CPU execution budget permits:
- **Run 3 (`fedprox_mu0.001_alpha0.3_noise40_seed42`):** Low proximal constraint ($\mu = 0.001$).
- **Run 4 (`fedprox_mu0.1_alpha0.3_noise40_seed42`):** Strong proximal constraint ($\mu = 0.1$).

---

## 8. Staged CPU Feasibility & Mutually Exclusive Round Calibration Policy

Host: Intel Pentium G3240 (2 cores, 2 threads, no CUDA), 3.92 GB RAM (~580 MB free).

### 8.1 Staged Execution Protocol `[PROPOSED]`
- **Stage A (Functional Smoke Test):** 1 round on synthetic mock data to verify tensor dimensions, backward passes, diagnostics, and logging without loading full datasets.
- **Stage B (2-Round CPU Pilot Benchmark):** 2 communication rounds on actual PathMNIST partitions. Empirically measures:
  - Per-batch step latency ($T_{\text{step}}$ in ms).
  - Full round duration across all 5 clients ($T_{\text{round}}$ in seconds).
  - Peak process RSS memory and minimum available physical system RAM via `psutil`.

### 8.2 Mutually Exclusive Round Horizon Calibration Policy `[PROPOSED]`
Let $T_{\text{round}}$ be the measured execution time per round across all 5 clients during Stage B:

$$\text{Decision}(T_{\text{round}}) = \begin{cases}
R = 100 \text{ rounds} & \text{if } T_{\text{round}} < 35 \text{ s} \quad (\text{Estimated total duration } < 60 \text{ min}) \\
R = 50 \text{ rounds} & \text{if } 35 \text{ s} \le T_{\text{round}} \le 75 \text{ s} \quad (\text{Estimated total duration } \approx 30\text{--}65 \text{ min}) \\
R = 30 \text{ rounds} & \text{if } 75 \text{ s} < T_{\text{round}} \le 120 \text{ s} \quad (\text{Estimated total duration } \approx 38\text{--}60 \text{ min}) \\
\text{Halt \& Review} & \text{if } T_{\text{round}} > 120 \text{ s} \quad (\text{Flag for mentor review; consider } E=1)
\end{cases}$$

### 8.3 System Memory Pressure Stop Condition `[PROPOSED]`
- If at any point during pilot or experimental execution, available physical system RAM drops below **150 MB** or persistent pagefile thrashing is detected:
  $$\text{Action} = \mathbf{EMERGENCY\_HALT}$$
  Execution will immediately cease, flush current logs, and report memory telemetry for architectural review rather than risking system freeze.
