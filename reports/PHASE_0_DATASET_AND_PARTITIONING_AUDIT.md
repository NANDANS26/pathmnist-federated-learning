# PHASE 0: DATASET AND PARTITIONING TECHNICAL AUDIT (CORRECTED)

**Project:** Federated Learning on PathMNIST (FedAvg vs. FedProx)  
**Fellowship Program:** IndiaAI Fellowship 2026–27 Screening Submission  
**Target Codebase:** `third_party/PFLlib` (`dataset/generate_MNIST.py`, `dataset/utils/dataset_utils.py`, `system/utils/data_utils.py`)  
**Audit Status:** Technical Correction Completed  
**Date:** November 7, 2026  

---

## 1. Executive Summary

This audit evaluates the dataset acquisition, partitioning, serialization, and deserialization mechanisms in PFLlib. We inspect its Dirichlet distribution implementation ($\text{Dir}(\alpha)$), evaluate the split-handling behavior of upstream dataset generators, examine the impact of rejection sampling on non-IID distributions, and formulate the exact specifications for PathMNIST split preservation, Dirichlet partitioning ($\alpha = 0.3$), and reproducible 40% symmetric label noise on Client 5 (zero-based index 4).

---

## 2. Upstream PFLlib Dataset Pipeline Analysis

### 2.1 The Existing Pipeline Architecture `[VERIFIED]`
In PFLlib, dataset preparation follows this sequence:
1. `dataset/generate_<Dataset>.py`: Downloads the raw dataset (e.g., MNIST) using `torchvision.datasets`.
2. Extracts images and labels into NumPy arrays.
3. Invokes `separate_data()` in `dataset/utils/dataset_utils.py`.
4. Splits client partitions using `split_data()`.
5. Serializes client shards to disk using `save_file()` as individual `.npz` files:
   - `dataset/<Dataset>/train/<client_id>.npz`
   - `dataset/<Dataset>/test/<client_id>.npz`
   - `dataset/<Dataset>/config.json`
6. During training, `system/utils/data_utils.py` reads these `.npz` files on demand per client.

---

## 3. Upstream MNIST Generator Split Pooling Audit `[VERIFIED]`

Line-by-line inspection of `dataset/generate_MNIST.py:53-61` reveals the following data extraction sequence:

```python
dataset_image = []
dataset_label = []

dataset_image.extend(trainset.data.cpu().detach().numpy())
dataset_image.extend(testset.data.cpu().detach().numpy())
dataset_label.extend(trainset.targets.cpu().detach().numpy())
dataset_label.extend(testset.targets.cpu().detach().numpy())
dataset_image = np.array(dataset_image)
dataset_label = np.array(dataset_label)
```

And in `dataset/utils/dataset_utils.py:11`:
```python
train_ratio = 0.75 # merge original training set and test set, then split it manually.
```

### Supported Technical Conclusion `[VERIFIED]`:
- **The inspected PFLlib MNIST generator does not preserve the original MNIST benchmark split boundaries.**
- By pooling `trainset` (60,000 samples) and `testset` (10,000 samples) into a single 70,000-sample array prior to Dirichlet partitioning and applying an arbitrary 75%/25% split, samples originally designated as official MNIST test benchmarks are redistributed into client training shards.
- This practice compromises using the resulting client shards as standard official MNIST held-out evaluations.

### Epistemic Boundary Regarding PathMNIST `[CORRECTION]`:
- *Important Clarification:* The observed pooling behavior in `generate_MNIST.py` directly proves only that the **MNIST generator** pools its input splits. It does **not** prove patient-level data leakage, whole-slide image contamination, or a verified defect in PathMNIST itself (PathMNIST is not present in PFLlib).
- However, as a foundational scientific precaution, our independent PathMNIST research pipeline must strictly preserve the official MedMNIST split boundaries.

**Mandatory Requirement for our PathMNIST Pipeline `[PROPOSED]`:**
We must **never** pool official training, validation, and test splits:
- **Official Training Split (89,996 images):** Used exclusively for client federated partitioning and local training.
- **Official Validation Split (10,004 images):** Clean, held out for round-level global and simulated per-client validation.
- **Official Test Split (7,180 images):** Clean, held out exclusively for final post-training evaluation.

---

## 4. Dirichlet Non-IID Partitioning Implementation Audit

PFLlib implements Dirichlet non-IID partitioning in `dataset/utils/dataset_utils.py:87-112`:

```python
    elif partition == "dir":
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
```

### 4.1 Rejection Sampling and Distribution Truncation `[INFERRED]`
- The research protocol specified Dirichlet non-IID partitioning with $\alpha = 0.3$, without introducing arbitrary client-size floors.
- Enforcing an artificial floor (such as $\ge 1,000$ samples) via rejection sampling truncates the tails of the Dirichlet distribution $\text{Dir}(\alpha=0.3)$, artificially dampening client heterogeneity.
- **Adopted Strategy `[PROPOSED]`:**
  1. Generate an unconstrained Dirichlet partition with $N=5$ clients and $\alpha=0.3$ using an explicit deterministic seed (`seed=42`).
  2. Inspect the resulting natural sample distribution and per-class frequencies across all 5 clients.
  3. Ensure that every client receives sufficient data to train without zero-sample degeneracies (which naturally occurs with 89,996 total samples across only 5 clients, averaging ~18,000 samples/client).
  4. If an empirical floor is ever deemed technically necessary, it must be documented, justified, and approved.

### 4.2 Invariant Partitioning Constraints `[PROPOSED]`
1. **Total Clients:** Exactly 5 clients ($c \in \{0, 1, 2, 3, 4\}$).
2. **Concentration Parameter:** $\alpha = 0.3$.
3. **Sample Conservation:** $\sum_{c=0}^4 M_c = 89,996$ (every official training sample is accounted for exactly once).
4. **Mutual Exclusivity:** $\bigcap_{c=0}^4 I_c = \emptyset$ (zero sample overlap across clients).
5. **Deterministic Reproducibility:** Governed by an explicit, documented random seed (`partition_seed=42`).

---

## 5. PathMNIST Dataset Specifications and MedMNIST Compatibility

### 5.1 Official PathMNIST Characteristics `[VERIFIED]`
- **Source:** MedMNIST v2 / NCT-CRC-HE-100K colorectal cancer histology.
- **Classes (9 total):**
  `0: adipose` (ADI), `1: background` (BACK), `2: debris` (DEB), `3: lymphocytes` (LYM), `4: mucus` (MUC), `5: smooth muscle` (MUS), `6: normal colon mucosa` (NORM), `7: cancer-associated stroma` (STR), `8: colorectal adenocarcinoma epithelium` (TUM).
- **Resolution:** $28 \times 28$ pixels, 3 channels (RGB).
- **Official Splits:**
  - Training set: **89,996 images**, labels shape `(89996, 1)`
  - Validation set: **10,004 images**, labels shape `(10004, 1)`
  - Test set: **7,180 images**, labels shape `(7180, 1)`
  - Total samples: **107,180**.
- **Data Types:** Images stored natively as `uint8` ($0 \dots 255$); labels stored as `uint8` ($0 \dots 8$).

---

## 6. Label Noise Implementation Design (40% Symmetric Noise on Client 5)

### 6.1 Specifications and Convention `[PROPOSED]`
- **Noisy Client:** Zero-based client index 4, corresponding to human-readable **Client 5** (among Clients 1, 2, 3, 4, 5).
- **Clean Clients:** Zero-based indices 0, 1, 2, 3 (Clients 1, 2, 3, 4) have **0% noise**.
- **Sample Selection:** Let $M_4$ be the total training samples allocated to Client 5. Exactly $M_{\text{noisy}}$ samples are selected without replacement using a documented integer rounding convention:
  $$M_{\text{noisy}} = \text{round}(0.40 \times M_4)$$
- **Symmetric Label Flipping:** For each selected sample index $i \in I_{\text{noisy}}$ with true label $y_i \in \{0, \dots, 8\}$, the corrupted label $\tilde{y}_i$ is drawn uniformly from the remaining 8 classes:
  $$\tilde{y}_i = (y_i + r_i) \pmod 9, \quad r_i \in \{1, 2, \dots, 8\}$$
  This guarantees $\tilde{y}_i \neq y_i$ with probability 1.
- **Noise Mask & Auditability:**
  A boolean array `noise_mask` of length $M_4$ is recorded:
  $$\text{noise\_mask}[i] = \begin{cases} \text{True} & \text{if } i \in I_{\text{noisy}} \\ \text{False} & \text{otherwise} \end{cases}$$
  Original clean labels are preserved in a parallel array `y_clean` for verification.
- **Strict Invariance:**
  - Client 5's images ($X$) are **never modified**.
  - All validation labels (global and per-client) remain **100% clean**.
  - All test labels remain **100% clean**.
  - Clients 1–4 training labels remain **100% clean**.
