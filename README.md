# Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Label Noise

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IndiaAI Fellowship](https://img.shields.io/badge/IndiaAI%20Fellowship-2026--27-orange.svg)](https://indiaai.gov.in/)

**Research Project:** Comparative Experimental Analysis of Federated Averaging (FedAvg) and Federated Proximal (FedProx) on Non-IID PathMNIST with 40% Client-Specific Label Noise
**Academic Supervision:** Dr. Sachin D N, Assistant Professor, Department of Computer Science and Engineering, IIIT Bhopal
**Fellowship Programme:** IndiaAI Fellowship 2026–27 (Research Screening Submission)
**Upstream Reference Framework:** [PFLlib](https://github.com/TsingZ0/PFLlib) (Tsinghua University; reference commit `0169ba7e412c9856a08bb3faefab1e35f538a3c1`)

---

## 1. Research Objective

This research repository provides an empirically rigorous, fully reproducible comparative analysis between two foundational federated optimization algorithms:
1. **Federated Averaging (FedAvg)** ([McMahan et al., AISTATS 2017](https://arxiv.org/abs/1602.05629))
2. **Federated Proximal (FedProx)** ([Li et al., MLSys 2020](https://arxiv.org/abs/1812.06127))

The study investigates how client-level data heterogeneity (non-IID Dirichlet distribution) interacts with asymmetric client-specific label corruption (40% label noise concentrated in a single client) during decentralized model convergence and held-out test generalization.

---

## 2. Dataset & Official Splits

The study uses the official **PathMNIST** dataset from the [MedMNIST v2](https://medmnist.com/) biomedical benchmark:
- **Modality:** Colorectal cancer histology ($28 \times 28 \times 3$ RGB images).
- **Classes:** 9 distinct histological tissue categories (Adipose, Background, Debris, Lymphocytes, Mucus, Smooth Muscle, Normal Colon Mucosa, Cancer-Associated Stroma, Colorectal Adenocarcinoma).
- **Official Splits & Integrity:**
  - **Training Split:** Exactly 89,996 images `(89996, 28, 28, 3)` and labels `(89996, 1)`, `uint8`.
  - **Validation Split:** Exactly 10,004 images `(10004, 28, 28, 3)` and labels `(10004, 1)`, `uint8`.
  - **Test Split:** Exactly 7,180 images `(7180, 28, 28, 3)` and labels `(7180, 1)`, `uint8`.
- **Cryptographic Verification:**
  - Official MedMNIST v2 archive: `data/pathmnist.npz` (205,615,438 bytes, ~196.09 MiB).
  - MD5: `a8b06965200029087d5bd730944a56c1` (Exact match with official benchmark).
  - SHA-256: `81823f52dc622e69db2db4c72f8e8e617938dd6864d3c1f23d4e49724a28ea72`.
- **Quarantine & Zero-Leakage Policy:** The 7,180-sample test split is strictly quarantined during training. It is evaluated exactly once post-training on the optimal validation checkpoint (zero exposure during model training or checkpoint selection).

---

## 3. Five-Client Non-IID Dirichlet Partitioning

The 89,996 training samples are partitioned across 5 federated clients using a class-wise Dirichlet distribution with concentration parameter $\alpha = 0.3$:

$$p_{k,c} \sim \text{Dir}(\alpha = 0.3), \quad \sum_{k=1}^5 p_{k,c} = 1$$

Where $p_{k,c}$ is the proportion of class $c$ allocated to client $k$.

| Client ID | Human Identifier | Sample Count | Percentage | Primary Enriched Classes |
| :---: | :---: | :---: | :---: | :--- |
| Client 0 | Client 1 | 24,650 | 27.39% | Class 0 (Adipose), Class 4 (Mucus) |
| Client 1 | Client 2 | 30,566 | 33.96% | Class 1 (Background), Class 8 (Colorectal Adenocarcinoma) |
| Client 2 | Client 3 | 14,388 | 15.99% | Class 7 (Stroma), Class 5 (Smooth Muscle) |
| Client 3 | Client 4 | 12,497 | 13.89% | Class 3 (Lymphocytes) |
| **Client 4** | **Client 5 (Noisy)**| **7,895** | **8.77%** | **Class 2 (Debris: 2,052), Class 6 (Normal Mucosa: 1,490)** |
| **Total** | | **89,996** | **100.00%** | **Exact partition; zero dropped or duplicate samples** |

### Mathematical Note on Client 5 Sample Count
Because $\alpha = 0.3 < 1.0$, the Dirichlet probability density function concentrates probability mass near the simplex boundaries. Across 9 independent draws (Seed 42), Client 5 naturally drew low shares in classes 5, 7, and 8, while drawing moderate allocations in classes 2 and 6. This skew is an authentic mathematical consequence of Dirichlet sampling, not a defect.

---

## 4. Client 5 Label-Noise Methodology

To model adversarial or low-grade edge client sensor error:
- **Target Client:** Client 5 (zero-based index 4).
- **Corruption Rate:** Exactly 40.000% of local training samples (3,158 corrupted samples out of 7,895).
- **Corruption Mechanism:** Deterministic symmetric label flipping modulo 9:
  $$\tilde{y} = (y_{\text{clean}} + r) \pmod 9, \quad r \in \{1, 2, \dots, 8\}$$
- **Isolation:** Clients 1–4, the official validation split, and the test split remain **100% clean**.

---

## 5. Model Architecture: `CompactPathCNN`

To enable reproducible execution on resource-constrained hardware without sacrificing representational capacity, the study implements `CompactPathCNN`:

```
Input: (B, 3, 28, 28)
  │
  ├── Conv2D(3 -> 32, kernel=3, padding=1) + ReLU() + MaxPool2D(2, 2)
  │     Output: (B, 32, 14, 14)
  │
  ├── Conv2D(32 -> 64, kernel=3, padding=1) + ReLU() + MaxPool2D(2, 2)
  │     Output: (B, 64, 7, 7)
  │
  ├── AdaptiveAvgPool2d((2, 2))
  │     Output: (B, 64, 2, 2) -> Flatten -> (B, 256)
  │
  ├── Linear(256 -> 64) + ReLU()
  │     Output: (B, 64)
  │
  └── Linear(64 -> 9)
        Output: Logits (B, 9)
```

### Exact Parameter Count Verification:
| Layer | Weight Shape | Bias Shape | Trainable Parameters |
| :--- | :--- | :--- | :---: |
| `conv1` | `[32, 3, 3, 3]` (864) | `[32]` (32) | **896** |
| `conv2` | `[64, 32, 3, 3]` (18,432) | `[64]` (64) | **18,496** |
| `fc1` | `[64, 256]` (16,384) | `[64]` (64) | **16,448** |
| `fc2` | `[9, 64]` (576) | `[9]` (9) | **585** |
| **Total Parameters** | | | **36,425** |

Strict state-dictionary loading (`load_state_dict(..., strict=True)`) is verified on all checkpoints.

---

## 6. Federated Algorithms & Optimization

### Federated Averaging (FedAvg)
- Local updates are computed via standard SGD:
  $$w_{k,t+1} = w_{k,t} - \eta \nabla F_k(w_{k,t})$$
- Central server performs sample-weighted aggregation:
  $$w_{t+1} = \sum_{k=1}^5 \frac{n_k}{N} w_{k,t+1}, \quad N = \sum_{k=1}^5 n_k = 89,996$$

### Federated Proximal (FedProx, $\mu = 0.01$)
- Employs a proximal regularization penalty to restrict client drift:
  $$\min_w h_k(w; w_t) = F_k(w) + \frac{\mu}{2} \|w - w_t\|^2$$
- Implemented via exact Perturbed SGD (`research/optimizers/perturbed_sgd.py`):
  $$d_p = \nabla F_k(w) + \mu (w - w_t), \quad w \leftarrow w - \eta d_p$$

---

## 7. Hyperparameter Configuration

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| Number of Clients ($K$) | 5 | Total federated participants |
| Client Participation ($C$) | 1.0 | Full synchronous client participation per round |
| Communication Rounds ($T$) | 30 | Global aggregation cycles |
| Local Epochs ($E$) | 2 | Training passes over local partition per round |
| Batch Size ($B$) | 32 | Mini-batch sample size |
| Learning Rate ($\eta$) | 0.01 | Client-side SGD step size (fixed, momentum=0, weight decay=0) |
| Proximal Parameter ($\mu$) | 0.01 | FedProx proximal contraction penalty |
| Loss Function | CrossEntropy | Standard multi-class cross-entropy |
| Loss Reduction | Mean (default) | Local client batches compute mean cross-entropy loss via nn.CrossEntropyLoss(), which is multiplied by batch sample count len(y) and accumulated to calculate exact sample-weighted mean epoch loss |

---

## 8. Installation & Setup

### Prerequisites
- Python 3.11.x (recommended)
- Git 2.30+

### Step 1: Clone Repository
```powershell
git clone https://github.com/NANDANS26/pathmnist-federated-learning.git
cd pathmnist-federated-learning
```

### Step 2: Environment Setup

#### Option A: CPU-Only Setup (Primary Study Platform)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### Option B: CUDA-Accelerated Setup (Compatible Machines)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 9. Dataset Acquisition & Verification

Download and generate contiguous memory-mapped arrays from official MedMNIST:
```powershell
python scripts/initialize_experiment.py
```
This script downloads `data/pathmnist.npz`, verifies the cryptographic SHA-256 and MD5 hashes, precomputes client Dirichlet partitions, injects Client 5 label noise, and compiles provenance manifests in `experiments/manifests/`.

---

## 10. Test & Regression Suite

Run the full automated test suite (22 tests covering model dimensions, backward pass, perturbed SGD, Dirichlet partitioning, deterministic mini-batch shuffling, and zero-leakage split isolation):
```powershell
pytest research/tests/ -v
```

---

## 11. Running Experiments

### CLI Arguments & Orchestrator Flags
The orchestrator `scripts/run_multiseed.py` supports the following explicit command-line flags:
- `--seeds [INT ...]`: One or more random seeds executed sequentially (default: `42`).
- `--rounds INT`: Number of communication rounds per algorithm (default: `30`).
- `--attempt INT`: Unique attempt index to prevent overwriting previous runs (default: `1`). If `attempt > 1`, outputs are written to `experiments/seed_<id>_attempt_<attempt>/`.
- `--force`: Force overwrite existing experiment directory. Without this flag, the runner aborts if destination directory contains artifacts.

### Execute Corrected Seed 42 Run (Unbuffered Streaming)
```powershell
python -u scripts/run_multiseed.py --seeds 42 --rounds 30 --attempt 1
```

### Execute Multi-Seed Benchmark (Seeds 42, 123, 2026, 3407, 7777)
```powershell
python -u scripts/run_multiseed.py --seeds 42 123 2026 3407 7777 --rounds 30 --attempt 1
```

---

## 12. Directory Structure

```
pathmnist-federated-learning/
│
├── research/                         # Modular research framework
│   ├── core/                         # FederatedClient, FederatedServer, FedAvg & FedProx runners
│   ├── data/                         # PathMNIST loader, Dirichlet partitioner, noise injector
│   ├── diagnostics/                  # Update norms, cosine similarity, memory tracker
│   ├── models/                       # CompactPathCNN architecture
│   ├── optimizers/                   # Exact PerturbedSGD optimizer
│   └── tests/                        # 22 automated unit and regression tests
│
├── scripts/                          # Research tooling & execution scripts
│   ├── initialize_experiment.py      # Dataset acquisition & manifest initialization
│   ├── independent_verification.py   # Held-out test evaluation & confusion matrix generator
│   ├── run_multiseed.py              # Multi-seed orchestration engine with unbuffered telemetry
│   └── sync_to_wandb.py              # Offline-to-cloud W&B synchronizer
│
├── experiments/                      # Persisted experiment artifacts
│   ├── configs/                      # Seed manifests (partition, validation, noise)
│   ├── checkpoints/                  # Trained model checkpoints (.pt)
│   ├── logs/                         # Durable JSONL round logs & summary CSVs
│   ├── results/                      # Confusion matrices, test predictions, benchmark JSONs
│   ├── figures/                      # High-resolution (300 DPI) publication plots
│   ├── legacy_seed42_pre_batch_fix/  # Preserved Phase 3 experimental run
│   └── interrupted_attempts/         # Quarantined partial runs
│
├── reports/                          # Comprehensive Phase 0 to Phase 4B research audits
├── requirements.txt                  # Portable dependency manifest
├── PROJECT_STATUS.md                 # Live milestone and execution tracking document
└── README.md                         # Master documentation
```

---

## 13. Experiment Tracking & Weights & Biases (W&B)

All rounds durably log telemetry to disk as JSON Lines (`.jsonl`) files with synchronous OS flushes.

To synchronize local runs to a public W&B dashboard:
```powershell
wandb login <YOUR_WANDB_API_KEY>
python scripts/sync_to_wandb.py --project pathmnist-fedavg-fedprox
```

---

## 14. Hardware Platform & Resource Constraints

All experiments in this submission were executed on an entry-level physical CPU host:
- **Processor:** Intel Pentium G3240 @ 3.10 GHz (Dual-Core, 2 Threads, no AVX2/FMA).
- **Memory:** 4 GB Physical DDR3 RAM.
- **Operating System:** Windows 10 Pro 64-bit.
- **Memory Profiling:** Process RSS strictly bounded to **289.8 MB – 659.3 MB**.
- **Wall-Clock Duration:**
  - 1 paired round $\approx 9.5$ to $12.0$ minutes.
  - 30 paired rounds $\approx 4.5$ to $6.0$ hours.

---

## 15. Experimental Results & Current Status

### Summary of Completed Experimental Evidence:

| Experiment Identifier | Status | FedAvg Val Acc | FedProx Val Acc | FedAvg Test Acc | FedProx Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Legacy Phase 3 (Seed 42)** | **Archived** | 73.29% (R26) | 73.36% (R30) | **79.46%** | 76.07% |
| **Interrupted Attempt (Seed 42)** | **Quarantined** (Rounds 1–20) | 68.27% (R20) | — | — | — |
| **Corrected Multi-Seed (Seed 42)**| **Ready for Execution** | Pending | Pending | Pending | Pending |

### Key Scientific Findings (Legacy Run):
- **Generalization Gap:** While FedProx achieved marginally higher peak validation accuracy (+0.07%), FedAvg exhibited superior held-out test generalization (+3.39% accuracy, +0.0474 macro-F1).
- **Hypothesized Mechanism:** Proximal contraction penalty ($\mu = 0.01$) constrained clean clients from taking sufficiently large update steps away from the noisy global average $w_t$, contributing to recall collapse on classes heavily represented on the noisy client (Class 2: 28.32% vs 60.47%; Class 6: 48.18% vs 77.33%).

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for live milestone tracking and [`reports/`](reports/) for complete technical audits.

---

## 16. Licensing & Dataset Provenance

### Software Code License
The source code, scripts, unit tests, configuration files, and experimental orchestration pipelines in this repository are licensed under the [MIT License](LICENSE) (Copyright (c) 2026 NANDAN S).

### Dataset License & Attribution
The **PathMNIST** dataset used in this research is part of the [MedMNIST v2](https://medmnist.com/) biomedical benchmark suite (Yang et al., *Scientific Data* 2023; Kather et al., *PLOS Medicine* 2019) and is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license via Zenodo ([Record 10519652](https://zenodo.org/records/10519652)).
- The project's MIT license applies exclusively to the research code, scripts, configuration, and documentation developed in this repository.
- The PathMNIST dataset is **not** covered by the MIT license; it is downloaded separately from official sources and remains governed by its native CC BY 4.0 license.

