# Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Label Noise

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IndiaAI Fellowship](https://img.shields.io/badge/IndiaAI%20Fellowship-2026--27-orange.svg)](https://indiaai.gov.in/)

**Research Project:** Comparative Experimental Analysis of Federated Averaging (FedAvg) and Federated Proximal Optimization (FedProx) on Non-IID PathMNIST with 40% Symmetric Label Noise
**Author:** Nandan S
**Fellowship Track:** IndiaAI Fellowship 2026–2027 (Research Screening Submission)
**Primary Repository:** [https://github.com/NANDANS26/pathmnist-federated-learning](https://github.com/NANDANS26/pathmnist-federated-learning)

---

## 1. Executive Summary & Deliverables

This repository contains the complete, seed-controlled paired 5-seed benchmark comparing **FedAvg** against **FedProx ($\mu = 0.01$)** across 300 total communication rounds (30 rounds per algorithm across 5 random seeds: 42, 123, 2026, 3407, 7777).

### Research Scope & Operational Boundaries
- **Simulated Federation:** This benchmark evaluates the public PathMNIST dataset partitioned synthetically into five client shards. No real-world multi-hospital clinical network was deployed.
- **Privacy Controls:** Differential privacy (DP) and secure multiparty computation (SMPC) were not implemented.
- **Algorithmic Scope:** Conclusions regarding FedProx are strictly bounded to the evaluated configuration ($\mu = 0.01$, CompactPathCNN architecture, PathMNIST dataset, Dirichlet $\alpha = 0.3$, single-client $40\%$ symmetric label noise, and 30 rounds). We do not claim generic failure of FedProx across all medical FL configurations.

### Key Submission Deliverables (`submission/` directory)
- **Full Research Report (PDF):** [`submission/PathMNIST_Federated_Learning_Final_Report.pdf`](submission/PathMNIST_Federated_Learning_Final_Report.pdf) *(6 pages)*
- **Full Research Report (Markdown):** [`submission/PathMNIST_Federated_Learning_Final_Report.md`](submission/PathMNIST_Federated_Learning_Final_Report.md)
- **Executive One-Page Summary (PDF):** [`submission/PathMNIST_Federated_Learning_One_Page_Summary.pdf`](submission/PathMNIST_Federated_Learning_One_Page_Summary.pdf) *(strictly 1 page A4)*
- **Executive One-Page Summary (Markdown):** [`submission/PathMNIST_Federated_Learning_One_Page_Summary.md`](submission/PathMNIST_Federated_Learning_One_Page_Summary.md)
- **Reactive Optimizers Article Summary:** [`submission/Reactive_Optimizers_Article_Summary.md`](submission/Reactive_Optimizers_Article_Summary.md)
- **Reactive Optimizers Source Status:** [`submission/REACTIVE_OPTIMIZERS_SOURCE_REQUIRED.md`](submission/REACTIVE_OPTIMIZERS_SOURCE_REQUIRED.md)
- **Weights & Biases Guide:** [`submission/WANDB_PUBLICATION_REQUIRED.md`](submission/WANDB_PUBLICATION_REQUIRED.md) *(runs tracked offline locally; publish via `python scripts/publish_to_wandb.py` upon user login)*
- **Consolidated Benchmark CSV:** [`experiments/results/multiseed_consolidated_summary.csv`](experiments/results/multiseed_consolidated_summary.csv)
- **Consolidated Benchmark JSON:** [`experiments/results/multiseed_consolidated_summary.json`](experiments/results/multiseed_consolidated_summary.json)
- **Publication Figures:** [`reports/figures/`](reports/figures/) and [`submission/figures/`](submission/figures/)

---

## 2. Benchmark Findings (5-Seed Paired Evaluation)

Over the five seeds on the quarantined held-out test split (7,180 colorectal histology images):

| Metric | FedAvg (Mean ± Sample SD) | FedProx ($\mu = 0.01$) | Paired Difference ($\Delta$: Prox - Avg) | Paired $t$-test ($df=4$) | $p$-value (two-sided) | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Accuracy** | **$68.880\% \pm 9.266\text{ pp}$** | **$68.279\% \pm 8.551\text{ pp}$** | **$-0.602 \pm 1.685\text{ pp}$** | $t = -0.7983$ | $p = 0.46938$ | $[-2.694\text{ pp}, +1.491\text{ pp}]$ |
| **Test Macro-F1** | **$0.6008 \pm 0.0948$** | **$0.5934 \pm 0.0960$** | **$-0.0075 \pm 0.0105$** | $t = -1.5908$ | $p = 0.18686$ | $[-0.0205, +0.0056]$ |
| **Test Balanced Acc** | **$63.176\% \pm 7.924\text{ pp}$** | **$62.902\% \pm 8.072\text{ pp}$** | **$-0.273 \pm 1.344\text{ pp}$** | $t = -0.4546$ | $p = 0.67294$ | $[-1.942\text{ pp}, +1.395\text{ pp}]$ |
| **Test Loss** | $0.9082 \pm 0.1897$ | $0.9239 \pm 0.1877$ | $+0.0156 \pm 0.0407$ | $t = +0.8566$ | $p = 0.43994$ | $[-0.0350, +0.0662]$ |
| **Best Val Accuracy** | $65.986\% \pm 4.292\text{ pp}$ | $65.466\% \pm 4.420\text{ pp}$ | **$-0.520 \pm 0.166\text{ pp}$** | $t = -7.0096$ | $p = 0.00218$ | $[-0.726\text{ pp}, -0.314\text{ pp}]$ |
| **Wall-Clock Time / Seed**| $42.86 \pm 16.08\text{ min}$ | $51.48 \pm 24.38\text{ min}$ | $+8.62 \pm 31.39\text{ min}$ | — | — | Total: $471.7\text{ min}$ |

### Statistical & Scientific Takeaway
- **No statistically significant difference in test performance was detected between FedAvg and FedProx under the evaluated configuration across five random seeds ($p = 0.469$).**
- Because five seeds provide limited statistical power and no formal equivalence testing with predefined equivalence margins was conducted, **this lack of statistical significance does not establish equivalence**.
- **Mechanistic Cause:** The proximal term $\frac{\mu}{2} \|\mathbf{w} - \mathbf{w}^t\|^2$ penalizes the magnitude of parameter drift away from $\mathbf{w}^t$, but does not alter the erroneous gradient orientation induced by mislabeled samples. When aggregated via sample weighting, Client 5's contaminated model update still enters the global consensus proportionally to its sample weight. Furthermore, negative cosine similarities between client updates and the global model reflect both label noise and Dirichlet class distribution skew ($\alpha = 0.3$).

---

## 3. Dataset & Partitioning Methodology

- **Dataset:** Official **PathMNIST** (MedMNIST v2 benchmark, 9 colorectal tissue categories, $28 \times 28 \times 3$ RGB).
  - Train split: 89,996 images
  - Validation split: 10,004 images
  - Quarantined test split: 7,180 images (evaluated strictly post-training)
  - Dataset SHA-256: `81823f52dc622e69db2db4c72f8e8e617938dd6864d3c1f23d4e49724a28ea72`
- **Non-IID Partitioning:** Class-wise Dirichlet distribution ($\alpha = 0.3$) across 5 clients.
- **Symmetric Label Noise:** Exactly 40.00% symmetric label noise on Client 5 (e.g. 3,158 of 7,895 samples in Seed 42), generated via deterministic modulo-9 replacement. Clients 1–4 remain 100% clean.

---

## 4. Model Architecture: `CompactPathCNN`

To enable reproducible execution on standard hardware, the study uses `CompactPathCNN`, containing exactly **TWO convolutional layers** and **36,425 trainable parameters**:

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

| Layer | Parameters |
| :--- | :---: |
| `conv1` (Conv2D $3 \to 32$) | 896 |
| `conv2` (Conv2D $32 \to 64$) | 18,496 |
| `fc1` (Linear $256 \to 64$) | 16,448 |
| `fc2` (Linear $64 \to 9$) | 585 |
| **Total Trainable Parameters** | **36,425** |

---

## 5. Hardware Platform & Compute Distinction

- **Experiment Compute Engine:** **PyTorch CPU Execution** with 2 compute threads and 1 interop thread (`torch.set_num_threads(2)`).
- **Physical Host Machine:** AMD Ryzen 5 5600H (6 cores / 12 threads @ 3.30 GHz base / 4.25 GHz boost), 16 GB DDR4 RAM, Windows 11 x64.
- **GPU Diagnostics vs Training Execution:** While CUDA availability and NVIDIA RTX 3050 Laptop GPU diagnostics were verified during pre-flight checks, **the entire federated training pipeline was executed on the CPU** in compliance with the standardized project configuration.

---

## 6. Reproduction Instructions

### Step 1: Environment Setup
```powershell
git clone https://github.com/NANDANS26/pathmnist-federated-learning.git
cd pathmnist-federated-learning
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 2: Initialize Dataset & Manifests
```powershell
python scripts/initialize_experiment.py
```

### Step 3: Run Test Suite (22 Unit & Regression Tests)
```powershell
python -m pytest -v
```

### Step 4: Execute Full 5-Seed Benchmark
```powershell
python -u scripts/run_multiseed.py --seeds 42 123 2026 3407 7777 --rounds 30
```

### Step 5: Consolidate Metrics & Re-plot Figures
```powershell
python scripts/consolidate_and_plot_phase7.py
```

---

## 7. License & Attribution

- **Source Code:** MIT License (Copyright (c) 2026 NANDAN S).
- **PathMNIST Dataset:** MedMNIST v2 suite, licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
