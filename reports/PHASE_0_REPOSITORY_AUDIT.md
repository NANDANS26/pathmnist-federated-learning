# PHASE 0: PFLlib REPOSITORY TECHNICAL AUDIT REPORT (REVISED)

**Project:** Federated Learning on PathMNIST (FedAvg vs. FedProx)  
**Fellowship Program:** IndiaAI Fellowship 2026–27 Screening Submission  
**Audit Author:** Senior ML Systems Engineer & Federated Learning Technical Auditor  
**Audit Status:** Technical Correction & Research Validation Completed  
**Date of Audit:** November 6, 2026  
**Audited Target:** [TsingZ0/PFLlib](https://github.com/TsingZ0/PFLlib) (Official Framework)

---

## 1. Executive Repository Overview

PFLlib (*Personalized Federated Learning Library*) is an open-source academic benchmarking framework developed primarily by researchers at Tsinghua University and the FL community. It is designed to evaluate both standard and personalized federated learning algorithms across diverse non-IID data partitioning schemes.

This audit evaluates the real upstream codebase acquired directly from GitHub via Git clone into `third_party/PFLlib`.

---

## 2. Git Clone and Repository Provenance

The official upstream repository was acquired using Git. Below are the verified provenance metadata:

| Attribute | Verified Value `[VERIFIED]` | Verification Method / Command |
|---|---|---|
| **Upstream URL** | `https://github.com/TsingZ0/PFLlib.git` | `git remote -v` |
| **Fetch Remote** | `origin https://github.com/TsingZ0/PFLlib.git (fetch)` | `git remote -v` |
| **Push Remote** | `origin https://github.com/TsingZ0/PFLlib.git (push)` | `git remote -v` |
| **Default Remote Branch** | `master` *(Note: Verified upstream default is `master`, not `main`)* | `git branch -a` |
| **Checked-out Branch** | `master` | `git branch --show-current` |
| **HEAD Commit SHA** | `0169ba7e412c9856a08bb3faefab1e35f538a3c1` | `git rev-parse HEAD` |
| **Commit Timestamp** | `Sun Jan 25 10:31:48 2026 +0800` | `git log -1` |
| **Commit Author** | `TsingZ0 <2719584131@qq.com>` | `git log -1` |
| **Commit Message** | `Merge pull request #257 from yucheng514/master \n fix typo in README.md` | `git log -1` |
| **Working Tree Status** | Clean (`0` untracked files, `0` staged changes, `0` modified files) | `git status --short` |
| **License** | Apache License 2.0 (January 2004) | Full text in `LICENSE` file |

---

## 3. Physical Footprint and File Inventory Statistics `[VERIFIED]`

An exhaustive recursive audit of the acquired repository was performed using automated PowerShell and Git index inspections:

- **Total Tracked Files:** 196
- **Total Physical Size on Disk:** 151.31 MB (158,661,120 bytes)
- **Total Tracked File Content Size:** 151.31 MB
- **Total Python Source Files (`.py`):** 122 files
- **Total YAML Configuration Files (`.yaml`/`.yml`):** 2 files (`env_cuda_latest.yaml`, `.github/workflows/deploy.yml`)
- **Total Shell Scripts (`.sh`):** 1 file (`prepare.sh`)
- **Total Markdown Files (`.md`):** 1 file (`README.md`)
- **Total Client Implementations (`client*.py`):** 40 files in `system/flcore/clients/`
- **Total Server Implementations (`server*.py`):** 41 files in `system/flcore/servers/`
- **Total Custom Optimizer Implementations:** 1 file (`system/flcore/optimizers/fedoptimizer.py` containing 5 optimizers)
- **Total Model Architecture Definitions:** 6 files in `system/flcore/trainmodel/`
- **Total Dataset Generator Scripts:** 24 files (`dataset/generate_*.py`)

### Breakdown of Repository Size Discrepancy `[VERIFIED]`
Unlike typical pure-source codebases which are under 5 MB, PFLlib is **151.31 MB** because the authors have pre-committed binary dataset assets directly into Git tracking:
1. `dataset/MNIST/rawdata/MNIST/raw/*-ubyte.gz`: The full raw MNIST binary gzip archives (~11.6 MB).
2. `dataset/MNIST/train/*.npz` & `test/*.npz`: 20 pre-partitioned client NumPy `.npz` files (~45 MB).
3. `dataset/utils/LEAF/data/shakespeare/data/...`: Synthetic Shakespeare NLP datasets (~94 MB).

---

## 4. Complete Directory Classification and Functional Architecture

The repository is structured into four primary top-level folders:

```
PFLlib/
├── .github/          # GitHub Actions workflow for documentation deployment (gh-pages)
├── dataset/          # Dataset acquisition, non-IID partitioning, and serialisation
│   └── utils/        # Mathematical partitioning routines (Dirichlet, pathological)
├── docs/             # Static HTML documentation published to GitHub Pages
└── system/           # Core federated learning execution engine
    ├── flcore/       # Algorithmic implementations
    │   ├── clients/  # Local client training routines
    │   ├── servers/  # Server aggregation and orchestration routines
    │   ├── optimizers/ # Custom PyTorch federated optimizers
    │   └── trainmodel/ # Neural network backbone definitions
    └── utils/        # System utilities (data loaders, memory profilers, result saving)
```

### Detailed Functional Responsibility Matrix

| Subdirectory / Component | Functional Responsibility `[VERIFIED]` | Relevance to IndiaAI Screening `[INFERRED]` |
|---|---|---|
| `system/flcore/servers/` | Orchestrates the communication rounds, client sampling, model broadcast, weight aggregation, and evaluation dispatching. Contains `serverbase.py`, `serveravg.py`, and `serverprox.py`. | **Direct Reference / Reusable Logic** |
| `system/flcore/clients/` | Implements local client execution: local SGD steps, forward/backward passes, local loss computation, and metric logging. Contains `clientbase.py`, `clientavg.py`, and `clientprox.py`. | **Direct Reference / Reusable Logic** |
| `system/flcore/optimizers/` | Contains `PerturbedGradientDescent` in `fedoptimizer.py`, implementing the proximal gradient update: $w \leftarrow w - \eta (\nabla F_k(w) + \mu(w - w_t))$. | **Direct Reference / Mathematical Core** |
| `system/flcore/trainmodel/` | Contains CNN, ResNet, MobileNet, MLP, and Transformer models. `models.py` defines `FedAvgCNN`, a standard 2-layer ConvNet with 2 FC layers. | **Direct Reference / Architectural Baseline** |
| `dataset/` | Contains standalone scripts to download and slice datasets into `.npz` client shards. Includes `generate_MNIST.py`. | **Must Audit & Redesign** (flawed split logic) |
| `dataset/utils/` | Contains `dataset_utils.py` with `separate_data()`, which implements Dirichlet partitioning `np.random.dirichlet()`. | **Direct Reference for Dirichlet Logic** |
| `system/utils/` | Contains `data_utils.py` (data loading), `mem_utils.py` (memory tracking), `result_utils.py` (HDF5 result aggregation), and `dlg.py` (Deep Leakage from Gradients evaluation). | **High Memory Fragmentation Risk** |

---

## 5. Key Source Files Directly Relevant to FedAvg and FedProx

1. `system/main.py`: Entry point for CLI argument parsing, model instantiation, algorithm selection, and top-level execution loop.
2. `system/flcore/servers/serverbase.py`: Abstract base class `Server` managing client lists, communication budgets, weighted aggregation (`add_parameters`), and HDF5 serialization (`save_results`).
3. `system/flcore/servers/serveravg.py`: Concrete class `FedAvg` overriding `train()` to run communication rounds.
4. `system/flcore/servers/serverprox.py`: Concrete class `FedProx` overriding `train()` to orchestrate proximal client training.
5. `system/flcore/clients/clientbase.py`: Abstract base class `Client` handling mini-batch DataLoader instantiation, parameter copying, and test/train metrics.
6. `system/flcore/clients/clientavg.py`: Concrete class `clientAVG` implementing standard local SGD.
7. `system/flcore/clients/clientprox.py`: Concrete class `clientProx` tracking global model parameter clones and calling `PerturbedGradientDescent`.
8. `system/flcore/optimizers/fedoptimizer.py`: Defines `PerturbedGradientDescent(Optimizer)`.
9. `dataset/generate_MNIST.py` & `dataset/utils/dataset_utils.py`: Generation scripts implementing Dirichlet non-IID partitioning.

---

## 6. Upstream Status and Architectural Health Assessment

The upstream repository is functionally mature for academic experimentation on Linux GPU workstations. However, our technical audit reveals several architectural and operational assumptions that conflict with our production research goals:
1. **Linux / Shell Specifics `[VERIFIED]`:** `prepare.sh` assumes Ubuntu bash, conda, and `nvidia-smi`.
2. **GPU Bias `[VERIFIED]`:** Defaults in `main.py` hardcode `-dev cuda`.
3. **Memory Allocator Churn `[INFERRED]`:** Image data is eagerly converted to full `float32` tensors on a per-client shard basis and stored as Python lists of individual tuples, creating substantial object allocation churn and heap fragmentation.
4. **Data Leakage in Generation `[VERIFIED]`:** The existing dataset generators merge official train and test sets into a single pool before Dirichlet allocation, destroying official benchmark boundaries.
5. **No Per-Client History Tracking `[VERIFIED]`:** Upstream discards per-client trajectories after calculating the cross-client standard deviation.
6. **No W&B or Update Diagnostics `[VERIFIED]`:** PFLlib lacks experiment tracking integration, client update $L_2$ norm tracking, and pairwise cosine similarity computation.

These audit findings dictate that we must **not** hack directly into PFLlib, but rather establish an independent, modular research implementation that adapts verified algorithmic formulas while eliminating upstream memory bugs and experimental flaws.
