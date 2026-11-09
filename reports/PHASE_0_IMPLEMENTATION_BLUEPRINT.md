# PHASE 0: RESEARCH IMPLEMENTATION BLUEPRINT (REVISED)

**Project:** Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Label Noise  
**Fellowship Program:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Author:** Senior ML Systems Engineer & Federated Learning Researcher  
**Target Environment:** Windows 10, Intel Pentium G3240 (2-Core CPU), 4 GB RAM  
**Audit Status:** Technical Correction & Research Validation Completed  
**Date:** November 6, 2026  

---

## 1. Executive Vision & Architectural Tenets

The goal of this research project is to provide an empirical, mathematically grounded comparison between **Federated Averaging (FedAvg)** and **Federated Proximal (FedProx)** under realistic pathological conditions:
1. **Severe non-IID class skew:** 5 clients, partitioned via Dirichlet distribution with concentration parameter $\alpha = 0.3$.
2. **Adversarial / Noisy Local Updates:** Exactly one client (Client 5, zero-based index 4) corrupted with 40% symmetric label noise in its training partition.
3. **Rigorous Telemetry:** Global validation accuracy, per-client validation trajectories, client parameter update $L_2$ norms, pairwise cosine similarity dynamics, and W&B experiment tracking.
4. **Strict Environmental Constraints:** Stable, non-thrashing execution on Windows with 4 GB RAM and 2 CPU cores.

To preserve scientific validity, we establish an **independent, modular research implementation** within `research/` that draws verified mathematical logic from `third_party/PFLlib` while eliminating its memory bloat, data leakage defects, and missing telemetry.

---

## 2. Directory Structure Blueprint `[PROPOSED]`

```
pathmnist-federated-learning/
│
├── third_party/
│   └── PFLlib/               # Untouched official upstream repository (Apache-2.0)
│
├── reports/                  # Phase audit and research documentation reports
│   ├── PHASE_0_REPOSITORY_AUDIT.md
│   ├── PHASE_0_FEDERATED_ALGORITHMS_AUDIT.md
│   ├── PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md
│   ├── PHASE_0_RESOURCE_FEASIBILITY_REPORT.md
│   ├── PHASE_0_EVALUATION_AND_DIAGNOSTICS_AUDIT.md
│   ├── PHASE_0_RISK_REGISTER.md
│   ├── PHASE_0_IMPLEMENTATION_BLUEPRINT.md
│   ├── PHASE_0_COMMANDS_AND_VERIFICATION.md
│   └── PHASE_0_FINAL_REVIEW_REPORT.md
│
├── research/                 # Original, modular federated learning framework
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pathmnist_loader.py    # MedMNIST acquisition & split preservation
│   │   ├── partitioner.py         # Dirichlet non-IID partitioner (alpha=0.3, seed=42)
│   │   └── noise_injector.py      # 40% symmetric label noise on Client 5
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── compact_cnn.py         # CompactPathCNN: 3-channel 28x28 CNN (36,425 params)
│   │
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── perturbed_sgd.py       # Exact FedProx proximal optimizer
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── client.py              # Zero-waste client (uint8 lazy-load)
│   │   ├── server.py              # Server orchestrator & weighted aggregation
│   │   ├── fedavg.py              # FedAvg pipeline implementation
│   │   └── fedprox.py             # FedProx pipeline implementation
│   │
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── update_metrics.py      # L2 norms, pairwise cosine sim & sim to weighted mean
│   │   └── tracking.py            # Local JSONL logger + Weights & Biases hook
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_splits.py         # Zero-leakage verification
│   │   ├── test_partition.py      # Dirichlet distribution verification
│   │   ├── test_noise.py          # Exact 40% noise & mask verification
│   │   ├── test_proximal.py       # Proximal gradient mathematical equivalence
│   │   └── test_memory.py         # RAM budget pilot measurement
│   │
│   └── run_experiment.py          # Unified CLI experiment runner
│
├── experiments/              # Experimental outputs, logs, charts, and metrics
│   ├── configs/              # Reproducible experiment YAML / JSON configs
│   ├── logs/                 # Local durable JSONL / CSV execution logs
│   ├── checkpoints/          # Model checkpoints (best and final)
│   └── figures/              # Publication-ready plots (PDF / PNG)
│
├── documentation/            # Extended research notes, research briefs, slides
│
├── requirements.txt          # Minimal CPU-only dependencies
└── README.md                 # Project presentation, results summary, instructions
```

---

## 3. Reuse vs. Custom Implementation Decisions

| Component | PFLlib Upstream Status | Blueprint Decision | Technical Justification |
|---|---|---|---|
| **FedProx Gradient Update** | Implemented in `fedoptimizer.py` as `PerturbedGradientDescent` `[VERIFIED]` | **REUSE LOGIC** `[PROPOSED]` | The in-place gradient perturbation $d_p = \nabla F + \mu(w - w_t)$ is mathematically exact and optimal. |
| **FedAvg Aggregation** | Implemented in `serverbase.py` as `add_parameters` `[VERIFIED]` | **REUSE LOGIC** `[PROPOSED]` | Weighted aggregation based on client training samples $\sum \frac{n_k}{N} \theta_k$ is standard and correct. |
| **Dataset Loading & Storage** | Converts each shard to float32 Python lists in `data_utils.py` `[VERIFIED]` | **CUSTOM RE-IMPLEMENTATION** `[PROPOSED]` | Storing client partitions in contiguous uint8 NumPy arrays (training split: 211.67 MB / 201.86 MiB; validation: 23.53 MB / 22.44 MiB; total 3 splits: 252.09 MB / 240.41 MiB) and converting lazily to float32 per batch minimizes memory churn and fragmentation. |
| **Dataset Splitting** | Merges train and test sets in `generate_MNIST.py` `[VERIFIED]` | **CUSTOM RE-IMPLEMENTATION** `[PROPOSED]` | Non-preservation defect identified in upstream MNIST script. Custom loader preserves official MedMNIST train (89,996), val (10,004), and test (7,180) splits as a strict scientific precaution. |
| **Dirichlet Partitioning** | `dataset_utils.py` with unbalanced loop `[VERIFIED]` | **ADAPT & ENCAPSULATE** `[PROPOSED]` | Adapt Dirichlet allocation with fixed seed (`seed=42`) without arbitrary rejection floors. Validate natural class distribution; assert no client receives 0 total samples. |
| **Label Noise Injection** | Absent in PFLlib `[VERIFIED]` | **CUSTOM IMPLEMENTATION** `[PROPOSED]` | Build dedicated 40% symmetric label noise injector on Client 5 (zero-based index 4) with deterministic seed and verifiable noise mask. |
| **Model Backbone** | Inconsistent `BaseHeadSplit` wrapper in `main.py` `[VERIFIED]` | **CUSTOM RE-IMPLEMENTATION** `[PROPOSED]` | Build uniform monolithic `CompactPathCNN` (3 channels, 9 classes, 36,425 params; 4.31 MMACs / 8.61 MFLOPs per image) used identically by both FedAvg and FedProx. |
| **Update Diagnostics** | Absent in PFLlib `[VERIFIED]` | **CUSTOM IMPLEMENTATION** `[PROPOSED]` | Build zero-copy update extractor computing $L_2$ norms, pairwise cosine similarities, and similarity to sample-weighted mean update each round. |
| **Experiment Tracking** | Absent in PFLlib (only HDF5 scalar dump) `[VERIFIED]` | **CUSTOM IMPLEMENTATION** `[PROPOSED]` | Build unified logger streaming to local durable JSONL/CSV tables and supplementary Weights & Biases (W&B). |

---

## 4. Hardware Constraints and Optimization Rules (4 GB RAM / 2-Core CPU)

To guarantee stable execution on the target Windows system:
1. **PyTorch Threads:** Limit CPU thread contention via `torch.set_num_threads(2)` and `torch.set_num_interop_threads(1)`.
2. **DataLoader:** Always set `num_workers=0` and `pin_memory=False`.
3. **Sequential Processing:** Never execute clients in parallel threads. Execute clients serially.
4. **Memory Cleanup:** Invoke `gc.collect()` at the conclusion of every communication round.
5. **Mini-Batch Size:** Set local training batch size to $32$ (or $64$).
6. **Peak Memory Target `[PROPOSED TARGET / UNVERIFIED]`:** Maintain process RSS memory below **1.2 GB**. Actual peak memory will be measured during the Phase 1 pilot test using `psutil`.

---

## 5. Experimental Configuration Matrix `[PROPOSED]`

| Experimental Hyperparameter | Planned Value | Justification |
|---|---|---|
| **Dataset** | PathMNIST (MedMNIST v2) `[VERIFIED]` | Colorectal cancer histology, 9 classes, $28 \times 28 \times 3$ RGB. |
| **Total Clients ($N$)** | 5 | Specified in screening prompt. |
| **Client Selection Ratio ($C$)** | 1.0 (Full participation: all 5 clients active every round) | Isolates algorithmic behavior from client subsampling variance. |
| **Non-IID Partitioning** | Dirichlet distribution ($\alpha = 0.3$, seed=42) | Specified in screening prompt. Evaluates severe class heterogeneity. |
| **Label Noise** | Client 5 (zero-based index 4) with 40% symmetric noise | Corrupts training set of Client 5 only; validation and test sets remain 100% clean. |
| **Base Model Architecture** | CompactPathCNN (36,425 parameters; 4.31 MMACs / 8.61 MFLOPs per image) | Lightweight, fast forward/backward passes on CPU without BatchNorm. |
| **Local Optimizer** | SGD (momentum=0.0, weight decay=0.0) | Standard baseline for clean proximal comparison. |
| **Local Learning Rate ($\eta$)** | 0.01 | Standard stable step size for federated CNN training. |
| **Local Epochs ($E$)** | 2 epochs per client per round | Produces realistic client drift without immediate divergence. |
| **Batch Size ($B$)** | 32 | Memory-friendly (raw RGB batch: 294.0 KiB). |
| **Communication Rounds ($R$)** | Mutually exclusive round calibration (100, 50, 30 rounds) `[PROPOSED]` | Horizon calibrated from Phase 2 pilot: $<35\text{s} \to 100\text{r}$, $35\text{--}75\text{s} \to 50\text{r}$, $75\text{--}120\text{s} \to 30\text{r}$. |
| **Proximal Parameter ($\mu$)** | $\mu \in \{0.001, 0.01, 0.1\}$ (Core: $\mu = 0.01$) | Standard range to evaluate proximal regularization vs. unregularized FedAvg. |

---

## 6. Implementation Phasing and Acceptance Criteria

### Phase 1: Data Infrastructure & Core Modeling (Next Phase)
- Implement `pathmnist_loader.py`: Acquire MedMNIST data, preserve official splits (train: 89,996; val: 10,004; test: 7,180).
- Implement `partitioner.py`: Dirichlet partitioner ($\alpha = 0.3$, $N=5$, seed=42). Inspect natural client sample sizes.
- Implement `noise_injector.py`: Exactly 40% symmetric noise on Client 5 (index 4) with audit mask.
- Implement `compact_cnn.py`: Lightweight 3-channel 9-class CNN backbone.
- Run complete test suite (`test_splits.py`, `test_partition.py`, `test_noise.py`).
- Run a 2-round CPU pilot test to measure empirical per-epoch execution time and verify actual process RSS memory.
- **Acceptance Gate:** All unit tests pass; zero data leakage verified; empirical step latency and memory measured and documented.

### Phase 2: Federated Optimization Engine & Diagnostics
- Implement `perturbed_sgd.py`: Exact FedProx optimizer.
- Implement `client.py` and `server.py`: Zero-waste sequential federated orchestration.
- Implement `update_metrics.py`: $L_2$ norm, pairwise cosine similarity, and similarity to sample-weighted mean update.
- Implement `tracking.py`: Local durable JSONL/CSV logger + supplementary W&B logger.
- Run 2-round integration sanity run of FedAvg and FedProx.
- **Acceptance Gate:** Successful 2-round run of FedAvg and FedProx with verified metric logging and measured CPU throughput.

### Phase 3: Full Experimental Execution & W&B Tracking
- Execute Baseline Run: FedAvg ($\alpha = 0.3$, 40% noise on Client 5, 50 rounds).
- Execute Primary Comparison: FedProx ($\mu = 0.01$, $\alpha = 0.3$, 40% noise on Client 5, 50 rounds).
- Execute Sensitivity Runs: FedProx ($\mu = 0.001$, $\mu = 0.1$).
- Stream metrics to local durable JSONL/CSV logs and supplementary public Weights & Biases project `pathmnist-fedavg-fedprox`.
- Save all raw checkpoints and round-by-round validation metrics.
- **Acceptance Gate:** All runs completed without memory thrashing; local and remote logs verified complete.

### Phase 4: Comparative Analysis, Visualizations & Documentation
- Evaluate the final global model once on the official held-out PathMNIST test split (7,180 samples).
- Generate publication-quality figures:
  1. Global Validation Accuracy vs. Communication Rounds (FedAvg vs. FedProx).
  2. Per-Client Validation Accuracy Trajectories (highlighting noisy Client 5).
  3. Client Update $L_2$ Norm Trajectories over rounds.
  4. Pairwise Cosine Similarity Heatmaps and Trajectories (Clean vs. Clean and Clean vs. Noisy).
- Generate final research submission report and comprehensive GitHub `README.md`.
- **Acceptance Gate:** Submission package complete, reproducible, and compliant with all screening guidelines for the IndiaAI Fellowship screening submission.
