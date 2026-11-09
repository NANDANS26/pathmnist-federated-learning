# PHASE 1: SOFTWARE ARCHITECTURE SPECIFICATION (CORRECTED)
## Modular Component Hierarchy, Object-Oriented Design, Data Flow Pipelines, and Interfaces

**Project:** Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Client-Specific Label Noise  
**Fellowship Program:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Author:** Senior Federated Learning Systems Architect & PyTorch Engineer  
**Document Status:** Formal Architecture Specification (`Phase 1`)  
**Date:** November 7, 2026  

---

## 1. Architectural Philosophy and Design Principles

The research implementation adheres to five software engineering principles:
1. **Separation of Concerns:** Dataset acquisition, data partitioning, noise corruption, neural network modeling, optimization, and metric diagnostics are completely decoupled into dedicated modules.
2. **Untouched Reference Attribution:** `third_party/PFLlib` remains completely untouched. Verified mathematical logic (`PerturbedGradientDescent`, aggregation formulas) is adapted and attributed within the original `research/` package.
3. **Zero-Waste Memory Management:** Client image partitions reside in contiguous NumPy `uint8` arrays in RAM (training split: 211.67 MB / 201.86 MiB; validation: 23.53 MB / 22.44 MiB; total 3 splits: 252.09 MB / 240.41 MiB); tensor conversion to `float32` occurs lazily per mini-batch (batch 32 RGB raw input is 294.0 KiB), eliminating object churn and fragmentation on 4 GB RAM.
4. **Deterministic Single-Thread Execution:** Execution is sequential across the 5 clients, limiting PyTorch threads to 2 to eliminate multi-process memory duplication and OS thread thrashing on the dual-core Intel Pentium G3240.
5. **No Over-Engineering:** No microservices, Docker daemons, REST endpoints, or message queues. The system is a clean, modular Python/PyTorch federated simulation.

---

## 2. Full Directory Blueprint

```
pathmnist-federated-learning/
│
├── third_party/
│   └── PFLlib/                       # Untouched upstream reference library (Apache-2.0)
│
├── reports/                          # Formal research reports and audits
│   ├── PHASE_0_REPOSITORY_AUDIT.md
│   ├── PHASE_0_FEDERATED_ALGORITHMS_AUDIT.md
│   ├── PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md
│   ├── PHASE_0_RESOURCE_FEASIBILITY_REPORT.md
│   ├── PHASE_0_EVALUATION_AND_DIAGNOSTICS_AUDIT.md
│   ├── PHASE_0_RISK_REGISTER.md
│   ├── PHASE_0_IMPLEMENTATION_BLUEPRINT.md
│   ├── PHASE_0_COMMANDS_AND_VERIFICATION.md
│   ├── PHASE_0_FINAL_REVIEW_REPORT.md
│   ├── PHASE_1_RESEARCH_PROTOCOL.md
│   ├── PHASE_1_EXPERIMENT_DESIGN.md
│   ├── PHASE_1_METRIC_SPECIFICATION.md
│   ├── PHASE_1_REPRODUCIBILITY_PLAN.md
│   ├── PHASE_1_ARCHITECTURE_SPECIFICATION.md
│   ├── PHASE_1_ACCEPTANCE_CRITERIA.md
│   ├── PHASE_1_DECISIONS_AND_OPEN_QUESTIONS.md
│   └── PHASE_1_FINAL_TECHNICAL_CORRECTION_REPORT.md
│
├── configs/
│   └── experiment_protocol.yaml      # Machine-readable protocol specification
│
├── research/                         # Original, modular federated learning framework
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pathmnist_loader.py        # MedMNIST downloader & split separator
│   │   ├── partitioner.py             # Dirichlet non-IID partitioner (alpha=0.3)
│   │   ├── validation_splitter.py     # Clean simulated validation shard generator
│   │   └── noise_injector.py          # 40% symmetric noise on Client 5 with audit mask
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── compact_cnn.py             # CompactPathCNN architecture (~36,425 params)
│   │
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── perturbed_sgd.py           # Exact FedProx proximal optimizer
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── client.py                  # Federated client entity (local SGD / Perturbed)
│   │   ├── server.py                  # Parameter server (broadcasting & aggregation)
│   │   ├── fedavg.py                  # FedAvg runner orchestrator
│   │   └── fedprox.py                 # FedProx runner orchestrator
│   │
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── update_metrics.py          # L2 norm, pairwise cosine, & mean-sim calculators
│   │   └── tracking.py                # Local JSONL logger & W&B synchronization
│   │
│   └── tests/
│       ├── __init__.py
│       ├── test_splits.py             # Split isolation & zero-leakage assertions
│       ├── test_partition.py          # Dirichlet partition & conservation assertions
│       ├── test_noise.py              # Exact 40% noise & mask assertions
│       ├── test_model.py              # CNN forward/backward & parameter count tests
│       ├── test_optimizer.py          # FedProx gradient perturbation equivalence test
│       └── test_pilot.py              # 1-round smoke test & memory verification
│
├── experiments/                      # Persisted outputs
│   ├── configs/                      # Partition and noise manifests
│   │   ├── partition_manifest.json
│   │   └── noise_manifest.json
│   ├── checkpoints/                  # Serialized PyTorch models (.pt)
│   │   └── initial_model.pt
│   ├── logs/                         # Local durable logs (.jsonl and .csv)
│   └── figures/                      # Publication plots (.png and .pdf)
│
├── requirements.txt                  # Minimal CPU-only dependencies
└── README.md                         # Project documentation and current status
```

---

## 3. Module Responsibilities, Classes, and Interfaces

### 3.1 Data Pipeline Modules (`research/data/`)

#### `pathmnist_loader.py`
- **Class:** `PathMNISTDataset(torch.utils.data.Dataset)`
- **Function:** `load_official_pathmnist(root_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
- **Signature:**
  ```python
  def load_official_pathmnist(root_dir: str = "./data"):
      """Loads PathMNIST via official MedMNIST API without altering splits.
      Returns:
          train_images (89996, 28, 28, 3) uint8
          train_labels (89996, 1) uint8
          val_images   (10004, 28, 28, 3) uint8
          val_labels   (10004, 1) uint8
          test_images  (7180, 28, 28, 3) uint8
          test_labels  (7180, 1) uint8
      """
  ```

#### `partitioner.py`
- **Function:** `generate_dirichlet_partitions(labels: np.ndarray, num_clients: int = 5, alpha: float = 0.3, seed: int = 42) -> Dict[int, np.ndarray]`
- **Responsibility:** Implements class-wise Dirichlet allocation without truncation floors, records client indices, and asserts sample conservation ($\sum |I_c| = 89,996$) and mutual exclusivity ($\bigcap I_c = \emptyset$).

#### `validation_splitter.py`
- **Function:** `generate_client_validation_shards(val_labels: np.ndarray, train_partitions: Dict[int, np.ndarray], train_labels: np.ndarray, seed: int = 42) -> Dict[int, np.ndarray]`
- **Responsibility:** Applies the deterministic largest-remainder (Hamilton-Hare) apportionment procedure to divide the 10,004 validation samples into 5 disjoint clean shards.

#### `noise_injector.py`
- **Function:** `inject_symmetric_label_noise(client_labels: np.ndarray, target_client: int = 4, noise_rate: float = 0.40, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
- **Responsibility:** Selects $\text{round}(0.40 \times M_5)$ training samples on Client 5, flips labels symmetrically via $\tilde{y} = (y + r) \pmod 9$, and generates a boolean `noise_mask`.

---

### 3.2 Model Backbone (`research/models/compact_cnn.py`)

- **Class:** `CompactPathCNN(nn.Module)`
- **Architecture:** 2 Conv layers + AdaptiveAvgPool2d((2, 2)) + 2 Linear layers. Total trainable parameters: **36,425**. Total MACs per image: **4,307,008 MACs** (~8.61 MFLOPs under $1\text{ MAC} = 2\text{ FLOPs}$).
- **Activation Function Policy:** Standard out-of-place activations (`nn.ReLU(inplace=False)`) for backward graph clarity and numerical stability.
- **Flatten Policy:** Zero-copy tensor view via `torch.flatten(x, 1)` that does not allocate independent buffer memory.
- **Batch Activation Dimensions ($B=32$):** Individual forward tensor dimensions: Conv1 ($802,816$), ReLU1 ($802,816$), MaxPool1 ($200,704$), Conv2 ($401,408$), ReLU2 ($401,408$), MaxPool2 ($100,352$), AdaptiveAvgPool2d ($8,192$), Flatten ($8,192$ view), FC1 ($2,048$), ReLU3 ($2,048$), FC2 logits ($288$). The arithmetic sum of all 11 individual forward shapes is $2,730,272$ elements (~10.42 MiB / 10.92 MB).
- **Peak RSS Caveat:** Static forward tensor dimensions do not determine peak training process memory. Peak process RSS will be measured dynamically using `psutil` in the Phase 2 pilot test.

#### Image Preprocessing Specification:
Mini-batch tensor construction from stored uint8 arrays:
```python
x = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
y = torch.from_numpy(batch_labels).squeeze(1).to(torch.int64)
```
- Input shape: $(B, 3, 28, 28)$, dtype `float32`, range $[0.0, 1.0]$.
- Target shape: $(B,)$, dtype `int64`, class indices $0\text{--}8$.
- Identical for training, validation, and test splits. No data augmentation or channel-wise mean/std normalization.

---

### 3.3 Optimizer (`research/optimizers/perturbed_sgd.py`)

- **Class:** `PerturbedSGD(torch.optim.Optimizer)`
- **Responsibility:** Adapts PFLlib's verified `PerturbedGradientDescent` to execute the exact FedProx update:
  $$d_p = \nabla F_k(w) + \mu(w - w_t), \quad w \leftarrow w - \eta d_p$$

---

### 3.4 Diagnostics and Logging (`research/diagnostics/`)

#### `update_metrics.py`
- **Functions:**
  - `compute_update_vector(local_model: nn.Module, global_model: nn.Module) -> torch.Tensor`: Extracts $\Delta_{k,t} = w_{k,t} - w_t$ as a flattened 1D tensor.
  - `compute_update_l2_norm(delta: torch.Tensor) -> float`: Computes $\|\Delta\|_2$.
  - `compute_pairwise_cosine_similarity(deltas: List[torch.Tensor]) -> Dict[str, float]`: Computes all 10 pairwise cosine similarities.
  - `compute_clean_noisy_alignment(deltas: List[torch.Tensor]) -> float`: Computes mean similarity between clean clients 1–4 and Client 5.
  - `compute_similarity_to_weighted_mean(deltas: List[torch.Tensor], weights: List[float]) -> Dict[str, float]`: Computes $\text{cos}(\Delta_k, \bar{\Delta})$.

---

## 4. End-to-End System Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PATHMNIST LOCAL STORAGE                            │
│                       (Images: uint8, Labels: uint8)                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATA INITIALIZATION (Executed Once)                     │
│  1. partitioner.py: 89,996 train samples -> 5 client train index arrays     │
│  2. validation_splitter.py: 10,004 val samples -> 5 client val index arrays │
│  3. noise_injector.py: 40% label flips on Client 5 training indices         │
│  4. Save partition_manifest.json & noise_manifest.json                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL INITIALIZATION (Executed Once)                     │
│  1. CompactPathCNN initialized with seed=42                                 │
│  2. Serialized to initial_model.pt (Canonical hash verified)                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             COMMUNICATION ROUND LOOP (t = 1 to R; Sequential Execution)     │
│                                                                             │
│  [Step 1] Server broadcasts w_t to Clients 1 ... 5                          │
│                                                                             │
│  [Step 2] Sequential Client Training:                                       │
│    For client in [Client 1, Client 2, Client 3, Client 4, Client 5]:        │
│      - Lazy batch generator yields (X, y) (Batch size 32, ~294 KiB float32) │
│      - Forward pass -> CrossEntropy Loss -> Backprop                        │
│      - FedAvg: SGD step: w = w - lr * grad                                  │
│      - FedProx: PerturbedSGD step: w = w - lr * [grad + mu * (w - w_t)]     │
│      - Return local model w_{k,t}                                           │
│                                                                             │
│  [Step 3] Diagnostics Extraction (Server-side, before aggregation):         │
│    - Compute Δ_{k,t} = w_{k,t} - w_t                                        │
│    - Calculate ||Δ_{k,t}||_2 for all 5 clients                              │
│    - Calculate 10 pairwise cosine similarities cos(Δ_i, Δ_j)                │
│    - Calculate clean-to-noisy alignment mean                                │
│    - Calculate similarity to sample-weighted mean cos(Δ_k, Δ_mean)          │
│                                                                             │
│  [Step 4] Global Model Aggregation:                                         │
│    - w_{t+1} = Σ p_k * w_{k,t} (p_k = n_{k,train} / 89,996)                 │
│                                                                             │
│  [Step 5] Round Evaluation on Clean Validation Data:                        │
│    - Global Validation Accuracy & Loss on all 10,004 samples                │
│    - Per-Client Validation Accuracy & Loss on clean shards V_1 ... V_5      │
│                                                                             │
│  [Step 6] Telemetry Persistence & Memory Check:                             │
│    - Atomic append to metrics.jsonl                                         │
│    - Synchronous push to Weights & Biases                                   │
│    - Monitor available physical RAM (Emergency halt if < 150 MB)            │
│    - gc.collect() to reset Python memory pool                               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FINAL POST-TRAINING BENCHMARK                         │
│  1. Each algorithm's selected best-validation checkpoint evaluated once on  │
│     official 7,180 test samples (zero test evaluation during training)      │
│  2. Export final summary CSV and render publication figures (FIG-01 to 07)  │
└─────────────────────────────────────────────────────────────────────────────┘
```
