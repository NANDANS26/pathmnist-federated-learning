# PHASE 0: RESOURCE FEASIBILITY & HARDWARE AUDIT REPORT (CORRECTED)

**Project:** Federated Learning on PathMNIST (FedAvg vs. FedProx)  
**Research Supervision:** Dr. Sachin D N, Assistant Professor, Dept. of CSE, IIIT Bhopal  
**Fellowship Program:** IndiaAI Fellowship 2026–27 Screening Submission  
**Execution Environment:** Windows 10 Host System  
**Audit Status:** Technical Correction Completed  
**Date:** November 7, 2026  

---

## 1. Physical Hardware and Operating System Diagnostics

Direct hardware telemetry was acquired via Windows CIM / WMI system instrumentation:

| Hardware / OS Parameter | Exact Verified Telemetry `[VERIFIED]` | Implication for Federated Experimentation `[INFERRED]` |
|---|---|---|
| **Operating System** | Microsoft Windows 10 Pro (Build 19045, 64-bit) | Must avoid POSIX/Linux-only shell scripts; use native Windows/Python execution. |
| **CPU Model** | Intel(R) Pentium(R) CPU G3240 @ 3.10GHz | 2 Cores, 2 Logical Processors, Haswell microarchitecture (No AVX2, supports SSE4.2). |
| **Physical Cores / Threads** | 2 Physical Cores, 2 Logical Processors | Compute-limited. Must constrain PyTorch threads (`torch.set_num_threads(2)`). |
| **Total Physical RAM** | 4,114,004 KB (**3.92 GB / ~4.00 GB**) | **Constrained Memory Environment**. Strict zero-waste memory budgeting required. |
| **Currently Free Physical RAM** | 593,924 KB (**~580 MB free**) | **Severe Memory Bottleneck Risk**. Careless array allocation will cause system thrashing / OOM. |
| **GPU / Video Controller** | Intel(R) HD Graphics (1 GB shared VRAM) | **No NVIDIA GPU, No CUDA**. Must run strictly on PyTorch CPU wheels. |
| **Disk Space (Drive C:)** | Total: 237.6 GB \| Free: **85.2 GB free** | Adequate for OS and caches. |
| **Disk Space (Drive D:)** | Total: 1,000.2 GB \| Free: **576.2 GB free** | Abundant disk space. All dataset artifacts and logs safely reside on D:. |
| **Git Version** | `git version 2.50.1.windows.1` | Modern Git client available. |
| **Installed Python Runtimes** | Python 3.14.3 (default), 3.13, 3.12, 3.11 | Python 3.11/3.12 are ideal targets for PyTorch CPU wheels (PyTorch has official wheels for 3.11/3.12). |

---

## 2. Rigorous Memory Analysis: Dataset Storage vs. Runtime Process Footprint

To ensure technically defensible claims, we distinguish between theoretical full-dataset representations, single-client data shards, and actual runtime process memory.

### 2.1 Theoretical Storage: `uint8` vs. Eager `float32` `[INFERRED]`
PathMNIST consists of $107,180$ images of size $28 \times 28 \times 3$ across three official splits:
- `uint8`: 1 byte per channel-pixel ($28 \times 28 \times 3 = 2,352$ bytes per image).
- `float32`: 4 bytes per channel-pixel ($28 \times 28 \times 3 \times 4 = 9,408$ bytes per image).

| Dataset Split | Sample Count | Resolution | `uint8` Storage (Decimal MB) | `uint8` Storage (Binary MiB) | Full Eager `float32` (Binary MiB) |
|---|---|---|---|---|---|
| **Training Split** | 89,996 | $28 \times 28 \times 3$ | **211.67 MB** | **201.86 MiB** | **807.46 MiB** |
| **Validation Split** | 10,004 | $28 \times 28 \times 3$ | **23.53 MB** | **22.44 MiB** | **89.76 MiB** |
| **Test Split** | 7,180 | $28 \times 28 \times 3$ | **16.89 MB** | **16.11 MiB** | **64.42 MiB** |
| **Entire PathMNIST** | **107,180** | $28 \times 28 \times 3$ | **252.09 MB** | **240.41 MiB** | **961.64 MiB** |

### 2.2 Re-Examination of PFLlib's Dataset-Loading Lifecycle `[VERIFIED]`
Inspection of `system/flcore/servers/serverbase.py`, `system/flcore/clients/clientbase.py`, and `system/utils/data_utils.py` demonstrates:
1. **Server Initialization (`serverbase.py:67-76`):**
   `Server.set_clients` reads each client's `.npz` file via `read_client_data()` only to obtain `len(train_data)` and `len(test_data)`. Because these are local loop variables, the parsed data is dereferenced at each iteration, making it eligible for garbage collection (subject to Python allocator behavior and heap fragmentation).
2. **Client Execution (`clientavg.py:12-14` & `clientbase.py:55-59`):**
   During training round $t$, only the currently selected client calls `self.load_train_data()`. Therefore, **PFLlib does not simultaneously materialize all 5 clients' datasets in memory as float32 tensors**.
3. **Single Client Shard Footprint (`float32`):**
   For 5 clients, an average client shard contains $\approx 18,000$ training images:
   $$18,000 \times 28 \times 28 \times 3 \times 4 \text{ bytes} \approx 169.3 \text{ MB} \approx 161.5 \text{ MiB}$$
   Adding Python list and PyTorch tensor wrapper overhead ($\approx 18,000 \times 300\text{--}500 \text{ bytes} \approx 5.4\text{--}9 \text{ MB}$), an individual client's materialized dataset in PFLlib occupies $\approx \mathbf{175\text{--}180 \text{ MB}}$, rather than the full-dataset figure of 807 MB.

### 2.3 Clarification of Memory Constraints `[CORRECTION]`
- **Corrected Technical Assessment:** PFLlib does **not** simultaneously load all client partitions into memory. However, eagerly materializing an individual 18,000-sample client shard as 18,000 distinct `(Tensor, Tensor)` tuple objects in a Python list (~175–180 MB) every communication round induces repeated heap allocations, garbage collector churn, and memory fragmentation. Given that the host system currently has **~580 MB of free physical RAM**, running PFLlib's unoptimized data loader alongside the operating system, model parameters, and runtime caches leaves dangerously slim margins (~300–400 MB) and risks triggering Windows virtual-memory pagefile swapping.
- **Recommended Architecture `[PROPOSED]`:** Storing client partitions as contiguous `uint8` NumPy arrays in RAM (training images: 211.67 MB / 201.86 MiB; validation images: 23.53 MB / 22.44 MiB; total in RAM during training: 235.20 MB / 224.30 MiB) and converting lazily to `float32` per mini-batch avoids tuple fragmentation and minimizes allocation churn. *Note: This lazy-loading pipeline is a proposed design and will be empirically benchmarked in a CPU pilot run during Phase 2.*

---

## 3. Comprehensive Breakdown of Runtime Memory Components

The previously mentioned peak process RSS value of **1.2 GB is an unverified operational target `[PROPOSED TARGET / UNVERIFIED]`, NOT an established ceiling.**

To provide a sound baseline for subsequent measurements, we categorize all runtime memory components:

1. **Input Batch Tensor Memory `[INFERRED]`:**
   For a mini-batch of size 32 of 3-channel RGB float32 images:
   $$32 \times 3 \times 28 \times 28 \times 4 \text{ bytes} = 301,056 \text{ bytes} = \mathbf{294.0 \text{ KiB}} \approx \mathbf{0.301 \text{ MB}}$$
   *(Crucial note: 294 KiB represents ONLY the raw input batch tensor $X$. It is NOT total peak training memory).*
2. **Intermediate Activations (Forward Pass Tensors) `[INFERRED]`:**
   For `CompactPathCNN` ($3 \to 32 \to 64$ filters, MaxPool $2\times 2$, AdaptiveAvgPool2d((2, 2)), batch size 32), the 11 individual forward tensor shapes and element counts are:
   - Conv1 feature maps: $(32, 32, 28, 28) \implies 802,816 \text{ elements}$ ($3,211,264 \text{ bytes} \approx \mathbf{3.06 \text{ MiB}}$)
   - ReLU1 (`inplace=False`): $(32, 32, 28, 28) \implies 802,816 \text{ elements}$ ($3,211,264 \text{ bytes} \approx \mathbf{3.06 \text{ MiB}}$)
   - Pool1 feature maps: $(32, 32, 14, 14) \implies 200,704 \text{ elements}$ ($802,816 \text{ bytes} \approx \mathbf{784 \text{ KiB}}$)
   - Conv2 feature maps: $(32, 64, 14, 14) \implies 401,408 \text{ elements}$ ($1,605,632 \text{ bytes} \approx \mathbf{1.53 \text{ MiB}}$)
   - ReLU2 (`inplace=False`): $(32, 64, 14, 14) \implies 401,408 \text{ elements}$ ($1,605,632 \text{ bytes} \approx \mathbf{1.53 \text{ MiB}}$)
   - Pool2 feature maps: $(32, 64, 7, 7) \implies 100,352 \text{ elements}$ ($401,408 \text{ bytes} \approx \mathbf{392 \text{ KiB}}$)
   - AdaptiveAvgPool2d((2, 2)) maps: $(32, 64, 2, 2) \implies 8,192 \text{ elements}$ ($32,768 \text{ bytes} = \mathbf{32 \text{ KiB}}$)
   - Flatten output: $(32, 256) \implies 8,192 \text{ elements}$ *(view of pooled tensor; zero separate buffer allocation)*
   - FC1 output: $(32, 64) \implies 2,048 \text{ elements}$ ($8,192 \text{ bytes} = \mathbf{8 \text{ KiB}}$)
   - ReLU3 (`inplace=False`): $(32, 64) \implies 2,048 \text{ elements}$ ($8,192 \text{ bytes} = \mathbf{8 \text{ KiB}}$)
   - FC2 (Logits) output: $(32, 9) \implies 288 \text{ elements}$ ($1,152 \text{ bytes} \approx \mathbf{1.1 \text{ KiB}}$)
   - **Sum of All 11 Individual Forward Shapes:** $2,730,272 \text{ elements}$ ($10,921,088 \text{ bytes} \approx \mathbf{10.42 \text{ MiB}} \approx \mathbf{10.92 \text{ MB}}$ if materialized independently).
   *(Important technical note: Flatten is a view, and ReLUs are specified out-of-place for autograd graph clarity. Static tensor sizes do NOT establish peak training memory. Autograd-saved backward execution graphs, gradient tensors, optimizer states, temporary memory allocator buffers, and operating-system paging heavily influence actual peak RSS).*
3. **Model Parameters & Gradients `[INFERRED]`:**
   For `CompactPathCNN` with $36,425$ trainable parameters:
   - Parameter tensors (`float32`): $36,425 \times 4 \text{ bytes} \approx 145.7 \text{ KB}$
   - Gradient tensors (`float32`): $36,425 \times 4 \text{ bytes} \approx 145.7 \text{ KB}$
4. **Optimizer State `[INFERRED]`:**
   - Standard SGD (momentum=0): 0 bytes extra state.
   - FedProx: Holds a frozen clone of global parameters: $36,425 \times 4 \text{ bytes} \approx 145.7 \text{ KB}$.
5. **Model Replicas in Memory `[INFERRED]`:**
   - Server global model: ~146 KB
   - Active client model: ~146 KB
   - FedProx cached global parameters: ~146 KB
   - Total model instances: $< 1 \text{ MB}$.
6. **Dataset Storage in RAM (Proposed uint8) `[INFERRED]`:**
   - Client training image partitions: 211.67 MB / 201.86 MiB
   - Clean validation image partitions: 23.53 MB / 22.44 MiB
   - Total dataset image arrays in RAM during training: $\approx \mathbf{235.20 \text{ MB}} \approx \mathbf{224.30 \text{ MiB}}$.
   - (All 3 splits combined, including test: 252.09 MB / 240.41 MiB).
7. **Python Runtime & Framework Overhead `[ESTIMATED]`:**
   - CPython runtime + imported PyTorch CPU C++ backend libraries: $\approx \mathbf{150\text{--}250 \text{ MB}}$.
8. **Summary of Target Memory Footprint `[PROPOSED / UNVERIFIED]`:**
   $$\text{Dataset (235 MB)} + \text{Runtime (200 MB)} + \text{Model/Batch/Activations (15 MB)} \approx \mathbf{450\text{--}500 \text{ MB}}$$
   Targeting peak process RSS $< 1.2 \text{ GB}$ provides a safety factor of $> 2\times$, leaving sufficient headroom for the Windows 10 OS on a 4 GB RAM machine. Actual peak RSS must be measured using `psutil` during the Phase 2 pilot execution.
   Targeting peak process RSS $< 1.2 \text{ GB}$ provides a safety factor of $> 2\times$, leaving sufficient headroom for the Windows 10 OS on a 4 GB RAM machine. Actual peak RSS must be measured using `psutil` during the Phase 2 pilot execution.

---

## 4. Execution Time Projections on Intel Pentium G3240 (2 Cores, 3.10 GHz, No CUDA)

### 4.1 Workload Arithmetic `[INFERRED]`
- **Dataset:** 89,996 training samples partitioned among 5 clients $\approx 18,000$ samples/client.
- **Batch Size:** 32.
- **Iterations per Epoch:** $18,000 / 32 \approx 562$ mini-batch updates per client-epoch.
- **Local Epochs per Round:** 2 epochs = $1,124$ mini-batch updates per client per round.
- **Updates per Round across 5 Clients:** $1,124 \times 5 = 5,620$ mini-batch updates.
- **Updates across a 50-Round Run:** $5,620 \times 50 = \mathbf{281,000 \text{ mini-batch updates}}$.

### 4.2 Provisional Runtime Projections `[ESTIMATED]`
The Intel Pentium G3240 lacks AVX2 (supports SSE4.2). A mini-batch forward+backward pass of a CNN on 2 CPU cores typically requires 10 to 25 milliseconds per batch of 32:

| Assumed Step Latency | Time per Client Epoch (562 batches) | Time per Client Round (2 epochs) | Time per Round (5 clients) | Estimated 50-Round Run Duration |
|---|---|---|---|---|
| **Optimistic (10 ms/batch)** | 5.6 seconds | 11.2 seconds | 56 seconds | **~47 minutes** |
| **Moderate (15 ms/batch)** | 8.4 seconds | 16.9 seconds | 84 seconds | **~70 minutes (1 hr 10 min)** |
| **Conservative (25 ms/batch)** | 14.1 seconds | 28.1 seconds | 141 seconds | **~117 minutes (< 2 hours)** |

- **Methodological Stance:** All execution time figures are **provisional estimates `[ESTIMATED]`**.
- **Requirement:** Actual per-epoch and per-round execution times must be measured during a controlled CPU pilot run (e.g., 2 rounds) before committing to a final 50- or 100-round experimental schedule.

---

## 5. Minimal Dependency Strategy for Windows CPU

| Package | Recommended Version `[PROPOSED]` | Purpose & Justification `[INFERRED]` |
|---|---|---|
| `python` | 3.11.x or 3.12.x | PyTorch has official, stable CPU wheels. |
| `torch` (CPU) | `2.2.x` or `2.3.x+cpu` | `--index-url https://download.pytorch.org/whl/cpu`. No CUDA downloads. |
| `torchvision` (CPU) | `0.17.x` or `0.18.x+cpu` | Paired CPU wheel. |
| `numpy` | `~1.26.4` | Stable NumPy 1.x without C-API compatibility issues. |
| `medmnist` | `~3.0.1` | Official MedMNIST Python package for PathMNIST acquisition. |
| `wandb` | `~0.17.x` | Lightweight experiment tracking (used alongside local JSONL). |
| `matplotlib` | `~3.8.x` | Offline figure generation for research submission reports. |
| `scikit-learn` | `~1.4.x` | Multi-class Macro-F1, ROC-AUC, and Confusion Matrix metrics. |
