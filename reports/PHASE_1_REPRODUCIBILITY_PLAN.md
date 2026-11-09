# PHASE 1: RESEARCH REPRODUCIBILITY PLAN (CORRECTED)
## Deterministic Random Seeding, Canonical Model Hashing, Partition Manifests, and Durable Telemetry Protocols

**Project:** Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Client-Specific Label Noise  
**Fellowship Program:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Author:** Senior Federated Learning Research Scientist & Reproducibility Auditor  
**Document Status:** Formal Reproducibility Plan (`Phase 1`)  
**Date:** November 7, 2026  

---

## 1. Executive Reproducibility Mandate

Scientific credibility in machine learning requires that independent researchers can regenerate the exact numerical results, client partitions, noise injections, and optimization trajectories on equivalent hardware. 

This Reproducibility Plan guarantees that all experimental arms (FedAvg vs. FedProx) share identical initial states, identical data shards, identical noise masks, identical mini-batch shuffling sequences, and transparent checkpoint provenance.

---

## 2. Multi-Level Random Seeding Strategy `[PROPOSED]`

A single master seed (**`42`**) anchors the entire experimental pipeline. Randomness streams are strictly decoupled to prevent cross-subsystem state leakage:

| Stochastic Subsystem | Seed Formula / Value | Generator Instance | Purpose & Isolation Rule |
|---|---|---|---|
| **Python Standard Library** | `seed = 42` | `random.seed(42)` | General non-data logic. |
| **NumPy Global RNG** | `seed = 42` | `np.random.seed(42)` | Anchors global NumPy routines. |
| **Data Partitioning** | `seed = 42` | `np.random.default_rng(42)` | Dedicated generator for Dirichlet class-wise slicing. |
| **Validation Shard Splitting** | `seed = 42` | `np.random.default_rng(42)` | Dedicated generator for clean validation allocation. |
| **Label Noise Selection** | `seed = 42` | `np.random.default_rng(42)` | Dedicated generator for selecting 40% sample indices on Client 5. |
| **Noise Class Replacement** | `seed = 1042` | `np.random.default_rng(1042)` | Dedicated generator for uniform class replacement $\tilde{y} \in [1, 8]$. |
| **Model Weight Initialization** | `seed = 42` | `torch.manual_seed(42)` | Deterministic weight initialization of `CompactPathCNN`. |
| **Client Mini-Batch Shuffling** | `seed = 42 + 1000*t + c` | `torch.Generator().manual_seed()` | Deterministic mini-batch ordering per client $c$ per round $t$. |

### 2.1 Identical Mini-Batch Ordering Across Algorithms
To ensure that local optimization differences between FedAvg and FedProx stem solely from proximal regularization and not different mini-batch sample orderings:
- For communication round $t$ and client $c \in \{1, 2, 3, 4, 5\}$:
  $$\text{batch\_seed}_{c,t} = 42 + 1000 \times t + c$$
- The PyTorch `DataLoader` for client $c$ at round $t$ will be instantiated with:
  ```python
  g = torch.Generator()
  g.manual_seed(42 + 1000 * t + c)
  loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)
  ```
- This guarantees that **FedAvg and FedProx process identical mini-batch sequences** at every local epoch of every communication round.

---

## 3. Canonical Model-State Hashing and Exact Verification `[PROPOSED]`

To eliminate risks associated with file-system serialization metadata (e.g., zip timestamp discrepancies in PyTorch `.pt` files):

### 3.1 Canonical State Hash Algorithm
A deterministic cryptographic hash of the initial model state $w_0$ is computed over raw tensor bytes:
```python
import hashlib

def compute_canonical_model_hash(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    state_dict = model.state_dict()
    for key in sorted(state_dict.keys()):
        param = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode('utf-8'))
        hasher.update(str(param.dtype).encode('utf-8'))
        hasher.update(str(list(param.shape)).encode('utf-8'))
        hasher.update(param.numpy().tobytes())
    return hasher.hexdigest()
```

### 3.2 Pre-Flight Initialization Assertion
Before Run 1 (FedAvg) or Run 2 (FedProx) begins:
1. `initial_model.pt` is loaded into the model.
2. The runtime asserts:
   $$\text{compute\_canonical\_model\_hash}(\text{model}) \equiv \text{CANONICAL\_HASH\_MANIFEST}$$
3. Furthermore, when comparing FedAvg and FedProx initializations in automated tests:
   ```python
   for p1, p2 in zip(model_fedavg.parameters(), model_fedprox.parameters()):
       assert torch.equal(p1, p2), "Initial parameter mismatch detected!"
   ```

---

## 4. Persistent Manifests

### 4.1 Partition Manifest (`experiments/configs/partition_manifest.json`)
Records:
- Global random seed (`42`).
- Dataset identifier (`PathMNIST v2`).
- Alpha parameter (`0.3`).
- Total training samples ($89,996$).
- Client sample counts $M_1 \dots M_5$.
- Exact original dataset indices assigned to each client partition.
- Mutual exclusivity and sample conservation verification booleans.

### 4.2 Noise Manifest (`experiments/configs/noise_manifest.json`)
Records:
- Target client index (Client 5, zero-based index 4).
- Deterministic noise seed (`42`).
- Nominal noise rate ($0.40$).
- Total training samples $M_5$.
- Corrupted count $M_{\text{noisy}} = \text{round}(0.40 \times M_5)$.
- Exact corrupted index list, clean labels, and flipped labels.
- Full boolean `noise_mask`.

---

## 5. Durable Local Logging and W&B Synchronization

1. **Local Primary Log:** Every communication round appends a complete JSON record to `experiments/logs/{run_name}_metrics.jsonl` using synchronous atomic file flushes.
2. **Offline Fallback:** If network drops or credentials are absent, W&B operates in `offline` mode.
3. **Zero Credential Exposure:** No API keys or credentials will be stored in source code or Git.

---

## 6. Preprocessing Reproducibility & Benchmark Evaluation Rules `[PROPOSED]`

### 6.1 Deterministic Preprocessing Pipeline
To guarantee complete data pipeline equivalence across all arms:
```python
x = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
y = torch.from_numpy(batch_labels).squeeze(1).to(torch.int64)
```
- Input shape: $(B, 3, 28, 28)$, dtype `float32`, pixel range $[0.0, 1.0]$.
- Target shape: $(B,)$, dtype `int64`, class range $\{0, 1, \dots, 8\}$.
- No data augmentation, random cropping, or channel normalization is permitted.

### 6.2 Post-Training Benchmark Evaluation Invariance
- The official PathMNIST test set (7,180 images) is excluded from training, validation, and hyperparameter tuning.
- **Evaluation Rule:** Each algorithm's selected best-validation checkpoint is evaluated once on the official PathMNIST test set after training has finished.
- One evaluation for FedAvg and one evaluation for FedProx, ensuring zero test contamination.
