# PHASE 0: EVALUATION AND DIAGNOSTICS TECHNICAL AUDIT (REVISED)

**Project:** Federated Learning on PathMNIST (FedAvg vs. FedProx)  
**Research Supervision:** Dr. Sachin D N, Assistant Professor, Dept. of CSE, IIIT Bhopal  
**Fellowship Program:** IndiaAI Fellowship 2026–27 Screening Submission  
**Target Codebase:** `third_party/PFLlib` (`system/flcore/servers/serverbase.py`, `system/flcore/clients/clientbase.py`, `system/utils/result_utils.py`)  
**Audit Status:** Technical Correction & Research Validation Completed  
**Date:** November 6, 2026  

---

## 1. Executive Summary

This audit evaluates the evaluation mechanics, metric calculations, and telemetry of PFLlib. We identify deficiencies in upstream metric tracking (notably the absence of per-client trajectory persistence, lack of client update norm diagnostics, and absence of experiment tracking frameworks), and formulate the standardized mathematical and algorithmic specifications for round-level validation metrics, client update diagnostics, and reproducible local/W&B logging.

---

## 2. Upstream Evaluation Mechanics Audit

### 2.1 Existing Metric Calculation in `serverbase.py` `[VERIFIED]`
In `system/flcore/servers/serverbase.py:228-254`:
```python
def evaluate(self, acc=None, loss=None):
    stats = self.test_metrics()
    stats_train = self.train_metrics()

    test_acc = sum(stats[2])*1.0 / sum(stats[1])
    test_auc = sum(stats[3])*1.0 / sum(stats[1])
    train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
    accs = [a / n for a, n in zip(stats[2], stats[1])]
    aucs = [a / n for a, n in zip(stats[3], stats[1])]
    
    if acc == None:
        self.rs_test_acc.append(test_acc)
    if loss == None:
        self.rs_train_loss.append(train_loss)
```

### 2.2 Critical Flaws and Omissions in Upstream Evaluation `[VERIFIED]`
1. **Per-Client Trajectories are Discarded:**
   - In line 235, `accs` (the list of per-client accuracies) is computed solely to print `np.std(accs)`.
   - `accs` is **never appended to any list or saved to disk**.
   - The HDF5 file saved at the end (`save_results` in `serverbase.py:170-185`) only writes global scalar arrays `rs_test_acc`, `rs_test_auc`, and `rs_train_loss`.
   - **Research Invalidation Risk:** The IndiaAI screening submission mandates per-client accuracy tracking. Relying on PFLlib as-is would fail to provide per-client curves.
2. **Evaluation Timing vs. Aggregation:**
   - In `serveravg.py` and `serverprox.py`:
     ```python
     for i in range(self.global_rounds + 1):
         self.selected_clients = self.select_clients()
         self.send_models()
         if i % self.eval_gap == 0:
             self.evaluate()
         for client in self.selected_clients:
             client.train()
         self.receive_models()
         self.aggregate_parameters()
     ```
   - Notice: `self.evaluate()` runs **before** local training in round $i$.
   - At round $0$, it evaluates the random initial model.
   - At round $t$, it evaluates the model resulting from round $t-1$'s aggregation.
   - After the loop finishes (round `global_rounds`), the final aggregated model is **never evaluated** unless an explicit post-loop evaluation is added.
3. **Absence of Client Update Diagnostics:**
   - Upstream provides zero functionality to track parameter updates $\Delta_k = w_k - w_{\text{global}}$, update $L_2$ norms, or cosine similarity.
4. **Absence of Experiment Tracking Integration:**
   - PFLlib has no Weights & Biases (W&B) integration; it only prints to stdout and writes local `.h5` files.

---

## 3. Standardized Evaluation Methodology

To guarantee scientific rigor, we enforce strict separation between round-level validation and final benchmark testing.

### 3.1 Split Utilization Strategy `[PROPOSED]`
1. **Official Training Split (89,996 images):** Used strictly for client local training ($N=5$ non-IID shards).
2. **Official Validation Split (10,004 images):** Clean, partitioned into 5 clean client-specific validation shards independently from the training data. Evaluated at each communication round $t$.
3. **Official Test Split (7,180 images):** Clean, held out completely during the entire training run. Evaluated **once** on the final converged global model.

### 3.2 Terminology Standardization `[CORRECTION]`
- All round-level metrics must be explicitly designated as **Validation** metrics (e.g., `val_accuracy`, `val_loss`).
- The term **Test** is reserved exclusively for the final held-out evaluation on PathMNIST's official test split.

### 3.3 Sample-Weighted Aggregation of Client Validation Metrics `[INFERRED]`
Let $n_{k,\text{val}}$ be the number of validation samples in client $k$'s clean validation shard.  
Let $\text{Val\_Acc}_k$ and $\text{Val\_Loss}_k$ be client $k$'s evaluated validation accuracy and loss.  
Global validation metrics are aggregated via exact sample-weighting:
$$\text{Val\_Acc}_{\text{global}} = \frac{\sum_{k=1}^5 n_{k,\text{val}} \cdot \text{Val\_Acc}_k}{\sum_{k=1}^5 n_{k,\text{val}}}$$
$$\text{Val\_Loss}_{\text{global}} = \frac{\sum_{k=1}^5 n_{k,\text{val}} \cdot \text{Val\_Loss}_k}{\sum_{k=1}^5 n_{k,\text{val}}}$$

---

## 4. Client Update Diagnostics Design

### 4.1 Mathematical Formulations `[INFERRED]`

#### 1. Client Parameter Update Vector:
Let $w_t$ be the global model parameter vector broadcast at round $t$.  
Let $w_{k,t}$ be the local model parameter vector of client $k$ after completing local training.  
The parameter update vector is:
$$\Delta_{k,t} = w_{k,t} - w_t \in \mathbb{R}^D$$
computed by flattening and concatenating differences across all trainable parameter tensors:
$$\Delta_{k,t} = \text{flatten}\left( [ p_{k,t}^{(l)} - p_t^{(l)} \quad \forall l \in \text{layers} ] \right)$$

#### 2. Client Update $L_2$ Norm:
$$\|\Delta_{k,t}\|_2 = \sqrt{\sum_{j=1}^D (\Delta_{k,t}[j])^2} = \sqrt{\sum_{l} \|p_{k,t}^{(l)} - p_t^{(l)}\|_F^2}$$

#### 3. Pairwise Cosine Similarity:
The directional alignment between client $i$'s update and client $j$'s update is:
$$\text{cos}(\Delta_{i,t}, \Delta_{j,t}) = \frac{\langle \Delta_{i,t}, \Delta_{j,t} \rangle}{\|\Delta_{i,t}\|_2 \|\Delta_{j,t}\|_2 + \epsilon}$$
where $\epsilon = 10^{-12}$. For $N=5$ clients, there are $\binom{5}{2} = 10$ unique pairs:
$$\{(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)\}$$
where Client 5 corresponds to zero-based index 4 (the noisy client).

#### 4. Cosine Similarity to Sample-Weighted Mean Update:
Let $p_k = n_{k,\text{train}} / \sum_{j=1}^5 n_{j,\text{train}}$ be the aggregation weight.  
The sample-weighted consensus update is:
$$\bar{\Delta}_t = \sum_{k=1}^5 p_k \Delta_{k,t}$$
The directional agreement of each client $k$ with the consensus update is:
$$\text{cos}(\Delta_{k,t}, \bar{\Delta}_t) = \frac{\langle \Delta_{k,t}, \bar{\Delta}_t \rangle}{\|\Delta_{k,t}\|_2 \|\bar{\Delta}_t\|_2 + \epsilon}$$

---

## 5. Experiment Tracking and Logging Architecture

### 5.1 Local Durable Log as Primary Source of Truth `[PROPOSED]`
- Every round appends a complete JSON record to a local `metrics.jsonl` file.
- Guarantees 100% offline durability, independent of network status or remote services.
- A summary CSV table is generated upon run completion.

### 5.2 Weights & Biases (W&B) Logging Schema `[PROPOSED]`
W&B supplements local logging synchronously (avoiding fragile background threads). Metric naming strictly adheres to validation terminology:

```json
{
  "round": t,
  "val/global_accuracy": 0.824,
  "val/global_loss": 0.452,
  
  "val/client_1_accuracy": 0.865,
  "val/client_1_loss": 0.380,
  "diagnostics/client_1_update_norm": 1.42,
  "diagnostics/client_1_sim_to_weighted_mean": 0.81,

  "val/client_5_noisy_accuracy": 0.695,
  "val/client_5_noisy_loss": 0.620,
  "diagnostics/client_5_noisy_update_norm": 3.85,
  "diagnostics/client_5_noisy_sim_to_weighted_mean": 0.22,

  "diagnostics/sim_c1_c2": 0.68,
  "diagnostics/sim_c1_c5_noisy": 0.12,
  "diagnostics/mean_clean_clean_sim": 0.65,
  "diagnostics/mean_clean_noisy_sim": 0.11
}
```

- **Credential Hygiene:** API keys will never be hardcoded or committed to Git.
