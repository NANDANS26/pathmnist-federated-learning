# PHASE 1: METRIC AND DIAGNOSTICS SPECIFICATION (CORRECTED)
## Mathematical Formulations, Aggregation Rules, Telemetry Schemas, and Visualization Specifications

**Project:** Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Client-Specific Label Noise  
**Research Supervision:** Dr. Sachin D N, Assistant Professor, Department of CSE, IIIT Bhopal  
**Fellowship Program:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Author:** Senior Federated Learning Research Scientist & ML Systems Architect  
**Document Status:** Formal Metric Specification (`Phase 1`)  
**Date:** November 7, 2026  

---

## 1. Executive Telemetry Overview

Every communication round $t \in \{1, 2, \dots, R\}$ evaluates the newly aggregated global model $w_t$ across both global and per-client validation partitions, and captures parameter update vector diagnostics $\Delta_{k,t}$.

All metrics are recorded synchronously into a local durable `metrics.jsonl` log and streamed to Weights & Biases (W&B).

---

## 2. Validation Metrics Mathematical Definitions

### 2.1 Global Validation Accuracy and Loss `[INFERRED]`
Evaluated across all $N_{\text{val}} = 10,004$ samples of the official PathMNIST validation split.

Let $(x_i, y_i)$ denote validation sample $i$, where $y_i \in \{0, \dots, 8\}$.  
Let $\hat{z}_i = f(x_i; w_t) \in \mathbb{R}^9$ denote the output logits from the global model.  
Predicted class is $\hat{y}_i = \arg\max_{c} \hat{z}_{i,c}$.

- **Global Validation Accuracy:**
  $$\text{Val\_Acc}_{\text{global}}(t) = \frac{1}{10,004} \sum_{i=1}^{10,004} \mathbb{I}(\hat{y}_i = y_i)$$
- **Global Validation Loss (Cross-Entropy):**
  $$\text{Val\_Loss}_{\text{global}}(t) = -\frac{1}{10,004} \sum_{i=1}^{10,004} \sum_{c=0}^8 y_{i,c} \log \left( \frac{\exp(\hat{z}_{i,c})}{\sum_{j=0}^8 \exp(\hat{z}_{i,j})} \right)$$

### 2.2 Per-Client Validation Accuracy and Loss `[INFERRED]`
Evaluated on each client's disjoint clean simulated validation shard $V_k$ of size $n_{k,\text{val}}$, where $\sum_{k=1}^5 n_{k,\text{val}} = 10,004$.

For client $k \in \{1, 2, 3, 4, 5\}$:
- **Client Validation Accuracy:**
  $$\text{Val\_Acc}_k(t) = \frac{1}{n_{k,\text{val}}} \sum_{i \in V_k} \mathbb{I}(\hat{y}_i = y_i)$$
- **Client Validation Loss:**
  $$\text{Val\_Loss}_k(t) = -\frac{1}{n_{k,\text{val}}} \sum_{i \in V_k} \sum_{c=0}^8 y_{i,c} \log \left( \frac{\exp(\hat{z}_{i,c})}{\sum_{j=0}^8 \exp(\hat{z}_{i,j})} \right)$$

*Important Interpretation `[PROPOSED]`:*
Each client validation metric evaluates the **identical global model $w_t$** on client $k$'s clean validation shard. It does **not** evaluate personalized local models.

### 2.3 Sample-Weighted Aggregation Equivalence `[INFERRED]`
The global validation metrics must satisfy sample-weighted conservation:
$$\text{Val\_Acc}_{\text{global}}(t) \equiv \sum_{k=1}^5 \left( \frac{n_{k,\text{val}}}{10,004} \right) \text{Val\_Acc}_k(t)$$
$$\text{Val\_Loss}_{\text{global}}(t) \equiv \sum_{k=1}^5 \left( \frac{n_{k,\text{val}}}{10,004} \right) \text{Val\_Loss}_k(t)$$

---

## 3. Client Update Diagnostics Specification

At communication round $t$, let $w_t \in \mathbb{R}^D$ denote the global model parameters broadcast to all clients ($D = 36,425$).  
Let $w_{k,t}^{\text{local}} \in \mathbb{R}^D$ denote the local model parameters of client $k$ after completing 2 local epochs.

The client parameter update vector is:
$$\Delta_{k,t} = w_{k,t}^{\text{local}} - w_t \in \mathbb{R}^D$$
*(Note: $\Delta_{k,t}$ represents parameter displacement after local training, not a raw stochastic gradient).*

### 3.1 Update $L_2$ Norm `[INFERRED]`
For each client $k \in \{1, 2, 3, 4, 5\}$:
$$\|\Delta_{k,t}\|_2 = \sqrt{\sum_{j=1}^D (\Delta_{k,t}[j])^2}$$

*Scientific Interpretation:*
- Measures the parameter displacement of client $k$ during local optimization.
- In FedAvg, Client 5 (40% label noise) is expected to exhibit an abnormally large $\|\Delta_{5,t}\|_2$ due to overfitting to noisy labels.
- In FedProx, the proximal term $\frac{\mu}{2}\|w - w_t\|_2^2$ should act as a shrinkage regularizer, contracting $\|\Delta_{5,t}\|_2$ toward clean client updates.

### 3.2 Pairwise Cosine Similarity `[INFERRED]`
For any two clients $i, j \in \{1, 2, 3, 4, 5\}$ with $i < j$:
$$\text{cos}(\Delta_{i,t}, \Delta_{j,t}) = \frac{\langle \Delta_{i,t}, \Delta_{j,t} \rangle}{\|\Delta_{i,t}\|_2 \|\Delta_{j,t}\|_2 + \epsilon}$$
where $\epsilon = 10^{-12}$ guarantees numerical stability against zero-norm updates.

There are $\binom{5}{2} = 10$ unique client pairs evaluated every round:
`c1_c2`, `c1_c3`, `c1_c4`, `c1_c5`, `c2_c3`, `c2_c4`, `c2_c5`, `c3_c4`, `c3_c5`, `c4_c5`.

### 3.3 Client Update-Direction Alignment Metrics `[INFERRED]`
1. **Clean-to-Noisy Update Alignment:**
   Defined mathematically as the unweighted mean of the four pairwise cosine similarities between clean clients (1–4) and Client 5:
   $$\overline{\text{cos}}(\Delta_{\text{clean}}, \Delta_{5,t}) = \frac{1}{4} \sum_{k=1}^4 \text{cos}(\Delta_{k,t}, \Delta_{5,t})$$
2. **Clean-to-Clean Consensus:**
   Defined as the mean of the 6 clean-clean pairwise cosine similarities:
   $$\overline{\text{cos}}(\Delta_{\text{clean}}, \Delta_{\text{clean},t}) = \frac{1}{6} \left( \text{cos}_{1,2} + \text{cos}_{1,3} + \text{cos}_{1,4} + \text{cos}_{2,3} + \text{cos}_{2,4} + \text{cos}_{3,4} \right)$$

### 3.4 Cosine Similarity to Sample-Weighted Mean Update `[INFERRED]`
Let $p_k = \frac{n_{k,\text{train}}}{89,996}$ be the aggregation weight.  
The sample-weighted consensus update vector is:
$$\bar{\Delta}_t = \sum_{k=1}^5 p_k \Delta_{k,t} \in \mathbb{R}^D$$

The directional agreement of each client $k$ with the consensus update is:
$$\text{sim\_to\_mean}_k(t) = \frac{\langle \Delta_{k,t}, \bar{\Delta}_t \rangle}{\|\Delta_{k,t}\|_2 \|\bar{\Delta}_t\|_2 + \epsilon}$$
Logged for all 5 clients: `client_1/sim_to_weighted_mean` through `client_5/sim_to_weighted_mean`.

---

## 4. Telemetry Logging Cadence and W&B Schema

Every communication round produces one atomic JSON record.

> [!WARNING]
> **ILLUSTRATIVE SCHEMA ONLY — NOT EXPERIMENTAL RESULTS.**  
> The numerical values in the schema below are fictitious syntax examples showing keys and floating-point representations. They must never be cited as empirical findings or copied into research results.

```json
{
  "round": 1,
  
  "val/global_accuracy": 0.6241,
  "val/global_loss": 1.1402,
  
  "val/client_1_accuracy": 0.6850,
  "val/client_1_loss": 0.9820,
  "diagnostics/client_1_update_l2_norm": 0.4215,
  "diagnostics/client_1_sim_to_weighted_mean": 0.8124,

  "val/client_2_accuracy": 0.6410,
  "val/client_2_loss": 1.0512,
  "diagnostics/client_2_update_l2_norm": 0.3980,
  "diagnostics/client_2_sim_to_weighted_mean": 0.7942,

  "val/client_3_accuracy": 0.6120,
  "val/client_3_loss": 1.1890,
  "diagnostics/client_3_update_l2_norm": 0.4105,
  "diagnostics/client_3_sim_to_weighted_mean": 0.7651,

  "val/client_4_accuracy": 0.6500,
  "val/client_4_loss": 1.0230,
  "diagnostics/client_4_update_l2_norm": 0.4350,
  "diagnostics/client_4_sim_to_weighted_mean": 0.8310,

  "val/client_5_accuracy": 0.4950,
  "val/client_5_loss": 1.5420,
  "diagnostics/client_5_update_l2_norm": 1.1250,
  "diagnostics/client_5_sim_to_weighted_mean": 0.2850,

  "pairwise_cosine/c1_c2": 0.6521,
  "pairwise_cosine/c1_c3": 0.5894,
  "pairwise_cosine/c1_c4": 0.6812,
  "pairwise_cosine/c1_c5": 0.1245,
  "pairwise_cosine/c2_c3": 0.6105,
  "pairwise_cosine/c2_c4": 0.6420,
  "pairwise_cosine/c2_c5": 0.0984,
  "pairwise_cosine/c3_c4": 0.6015,
  "pairwise_cosine/c3_c5": 0.1410,
  "pairwise_cosine/c4_c5": 0.1180,

  "diagnostics/mean_clean_clean_cosine": 0.6294,
  "diagnostics/mean_clean_noisy_cosine": 0.1205
}
```

---

## 5. Checkpoint Selection Policy `[PROPOSED]`

To ensure complete fairness and scientific rigor:
1. **Primary Evaluation Model:**
   - Both FedAvg and FedProx will preserve a checkpoint of every round: `model_round_{t}.pt`.
   - **Best Validation Checkpoint:** The round $t^*$ that achieves the highest global validation accuracy:
     $$t^* = \arg\max_{t \in [1, R]} \text{Val\_Acc}_{\text{global}}(t)$$
   - *Tie-Breaking Policy:* If multiple rounds achieve the identical peak validation accuracy, the tie is broken first by the lowest `val/global_loss`, and second by the earlier round index.
2. **Final Round Checkpoint:**
   - The final model at round $R$ (`model_final.pt`) will also be preserved.
3. **Test Split Exclusion & Final Benchmark Evaluation:**
   - The official PathMNIST test set (7,180 images) is **strictly excluded** from training, validation, and model selection.
   - **Evaluation Rule:** Each algorithm's selected best-validation checkpoint is evaluated once on the official PathMNIST test set after training has finished.
   - This ensures exactly one final test evaluation for FedAvg and one final test evaluation for FedProx, completely free from data leakage or selection bias.

---

## 6. Scope Control: Mandatory Deliverables vs. Optional Metrics

To protect project completion under hardware constraints:

### 6.1 Mandatory Experimental Deliverables
- Global validation accuracy and loss per round.
- Per-client validation accuracy and loss per round (Clients 1–5).
- Client update $L_2$ norms per round (Clients 1–5).
- 10 pairwise cosine similarities per round.
- Cosine similarity to sample-weighted mean update per round.
- Durable local JSONL logging + summary CSV.
- Supplementary Weights & Biases dashboard streaming.
- Final held-out test accuracy and cross-entropy loss.

### 6.2 Optional / Secondary Metrics (Executed only if compute budget permits)
- **Macro-F1 Score:** Unweighted average of per-class F1 scores.
- **Multiclass Macro-AUC:** One-vs-Rest (OvR) macro-average:
  $$\text{Macro\_AUC} = \frac{1}{9} \sum_{c=0}^8 \text{AUC}(y == c, \hat{P}(y=c))$$
- **Confusion Matrix ($9 \times 9$):** Plotted via matplotlib for the final test evaluation.
- Optional metrics will not block mandatory experiment completion.

---

## 7. Research Plots Specification `[PROPOSED]`

Seven publication-quality figures will be rendered automatically from logged JSONL files:

| Plot ID | Title | X-Axis | Y-Axis | Curves / Series | Purpose & Research Insight |
|---|---|---|---|---|---|
| **FIG-01** | Global Validation Accuracy vs. Communication Rounds | Rounds ($1 \dots R$) | Accuracy ($0 \dots 1.0$) | FedAvg vs. FedProx ($\mu=0.01$) | Compares global generalization under noisy federated aggregation. |
| **FIG-02** | Global Validation Loss vs. Communication Rounds | Rounds ($1 \dots R$) | Cross-Entropy Loss | FedAvg vs. FedProx ($\mu=0.01$) | Compares empirical loss optimization and stability. |
| **FIG-03** | Per-Client Validation Accuracy Trajectories | Rounds ($1 \dots R$) | Accuracy ($0 \dots 1.0$) | 5 curves per algorithm (Clients 1–4 solid, Client 5 dashed) | Shows whether noise degrades clean clients or isolates Client 5. |
| **FIG-04** | Per-Client Validation Loss Trajectories | Rounds ($1 \dots R$) | Cross-Entropy Loss | 5 curves per algorithm | Evaluates local convergence dynamics under noise. |
| **FIG-05** | Client Update $L_2$ Norms ($\|\Delta_{k,t}\|_2$) | Rounds ($1 \dots R$) | $L_2$ Norm Value | 5 curves per algorithm (highlighting Client 5) | Tests Hypothesis 1: Does FedProx contract Client 5's update magnitude? |
| **FIG-06** | Pairwise Cosine Similarity Dynamics | Rounds ($1 \dots R$) | Cosine Similarity ($-1 \dots 1$) | Mean clean-clean vs. Mean clean-noisy | Tests Hypothesis 2: Does noise cause gradient divergence? |
| **FIG-07** | Similarity to Sample-Weighted Mean Update | Rounds ($1 \dots R$) | Cosine Similarity ($-1 \dots 1$) | Clients 1–4 vs. Client 5 | Evaluates consensus alignment with the aggregated model. |
