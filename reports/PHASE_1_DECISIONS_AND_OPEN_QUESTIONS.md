# PHASE 1: RESEARCH DECISIONS AND OPEN QUESTIONS (CORRECTED)
## Finalized Protocols, Empirical Calibration Gates, Dataset Inspection Checkpoints, and Optional Enhancements

**Project:** Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Client-Specific Label Noise  
**Fellowship Program:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Author:** Senior Federated Learning Research Scientist & Quality Assurance Auditor  
**Document Status:** Decision Log and Open Questions (`Phase 1`)  
**Date:** November 7, 2026  

---

## 1. Executive Decision Framework

To prevent scope creep, unverified claims, and methodological regressions, decisions are categorized into five distinct operational domains:
1. **Decisions Finalized in Phase 1:** Fixed design specifications that are mathematically and experimentally frozen.
2. **Decisions Requiring CPU Pilot Measurement:** Parameters calibrated empirically during the Phase 2 pilot test.
3. **Decisions Requiring Dataset Inspection:** Structural observations conducted upon downloading PathMNIST in Phase 2.
4. **Decisions Requiring Protocol Clarification:** Potential areas for methodology refinement.
5. **Optional Enhancements:** Non-mandatory explorations that must not obstruct core deliverables.

---

## 2. Category 1: Decisions Finalized in Phase 1 (Frozen Specifications)

These decisions are locked and require no further changes:

1. **Benchmark Dataset & Splits:** Official PathMNIST (MedMNIST v2). Training split (89,996 images), validation split (10,004 images), and test split (7,180 images) remain strictly disjoint.
2. **Federated Topology:** Exactly 5 clients ($N=5$).
3. **Client Participation:** Full client participation ($C=1.0$) every communication round. No partial client sampling or dropouts.
4. **Data Partitioning Method:** Dirichlet non-IID class-wise allocation with concentration parameter $\alpha = 0.3$. No arbitrary client-size floors (e.g., 1,000 samples). Governed by fixed random seed `seed=42`.
5. **Sample Conservation:** All 89,996 training samples are partitioned across the 5 clients without duplication ($\bigcap I_c = \emptyset$, $\sum M_c = 89,996$).
6. **Validation Shard Apportionment:** Largest-remainder (Hamilton-Hare) apportionment procedure creates 5 clean, disjoint simulated validation shards from the 10,004 official validation images, reflecting local training class proportions.
7. **Noisy Client Identity:** Human-readable **Client 5** (zero-based client index 4). Clients 1, 2, 3, and 4 undergo 0% noise.
8. **Noise Model:** Symmetric multiclass label flipping across the remaining 8 classes: $\tilde{y} = (y + r) \pmod 9, r \in [1, 8]$. Exactly $\text{round}(0.40 \times M_5)$ training samples are corrupted. Original clean labels and boolean `noise_mask` are preserved.
9. **Oracle Validation Integrity:** Validation labels and test labels remain 100% clean.
10. **Neural Network Backbone:** `CompactPathCNN` (36,425 parameters; 2 Conv layers + Adaptive Average Pooling + 2 FC layers, no BatchNorm).
11. **Federated Optimizers:**
    - FedAvg: Standard PyTorch SGD ($\eta = 0.01$, momentum = 0.0, weight_decay = 0.0).
    - FedProx: Exact Perturbed SGD ($\eta = 0.01$, $\mu = 0.01$).
12. **Parameter Initialization:** Shared initial weights $w_0$ serialized to `initial_model.pt` with verified canonical model-state hash and `torch.equal()` tensor equality across both algorithms.
13. **Checkpoint Selection Policy:** Each algorithm's selected best-validation checkpoint is evaluated once on the official PathMNIST test set after training has finished (tie-breaking: lowest validation loss, then earlier round). Test split is strictly isolated during training.
14. **Telemetry Storage:** Local durable JSONL logging (`metrics.jsonl`) is the primary source of truth, supplemented synchronously by Weights & Biases (W&B). Zero credentials in source code.

---

## 3. Category 2: Decisions Requiring CPU Pilot Measurement (Phase 2 Gates)

These parameters cannot be reliably guessed on paper; they will be calibrated empirically during the Phase 2 pilot test:

### 3.1 Empirical Mini-Batch Step Latency
- **Measurement Procedure:** Measure 100 iterations of forward+backward passes during the 2-round pilot test. Compute mean latency and standard deviation.

### 3.2 System Memory Footprint & Safety Thresholds
- **Monitoring Reference:** Target process RSS $< 1.2 \text{ GB}$ is retained as a provisional monitoring reference, not an established ceiling.
- **System Memory Safety Threshold:** If available physical system RAM drops below **150 MB** during execution, trigger an immediate emergency halt. This 150 MB floor is strictly an emergency abort threshold and does not constitute proof that execution is safe above it. Additional stop triggers include observed OS hard paging/disk thrashing or rapid unrecovered collapse of available physical RAM.

### 3.3 Mutually Exclusive Communication Round Calibration Policy `[PROPOSED]`
The previously contradictory conditional logic has been replaced with an explicitly ordered, mutually exclusive calibration policy:

$$\text{Decision}(T_{\text{round}}) = \begin{cases}
R = 100 \text{ rounds} & \text{if } T_{\text{round}} < 35 \text{ s} \quad (\text{Estimated total duration } < 60 \text{ min}) \\
R = 50 \text{ rounds} & \text{if } 35 \text{ s} \le T_{\text{round}} \le 75 \text{ s} \quad (\text{Estimated total duration } \approx 30\text{--}65 \text{ min}) \\
R = 30 \text{ rounds} & \text{if } 75 \text{ s} < T_{\text{round}} \le 120 \text{ s} \quad (\text{Estimated total duration } \approx 38\text{--}60 \text{ min}) \\
\text{Halt \& Review} & \text{if } T_{\text{round}} > 120 \text{ s} \quad (\text{Flag for protocol review; consider } E=1)
\end{cases}$$

- Both algorithms will receive the identical calibrated number of communication rounds.
- Full official training data will be preserved without reduction.

---

## 4. Category 3: Decisions Requiring Dataset Inspection (Phase 2 Checkpoints)

These observations will be conducted immediately upon downloading PathMNIST in Phase 2:

### 4.1 Natural Sample Distribution Across Clients at $\alpha = 0.3$
- **Inspection Goal:** Verify the exact sample counts $M_1, M_2, M_3, M_4, M_5$ resulting from an unconstrained Dirichlet draw with `seed=42`.
- **Checkpoint:** Generate and print the client sample size summary table and per-class histograms. Verify that no client receives an empty total partition ($\sum_k N_{c,k} > 0$), while recognizing that individual class sparsity ($N_{c,k} = 0$) is expected and naturally handled under non-IID conditions. If any client receives zero total samples, halt partition acceptance and report immediately. Silent sample movement or arbitrary floors are prohibited.

### 4.2 Client 5 Histological Class Skew
- **Inspection Goal:** Observe which tissue classes dominate Client 5's partition before 40% noise injection.
- **Checkpoint:** Confirm how the 40% noise flipping distributes across tissue classes.

---

## 5. Category 4: Decisions for Protocol Refinement

These design aspects are well-defined by default, but remain open for reviewer input:

1. **Ablation Matrix Prioritization:**
   - *Default Plan:* Complete primary mandatory runs (FedAvg baseline and FedProx $\mu=0.01$). If CPU runtime permits, execute sensitivity runs $\mu=0.001$ and $\mu=0.1$.
   - *Protocol Clarification:* Confirm whether to explore exploring different $\mu$ values (0.001, 0.1) or local epoch variations ($E=1$ vs $E=2$).
2. **Per-Client Validation Shard Structure:**
   - *Default Plan:* Validation set (10,004) is partitioned into 5 clean client-specific validation shards representative of each client's training class distribution, while also reporting global pooled validation.
   - *Protocol Clarification:* Confirm that reporting both pooled global validation accuracy and per-client validation accuracy satisfies the screening requirements.

---

## 6. Category 5: Optional Enhancements (Non-Blocking)

These features provide additional scientific value but are strictly secondary to the core deliverables:
1. **Multiclass Macro-AUC:** One-vs-Rest (OvR) macro-average AUC evaluated post-training if compute permits.
2. **Per-Class Confusion Matrix:** $9 \times 9$ heatmap plotted on the final test evaluation.
3. **Client Update Norm Heatmap Matrix:** Visualizing the $5 \times 5$ pairwise cosine similarity matrix across communication rounds as a progression plot.
