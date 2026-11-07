# PHASE 0 FINAL REVIEW REPORT: TECHNICAL CORRECTION AND RESEARCH VALIDATION

**Project Title:** Comparative Experimental Analysis of Federated Averaging (FedAvg) and Federated Proximal (FedProx) on Non-IID PathMNIST with Label Noise  
**Research Supervision:** Dr. Sachin D N, Assistant Professor, Department of Computer Science and Engineering, IIIT Bhopal  
**Fellowship Program:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Auditor Role:** Senior Research Auditor & ML Systems Engineer  
**Audit Target:** Completed Phase 0 Audit Reports, Upstream PFLlib Codebase & Windows Host Telemetry  
**Date of Review:** November 6, 2026  

---

## 1. Executive Review Summary

As part of the final Phase 0 quality gate, an adversarial technical review of all Phase 0 audit reports was conducted to ensure that findings are technically defensible, mathematical formulations are rigorously verified, empirical estimates are clearly distinguished from verified facts, and no unjustified assumptions undermine the subsequent research phases.

This review systematically identified and corrected four key areas of technical overstatement or inconsistency:
1. **Memory Behavior Correction:** Refined the claim that PFLlib necessarily consumes >1 GB and immediately crashes. Re-examination of PFLlib's data lifecycle proved that client datasets are loaded sequentially per client shard (~175–180 MB per client in float32), not all 107k samples simultaneously. The uint8 lazy-loading pipeline was properly categorized as a proposed design to be measured, not a benchmarked claim.
2. **Memory Ceiling Classification:** Downgraded the 1.2 GB peak process RSS from an alleged "measured ceiling" to an **unverified operational target**, providing a comprehensive line-item breakdown of input batch tensors (294 KiB raw RGB float32 for batch 32; correcting earlier 98 KB single-channel grayscale errors), forward activations (11 individual tensor shapes summing to 2,730,272 elements if materialized; Flatten is a view, and ReLUs are out-of-place), model parameters/gradients (~400 KB), runtime overhead (~150–250 MB), and dataset storage (~235 MB in RAM during training).
3. **Execution Time Recalculation:** Identified and resolved a mathematical arithmetic inconsistency in earlier epoch runtime estimates. With 89,996 samples across 5 clients, each client trains on ~18,000 samples (~562 batches of size 32 per epoch). On a dual-core Pentium G3240 without AVX2, an epoch requires an estimated 5.6–14.1 seconds (not < 1 second). All runtime estimates were marked as provisional subject to a Phase 1 CPU pilot test.
4. **Dirichlet Partitioning & Rejection Sampling:** Removed the arbitrary 1,000-sample floor proposal, documenting how artificial truncation distorts the theoretical Dirichlet distribution $\text{Dir}(\alpha=0.3)$. Recommended unconstrained Dirichlet generation with fixed seed (`seed=42`) and natural distribution inspection.
5. **Evaluation Methodology Standardization:** Strictly separated official splits: Training (89,996) for client training; Validation (10,004) for round-level global and per-client validation metrics; Test (7,180) held out strictly for final post-training benchmark evaluation. Standardized nomenclature to "Validation" for round metrics.

---

## 2. Corrections Made

| Review Area | Initial Inaccurate Claim / Assumption | Technical Root Cause | Adopted Correction `[PROPOSED / INFERRED]` |
|---|---|---|---|
| **PFLlib Memory Footprint** | *"PFLlib necessarily consumes > 1 GB of RAM and immediately crashes."* | Conflated full-dataset theoretical float32 size (807 MB) with single-client training shard memory. | Verified from `serverbase.py` and `clientavg.py` that PFLlib loads shards sequentially per client (~175–180 MB). PFLlib poses heap fragmentation and allocation churn risks on ~580 MB free RAM, but does not load all 5 clients simultaneously. |
| **Peak Memory Ceiling** | *"1.2 GB peak process RSS ceiling established."* | Stated target memory as if it were an empirical ceiling. | Formally classified 1.2 GB as an **unverified target `[PROPOSED]`**, pending empirical validation via `psutil` in Phase 1. |
| **Input Batch vs. Total Memory** | *"Batch size 32 requires 98 KB peak memory."* | Omitted activations, graph tensors, and framework overhead, and calculated memory for single-channel grayscale instead of 3-channel RGB. | Explicitly noted that **294 KiB** ($32 \times 3 \times 28 \times 28 \times 4\text{ bytes} = 301,056\text{ bytes}$) is **strictly the raw input tensor $X$** for 3-channel RGB images. Forward activation shapes sum to $2,730,272$ elements (~10.42 MiB / 10.92 MB if materialized), but Flatten is a view, and static shapes do not determine peak RSS, which requires empirical measurement. |
| **Execution Time Arithmetic** | *"Less than 1 second per local epoch; 20–25 minutes total for 50 rounds."* | Flawed mental model of dataset size (89,996 samples / 5 clients = ~18,000 samples/client = 562 batches of 32). | Re-derived arithmetic: 562 batches at 10–25 ms/batch on dual-core CPU yields **5.6–14.1 s per epoch**; total 50-round duration is provisionally estimated at **47–117 minutes**, to be measured in a Phase 1 pilot run. |
| **Dirichlet Floor Constraint** | *"Enforce a minimum threshold of $\ge 1,000$ samples per client."* | Unjustified heuristic not specified by research supervisor Dr. Sachin D N. | Discarded arbitrary floor. Documented that rejection sampling distorts $\text{Dir}(\alpha=0.3)$. Preserved pure Dirichlet draw with seed=42 and natural distribution audit. |
| **Evaluation Terminology** | Round-level metrics referred to as "Test Accuracy" / "Test Loss". | Borrowed loose terminology from PFLlib's `rs_test_acc`. | Standardized all round-level metrics to **Validation** (`val/global_accuracy`, `val/client_k_accuracy`). Reserved **Test** strictly for the official held-out test split (7,180 samples). |
| **Noisy Client Indexing** | Mixed references to "Client 4" and "noisy client". | Zero-based vs. one-based indexing ambiguity. | Standardized: Zero-based client index 4 corresponds to human-readable **Client 5** (among Clients 1, 2, 3, 4, 5). |
| **Mean Update Diagnostic** | Unweighted mean update similarity. | PFLlib and FedAvg aggregate using sample weights $p_k = n_k / \sum n_j$. | Corrected metric to **similarity to the sample-weighted mean update** $\bar{\Delta}_t = \sum p_k \Delta_k$. |

---

## 3. Claims Retained as Technically Sound

The following core findings from the Phase 0 audit were reaffirmed as fully sound and retained:
1. **Mathematical Exactness of FedProx Update Rule `[VERIFIED]`:**
   Inspection of `system/flcore/optimizers/fedoptimizer.py:64-75` confirms that `PerturbedGradientDescent` executes:
   $$d_p = \nabla F_k(w) + \mu(w - w_t), \quad w \leftarrow w - \eta d_p$$
   which is the exact gradient of $h_k(w) = F_k(w) + \frac{\mu}{2}\|w - w_t\|_2^2$.
2. **Proximal Scalar Loss Squared-Norm Defect `[VERIFIED]`:**
   Inspection of `clientprox.py:86` confirms that PFLlib computes `loss += 0.5 * self.mu * torch.norm(gm - pm, p=2)` without squaring the $L_2$ norm.
3. **Upstream Data Leakage Flaw `[VERIFIED]`:**
   Inspection of `dataset/generate_MNIST.py:53-61` and `dataset_utils.py:11` proves that PFLlib pools official train and test sets into a 70k pool before Dirichlet partitioning.
4. **Discarded Per-Client Telemetry `[VERIFIED]`:**
   Inspection of `serverbase.py:235` confirms that per-client metric lists are computed solely for standard deviation and immediately discarded.
5. **Apache License 2.0 and Git Provenance `[VERIFIED]`:**
   Confirmed commit SHA `0169ba7e412c9856a08bb3faefab1e35f538a3c1` on `master` branch.
6. **Hardware Telemetry `[VERIFIED]`:**
   Confirmed 3.92 GB physical RAM, ~580 MB free memory, Intel Pentium G3240 (2 cores, 2 threads, no CUDA).
7. **PathMNIST Dataset Specifications `[VERIFIED]`:**
   Confirmed 9 classes, $28 \times 28 \times 3$ RGB, 89,996 train, 10,004 val, 7,180 test.

---

## 4. Claims Downgraded from Verified to Estimated / Proposed

To prevent premature conclusions before actual benchmarking, the following items were reclassified:

| Item | Previous Classification | Revised Epistemic Classification | Justification |
|---|---|---|---|
| **1.2 GB Peak Process Memory Ceiling** | Treated as established ceiling | **`[PROPOSED TARGET / UNVERIFIED]`** | An analytical budget target. Must be empirically verified using `psutil` during a Phase 1 CPU execution. |
| **Epoch & Round Runtime Durations** | Treated as definitive benchmark (<1s/epoch) | **`[ESTIMATED / PROVISIONAL]`** | Derived from assumed per-batch latencies (10–25 ms). True hardware execution times must be measured during a 2-round pilot test. |
| **Lazy uint8 DataLoader Performance** | Treated as benchmarked optimization | **`[PROPOSED DESIGN / UNVERIFIED]`** | The architectural concept is sound, but its exact throughput and memory advantages over eager loading must be verified in Phase 1. |
| **Optimal Proximal Coefficient ($\mu=0.01$)** | Treated as definitive | **`[PROPOSED DEFAULT]`** | The literature recommends $\mu \in [0.001, 0.1]$. Final optimal $\mu$ will be established empirically via sensitivity runs in Phase 3. |

---

## 5. Mathematical Consistency Checks

### Check 1: Sample Conservation in Partitioning
$$\sum_{k=1}^5 M_k = 89,996 \quad \text{and} \quad \bigcap_{k=1}^5 I_k = \emptyset$$
Every official training sample must be allocated to exactly one client. No training sample may be discarded or duplicated.

### Check 2: Exact Label Noise Percentage
For Client 5 (zero-based index 4) with $M_4$ training samples:
$$M_{\text{noisy}} = \text{round}(0.40 \times M_4)$$
Corrupted label generation:
$$\tilde{y}_i = (y_i + r_i) \pmod 9, \quad r_i \sim \text{Uniform}(\{1, 2, \dots, 8\})$$
Since $r_i \not\equiv 0 \pmod 9$, $\tilde{y}_i \neq y_i$ with probability 1.0.  
Verified noise ratio:
$$\frac{1}{M_4} \sum_{i=1}^{M_4} \mathbb{I}(\tilde{y}_i \neq y_i) = \frac{M_{\text{noisy}}}{M_4} \approx 0.4000$$

### Check 3: Sample-Weighted Validation Aggregation
$$\text{Val\_Acc}_{\text{global}} = \frac{\sum_{k=1}^5 n_{k,\text{val}} \cdot \text{Val\_Acc}_k}{\sum_{k=1}^5 n_{k,\text{val}}}, \quad \text{Val\_Loss}_{\text{global}} = \frac{\sum_{k=1}^5 n_{k,\text{val}} \cdot \text{Val\_Loss}_k}{\sum_{k=1}^5 n_{k,\text{val}}}$$
This guarantees that global validation metrics exactly match the evaluation of the pooled validation set.

### Check 4: Client Update Diagnostics
$$\Delta_{k,t} = w_{k,t} - w_t \in \mathbb{R}^D$$
$$\|\Delta_{k,t}\|_2 = \sqrt{\sum_{j=1}^D (\Delta_{k,t}[j])^2}$$
$$\text{cos}(\Delta_{i,t}, \Delta_{j,t}) = \frac{\langle \Delta_{i,t}, \Delta_{j,t} \rangle}{\|\Delta_{i,t}\|_2 \|\Delta_{j,t}\|_2 + 10^{-12}}$$
$$\bar{\Delta}_t = \sum_{k=1}^5 p_k \Delta_{k,t}, \quad \text{where } p_k = \frac{n_{k,\text{train}}}{\sum_{j=1}^5 n_{j,\text{train}}}$$
$$\text{cos}(\Delta_{k,t}, \bar{\Delta}_t) = \frac{\langle \Delta_{k,t}, \bar{\Delta}_t \rangle}{\|\Delta_{k,t}\|_2 \|\bar{\Delta}_t\|_2 + 10^{-12}}$$

---

## 6. Updated Experimental Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OFFICIAL MEDMNIST PATHMNIST DATASET                    │
│                                (107,180 Images)                             │
└──────────────┬───────────────────────────────┬──────────────────────────────┘
               │                               │                              │
               ▼                               ▼                              ▼
    ┌──────────────────────┐        ┌──────────────────────┐       ┌──────────────────────┐
    │  OFFICIAL TRAINING   │        │ OFFICIAL VALIDATION  │       │    OFFICIAL TEST     │
    │     (89,996 Imgs)    │        │    (10,004 Imgs)     │       │     (7,180 Imgs)     │
    └──────────┬───────────┘        └──────────┬───────────┘       └──────────┬───────────┘
               │                               │                              │
               ▼ (Dirichlet α=0.3, seed=42)    ▼ (Clean Independent 5 Shards) │
    ┌──────────────────────┐        ┌──────────────────────┐                  │
    │ 5 Client Train Shards│        │  5 Client Val Shards │                  │
    │  Client 1 (Clean)    │        │   Client 1 Val       │                  │
    │  Client 2 (Clean)    │        │   Client 2 Val       │                  │
    │  Client 3 (Clean)    │        │   Client 3 Val       │                  │
    │  Client 4 (Clean)    │        │   Client 4 Val       │                  │
    │  Client 5 (40% NOISE)│        │   Client 5 Val (100% │                  │
    │                      │        │         CLEAN)       │                  │
    └──────────┬───────────┘        └──────────┬───────────┘                  │
               │                               │                              │
               ▼                               ▼                              │
    ┌──────────────────────────────────────────────────────┐                  │
    │          COMMUNICATION ROUNDS (t = 1 ... 50)         │                  │
    │  1. Broadcast Global Model w_t                       │                  │
    │  2. Local Training (FedAvg SGD / FedProx Perturbed)  │                  │
    │  3. Compute Diagnostics:                             │                  │
    │     - Client Update L2 Norms ||Δ_k||                 │                  │
    │     - Pairwise Cosine Similarities cos(Δ_i, Δ_j)     │                  │
    │     - Similarity to Sample-Weighted Mean cos(Δ_k, Δ) │                  │
    │  4. Aggregate Global Model w_{t+1} = Σ p_k w_{k,t}   │                  │
    │  5. Evaluate Round Validation Metrics:               │                  │
    │     - Global Val Accuracy & Loss                     │                  │
    │     - Client 1–5 Val Accuracy & Loss                 │                  │
    │  6. Synchronous Logging: Local JSONL + W&B           │                  │
    └──────────────────────────┬───────────────────────────┘                  │
                               │                                              │
                               ▼ (Final Global Model w_final)                 │
    ┌─────────────────────────────────────────────────────────────────────────┴┐
    │                       HELD-OUT BENCHMARK EVALUATION                      │
    │          Evaluated ONCE at the end of training on Official Test Split    │
    │               (7,180 Unseen Images: Global Test Accuracy)                │
    └──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Remaining Uncertainties to Resolve in Phase 1

1. **Empirical CPU Step Latency:**
   The exact forward+backward latency per batch of 32 for our Compact CNN on the Intel Pentium G3240 (SSE4.2, no AVX2) is unmeasured. It will be empirically determined during a 2-round pilot test in Phase 1.
2. **Actual Peak Process RSS:**
   The exact process RSS under PyTorch CPU memory caching and DataLoader batch generation will be measured using `psutil` during the Phase 1 pilot test to confirm that memory remains safely below the 1.2 GB budget.
3. **Natural Dirichlet Allocation at $\alpha = 0.3$:**
   The exact per-client sample counts and per-class frequencies resulting from an unconstrained Dirichlet draw (`seed=42`) will be inspected upon running the partitioner script in Phase 1.

---

## 8. Final Phase 0 Acceptance Checklist

| Item | Verification Status | Epistemic Status | Reference Report |
|---|---|---|---|
| PFLlib cloned via official URL | **ACCEPTED** | `[VERIFIED]` | [PHASE_0_REPOSITORY_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_REPOSITORY_AUDIT.md) |
| Upstream branch identified as `master` | **ACCEPTED** | `[VERIFIED]` | [PHASE_0_REPOSITORY_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_REPOSITORY_AUDIT.md) |
| Upstream commit SHA recorded | **ACCEPTED** | `[VERIFIED]` | [PHASE_0_REPOSITORY_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_REPOSITORY_AUDIT.md) |
| Git working tree 100% clean | **ACCEPTED** | `[VERIFIED]` | [PHASE_0_COMMANDS_AND_VERIFICATION.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_COMMANDS_AND_VERIFICATION.md) |
| FedAvg implementation deeply audited | **ACCEPTED** | `[VERIFIED]` | [PHASE_0_FEDERATED_ALGORITHMS_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_FEDERATED_ALGORITHMS_AUDIT.md) |
| FedProx update formula verified | **ACCEPTED** | `[VERIFIED]` | [PHASE_0_FEDERATED_ALGORITHMS_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_FEDERATED_ALGORITHMS_AUDIT.md) |
| PFLlib scalar loss defect documented | **ACCEPTED** | `[VERIFIED]` | [PHASE_0_FEDERATED_ALGORITHMS_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_FEDERATED_ALGORITHMS_AUDIT.md) |
| Upstream data leakage documented | **ACCEPTED** | `[VERIFIED]` | [PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md) |
| Memory lifecycle corrected & classified | **ACCEPTED** | `[VERIFIED / INFERRED]` | [PHASE_0_RESOURCE_FEASIBILITY_REPORT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_RESOURCE_FEASIBILITY_REPORT.md) |
| Execution time arithmetic corrected | **ACCEPTED** | `[INFERRED / ESTIMATED]` | [PHASE_0_RESOURCE_FEASIBILITY_REPORT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_RESOURCE_FEASIBILITY_REPORT.md) |
| Dirichlet floor rejection sampling removed | **ACCEPTED** | `[PROPOSED]` | [PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md) |
| Validation vs. Test standardized | **ACCEPTED** | `[PROPOSED]` | [PHASE_0_EVALUATION_AND_DIAGNOSTICS_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_EVALUATION_AND_DIAGNOSTICS_AUDIT.md) |
| Client 5 (index 4) noise design verified | **ACCEPTED** | `[PROPOSED]` | [PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md) |
| Update diagnostics formulated | **ACCEPTED** | `[INFERRED]` | [PHASE_0_EVALUATION_AND_DIAGNOSTICS_AUDIT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_EVALUATION_AND_DIAGNOSTICS_AUDIT.md) |
| All 8 Phase 0 reports updated & aligned | **ACCEPTED** | `[VERIFIED]` | All files under `reports/` |
| Research implementation blueprint updated | **ACCEPTED** | `[PROPOSED]` | [PHASE_0_IMPLEMENTATION_BLUEPRINT.md](file:///d:/NANDAN.S/ENGINEERING%20ISSUES/pathmnist-federated-learning/reports/PHASE_0_IMPLEMENTATION_BLUEPRINT.md) |
| Zero changes to PFLlib upstream | **ACCEPTED** | `[VERIFIED]` | `git status --short` verified empty |
| Phase 0 final review report generated | **ACCEPTED** | `[VERIFIED]` | `PHASE_0_FINAL_REVIEW_REPORT.md` |

---

## 9. Conclusion & Auditor Sign-Off

The technical audit and review of Phase 0 are complete, rigorous, and technically defensible. All overstated claims have been scaled back to verifiable statements, memory and execution estimates have been re-anchored in sound mathematics, and the evaluation methodology preserves official benchmark integrity.

The repository and workspace remain in a clean, non-executable state:
- No code has been modified in PFLlib.
- No packages have been installed.
- PathMNIST has not been downloaded.
- No W&B runs or projects have been initiated.

Phase 0 is formally concluded. I await your review and explicit approval before any Phase 1 activities commence.
