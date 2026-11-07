# Project Status: PathMNIST Federated Learning Benchmark

**Project:** Comparative Experimental Analysis of Federated Averaging (FedAvg) and Federated Proximal (FedProx, $\mu = 0.01$) on Non-IID PathMNIST with 40% Client-Specific Label Noise  
**Academic Context:** IndiaAI Fellowship 2026–27 Research Screening  
**Faculty Mentor:** Dr. Sachin D N, Assistant Professor, Department of CSE, IIIT Bhopal  
**Last Updated:** November 2026  
**Status:** Architecture Hardened, Legacy Evidence Archived, Ready for Full Multi-Seed Execution  

---

## 1. Executive Summary & Experimental State

This repository hosts a scientifically rigorous, fully reproducible comparative study between **FedAvg** and **FedProx** on the histological **PathMNIST** dataset under realistic non-IID data distribution and client-specific label corruption.

### Clear Distinction of Experimental Artifacts:
1. **Legacy Experiment (`experiments/legacy_seed42_pre_batch_fix/`):**  
   The original Phase 3 experimental run (30 rounds FedAvg, 30 rounds FedProx) conducted before the minibatch generator hardening. Complete metrics, test evaluation (79.46% FedAvg vs 76.07% FedProx), and figures are preserved for historical provenance.
2. **Interrupted Corrected Attempt (`experiments/interrupted_attempts/seed_42_attempt_1_rounds_1_to_20/`):**  
   The initial attempt of the corrected Seed 42 execution, which completed 20 rounds of FedAvg (Val Acc rising to 68.27%) before being gracefully interrupted due to pipe buffer latency. All 20 valid JSONL records and checkpoints are quarantined.
3. **Corrected Multi-Seed Benchmark (`experiments/seed_<id>/`):**  
   The fully hardened, unbuffered orchestrator (`scripts/run_multiseed.py`) ready to execute fresh 30-round runs with exact monotonic timing and unbuffered stdout streaming.

---

## 2. Completed Work Matrix

| Category | Milestone | Status | Key Deliverable |
| :--- | :--- | :---: | :--- |
| **Framework Audit** | PFLlib Upstream Study | **COMPLETE** | Technical audit of PFLlib client-server lifecycle; reference preserved untouched. |
| **Dataset Acquisition** | PathMNIST Verification | **COMPLETE** | Verified official MD5 (`a8b069...`) and SHA-256 (`81823f...`); zero split overlap. |
| **Data Partitioning** | Dirichlet Non-IID ($\alpha = 0.3$) | **COMPLETE** | 5 clients partitioned; mathematical proof of Client 5 partition size (7,895 samples). |
| **Noise Injection** | Symmetric Label Noise (40%) | **COMPLETE** | Deterministic modulo-9 label flipping ($\tilde{y} = (y + r) \pmod 9$) on Client 5. |
| **Model Architecture** | `CompactPathCNN` | **COMPLETE** | Reconciled parameter count to exactly 36,425 parameters; strict state loading verified. |
| **Local Optimization** | Exact Perturbed SGD | **COMPLETE** | Analytical equivalence and gradient invariance verified for FedProx ($\mu = 0.01$). |
| **Minibatch Generator**| Deterministic Permutation | **COMPLETE** | Fixed sub-epoch generator; 22 unit & regression tests passing in 26.78s. |
| **Evaluation Suite** | Independent Verification | **COMPLETE** | Reconstructed confusion matrices, macro-F1, balanced accuracy, and 7,180 predictions. |
| **Process Visibility** | Unbuffered Streaming & Tracking| **COMPLETE** | Python `-u`, per-round `flush=True`, and real-time disk persistence of `run_status.json`. |
| **Repository Setup** | Clean Git Structure & Portability| **COMPLETE** | Portable `requirements.txt`, relative pathing, security scan passed (zero leaked tokens). |

---

## 3. Incomplete Work & Next Steps

1. **Full Corrected Seed 42 Run:**
   - Execute the complete 30 rounds of FedAvg and 30 rounds of FedProx under the corrected minibatch pipeline.
   - Command:
     ```powershell
     .venv\Scripts\python -u scripts/run_multiseed.py --seeds 42 --rounds 30
     ```
2. **Multi-Seed Robustness Evaluation (Seeds 123, 2026, 3407, 7777):**
   - Execute the 4 remaining configured seeds sequentially to calculate mean $\pm$ standard deviation for global test accuracy, macro-F1, and paired deltas ($\Delta_{\text{Test Acc}} = \text{FedProx} - \text{FedAvg}$).
3. **Weights & Biases Public Dashboard Push:**
   - Currently, all metrics are durably preserved in local offline JSONL files.
   - Requires setting `WANDB_API_KEY` and running `python scripts/sync_to_wandb.py` to push to project `pathmnist-fedavg-fedprox`.

---

## 4. Hardware Constraints & Runtime Realities

- **CPU Host:** Intel Pentium G3240 (Dual-Core, 2 Threads, 3.10 GHz, 4 GB Physical RAM, no AVX2/FMA).
- **Per-Round Duration:** ~4.5 to 6.0 minutes per algorithm (~9.5 to 12.0 minutes per paired communication round).
- **Per-Seed Duration:** 30 rounds paired $\approx$ 4.5 to 6.0 hours of continuous CPU execution.
- **5-Seed Study Duration:** 5 paired seeds $\approx$ 22.5 to 30.0 hours of sequential compute.
- **Memory Consumption:** Peak process RSS strictly bounded to 289–659 MB (well within 4 GB physical RAM).

---

## 5. Known Scientific Limitations & Boundary Conditions

1. **Proximal Penalty Generalization Hypotheses:**  
   The observed collapse in recall on Class 2 (28.32% vs 60.47%) and Class 6 (48.18% vs 77.33%) for FedProx is documented as a **working hypothesis** regarding proximal contraction dynamics under asymmetric class-correlated noise. A formal proof of causality would require an explicit clean-label control ablation.
2. **Timing Reconciliation:**  
   Precise reconciliation between monotonic process-level timers (`time.perf_counter()`) and round duration sums is instrumented in `scripts/run_multiseed.py` and will be populated upon completion of fresh runs.
