# PHASE 1: ACCEPTANCE CRITERIA MATRIX (CORRECTED)
## Objective Verification Gates and Acceptance Criteria for Implementation and Experimental Phases

**Project:** Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Client-Specific Label Noise  
**Research Supervision:** Dr. Sachin D N, Assistant Professor, Department of CSE, IIIT Bhopal  
**Fellowship Program:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Author:** Senior Federated Learning Research Scientist & Quality Assurance Auditor  
**Document Status:** Formal Quality Gate Specifications (`Phase 1`)  
**Date:** November 7, 2026  

---

## 1. Executive Quality Assurance Framework

To maintain scientific integrity and prevent regression or unverified assumptions across subsequent implementation phases, we establish an explicit, checklist-driven acceptance framework. 

No phase may begin until all acceptance criteria of the preceding phase have been empirically satisfied.

---

## 2. Phase 2 Acceptance Criteria: Core Implementation & Pilot Verification

Phase 2 builds the modular software architecture in `research/` and executes the initial CPU pilot tests.

| Gate ID | Component | Objective Acceptance Criterion | Verification Method |
|---|---|---|---|
| **P2-G01** | Split Isolation | PathMNIST train (89,996), val (10,004), and test (7,180) splits remain 100% disjoint with zero sample overlap. | `pytest research/tests/test_splits.py` |
| **P2-G02** | Sample Conservation & Non-Empty Partitions | Dirichlet partition ($\alpha=0.3$, $N=5$, seed 42) allocates exactly 89,996 samples with $\bigcap I_c = \emptyset$ without artificial floors, asserting each client partition is non-empty ($\sum_k N_{c,k} > 0$). | `pytest research/tests/test_partition.py` |
| **P2-G03** | Validation Apportionment | Simulated clean validation shards $V_1 \dots V_5$ partition 10,004 samples via largest-remainder procedure with exact conservation and disjointness. | `pytest research/tests/test_splits.py` |
| **P2-G04** | Label Noise Precision | Client 5 (zero-based index 4) contains exactly $\text{round}(0.40 \times M_5)$ corruptions; Clients 1–4, val, and test remain 100% clean. | `pytest research/tests/test_noise.py` |
| **P2-G05** | Noise Flipping Validity | All corrupted labels satisfy $\tilde{y}_i \neq y_i$ and $\tilde{y}_i \in [0, 8]$. | `pytest research/tests/test_noise.py` |
| **P2-G06** | Model Architecture | `CompactPathCNN` forward pass accepts $(B, 3, 28, 28)$, outputs $(B, 9)$, and has exactly $36,425$ parameters. | `pytest research/tests/test_model.py` |
| **P2-G07** | Optimizer Equivalence | `PerturbedSGD` gradient step mathematically matches $\nabla F + \mu(w - w_t)$ within $10^{-7}$ numerical tolerance. | `pytest research/tests/test_optimizer.py` |
| **P2-G08** | Canonical Weight Identity | Initial model weights $w_0$ match across FedAvg and FedProx via canonical state hash and `torch.equal()` tensor verification. | Pre-flight assertion check |
| **P2-G09** | System Memory Monitoring | 1-round smoke test verifies process RSS monitoring, available system RAM tracking, and emergency stop threshold ($<150 \text{ MB}$, qualified as an emergency halt rather than proof of safety above it). | `psutil` instrumentation in smoke test |
| **P2-G10** | CPU Pilot Benchmark | 2-round pilot execution measures empirical batch step latency (ms), round duration (s), and peak RSS, calibrating the final round horizon. | Execution telemetry in `pilot_benchmark.json` |
| **P2-G11** | Upstream PFLlib Integrity | `third_party/PFLlib` remains completely clean with zero modifications. | `git status --short` returns zero output |

---

## 3. Phase 3 Acceptance Criteria: Full Experimental Execution & Telemetry

Phase 3 executes the primary experimental arms (FedAvg vs. FedProx) and logs telemetry.

| Gate ID | Component | Objective Acceptance Criterion | Verification Method |
|---|---|---|---|
| **P3-G01** | Full Client Participation | Every communication round ($t=1 \dots R$) trains all 5 clients sequentially with full participation ($C=1.0$). | Assert 5 client logs per round in JSONL |
| **P3-G02** | Identical Batch Shuffling | Both algorithms use identical mini-batch sample orderings derived deterministically from `seed = 42 + 1000*t + c`. | Batch ordering test assertion |
| **P3-G03** | Algorithmic Fairness | Both FedAvg and FedProx execute on identical partition manifests, identical noise manifests, and identical $w_0$. | Manifest checksum inspection in run receipt |
| **P3-G04** | Telemetry Completeness | Every communication round records all global validation metrics, per-client validation metrics, $L_2$ norms, 10 pairwise cosine similarities, and similarity to mean update. | Automated JSON schema validator on `metrics.jsonl` |
| **P3-G05** | Durable Local Persistence | Local `metrics.jsonl` contains exactly $R$ complete records; summary CSV matches JSONL records. | File existence and row-count assertion |
| **P3-G06** | W&B Live Mirroring | Public W&B project `pathmnist-fedavg-fedprox` reflects runs `fedavg_alpha0.3_noise40_seed42` and `fedprox_mu0.01_alpha0.3_noise40_seed42`. | W&B dashboard verification |
| **P3-G07** | Memory Safety Maintained | Process runs without triggering OS pagefile thrashing, memory crashes, or dropping available system RAM below 150 MB. | System telemetry logging |
| **P3-G08** | Checkpoint Archival | Best validation checkpoint and final round checkpoint are saved with canonical state hashes recorded. | Checkpoint existence in `experiments/checkpoints/` |

---

## 4. Phase 4 Acceptance Criteria: Analysis, Evaluation & Documentation

Phase 4 conducts the final held-out test evaluation, generates publication figures, and compiles the submission dossier.

| Gate ID | Component | Objective Acceptance Criterion | Verification Method |
|---|---|---|---|
| **P4-G01** | Held-Out Test Integrity | Official PathMNIST test set (7,180 images) is evaluated once per algorithm on the selected best validation checkpoint. | Execution of `test_evaluation.py` |
| **P4-G02** | Mandatory Metric Delivery | Reports final global test accuracy and test loss for both algorithms. | Output table in final submission report |
| **P4-G03** | Publication Plots | Renders all 7 required publication-quality figures (`FIG-01` to `FIG-07`) in high-resolution PNG format. | Visual inspection and file verification |
| **P4-G04** | Exploratory Analysis | Evaluates Hypotheses 1, 2, and 3 using single-seed empirical trajectories, update norm values, and cosine similarities without unsupported significance claims. | Analysis section in final report |
| **P4-G05** | Repository Documentation | Project `README.md` provides complete reproduction commands, hardware requirements, methodology explanations, and results summaries. | Peer review against IndiaAI guidelines |
| **P4-G06** | Zero Credential Exposure | Repository contains zero API keys, personal tokens, or hardcoded credentials. | Automated scan (`git log -p | grep -E "api_key|token"`) |
