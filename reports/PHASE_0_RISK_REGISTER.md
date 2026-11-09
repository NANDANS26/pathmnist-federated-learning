# PHASE 0: COMPREHENSIVE TECHNICAL RISK REGISTER (CORRECTED)

**Project:** Federated Learning on PathMNIST (FedAvg vs. FedProx)  
**Fellowship Program:** IndiaAI Fellowship 2026–27 Screening Submission  
**Audit Status:** Technical Correction Completed  
**Date:** November 7, 2026  

---

## 1. Risk Evaluation Matrix

Risks are categorized across six domains:
1. **DATA:** Data leakage, split corruption, partition anomalies.
2. **ALGO:** Algorithmic divergence, proximal update correctness, aggregation flaws.
3. **EVAL:** Evaluation timing, missing metrics, diagnostic distortion.
4. **ENV:** RAM exhaustion (4 GB), CPU compute limits, Windows OS path issues.
5. **REPRO:** Random seed divergence, unpinned stochasticity.
6. **INTEG:** W&B failure, dependency conflicts.

Severity is ranked as **CRITICAL**, **HIGH**, **MEDIUM**, or **LOW**.

---

## 2. Structured Risk Register

| Risk ID | Domain | Severity | Risk Description & Cause `[VERIFIED / INFERRED]` | Upstream File / Location | Impact on IndiaAI Submission | Mitigation Strategy `[PROPOSED]` | Future Verification Method `[PROPOSED]` |
|---|---|---|---|---|---|---|---|
| **RSK-01** | DATA | **CRITICAL** | **Benchmark Split Pooling in Upstream MNIST Generator:** Upstream `generate_MNIST.py` pools original train and test arrays before partitioning. | `dataset/generate_MNIST.py:53-61`, `dataset_utils.py:11` | Compromises benchmark evaluation protocol. Precautionary measure required for PathMNIST. | Strictly isolate official PathMNIST splits: train (89,996) partitioned to clients; val (10,004) for round metrics; test (7,180) held out. | Unit test asserting zero sample overlap between client sets and test/val sets: `set(train_hashes) & set(test_hashes) == empty`. |
| **RSK-02** | ENV | **HIGH** | **Memory Allocation Churn on 4 GB RAM:** Eager float32 conversion of individual 18k-sample client shards creates memory fragmentation. | `system/utils/data_utils.py:39-42` | Windows OS paging, system lag, or OOM if multiple allocations persist. | Store client images as contiguous `uint8` NumPy arrays. Convert to `float32` lazily per mini-batch (batch 32 RGB is ~294 KiB). | Monitor Python RSS and available physical RAM using `psutil`. Halt if available RAM drops below 150 MB. |
| **RSK-03** | DATA | **HIGH** | **Label Noise Leaking into Validation/Test Sets:** Accidental noise injection into client evaluation sets. | Custom Noise Logic / `clientbase.py` | Distorts ground-truth evaluation, corrupting scientific comparison. | Apply noise strictly to `client_5.train_labels`. Assert validation and test labels are 100% clean. | Automated assertions verifying test and validation labels match original MedMNIST ground truth. |
| **RSK-04** | ALGO | **HIGH** | **Proximal Loss Squared-Norm Defect:** PFLlib computes $L_2$ norm without square: `0.5 * mu * torch.norm(gm-pm, p=2)`. | `system/flcore/clients/clientprox.py:86` | Incorrect logged proximal loss scalar value, mathematical inconsistency. | Implement correct squared loss: `0.5 * self.mu * torch.sum((pm - gm) ** 2)`. | Mathematical unit test verifying analytical equivalence of objective value. |
| **RSK-05** | EVAL | **HIGH** | **Evaluation Timing Lag & Missing Final Round:** Evaluation occurs *before* round $t$ training, omitting final aggregated model evaluation. | `system/flcore/servers/serveravg.py:28-32` | Final submission metrics will report round $R-1$ performance rather than final model. | Perform explicit evaluation after round aggregation and an extra final round evaluation. | Assert evaluation log length equals `num_rounds + 1`. |
| **RSK-06** | EVAL | **HIGH** | **Per-Client Trajectory Omission:** Upstream discards per-client validation metrics after calculating standard deviation. | `system/flcore/servers/serverbase.py:235` | Inability to plot per-client learning curves, failing prompt requirements. | Explicitly log and persist per-client validation metrics each round to JSONL, CSV, and W&B. | Verify export contains full $(R+1) \times K$ validation matrix. |
| **RSK-07** | ALGO | **MEDIUM** | **Model Architecture Inconsistency:** PFLlib wraps FedAvg in `BaseHeadSplit` but leaves FedProx monolithic. | `system/main.py:187-190`, `204-205` | Confounding variable; model parameter count or layer access differs between algorithms. | Use an identical monolithic lightweight CNN for both FedAvg and FedProx. | Unit test verifying identical state_dict keys and parameter counts for both models. |
| **RSK-08** | ENV | **MEDIUM** | **CPU Latency & Thread Saturation:** 281,000 mini-batch updates across 50 rounds on 2-core Pentium CPU without AVX2. | Execution Engine | Extended run duration (> 1-2 hours) if batch latency is unmonitored. | Measure empirical step latency during Phase 2 pilot; calibrate round horizon using mutually exclusive conditional policy. | Benchmark 2-round pilot execution before committing to full 50-round runs. |
| **RSK-09** | DATA | **MEDIUM** | **Distortion via Rejection Sampling in Dirichlet:** Arbitrary sample floors distort the natural Dirichlet distribution $\text{Dir}(\alpha=0.3)$. | Partitioning Logic | Truncates distribution tails, artificially reducing client heterogeneity. | Draw unconstrained Dirichlet partition with fixed seed; inspect natural client sizes before applying any threshold. | Document per-client sample counts and per-class distribution tables. |
| **RSK-10** | REPRO | **MEDIUM** | **Stochastic Partition Drift:** Non-deterministic partition sampling prevents fair FedAvg vs. FedProx comparison. | `dataset/generate_MNIST.py:11-12` | FedAvg and FedProx run on different client data distributions, invalidating comparison. | Persist the generated partition indices to disk as fixed manifest `.npz` files loaded by all runs. | Hash check: `hash(partition_fedavg) == hash(partition_fedprox)`. |
| **RSK-11** | INTEG | **LOW** | **W&B Network Timeout or Missing Credentials:** Network failure aborting training runs mid-experiment. | Upstream (None) / Custom runner | Lost experimental data if training crashes on W&B network failure. | Implement local durable JSONL/CSV file logger as primary source of truth; W&B is supplementary. | Verify local JSONL file writes complete independently of network status. |
| **RSK-12** | ENV | **LOW** | **Windows Path Separators & File Locks:** Hardcoded Unix `/` slashes and open file descriptor locks on `.pt` files. | `system/flcore/servers/serverbase.py:172` | `FileNotFoundError` or permission denied errors on Windows NTFS. | Use `pathlib.Path` or `os.path.join` universally across all paths. Explicitly close file handles. | Run automated test suite across Windows PowerShell environment. |
