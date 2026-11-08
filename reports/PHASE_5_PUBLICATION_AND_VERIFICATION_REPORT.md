# Phase 5: Publication, Verification & Integrity Reconciliation Report

**Project:** Comparative Experimental Analysis of Federated Averaging (FedAvg) and Federated Proximal (FedProx, $\mu = 0.01$) on Non-IID PathMNIST with 40% Client-Specific Label Noise  
**Academic Supervision:** Dr. Sachin D N, Assistant Professor, Department of CSE, IIIT Bhopal  
**Fellowship Programme:** IndiaAI Fellowship 2026–27 (Research Screening Submission)  
**Repository:** [https://github.com/NANDANS26/pathmnist-federated-learning](https://github.com/NANDANS26/pathmnist-federated-learning)  
**Date:** November 2026  

---

## 1. Executive Summary

This report documents the final publication, independent verification, and technical consistency audits conducted across Phases 5, 5B, and 5C for the public GitHub release of the PathMNIST Federated Learning benchmark.

All repository components—including the core federated training algorithms, dataset acquisition and partitioning pipelines, local perturbation optimizers, unit test suites, and historical artifacts—have been reconciled, cryptographically audited, and confirmed to be fully functional and reproducible.

---

## 2. Model Architecture Reconciliation (`CompactPathCNN`)

### Actual Canonical Architecture in Implementation and Documentation:
Inspection of `research/models/compact_cnn.py` and `README.md` confirms the exact layer-by-layer structure:

1. **Stage 1 (Feature Extraction):**
   - `conv1`: `nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)`
   - `relu1`: `nn.ReLU(inplace=False)`
   - `pool1`: `nn.MaxPool2d(kernel_size=2, stride=2)` $\to$ Feature map: `(B, 32, 14, 14)`
2. **Stage 2 (Feature Extraction):**
   - `conv2`: `nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)`
   - `relu2`: `nn.ReLU(inplace=False)`
   - `pool2`: `nn.MaxPool2d(kernel_size=2, stride=2)` $\to$ Feature map: `(B, 64, 7, 7)`
3. **Adaptive Pooling & Dimension Reduction:**
   - `adaptive_pool`: `nn.AdaptiveAvgPool2d((2, 2))` $\to$ Output: `(B, 64, 2, 2)`
   - `flatten`: `nn.Flatten()` $\to$ Feature vector: `(B, 256)`
4. **Classification Head:**
   - `fc1`: `nn.Linear(in_features=256, out_features=64)`
   - `relu3`: `nn.ReLU(inplace=False)`
   - `fc2`: `nn.Linear(in_features=64, out_features=9)` $\to$ Output Logits: `(B, 9)`

### Exact Parameter Count Verification:
$$\begin{aligned}
\text{conv1 (weights)} &= 32 \times 3 \times 3 \times 3 = 864 \\
\text{conv1 (biases)} &= 32 \\
\text{conv2 (weights)} &= 64 \times 32 \times 3 \times 3 = 18{,}432 \\
\text{conv2 (biases)} &= 64 \\
\text{fc1 (weights)} &= 64 \times 256 = 16{,}384 \\
\text{fc1 (biases)} &= 64 \\
\text{fc2 (weights)} &= 9 \times 64 = 576 \\
\text{fc2 (biases)} &= 9 \\
\hline
\mathbf{Total\ Trainable\ Parameters} &= \mathbf{36{,}425}
\end{aligned}$$

### Note on Prior Phase 5B Textual Summary:
In an intermediate textual summary of Phase 5B, the architecture was erroneously described in prose as `3 -> 16 -> 32`. The codebase (`research/models/compact_cnn.py`), the canonical initial model hash (`b30f8ea6...`), the unit test suite (`research/tests/test_model_architecture.py`), and the published `README.md` have consistently and exclusively implemented and documented the true `3 -> 32 -> 64` configuration with 36,425 parameters. No model alterations were required or permitted.

---

## 3. Dataset Integrity, Checksums & Official Metadata

### Environmental Audit:
- **Installed MedMNIST Package:** Version `3.0.2`
- **Official Package Metadata (`medmnist.INFO["pathmnist"]`):**
  - URL: `https://zenodo.org/records/10519652/files/pathmnist.npz?download=1`
  - Expected MD5: `a8b06965200029087d5bd730944a56c1`
  - Samples: Train: 89,996; Val: 10,004; Test: 7,180 (Total: 107,180)
- **Local File Inspection (`data/pathmnist.npz`):**
  - Actual File Size: **205,615,438 bytes** (~196.09 MiB)
  - Actual MD5: **`a8b06965200029087d5bd730944a56c1`**
  - Actual SHA-256: **`81823f52dc622e69db2db4c72f8e8e617938dd6864d3c1f23d4e49724a28ea72`**
  - Integrity Check Status: **100% MATCH** (`actual_md5 == expected_md5` evaluates to `True`)

### Clarification on Checksum Provenance:
All repository-persisted manifests (`experiments/manifests/dataset_manifest.json`, `experiments/legacy_seed42_pre_batch_fix/manifests/dataset_manifest.json`), historical phase reports (Phases 2, 3, and 4), and `PROJECT_STATUS.md` have consistently recorded `a8b06965200029087d5bd730944a56c1`. The string `12fa4b1b36913c19659b870a24a5be9a` cited in the Phase 5B narrative was an isolated conversational hallucination and did not originate from any file, metadata, or download.

---

## 4. Licensing and Dataset Provenance Separation

The repository maintains an unambiguous separation between software intellectual property and scientific dataset licensing:
1. **Source Code & Pipelines:**
   - Licensed under the [MIT License](LICENSE) (Copyright (c) 2026 NANDAN S).
   - Covers all Python modules (`research/`), orchestration scripts (`scripts/`), configurations, unit tests, and documentation.
2. **PathMNIST Dataset:**
   - Governed by the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
   - Originated from the MedMNIST v2 project (Yang et al., *Scientific Data* 2023) and the NCT-CRC-HE-100K / CRC-VAL-HE-7K cohorts (Kather et al., *PLOS Medicine* 2019).
   - Distributed via Zenodo (Record 10519652).
   - Clarified explicitly in `README.md` Section 16 that the repository's MIT license does not extend to the dataset.

---

## 5. Verification Checklist & Repository Health

| Requirement | Audit Command / Method | Result | Status |
| :--- | :--- | :--- | :---: |
| **Data Modules Tracked** | `git ls-files research/data/` | All 5 `.py` modules tracked | **PASS** |
| **Root Data Ignored** | `git check-ignore -v data/pathmnist.npz` | Ignored by rule `.gitignore:35:/data/` | **PASS** |
| **MMAP Ignored** | `git check-ignore -v data/pathmnist_mmap/*.npy` | Ignored by rule `.gitignore:35:/data/` | **PASS** |
| **Unit Test Suite** | `pytest research/tests/ -v` | 22 / 22 unit tests passing | **PASS** |
| **Test Split Isolation** | Code audit in `scripts/run_multiseed.py` | Quarantined to Step 6 post-training | **PASS** |
| **Directory Protection** | Safety check in `scripts/run_multiseed.py` | Aborts with `FileExistsError` unless `--force` | **PASS** |
| **CLI Documentation** | `python scripts/run_multiseed.py --help` | Flags `--seeds`, `--rounds`, `--attempt`, `--force` verified | **PASS** |
| **Clean Clone Setup** | `python scripts/initialize_experiment.py` | Successfully initializes partitions and initial checkpoint | **PASS** |

---

## 6. Conclusion

The repository is mathematically sound, fully documented, cryptographically verified, and ready for review and multi-seed execution under IndiaAI Fellowship guidelines.
