# Repository Attribution Update: Removal of Unconfirmed Academic Supervision Attribution

**Project:** Comparative Experimental Analysis of Federated Averaging (FedAvg) and Federated Proximal Optimization (FedProx) on Non-IID PathMNIST with 40% Symmetric Label Noise  
**Author / Fellowship Candidate:** Nandan S  
**Fellowship Program:** IndiaAI Fellowship 2026–2027 Screening Submission  
**Repository:** [https://github.com/NANDANS26/pathmnist-federated-learning](https://github.com/NANDANS26/pathmnist-federated-learning)  
**Date of Execution:** November 9, 2026  
**Starting Commit SHA:** `2cb1228806e378315d8978dddd8a4af0627f6960`  

---

## 1. Executive Summary & Purpose

Following an explicit request from Dr. Sachin D N ("Please remove my name and the 'Academic Supervision' line from your GitHub repository"), this repository audit and documentation update cleanly removed all unconfirmed academic supervision, mentorship, and oversight attribution associated with his name from all current repository files.

In accordance with strict research integrity and publication guidelines:
- All supervisory attribution lines and fields have been removed cleanly from document headers, templates, metadata, docstrings, and report narratives.
- No substitute supervisor was introduced; no alternative academic endorsement was fabricated.
- The research candidate and primary author identity (**Nandan S**) is preserved across all documents.
- Historical Git commit history has been preserved untouched (no history rewriting or force-pushing).
- All empirical experimental findings, 5-seed metrics, mathematical formulations, model architecture parameters (CompactPathCNN, 36,425 parameters), and official dataset citations remain identical and frozen.

---

## 2. Inventory of Files Audited & Modified

A repository-wide audit was conducted across all Git-tracked files. A total of **75 files** were modified:

### A. Root Documentation & Configurations
1. `README.md` — Replaced `Academic Supervision: Dr. Sachin D N...` with `Author: Nandan S`.
2. `PROJECT_STATUS.md` — Removed `Faculty Mentor: Dr. Sachin D N...`.
3. `SYSTEM_COMPATIBILITY_AUDIT.md` — Removed `Faculty Mentor: Dr. Sachin D N...`.
4. `configs/experiment_protocol.yaml` — Removed mentor attribution comment line.
5. `experiments/configs/experiment_protocol.yaml` — Removed mentor attribution comment line.
6. `experiments/legacy_seed42_pre_batch_fix/configs/experiment_protocol.yaml` — Removed mentor attribution comment line.

### B. Experiment Metadata JSON Files
7. `experiments/seed_123_attempt_2/metadata.json` — Removed `"mentor"` attribution; attributed `"author": "Nandan S"`.
8. `experiments/seed_2026/metadata.json` — Removed `"mentor"` attribution; attributed `"author": "Nandan S"`.
9. `experiments/seed_3407/metadata.json` — Removed `"mentor"` attribution; attributed `"author": "Nandan S"`.
10. `experiments/seed_42_attempt_3/metadata.json` — Removed `"mentor"` attribution; attributed `"author": "Nandan S"`.
11. `experiments/seed_7777/metadata.json` — Removed `"mentor"` attribution; attributed `"author": "Nandan S"`.

### C. Submission Package & Templates
12. `scripts/package_submission_phase8.py` — Removed Dr. Sachin's name from Final Report Markdown header, HTML PDF author block, One-Page Summary Markdown header, HTML PDF subtitle, Reactive Optimizers review header, Source requirement notice header, and Phase 8.2 release report header.
13. `submission/PathMNIST_Federated_Learning_Final_Report.md` — Removed `Academic Supervision` header line.
14. `submission/PathMNIST_Federated_Learning_Final_Report.pdf` — Regenerated from clean HTML template (exactly 6 pages).
15. `submission/PathMNIST_Federated_Learning_One_Page_Summary.md` — Removed supervisory attribution from candidate header.
16. `submission/PathMNIST_Federated_Learning_One_Page_Summary.pdf` — Regenerated from clean HTML template (exactly 1 A4 page).
17. `submission/PHASE_8_FINAL_RELEASE_REPORT.md` — Removed `Academic Oversight` header line.
18. `submission/Reactive_Optimizers_Article_Summary.md` — Removed `Faculty Mentor` header line.
19. `submission/REACTIVE_OPTIMIZERS_SOURCE_REQUIRED.md` — Removed `Academic Supervision` header line.

### D. Phase Audit & Technical Reports
20. `reports/PHASE_0_COMMANDS_AND_VERIFICATION.md`
21. `reports/PHASE_0_DATASET_AND_PARTITIONING_AUDIT.md`
22. `reports/PHASE_0_EVALUATION_AND_DIAGNOSTICS_AUDIT.md`
23. `reports/PHASE_0_FEDERATED_ALGORITHMS_AUDIT.md`
24. `reports/PHASE_0_FINAL_REVIEW_REPORT.md`
25. `reports/PHASE_0_IMPLEMENTATION_BLUEPRINT.md`
26. `reports/PHASE_0_REPOSITORY_AUDIT.md`
27. `reports/PHASE_0_RESOURCE_FEASIBILITY_REPORT.md`
28. `reports/PHASE_0_RISK_REGISTER.md`
29. `reports/PHASE_1_ACCEPTANCE_CRITERIA.md`
30. `reports/PHASE_1_ARCHITECTURE_SPECIFICATION.md`
31. `reports/PHASE_1_DECISIONS_AND_OPEN_QUESTIONS.md`
32. `reports/PHASE_1_EXPERIMENT_DESIGN.md`
33. `reports/PHASE_1_FINAL_CONSISTENCY_AUDIT.md`
34. `reports/PHASE_1_FINAL_TECHNICAL_CORRECTION_REPORT.md`
35. `reports/PHASE_1_METRIC_SPECIFICATION.md`
36. `reports/PHASE_1_REPRODUCIBILITY_PLAN.md`
37. `reports/PHASE_1_RESEARCH_PROTOCOL.md`
38. `reports/PHASE_2_EXECUTION_REPORT.md`
39. `reports/PHASE_3_EXPERIMENT_EXECUTION_REPORT.md`
40. `reports/PHASE_4_SCIENTIFIC_HARDENING_REPORT.md`
41. `reports/PHASE_5_PUBLICATION_AND_VERIFICATION_REPORT.md`
42. `reports/PHASE_6B_ENVIRONMENT_READINESS_REPORT.md`
43. `reports/PHASE_6C1_METRICS_RECONCILIATION.md`
44. `reports/PHASE_6C_SMOKE_EXPERIMENT_REPORT.md`
45. `reports/PHASE_6D1_REPRODUCIBILITY_AUDIT.md`
46. `reports/PHASE_6D_SEED42_FULL_EXPERIMENT_REPORT.md`
47. `reports/PHASE_7_MULTISEED_EXPERIMENT_REPORT.md`

### E. Python Source Modules & Unit Tests
48. `research/core/client.py`
49. `research/core/fedavg.py`
50. `research/core/fedprox.py`
51. `research/core/server.py`
52. `research/data/noise_injector.py`
53. `research/data/partitioner.py`
54. `research/data/pathmnist_loader.py`
55. `research/data/validation_splitter.py`
56. `research/diagnostics/memory_monitor.py`
57. `research/diagnostics/tracking.py`
58. `research/diagnostics/update_metrics.py`
59. `research/models/compact_cnn.py`
60. `research/optimizers/perturbed_sgd.py`
61. `research/tests/test_aggregation.py`
62. `research/tests/test_minibatch.py`
63. `research/tests/test_model.py`
64. `research/tests/test_noise.py`
65. `research/tests/test_optimizer.py`
66. `research/tests/test_partition.py`
67. `research/tests/test_reproducibility.py`
68. `research/tests/test_splits.py`
69. `scripts/independent_verification.py`
70. `scripts/initialize_experiment.py`
71. `scripts/run_multiseed.py`
72. `scripts/run_phase3.py`
73. `scripts/run_pilot.py`
74. `scripts/run_smoke_test.py`
75. `scripts/sync_to_wandb.py`

---

## 3. PDF Regeneration & Inspection

Both submission PDFs were regenerated via `scripts/package_submission_phase8.py` using Microsoft Edge headless rendering (`--headless --disable-gpu --run-all-compositor-stages-before-draw --print-to-pdf`):

| Deliverable | Target Path | Page Count | File Size | Supervision Mentions | Author Confirmed |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Complete Research Report** | `submission/PathMNIST_Federated_Learning_Final_Report.pdf` | **6 Pages** | 1,508,527 bytes | **0** | **Nandan S** |
| **Executive One-Page Summary** | `submission/PathMNIST_Federated_Learning_One_Page_Summary.pdf` | **1 Page (A4)** | 147,846 bytes | **0** | **Nandan S** |

### Automated Text Extraction Audit (via `pypdf`):
- `PathMNIST_Federated_Learning_Final_Report.pdf`:
  - `sachin`: 0 occurrences
  - `supervis`: 0 occurrences
  - `mentor`: 0 occurrences
  - `academic oversight`: 0 occurrences
  - `Nandan S`: 1 occurrence
  - Local Windows file paths (`C:\`, `D:\`): 0 occurrences
- `PathMNIST_Federated_Learning_One_Page_Summary.pdf`:
  - `sachin`: 0 occurrences
  - `supervis`: 0 occurrences
  - `mentor`: 0 occurrences
  - `academic oversight`: 0 occurrences
  - `Nandan S`: 1 occurrence
  - Local Windows file paths (`C:\`, `D:\`): 0 occurrences

---

## 4. Verification Checksums & Quality Gates

| Verification Check | Tool / Command | Outcome | Status |
| :--- | :--- | :--- | :---: |
| **Case-Insensitive "Sachin" Search** | `git grep -i -n "Sachin"` | 0 matches | **PASS** |
| **"Academic Supervision" Search** | `git grep -i -n "Academic Supervision"` | 0 matches | **PASS** |
| **"Faculty Mentor" Search** | `git grep -i -n "Faculty Mentor"` | 0 matches | **PASS** |
| **"Supervised by" Search** | `git grep -i -n "Supervised by"` | 0 matches | **PASS** |
| **"Academic Oversight" Search** | `git grep -i -n "Academic Oversight"` | 0 matches | **PASS** |
| **Generic "Mentor" Search** | `git grep -i -n "mentor"` | 0 matches | **PASS** |
| **Git Whitespace & Formatting** | `git diff --check` | 0 errors, 0 warnings | **PASS** |
| **Unit Test Suite Integrity** | Logic Audit | Only non-functional docstrings & header text edited | **PASS** |
| **Working Tree Cleanliness** | `git status --short` | Clean post-commit | **PASS** |

---

## 5. Conclusion

All unconfirmed academic supervision and mentorship references have been completely and systematically removed from the current repository documents, templates, metadata, and generated PDFs. The repository remains fully reproducible, mathematically sound, and ready for IndiaAI Fellowship 2026–2027 screening review under candidate Nandan S.
