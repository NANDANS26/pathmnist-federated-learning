# PHASE 1: RESEARCH PROTOCOL (CORRECTED)
## Exploratory Empirical Comparison of Federated Averaging (FedAvg) and Federated Proximal (FedProx) on Non-IID PathMNIST with 40% Client-Specific Label Noise

**Project:** Comparative Experimental Analysis of FedAvg and FedProx on Non-IID PathMNIST with Client-Specific Label Noise  
**Research Supervision:** Dr. Sachin D N, Assistant Professor, Department of Computer Science and Engineering, IIIT Bhopal  
**Fellowship Program:** IndiaAI Fellowship 2026–27 (Technical Screening Submission)  
**Author:** Senior Federated Learning Research Scientist & ML Systems Architect  
**Document Status:** Formal Research Protocol (`Phase 1`)  
**Date:** November 7, 2026  

---

## 1. Executive Research Scope

Federated Learning (FL) enables collaborative model training across decentralized clients while preserving data locality. However, federated optimization faces two primary obstacles in real-world biomedical and clinical deployments:
1. **Statistical Heterogeneity (Non-IID Data):** Disparate disease prevalences, patient demographics, and staining protocols across clinical centres yield non-identical client data distributions ($P_i(x, y) \neq P_j(x, y)$). This causes local model updates to drift toward client-specific local optima, degrading the convergence and accuracy of standard parameter averaging (**Client Drift**).
2. **Systemic Data Quality Degradation (Label Noise):** In histopathological image classification, annotation errors frequently arise from inter-observer diagnostic discordance, ambiguous tissue morphology, or noisy crowdsourced labeling. In federated networks, if an individual client institution possesses substantial label noise, local gradient descent overfits to corrupted labels, transmitting divergent, destabilizing parameter updates to the server.

This study implements an **exploratory empirical comparison** of two foundational federated optimization algorithms:
- **Federated Averaging (FedAvg)** (McMahan et al., 2017)
- **Federated Proximal (FedProx)** (Li et al., 2020)

Experiments are benchmarked on **PathMNIST** (MedMNIST v2, 9 colorectal cancer histological tissue classes, $28 \times 28 \times 3$ RGB) partitioned across **five clients** under **Dirichlet non-IID class skew ($\alpha = 0.3$)**, with **Client 5 (zero-based index 4)** subjected to **40% symmetric label noise** in its local training partition.

---

## 2. Research Questions

### 2.1 Primary Research Question
> **How do FedAvg and FedProx differ in global validation performance, client-level accuracy, and client parameter update dynamics when trained across five non-IID PathMNIST clients where one client contains 40% symmetric training label noise?**

### 2.2 Secondary Research Questions
1. **RQ1 (Local Drift Mitigation):** Does the proximal regularization term $\frac{\mu}{2}\|w - w_t\|_2^2$ in FedProx noticeably constrain client update magnitudes ($\|\Delta_{k,t}\|_2$) relative to FedAvg, particularly for the noisy client?
2. **RQ2 (Client Update-Direction Alignment):** Does Client 5 (the noisy client) exhibit update-direction divergence (low or negative pairwise cosine similarity) relative to clean clients? Does FedProx improve directional consensus between clean and noisy updates?
3. **RQ3 (Global vs. Local Trade-Offs):** Does any observed global validation accuracy advantage in FedProx come at the expense of individual clean client performance, or does proximal regularization improve generalization uniformly across both clean and noisy participants?
4. **RQ4 (Dynamic Behavior Across Communication Rounds):** How do update norms and pairwise cosine similarities evolve over the communication trajectory?
5. **RQ5 (Proximal Coefficient Sensitivity):** How sensitive are convergence and update norm behavior to the magnitude of $\mu \in \{0.001, 0.01, 0.1\}$?

---

## 3. Exploratory Directional Hypotheses & Statistical Framing `[CORRECTION]`

### Methodological and Statistical Clarification
The mandatory primary comparison is conducted under a **controlled single-seed experimental configuration (`seed=42`)**.  
- *Crucial Statistical Principle:* Communication rounds within the same training trajectory are temporally dependent time series, not independent experimental replicates. Therefore, round-by-round observations cannot be subjected to classical independent-sample hypothesis tests (e.g., standard t-tests) to claim statistical significance across seeds.
- The project is formally framed as an **exploratory empirical comparison**. All comparative statements will describe observed single-seed trajectories without claiming statistical generalization across random seeds.

### Exploratory Directional Hypotheses

#### Hypothesis 1: Client Drift and Update Magnitude Contraction
- **Directional Proposition 1:** Under 40% label noise on Client 5 and $\alpha = 0.3$, the parameter update $L_2$ norm of Client 5 under FedProx ($\mu = 0.01$) will be visibly smaller across communication rounds than under FedAvg:
  $$\|\Delta_{5,t}^{\text{FedProx}}\|_2 < \|\Delta_{5,t}^{\text{FedAvg}}\|_2$$

#### Hypothesis 2: Client Update-Direction Alignment
- **Directional Proposition 2:** The mean pairwise cosine similarity between clean clients and Client 5 will be higher under FedProx than under FedAvg:
  $$\overline{\text{cos}}(\Delta_{\text{clean}}, \Delta_{5,t}^{\text{FedProx}}) > \overline{\text{cos}}(\Delta_{\text{clean}}, \Delta_{5,t}^{\text{FedAvg}})$$
  where clean-noisy alignment is mathematically defined as the unweighted mean of the four pairwise similarities:
  $$\overline{\text{cos}}(\Delta_{\text{clean}}, \Delta_{5,t}) = \frac{1}{4} \sum_{k=1}^4 \text{cos}(\Delta_{k,t}, \Delta_{5,t})$$

#### Hypothesis 3: Global Model Peak Validation Accuracy
- **Directional Proposition 3:** In the presence of Client 5's 40% label noise and $\alpha = 0.3$ non-IID partitioning, the observed FedProx ($\mu = 0.01$) global validation accuracy trajectory will reach a higher peak than FedAvg:
  $$\max_t \text{Acc}_{\text{val}}^{\text{FedProx}}(t) > \max_t \text{Acc}_{\text{val}}^{\text{FedAvg}}(t)$$

*Crucial Methodological Stance:* We do **not** assert that this result will occur; it is an exploratory directional proposition. Trajectory stability across communication rounds will be treated as a descriptive visual observation rather than a formal inferential test, as stability is not parameterized as a single scalar test statistic under the single-seed design. The empirical data collected during the experiment will determine the conclusion.

---

## 4. Experimental Scope and Boundaries

### 4.1 In-Scope Components `[VERIFIED / PROPOSED]`
- **Benchmark Dataset:** PathMNIST (official MedMNIST v2 release, colorectal histology, 9 classes).
- **Split Preservation:**
  - Official Training Split (89,996 images): Partitioned across 5 clients.
  - Official Validation Split (10,004 images): Partitioned into 5 clean disjoint validation shards for round-level global and per-client validation.
  - Official Test Split (7,180 images): Strictly held out from training and validation. Each algorithm's selected best-validation checkpoint is evaluated once on the official PathMNIST test set after training has finished.
- **Federated Setup:**
  - Client Count: $N = 5$.
  - Participation Ratio: $C = 1.0$ (Full client participation every round).
  - Data Partitioning: Non-IID Dirichlet distribution with $\alpha = 0.3$, fixed partition seed `seed=42`.
  - Noise Model: Exactly 40% symmetric label flipping on Client 5 (zero-based index 4) training labels only.
- **Algorithms:**
  - Primary Arm 1: FedAvg (Standard local SGD, normalized sample-weighted aggregation).
  - Primary Arm 2: FedProx (Perturbed Gradient Descent with $\mu = 0.01$, sample-weighted aggregation).
  - Optional Sensitivity Arms: FedProx with $\mu = 0.001$ and $\mu = 0.1$.
- **Telemetry & Diagnostics:**
  - Global validation accuracy and loss.
  - Client 1–5 validation accuracy and loss on clean validation shards.
  - Client update $L_2$ norms: $\|\Delta_{k,t}\|_2$.
  - Pairwise cosine similarity matrix (10 client pairs).
  - Cosine similarity to the sample-weighted mean update: $\text{cos}(\Delta_{k,t}, \bar{\Delta}_t)$.

### 4.2 Out-of-Scope Elements `[VERIFIED]`
- Inferential p-value claims across time-dependent communication rounds.
- Asynchronous aggregation schemes (all updates are synchronous).
- Dynamic client participation / random client dropouts (all 5 clients participate every round).
- Differential privacy, secure multiparty computation, or cryptographic gradient masking.
- Multi-GPU distributed runtime execution (host hardware is Windows 10 CPU-only).
- Modifying or altering the upstream PFLlib repository (`third_party/PFLlib` remains pristine).

---

## 5. Core Scientific Assumptions

1. **Stationary Client Partitions:** Client data partitions remain static across communication rounds; no dynamic data streaming or domain shift occurs during training.
2. **Symmetric Label Noise Model:** Noise on Client 5 is purely label-conditional (independent of image features), corrupting ground-truth histological classes with uniform probability across all other 8 classes.
3. **Clean Oracle Validation:** Validation and test splits represent clean histological ground truth, unaffected by any noise injection.
4. **Synchronous Server Coordination:** Communication rounds proceed in synchronized locks where the server waits for all 5 client updates before aggregating.
5. **Deterministic Baseline:** Controlling random seeds across data partitioning, model parameter initialization, and DataLoader batch generation enables identical starting conditions for FedAvg and FedProx.

---

## 6. Core Algorithm Comparison Matrix

| Component | Federated Averaging (FedAvg) | Federated Proximal (FedProx) |
|---|---|---|
| **Local Objective** | Empirical Cross-Entropy: $\min_w F_k(w)$ | Regularized Loss: $\min_w F_k(w) + \frac{\mu}{2}\|w - w_t\|_2^2$ |
| **Local Optimizer** | Standard SGD | Perturbed Gradient Descent |
| **Gradient Step** | $w \leftarrow w - \eta \nabla F_k(w)$ | $w \leftarrow w - \eta [\nabla F_k(w) + \mu(w - w_t)]$ |
| **Global Memory State** | Current global parameters $w_t$ | Current global parameters $w_t$ + cached snapshot per client |
| **Aggregation Rule** | $w_{t+1} = \sum_{k=1}^5 \frac{n_k}{N_{\text{train}}} w_{k,t+1}$ | $w_{t+1} = \sum_{k=1}^5 \frac{n_k}{N_{\text{train}}} w_{k,t+1}$ (Identical) |
| **Client Participation** | Full (5/5 clients every round) | Full (5/5 clients every round) |
| **Neural Architecture** | Identical CompactPathCNN (36,425 params) | Identical CompactPathCNN (36,425 params) |
| **Initial Weights $w_0$** | Identical seed-generated weights | Identical seed-generated weights |

---

## 7. Expected Analysis and Synthesis

The generated empirical records will be analyzed along three primary dimensions:
1. **Convergence Curves:** Plotting global validation accuracy and loss trajectories against communication rounds $t \in [1, R]$ to observe stability, rate of convergence, and peak accuracy.
2. **Robustness Against Corrupted Updates:** Evaluating whether the inclusion of Client 5 degrades clean client performance (Clients 1–4) more severely under FedAvg than FedProx.
3. **Update Geometry & Trajectory Diagnostics:** Examining how update $L_2$ norms and directional cosine similarities correlate with round-by-round validation accuracy, providing mathematical evidence for the observed trajectory differences.
