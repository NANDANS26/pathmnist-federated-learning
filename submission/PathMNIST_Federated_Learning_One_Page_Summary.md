# Executive Research Summary: FedAvg vs. FedProx on PathMNIST
**Research Study:** IndiaAI Fellowship 2026–2027 Screening Submission  
**Candidate:** Nandan S | **Supervision:** Dr. Sachin D N, Assistant Professor, CSE, IIIT Bhopal  
**Repository:** [github.com/NANDANS26/pathmnist-federated-learning](https://github.com/NANDANS26/pathmnist-federated-learning)  

---

### 1. Research Problem & Objective
Federated Learning (FL) enables decentralized training of diagnostic medical imaging models without centralizing patient biopsies. In digital histopathology, collaborative FL faces two simultaneous obstacles: severe inter-client class distribution skew (non-IID data) and local annotation noise from diagnostic ambiguity or tissue artifacts. We evaluate whether Federated Proximal Optimization (**FedProx**, $\mu = 0.01$), which penalizes local parameter deviation from the global model, improves test generalization over standard Federated Averaging (**FedAvg**) under extreme non-IID partitioning (Dirichlet $\alpha = 0.3$) and single-client $40\%$ symmetric label noise.

### 2. Experimental Setup & Protocol
- **Dataset:** PathMNIST ($89,996$ training, $10,004$ validation, $7,180$ quarantined held-out test patches, 9 colorectal tissue classes).
- **Federated Topology:** 5 clients, Dirichlet $\alpha = 0.3$. Client 5 trained on exactly $40.00\%$ symmetric label noise.
- **Model Architecture:** **CompactPathCNN** (2 convolutional layers, Adaptive Average Pooling, 2 fully-connected layers, **36,425 trainable parameters**).
- **Protocol:** 5 canonical random seeds (42, 123, 2026, 3407, 7777). 30 communication rounds per algorithm (300 rounds total), local epochs $E = 2$, batch size $B = 32$, learning rate $\eta = 0.01$, executed on 2 PyTorch CPU threads (AMD Ryzen 5 5600H).

### 3. Quantitative Findings (5-Seed Paired Benchmark)

| Benchmark Metric | FedAvg (Mean $\pm$ Std) | FedProx ($\mu = 0.01$) | Paired Difference ($\Delta$) | Paired $t$-test ($df=4$) | $p$-value | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-Out Test Accuracy** | **$68.880\% \pm 9.475\%$** | **$68.279\% \pm 8.683\%$** | **$-0.602 \pm 1.685\text{ pp}$** | $t = -0.7983$ | $p = 0.46938$ | $[-2.694\text{ pp}, +1.491\text{ pp}]$ |
| **Test Macro-F1 Score** | **$0.6008 \pm 0.0931$** | **$0.5934 \pm 0.0950$** | **$-0.0075 \pm 0.0105$** | $t = -1.5908$ | $p = 0.18686$ | $[-0.0205, +0.0056]$ |
| **Test Balanced Accuracy** | **$63.176\% \pm 8.016\%$** | **$62.902\% \pm 8.019\%$** | **$-0.273 \pm 1.344\text{ pp}$** | $t = -0.4546$ | $p = 0.67294$ | $[-1.942\text{ pp}, +1.395\text{ pp}]$ |
| **Peak Validation Accuracy** | $65.986\% \pm 4.305\%$ | $65.466\% \pm 4.420\%$ | **$-0.520 \pm 0.166\text{ pp}$** | $t = -6.9961$ | $p = 0.00219$ | $[-0.726\text{ pp}, -0.314\text{ pp}]$ |
| **Total Wall-Clock Time** | $214.3\text{ min}$ ($42.9\text{ m/seed}$) | $257.4\text{ min}$ ($51.5\text{ m/seed}$) | $+8.62 \pm 31.39\text{ min}$ | — | — | Total Compute: $471.7\text{ min}$ |

### 4. Statistical Interpretation & Core Conclusion
- **Statistical Interpretation:** **No statistically significant difference in test performance was detected between FedAvg and FedProx across the five random seeds ($p = 0.469$).** Because five seeds provide limited statistical power and no predefined equivalence testing was performed, this lack of statistical significance does not prove equivalence.
- **Why FedProx Does Not Neutralize Noise:** The proximal penalty $\frac{\mu}{2}\|\mathbf{w}-\mathbf{w}^t\|^2$ penalizes the magnitude of parameter drift but cannot rectify corrupted gradient directions induced by noisy annotations. Client 5's corrupted updates still contaminate global aggregation proportionally to its sample weight.
- **Main Conclusion:** Generic proximal regularization is insufficient to defend federated medical models against corrupted local annotators. Robust federated histopathology requires explicit noise filtering, robust loss functions (e.g., Generalized Cross-Entropy), or dynamic server-side update reweighting based on cosine similarity.
