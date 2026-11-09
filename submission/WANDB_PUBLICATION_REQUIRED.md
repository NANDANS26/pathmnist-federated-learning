# Weights & Biases (W&B) Publication Guide & Status

**Study:** PathMNIST Federated Learning (FedAvg vs. FedProx)  
**Submission:** IndiaAI Fellowship 2026–2027 Screening  
**Status:** **AWAITING USER AUTHENTICATION** (API Key Not Configured Locally)  
**Date:** November 9, 2026  

---

## 1. Authentication Status

During Phase 8 pre-flight verification, the local environment was queried for active Weights & Biases authentication credentials:
```
wandb.errors.errors.UsageError: No API key configured. Use `wandb login` to log in.
```
In strict compliance with Phase 8 Constraint #0 (*"Do not expose credentials, API keys, tokens, or private environment variables in files, logs, terminal output, or Git commits. If credentials are missing, do not ask the user to paste a token into the chat and do not fabricate a W&B URL"*), automated external publishing is held until local authentication is executed by the user.

All experimental metrics, checkpoints, and summaries have been preserved in local immutable storage and are 100% ready for one-command synchronization.

---

## 2. One-Command Publication Procedure

To publish the completed 5-seed benchmark to your personal or academic Weights & Biases workspace, perform the following two steps in your local terminal:

### Step 1: Authenticate with your W&B Account
Open a local PowerShell prompt in `pathmnist-federated-learning-audit` and run:
```powershell
.venv\Scripts\wandb.exe login
```
*(Paste your private W&B API key when prompted in your private terminal; do not share or paste this key into any shared chat).*

### Step 2: Run the Non-Destructive Publishing Script
Execute the dedicated publishing script (which reads the existing saved metrics without re-running any training):
```powershell
.venv\Scripts\python.exe scripts/publish_to_wandb.py --project "pathmnist-fedavg-vs-fedprox" --entity "<your-wandb-username>"
```

---

## 3. What Will Be Synchronized to W&B
The publishing script synchronizes the frozen source-of-truth artifacts:
1. **Per-Seed Round Logs:** All 300 executed rounds across Seeds 42, 123, 2026, 3407, and 7777 (Global Val Acc, Val Loss, Per-Client Accuracies, Client Update L2 Norms, Pairwise Cosine Similarities).
2. **Post-Training Test Metrics:** Final Test Accuracy, Macro-F1, Balanced Accuracy, and Test Loss for FedAvg and FedProx.
3. **Scientific Figures:** Publication-ready visual artifacts (`FIG-P7-01` through `FIG-P7-06`).
4. **Reproducibility Metadata:** Git commit hash (`64ce6da727c8f039b069ff030759e33f74abd69f`), Dirichlet concentration $\alpha=0.3$, label noise rate $40\%$, and manifest checksums.
