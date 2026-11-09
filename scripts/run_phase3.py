"""
Phase 3 Master Experiment Runner: Full Federated Learning Experiments.
Executes Primary Run 1 (FedAvg) and Primary Run 2 (FedProx, mu=0.01) on non-IID PathMNIST with 40% label noise on Client 5.
Evaluates final held-out test split, generates FIG-01 to FIG-07, and produces final comparison artifacts.
"""

import sys
import os
import json
import time
import hashlib
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from research.data.pathmnist_loader import get_training_data, get_validation_data, get_test_data
from research.data.noise_injector import inject_symmetric_label_noise
from research.models.compact_cnn import CompactPathCNN, preprocess_batch, compute_canonical_model_hash
from research.core.fedavg import run_fedavg
from research.core.fedprox import run_fedprox


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@torch.no_grad()
def evaluate_on_test_split(model: CompactPathCNN, test_images: np.ndarray, test_labels: np.ndarray, batch_size: int = 128):
    """
    Evaluates a trained model once on the official held-out test split (7,180 samples).
    Zero test-set tuning or exposure during training.
    """
    model.eval()
    total_test = len(test_labels)
    total_correct = 0
    total_loss = 0.0
    num_batches = int(np.ceil(total_test / batch_size))
    
    for b in range(num_batches):
        b_idx = slice(b * batch_size, min((b + 1) * batch_size, total_test))
        b_imgs = test_images[b_idx]
        b_lbls = test_labels[b_idx]
        
        x, y = preprocess_batch(b_imgs, b_lbls)
        logits = model(x)
        losses = nn.functional.cross_entropy(logits, y, reduction="sum")
        preds = torch.argmax(logits, dim=1)
        
        total_loss += float(losses.item())
        total_correct += int((preds == y).sum().item())
        
    return {
        "test_accuracy": float(round(total_correct / total_test, 5)),
        "test_loss": float(round(total_loss / total_test, 5)),
        "test_correct": total_correct,
        "test_total": total_test
    }


def generate_research_figures(fedavg_records, fedprox_records, output_dir: str):
    """
    Generates publication-quality research figures (FIG-01 through FIG-07) from experiment logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    rounds_avg = [r["round"] for r in fedavg_records]
    rounds_prox = [r["round"] for r in fedprox_records]
    
    # FIG-01: Global Validation Accuracy vs Communication Round
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(rounds_avg, [r["val/global_accuracy"] * 100 for r in fedavg_records], "b-o", linewidth=2, label="FedAvg Baseline")
    plt.plot(rounds_prox, [r["val/global_accuracy"] * 100 for r in fedprox_records], "r-s", linewidth=2, label="FedProx (mu=0.01)")
    plt.title("FIG-01: Global Validation Accuracy vs Communication Round (PathMNIST Non-IID)", fontsize=12, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Global Validation Accuracy (%)", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FIG-01_global_val_accuracy.png"))
    plt.close()
    
    # FIG-02: Global Validation Loss vs Communication Round
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(rounds_avg, [r["val/global_loss"] for r in fedavg_records], "b-o", linewidth=2, label="FedAvg Baseline")
    plt.plot(rounds_prox, [r["val/global_loss"] for r in fedprox_records], "r-s", linewidth=2, label="FedProx (mu=0.01)")
    plt.title("FIG-02: Global Validation Loss vs Communication Round (PathMNIST Non-IID)", fontsize=12, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Global Validation Loss (Cross-Entropy)", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FIG-02_global_val_loss.png"))
    plt.close()
    
    # FIG-03: Per-Client Validation Accuracy vs Communication Round
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300, sharey=True)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    for c in range(1, 6):
        lbl = f"Client {c} (Noisy 40%)" if c == 5 else f"Client {c} (Clean)"
        lw = 2.5 if c == 5 else 1.5
        ls = "--" if c == 5 else "-"
        axes[0].plot(rounds_avg, [r[f"val/client_{c}_accuracy"] * 100 for r in fedavg_records], color=colors[c-1], linewidth=lw, linestyle=ls, label=lbl)
        axes[1].plot(rounds_prox, [r[f"val/client_{c}_accuracy"] * 100 for r in fedprox_records], color=colors[c-1], linewidth=lw, linestyle=ls, label=lbl)
    axes[0].set_title("FedAvg Per-Client Validation Accuracy", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Communication Round", fontsize=10)
    axes[0].set_ylabel("Validation Accuracy (%)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[1].set_title("FedProx (mu=0.01) Per-Client Validation Accuracy", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Communication Round", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    fig.suptitle("FIG-03: Per-Client Validation Accuracy Evolution Across Clean & Noisy Shards", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FIG-03_per_client_accuracy.png"))
    plt.close()
    
    # FIG-04: Per-Client Validation Loss vs Communication Round
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300, sharey=True)
    for c in range(1, 6):
        lbl = f"Client {c} (Noisy 40%)" if c == 5 else f"Client {c} (Clean)"
        lw = 2.5 if c == 5 else 1.5
        ls = "--" if c == 5 else "-"
        axes[0].plot(rounds_avg, [r[f"val/client_{c}_loss"] for r in fedavg_records], color=colors[c-1], linewidth=lw, linestyle=ls, label=lbl)
        axes[1].plot(rounds_prox, [r[f"val/client_{c}_loss"] for r in fedprox_records], color=colors[c-1], linewidth=lw, linestyle=ls, label=lbl)
    axes[0].set_title("FedAvg Per-Client Validation Loss", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Communication Round", fontsize=10)
    axes[0].set_ylabel("Validation Loss", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[1].set_title("FedProx (mu=0.01) Per-Client Validation Loss", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Communication Round", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    fig.suptitle("FIG-04: Per-Client Validation Loss Evolution Across Clean & Noisy Shards", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FIG-04_per_client_loss.png"))
    plt.close()
    
    # FIG-05: Client Update L2 Norms vs Communication Round
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300, sharey=True)
    for c in range(1, 6):
        lbl = f"Client {c} (Noisy 40%)" if c == 5 else f"Client {c} (Clean)"
        lw = 2.5 if c == 5 else 1.5
        ls = "--" if c == 5 else "-"
        axes[0].plot(rounds_avg, [r[f"diagnostics/client_{c}_update_l2_norm"] for r in fedavg_records], color=colors[c-1], linewidth=lw, linestyle=ls, label=lbl)
        axes[1].plot(rounds_prox, [r[f"diagnostics/client_{c}_update_l2_norm"] for r in fedprox_records], color=colors[c-1], linewidth=lw, linestyle=ls, label=lbl)
    axes[0].set_title("FedAvg Client Update L2 Norms ||Delta_{k,t}||_2", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Communication Round", fontsize=10)
    axes[0].set_ylabel("Update L2 Norm", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[1].set_title("FedProx (mu=0.01) Client Update L2 Norms ||Delta_{k,t}||_2", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Communication Round", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    fig.suptitle("FIG-05: Client Parameter Update Displacement L2 Norms", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FIG-05_update_l2_norms.png"))
    plt.close()
    
    # FIG-06: Pairwise Client Update Cosine Similarities vs Round
    plt.figure(figsize=(10, 6), dpi=300)
    pair_keys = [k for k in fedavg_records[0].keys() if k.startswith("pairwise_cosine/")]
    for k in pair_keys:
        is_noisy = "c5" in k
        alpha = 0.9 if is_noisy else 0.4
        lw = 2.0 if is_noisy else 1.0
        ls = "-." if is_noisy else "-"
        lbl = f"FedAvg {k.replace('pairwise_cosine/', '')}"
        plt.plot(rounds_avg, [r[k] for r in fedavg_records], linewidth=lw, linestyle=ls, alpha=alpha, label=lbl)
    plt.title("FIG-06: Pairwise Client Update Cosine Similarities Across Communication Rounds", fontsize=12, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Cosine Similarity", fontsize=11)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FIG-06_pairwise_cosine_similarities.png"))
    plt.close()
    
    # FIG-07: Clean-to-Noisy and Clean-to-Clean Update Alignment
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(rounds_avg, [r["diagnostics/mean_clean_noisy_cosine"] for r in fedavg_records], "r-o", linewidth=2, label="FedAvg: Clean-to-Noisy Alignment")
    plt.plot(rounds_prox, [r["diagnostics/mean_clean_noisy_cosine"] for r in fedprox_records], "r--s", linewidth=2, label="FedProx: Clean-to-Noisy Alignment")
    plt.plot(rounds_avg, [r["diagnostics/mean_clean_clean_cosine"] for r in fedavg_records], "b-o", linewidth=2, label="FedAvg: Clean-to-Clean Consensus")
    plt.plot(rounds_prox, [r["diagnostics/mean_clean_clean_cosine"] for r in fedprox_records], "b--s", linewidth=2, label="FedProx: Clean-to-Clean Consensus")
    plt.title("FIG-07: Comparative Clean-to-Noisy Alignment & Clean Consensus (FedAvg vs FedProx)", fontsize=12, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Mean Cosine Similarity", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "FIG-07_clean_noisy_alignment.png"))
    plt.close()
    
    print(f"All 7 research figures successfully saved to {output_dir}")


def run_full_phase3_experiments(num_rounds: int = 30):
    print("======================================================================")
    print("      PHASE 3: FULL FEDERATED LEARNING EXPERIMENTAL EXECUTION        ")
    print("      Comparative Analysis: FedAvg vs FedProx (mu=0.01) on PathMNIST  ")
    print("======================================================================")
    
    # Hardware & PyTorch Configuration
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    
    # 1. Load Data
    print("\n1. Loading in-RAM uint8 PathMNIST datasets...")
    mmap_dir = os.path.join(PROJ_ROOT, "data", "pathmnist_mmap")
    train_images = np.load(os.path.join(mmap_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(mmap_dir, "train_labels.npy"))
    val_images = np.load(os.path.join(mmap_dir, "val_images.npy"))
    val_labels = np.load(os.path.join(mmap_dir, "val_labels.npy"))
    
    # 2. Load Manifests
    print("\n2. Loading verified experiment manifests...")
    partition_file = os.path.join(PROJ_ROOT, "experiments", "configs", "partition_manifest.json")
    val_file = os.path.join(PROJ_ROOT, "experiments", "configs", "validation_manifest.json")
    noise_file = os.path.join(PROJ_ROOT, "experiments", "configs", "noise_manifest.json")
    init_ckpt_path = os.path.join(PROJ_ROOT, "experiments", "checkpoints", "initial_model.pt")
    
    with open(partition_file, "r") as f:
        part_data = json.load(f)
    with open(val_file, "r") as f:
        val_data = json.load(f)
    with open(noise_file, "r") as f:
        noise_data = json.load(f)
        
    train_partitions = {int(k): np.array(v, dtype=np.int64) for k, v in part_data["client_indices"].items()}
    val_shards = {int(k): np.array(v, dtype=np.int64) for k, v in val_data["client_val_indices"].items()}
    
    # Inject exact label noise on Client 5
    client_5_indices = train_partitions[4]
    client_5_noisy_labels, noise_mask, _ = inject_symmetric_label_noise(
        client_train_indices=client_5_indices,
        train_labels=train_labels,
        target_client_human="Client 5",
        target_client_index=4,
        noise_rate=0.40,
        selection_seed=42,
        replacement_seed=1042,
        manifest_path=None
    )
    assert len(client_5_indices) == noise_data["total_client_samples"]
    assert np.sum(noise_mask) == noise_data["exact_corruptions_count"]
    client_noisy_labels = {4: client_5_noisy_labels}
    
    # Initial model state
    initial_state_dict = torch.load(init_ckpt_path)
    init_hash = compute_file_sha256(init_ckpt_path)
    print(f"   Initial Model Checkpoint SHA-256: {init_hash}")
    
    # ------------------------------------------------------------------
    # PRIMARY RUN 1: FedAvg
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print(f" PRIMARY RUN 1: FedAvg Baseline ({num_rounds} communication rounds)")
    print("==================================================================")
    model_fedavg = CompactPathCNN()
    model_fedavg.load_state_dict(initial_state_dict)
    
    fedavg_run_name = "fedavg_alpha0.3_noise40_seed42"
    fedavg_start_wall = time.perf_counter()
    
    res_fedavg = run_fedavg(
        global_model=model_fedavg,
        train_images=train_images,
        train_labels=train_labels,
        client_train_partitions=train_partitions,
        client_noisy_labels=client_noisy_labels,
        val_images=val_images,
        val_labels=val_labels,
        client_val_indices=val_shards,
        num_rounds=num_rounds,
        lr=0.01,
        batch_size=32,
        local_epochs=2,
        run_name=fedavg_run_name,
        resume=True,
        custom_jsonl_name="fedavg_metrics.jsonl"
    )
    fedavg_total_wall = time.perf_counter() - fedavg_start_wall
    print(f"FedAvg execution complete in {fedavg_total_wall / 60.0:.2f} minutes.")
    
    # Also sync to run_name jsonl
    log_dir = os.path.join(PROJ_ROOT, "experiments", "logs")
    src_avg_jsonl = os.path.join(log_dir, "fedavg_metrics.jsonl")
    tgt_avg_jsonl = os.path.join(log_dir, f"{fedavg_run_name}_metrics.jsonl")
    if os.path.exists(src_avg_jsonl) and src_avg_jsonl != tgt_avg_jsonl:
        import shutil
        shutil.copy2(src_avg_jsonl, tgt_avg_jsonl)
    
    # ------------------------------------------------------------------
    # PRIMARY RUN 2: FedProx (mu=0.01)
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print(f" PRIMARY RUN 2: FedProx (mu=0.01) ({num_rounds} communication rounds)")
    print("==================================================================")
    model_fedprox = CompactPathCNN()
    model_fedprox.load_state_dict(initial_state_dict)
    
    fedprox_run_name = "fedprox_mu0.01_alpha0.3_noise40_seed42"
    fedprox_start_wall = time.perf_counter()
    
    res_fedprox = run_fedprox(
        global_model=model_fedprox,
        train_images=train_images,
        train_labels=train_labels,
        client_train_partitions=train_partitions,
        client_noisy_labels=client_noisy_labels,
        val_images=val_images,
        val_labels=val_labels,
        client_val_indices=val_shards,
        num_rounds=num_rounds,
        lr=0.01,
        mu=0.01,
        batch_size=32,
        local_epochs=2,
        run_name=fedprox_run_name,
        resume=True,
        custom_jsonl_name="fedprox_metrics.jsonl"
    )
    fedprox_total_wall = time.perf_counter() - fedprox_start_wall
    print(f"FedProx execution complete in {fedprox_total_wall / 60.0:.2f} minutes.")
    
    src_prox_jsonl = os.path.join(log_dir, "fedprox_metrics.jsonl")
    tgt_prox_jsonl = os.path.join(log_dir, f"{fedprox_run_name}_metrics.jsonl")
    if os.path.exists(src_prox_jsonl) and src_prox_jsonl != tgt_prox_jsonl:
        import shutil
        shutil.copy2(src_prox_jsonl, tgt_prox_jsonl)
        
    # ------------------------------------------------------------------
    # HELD-OUT TEST EVALUATION (STRICT SINGLE PASS)
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print(" HELD-OUT TEST EVALUATION (OFFICIAL PATHMNIST TEST SPLIT: 7,180 SAMPLES)")
    print("==================================================================")
    test_images, test_labels = get_test_data()
    ckpt_dir = os.path.join(PROJ_ROOT, "experiments", "checkpoints")
    
    # 1. Evaluate FedAvg Best Checkpoint
    fedavg_best_ckpt = os.path.join(ckpt_dir, f"{fedavg_run_name}_best.pt")
    assert os.path.exists(fedavg_best_ckpt), f"Missing FedAvg best checkpoint: {fedavg_best_ckpt}"
    fedavg_best_hash = compute_file_sha256(fedavg_best_ckpt)
    
    eval_model_avg = CompactPathCNN()
    eval_model_avg.load_state_dict(torch.load(fedavg_best_ckpt))
    fedavg_test_results = evaluate_on_test_split(eval_model_avg, test_images, test_labels)
    
    # 2. Evaluate FedProx Best Checkpoint
    fedprox_best_ckpt = os.path.join(ckpt_dir, f"{fedprox_run_name}_best.pt")
    assert os.path.exists(fedprox_best_ckpt), f"Missing FedProx best checkpoint: {fedprox_best_ckpt}"
    fedprox_best_hash = compute_file_sha256(fedprox_best_ckpt)
    
    eval_model_prox = CompactPathCNN()
    eval_model_prox.load_state_dict(torch.load(fedprox_best_ckpt))
    fedprox_test_results = evaluate_on_test_split(eval_model_prox, test_images, test_labels)
    
    print(f"FedAvg Best Checkpoint (Round {res_fedavg['best_round']}): Test Acc = {fedavg_test_results['test_accuracy'] * 100:.2f}%, Test Loss = {fedavg_test_results['test_loss']:.4f}")
    print(f"FedProx Best Checkpoint (Round {res_fedprox['best_round']}): Test Acc = {fedprox_test_results['test_accuracy'] * 100:.2f}%, Test Loss = {fedprox_test_results['test_loss']:.4f}")
    
    # ------------------------------------------------------------------
    # LOAD RECORDS & GENERATE FIGURES
    # ------------------------------------------------------------------
    fedavg_records = []
    with open(src_avg_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fedavg_records.append(json.loads(line))
                
    fedprox_records = []
    with open(src_prox_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fedprox_records.append(json.loads(line))
                
    fig_dir = os.path.join(PROJ_ROOT, "experiments", "figures")
    generate_research_figures(fedavg_records, fedprox_records, fig_dir)
    
    # ------------------------------------------------------------------
    # COMPARATIVE SUMMARY ARTIFACTS
    # ------------------------------------------------------------------
    # Compute aggregate diagnostic metrics across all completed rounds
    avg_update_norm_fedavg = float(np.mean([
        np.mean([r[f"diagnostics/client_{c}_update_l2_norm"] for c in range(1, 6)])
        for r in fedavg_records
    ]))
    avg_update_norm_fedprox = float(np.mean([
        np.mean([r[f"diagnostics/client_{c}_update_l2_norm"] for c in range(1, 6)])
        for r in fedprox_records
    ]))
    
    mean_clean_noisy_fedavg = float(np.mean([r["diagnostics/mean_clean_noisy_cosine"] for r in fedavg_records]))
    mean_clean_noisy_fedprox = float(np.mean([r["diagnostics/mean_clean_noisy_cosine"] for r in fedprox_records]))
    
    mean_clean_clean_fedavg = float(np.mean([r["diagnostics/mean_clean_clean_cosine"] for r in fedavg_records]))
    mean_clean_clean_fedprox = float(np.mean([r["diagnostics/mean_clean_clean_cosine"] for r in fedprox_records]))
    
    mean_round_sec_fedavg = float(np.mean([r["round_duration_seconds"] for r in fedavg_records]))
    mean_round_sec_fedprox = float(np.mean([r["round_duration_seconds"] for r in fedprox_records]))
    
    peak_rss_fedavg = float(max(r["process_rss_mb"] for r in fedavg_records))
    peak_rss_fedprox = float(max(r["process_rss_mb"] for r in fedprox_records))
    
    final_comparison = {
        "study_metadata": {
            "dataset": "PathMNIST (MedMNIST v2)",
            "num_clients": 5,
            "partitioning": "Dirichlet (alpha=0.3, seed=42)",
            "label_noise": "Client 5 (40% symmetric multiclass, seeds 42/1042)",
            "completed_communication_rounds": num_rounds,
            "local_epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.01,
            "model_architecture": "CompactPathCNN (36,425 parameters)",
            "initial_model_hash": init_hash
        },
        "fedavg": {
            "run_name": fedavg_run_name,
            "best_validation_accuracy": res_fedavg["best_validation_accuracy"],
            "best_validation_loss": res_fedavg["best_validation_loss"],
            "best_round": res_fedavg["best_round"],
            "best_checkpoint_sha256": fedavg_best_hash,
            "final_test_accuracy": fedavg_test_results["test_accuracy"],
            "final_test_loss": fedavg_test_results["test_loss"],
            "mean_client_update_norm": avg_update_norm_fedavg,
            "mean_clean_noisy_alignment": mean_clean_noisy_fedavg,
            "mean_clean_clean_alignment": mean_clean_clean_fedavg,
            "mean_round_duration_sec": mean_round_sec_fedavg,
            "total_wall_clock_minutes": float(round(fedavg_total_wall / 60.0, 2)),
            "peak_process_rss_mb": peak_rss_fedavg
        },
        "fedprox": {
            "run_name": fedprox_run_name,
            "mu": 0.01,
            "best_validation_accuracy": res_fedprox["best_validation_accuracy"],
            "best_validation_loss": res_fedprox["best_validation_loss"],
            "best_round": res_fedprox["best_round"],
            "best_checkpoint_sha256": fedprox_best_hash,
            "final_test_accuracy": fedprox_test_results["test_accuracy"],
            "final_test_loss": fedprox_test_results["test_loss"],
            "mean_client_update_norm": avg_update_norm_fedprox,
            "mean_clean_noisy_alignment": mean_clean_noisy_fedprox,
            "mean_clean_clean_alignment": mean_clean_clean_fedprox,
            "mean_round_duration_sec": mean_round_sec_fedprox,
            "total_wall_clock_minutes": float(round(fedprox_total_wall / 60.0, 2)),
            "peak_process_rss_mb": peak_rss_fedprox
        }
    }
    
    results_dir = os.path.join(PROJ_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "final_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_comparison, f, indent=2)
        
    csv_path = os.path.join(results_dir, "final_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "FedAvg", "FedProx (mu=0.01)"])
        writer.writerow(["Best Validation Accuracy", f"{final_comparison['fedavg']['best_validation_accuracy']:.5f}", f"{final_comparison['fedprox']['best_validation_accuracy']:.5f}"])
        writer.writerow(["Best Validation Loss", f"{final_comparison['fedavg']['best_validation_loss']:.5f}", f"{final_comparison['fedprox']['best_validation_loss']:.5f}"])
        writer.writerow(["Selected Checkpoint Round", final_comparison['fedavg']['best_round'], final_comparison['fedprox']['best_round']])
        writer.writerow(["Final Test Accuracy", f"{final_comparison['fedavg']['final_test_accuracy']:.5f}", f"{final_comparison['fedprox']['final_test_accuracy']:.5f}"])
        writer.writerow(["Final Test Loss", f"{final_comparison['fedavg']['final_test_loss']:.5f}", f"{final_comparison['fedprox']['final_test_loss']:.5f}"])
        writer.writerow(["Mean Client Update Norm", f"{avg_update_norm_fedavg:.5f}", f"{avg_update_norm_fedprox:.5f}"])
        writer.writerow(["Mean Clean-to-Noisy Alignment", f"{mean_clean_noisy_fedavg:.5f}", f"{mean_clean_noisy_fedprox:.5f}"])
        writer.writerow(["Mean Clean-to-Clean Alignment", f"{mean_clean_clean_fedavg:.5f}", f"{mean_clean_clean_fedprox:.5f}"])
        writer.writerow(["Mean Round Duration (sec)", f"{mean_round_sec_fedavg:.2f}", f"{mean_round_sec_fedprox:.2f}"])
        writer.writerow(["Total Wall-Clock Time (min)", f"{final_comparison['fedavg']['total_wall_clock_minutes']:.2f}", f"{final_comparison['fedprox']['total_wall_clock_minutes']:.2f}"])
        writer.writerow(["Peak Process RSS (MB)", f"{peak_rss_fedavg:.2f}", f"{peak_rss_fedprox:.2f}"])
        
    print(f"Final comparison JSON saved to {json_path}")
    print(f"Final comparison CSV saved to {csv_path}")
    print("\n=== PHASE 3 EXPERIMENTAL EXECUTION COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    import csv
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_full_phase3_experiments(num_rounds=rounds)
