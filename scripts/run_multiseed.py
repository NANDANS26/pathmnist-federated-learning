"""
Multi-Seed Experiment Runner for Federated Learning.
Supports deterministic multi-seed runs (e.g., seeds 42, 123, 2026, 3407, 7777).
Generates isolated partition manifests, noise manifests, and initial checkpoints per seed.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import sys
import os
import json
import time
import torch
import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from research.data.pathmnist_loader import get_training_data, get_validation_data, get_test_data
from research.data.partitioner import generate_dirichlet_partitions
from research.data.validation_splitter import generate_client_validation_shards
from research.data.noise_injector import inject_symmetric_label_noise
from research.models.compact_cnn import CompactPathCNN, compute_canonical_model_hash
from research.core.fedavg import run_fedavg
from research.core.fedprox import run_fedprox
from scripts.independent_verification import evaluate_checkpoint_on_test, compute_file_sha256


def execute_seed_experiment(seed: int, num_rounds: int = 30):
    print(f"\n==================================================================")
    print(f"      EXECUTING PAIRED SEED EXPERIMENT: SEED {seed} ({num_rounds} ROUNDS)     ")
    print(f"==================================================================")
    
    config_dir = os.path.join(PROJ_ROOT, "experiments", "configs", f"seed_{seed}")
    ckpt_dir = os.path.join(PROJ_ROOT, "experiments", "checkpoints", f"seed_{seed}")
    log_dir = os.path.join(PROJ_ROOT, "experiments", "logs", f"seed_{seed}")
    results_dir = os.path.join(PROJ_ROOT, "experiments", "results")
    
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Data
    mmap_dir = os.path.join(PROJ_ROOT, "data", "pathmnist_mmap")
    train_images = np.load(os.path.join(mmap_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(mmap_dir, "train_labels.npy"))
    val_images = np.load(os.path.join(mmap_dir, "val_images.npy"))
    val_labels = np.load(os.path.join(mmap_dir, "val_labels.npy"))
    test_images, test_labels = get_test_data()
    
    # 2. Partition Data for this Seed
    part_manifest_path = os.path.join(config_dir, "partition_manifest.json")
    val_manifest_path = os.path.join(config_dir, "validation_manifest.json")
    noise_manifest_path = os.path.join(config_dir, "noise_manifest.json")
    
    train_partitions = generate_dirichlet_partitions(
        labels=train_labels,
        num_clients=5,
        alpha=0.3,
        partition_seed=seed,
        manifest_path=part_manifest_path
    )
    
    val_shards = generate_client_validation_shards(
        val_labels=val_labels,
        train_partitions=train_partitions,
        train_labels=train_labels,
        num_clients=5,
        val_seed=seed,
        manifest_path=val_manifest_path
    )
    
    c5_noisy, noise_mask, _ = inject_symmetric_label_noise(
        client_train_indices=train_partitions[4],
        train_labels=train_labels,
        target_client_human="Client 5",
        target_client_index=4,
        noise_rate=0.40,
        selection_seed=seed,
        replacement_seed=1000 + seed,
        manifest_path=noise_manifest_path
    )
    client_noisy_labels = {4: c5_noisy}
    
    # 3. Initialize Shared Canonical Model for this Seed
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    
    init_model = CompactPathCNN()
    init_hash = compute_canonical_model_hash(init_model)
    init_ckpt_path = os.path.join(ckpt_dir, "initial_model.pt")
    torch.save(init_model.state_dict(), init_ckpt_path)
    print(f"Initial model saved for Seed {seed}: SHA-256 = {init_hash}")
    
    # 4. Primary Run 1: FedAvg
    fedavg_run_name = f"fedavg_alpha0.3_noise40_seed{seed}"
    model_fedavg = CompactPathCNN()
    model_fedavg.load_state_dict(torch.load(init_ckpt_path))
    
    t0_avg = time.perf_counter()
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
        checkpoint_dir=ckpt_dir,
        resume=True
    )
    dur_avg = time.perf_counter() - t0_avg
    
    # 5. Primary Run 2: FedProx (mu=0.01)
    fedprox_run_name = f"fedprox_mu0.01_alpha0.3_noise40_seed{seed}"
    model_fedprox = CompactPathCNN()
    model_fedprox.load_state_dict(torch.load(init_ckpt_path))
    
    t0_prox = time.perf_counter()
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
        checkpoint_dir=ckpt_dir,
        resume=True
    )
    dur_prox = time.perf_counter() - t0_prox
    
    # 6. Evaluate on Test Split
    fedavg_best_ckpt = os.path.join(ckpt_dir, f"{fedavg_run_name}_best.pt")
    fedprox_best_ckpt = os.path.join(ckpt_dir, f"{fedprox_run_name}_best.pt")
    
    pred_avg_path = os.path.join(results_dir, f"fedavg_seed{seed}_test_predictions.jsonl")
    pred_prox_path = os.path.join(results_dir, f"fedprox_seed{seed}_test_predictions.jsonl")
    
    test_metrics_avg = evaluate_checkpoint_on_test(fedavg_best_ckpt, test_images, test_labels, pred_avg_path)
    test_metrics_prox = evaluate_checkpoint_on_test(fedprox_best_ckpt, test_images, test_labels, pred_prox_path)
    
    seed_summary = {
        "seed": seed,
        "completed_rounds": num_rounds,
        "fedavg": {
            "run_name": fedavg_run_name,
            "best_validation_accuracy": res_fedavg["best_validation_accuracy"],
            "best_validation_loss": res_fedavg["best_validation_loss"],
            "best_round": res_fedavg["best_round"],
            "test_accuracy": test_metrics_avg["overall_accuracy"],
            "test_macro_f1": test_metrics_avg["macro_f1"],
            "test_balanced_acc": test_metrics_avg["balanced_accuracy"],
            "test_loss": test_metrics_avg["test_loss"],
            "wall_clock_minutes": float(round(dur_avg / 60.0, 2))
        },
        "fedprox": {
            "run_name": fedprox_run_name,
            "mu": 0.01,
            "best_validation_accuracy": res_fedprox["best_validation_accuracy"],
            "best_validation_loss": res_fedprox["best_validation_loss"],
            "best_round": res_fedprox["best_round"],
            "test_accuracy": test_metrics_prox["overall_accuracy"],
            "test_macro_f1": test_metrics_prox["macro_f1"],
            "test_balanced_acc": test_metrics_prox["balanced_accuracy"],
            "test_loss": test_metrics_prox["test_loss"],
            "wall_clock_minutes": float(round(dur_prox / 60.0, 2))
        }
    }
    
    summary_path = os.path.join(results_dir, f"seed_{seed}_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(seed_summary, f, indent=2)
        
    print(f"\nSeed {seed} complete!")
    print(f"FedAvg  (Round {res_fedavg['best_round']}): Test Acc = {test_metrics_avg['overall_accuracy']*100:.2f}%, Test Macro-F1 = {test_metrics_avg['macro_f1']:.4f}")
    print(f"FedProx (Round {res_fedprox['best_round']}): Test Acc = {test_metrics_prox['overall_accuracy']*100:.2f}%, Test Macro-F1 = {test_metrics_prox['macro_f1']:.4f}")
    return seed_summary


if __name__ == "__main__":
    target_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 123
    target_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    execute_seed_experiment(seed=target_seed, num_rounds=target_rounds)
