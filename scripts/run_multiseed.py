"""
Multi-Seed Experiment Runner and Orchestration Engine for Federated Learning.
Supports deterministic multi-seed runs (e.g., seeds 42, 123, 2026, 3407, 7777).
Generates isolated partition manifests, noise manifests, initial checkpoints,
and immutable run metadata per seed.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import sys
import os
import json
import time
import datetime
import subprocess
import argparse
import numpy as np
import torch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

# Configure PyTorch CPU thread pools once at process startup
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

from research.data.pathmnist_loader import get_test_data
from research.data.partitioner import generate_dirichlet_partitions
from research.data.validation_splitter import generate_client_validation_shards
from research.data.noise_injector import inject_symmetric_label_noise
from research.models.compact_cnn import CompactPathCNN, compute_canonical_model_hash
from research.core.fedavg import run_fedavg
from research.core.fedprox import run_fedprox
from scripts.independent_verification import evaluate_checkpoint_on_test, compute_file_sha256


def get_git_commit_hash() -> str:
    """Retrieves current Git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "c6159b0_untracked"


def execute_seed_experiment(seed: int, num_rounds: int = 30) -> dict:
    """
    Executes a complete paired FedAvg and FedProx experiment for a specific seed.
    Guarantees isolated directory storage, immutable metadata, zero checkpoint reuse,
    and strict post-training test evaluation.
    """
    t_seed_start_mono = time.perf_counter()
    iso_start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_hash = get_git_commit_hash()
    
    print(f"\n{'='*75}")
    print(f"   EXECUTING IMMUTABLE PAIRED EXPERIMENT: SEED {seed} ({num_rounds} ROUNDS)   ")
    print(f"   Git Commit: {git_hash} | Start UTC: {iso_start_time}")
    print(f"{'='*75}")
    
    seed_dir = os.path.join(PROJ_ROOT, "experiments", f"seed_{seed}")
    config_dir = os.path.join(seed_dir, "configs")
    ckpt_dir = os.path.join(seed_dir, "checkpoints")
    log_dir = os.path.join(seed_dir, "logs")
    results_dir = os.path.join(seed_dir, "results")
    
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Dataset Checksum & In-RAM Data Loading
    t_data_0 = time.perf_counter()
    raw_npz_path = os.path.join(PROJ_ROOT, "data", "pathmnist.npz")
    dataset_sha256 = compute_file_sha256(raw_npz_path) if os.path.exists(raw_npz_path) else "unknown"
    
    mmap_dir = os.path.join(PROJ_ROOT, "data", "pathmnist_mmap")
    train_images = np.load(os.path.join(mmap_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(mmap_dir, "train_labels.npy"))
    val_images = np.load(os.path.join(mmap_dir, "val_images.npy"))
    val_labels = np.load(os.path.join(mmap_dir, "val_labels.npy"))
    test_images, test_labels = get_test_data()
    
    # 2. Partitioning & Noise Injection for this Seed
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
    partition_manifest_sha256 = compute_file_sha256(part_manifest_path)
    
    val_shards = generate_client_validation_shards(
        val_labels=val_labels,
        train_partitions=train_partitions,
        train_labels=train_labels,
        num_clients=5,
        val_seed=seed,
        manifest_path=val_manifest_path
    )
    val_manifest_sha256 = compute_file_sha256(val_manifest_path)
    
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
    noise_manifest_sha256 = compute_file_sha256(noise_manifest_path)
    client_noisy_labels = {4: c5_noisy}
    
    data_prep_overhead_sec = float(round(time.perf_counter() - t_data_0, 3))
    print(f"Data partitioning & setup complete ({data_prep_overhead_sec}s). Partition SHA: {partition_manifest_sha256[:12]}...")
    
    # 3. Canonical Shared Initial Model Checkpoint
    torch.manual_seed(seed)
    init_model = CompactPathCNN(in_channels=3, num_classes=9)
    init_hash = compute_canonical_model_hash(init_model)
    init_ckpt_path = os.path.join(ckpt_dir, "initial_model.pt")
    torch.save(init_model.state_dict(), init_ckpt_path)
    init_model_sha256 = compute_file_sha256(init_ckpt_path)
    print(f"Initial model instantiated: SHA-256 = {init_model_sha256}")
    
    # 4. Primary Run 1: FedAvg
    fedavg_run_name = f"fedavg_alpha0.3_noise40_seed{seed}"
    model_fedavg = CompactPathCNN(in_channels=3, num_classes=9)
    model_fedavg.load_state_dict(torch.load(init_ckpt_path))
    
    t0_avg = time.perf_counter()
    iso_avg_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
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
        log_dir=log_dir,
        resume=False,
        seed=seed
    )
    dur_avg_sec = float(round(time.perf_counter() - t0_avg, 3))
    iso_avg_end = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # 5. Primary Run 2: FedProx (mu=0.01)
    fedprox_run_name = f"fedprox_mu0.01_alpha0.3_noise40_seed{seed}"
    model_fedprox = CompactPathCNN(in_channels=3, num_classes=9)
    model_fedprox.load_state_dict(torch.load(init_ckpt_path))
    
    t0_prox = time.perf_counter()
    iso_prox_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
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
        log_dir=log_dir,
        resume=False,
        seed=seed
    )
    dur_prox_sec = float(round(time.perf_counter() - t0_prox, 3))
    iso_prox_end = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # 6. Strict Post-Training Held-Out Test Evaluation
    fedavg_best_ckpt = os.path.join(ckpt_dir, f"{fedavg_run_name}_best.pt")
    fedprox_best_ckpt = os.path.join(ckpt_dir, f"{fedprox_run_name}_best.pt")
    
    pred_avg_path = os.path.join(results_dir, "fedavg_test_predictions.jsonl")
    pred_prox_path = os.path.join(results_dir, "fedprox_test_predictions.jsonl")
    cm_avg_path = os.path.join(results_dir, "fedavg_confusion_matrix.json")
    cm_prox_path = os.path.join(results_dir, "fedprox_confusion_matrix.json")
    
    test_metrics_avg = evaluate_checkpoint_on_test(fedavg_best_ckpt, test_images, test_labels, pred_avg_path)
    test_metrics_prox = evaluate_checkpoint_on_test(fedprox_best_ckpt, test_images, test_labels, pred_prox_path)
    
    with open(cm_avg_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics_avg["confusion_matrix"], f, indent=2)
    with open(cm_prox_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics_prox["confusion_matrix"], f, indent=2)
        
    # 7. Exact Timing Reconciliation
    avg_jsonl = os.path.join(log_dir, f"{fedavg_run_name}_metrics.jsonl")
    prox_jsonl = os.path.join(log_dir, f"{fedprox_run_name}_metrics.jsonl")
    
    def parse_round_times(jsonl_path):
        round_times = []
        train_times = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                round_times.append(rec.get("round_duration_seconds", 0.0))
                # Sum client train times
                c_times = [rec.get(f"train/client_{k}_train_time_sec", 0.0) for k in range(1, 6)]
                train_times.append(sum(c_times))
        return round_times, train_times

    avg_round_times, avg_train_times = parse_round_times(avg_jsonl)
    prox_round_times, prox_train_times = parse_round_times(prox_jsonl)
    
    timing_reconciliation = {
        "fedavg": {
            "wall_clock_total_seconds": dur_avg_sec,
            "wall_clock_minutes": float(round(dur_avg_sec / 60.0, 2)),
            "sum_round_durations_seconds": float(round(sum(avg_round_times), 3)),
            "average_round_duration_seconds": float(round(np.mean(avg_round_times), 3)),
            "sum_client_training_seconds": float(round(sum(avg_train_times), 3)),
            "eval_and_server_overhead_seconds": float(round(dur_avg_sec - sum(avg_train_times), 3))
        },
        "fedprox": {
            "wall_clock_total_seconds": dur_prox_sec,
            "wall_clock_minutes": float(round(dur_prox_sec / 60.0, 2)),
            "sum_round_durations_seconds": float(round(sum(prox_round_times), 3)),
            "average_round_duration_seconds": float(round(np.mean(prox_round_times), 3)),
            "sum_client_training_seconds": float(round(sum(prox_train_times), 3)),
            "eval_and_server_overhead_seconds": float(round(dur_prox_sec - sum(prox_train_times), 3))
        },
        "data_prep_overhead_seconds": data_prep_overhead_sec,
        "total_seed_wall_clock_seconds": float(round(time.perf_counter() - t_seed_start_mono, 3)),
        "total_seed_wall_clock_minutes": float(round((time.perf_counter() - t_seed_start_mono) / 60.0, 2))
    }
    
    # 8. Immutable Experiment Identity Metadata
    iso_end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    seed_metadata = {
        "study": "IndiaAI Fellowship 2026-27 Screening Submission",
        "mentor": "Dr. Sachin D N, IIIT Bhopal",
        "seed": seed,
        "git_commit_hash": git_hash,
        "model_architecture": "CompactPathCNN",
        "trainable_parameters": 36425,
        "hyperparameters": {
            "num_clients": 5,
            "dirichlet_alpha": 0.3,
            "noisy_client": "Client 5",
            "noisy_client_index": 4,
            "noise_rate": 0.40,
            "noise_type": "symmetric_deterministic_modulo_9",
            "communication_rounds": num_rounds,
            "local_epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.01,
            "fedprox_mu": 0.01
        },
        "checksums": {
            "dataset_sha256": dataset_sha256,
            "partition_manifest_sha256": partition_manifest_sha256,
            "validation_manifest_sha256": val_manifest_sha256,
            "noise_manifest_sha256": noise_manifest_sha256,
            "initial_model_checkpoint_sha256": init_model_sha256
        },
        "timestamps": {
            "seed_start_utc": iso_start_time,
            "fedavg_start_utc": iso_avg_start,
            "fedavg_end_utc": iso_avg_end,
            "fedprox_start_utc": iso_prox_start,
            "fedprox_end_utc": iso_prox_end,
            "seed_end_utc": iso_end_time
        },
        "timing_reconciliation": timing_reconciliation
    }
    
    meta_path = os.path.join(seed_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(seed_metadata, f, indent=2)
        
    seed_summary = {
        "seed": seed,
        "git_commit_hash": git_hash,
        "completed_rounds": num_rounds,
        "fedavg": {
            "run_name": fedavg_run_name,
            "best_validation_round": res_fedavg["best_round"],
            "best_validation_accuracy": res_fedavg["best_validation_accuracy"],
            "best_validation_loss": res_fedavg["best_validation_loss"],
            "test_accuracy": test_metrics_avg["overall_accuracy"],
            "test_macro_f1": test_metrics_avg["macro_f1"],
            "test_balanced_accuracy": test_metrics_avg["balanced_accuracy"],
            "test_loss": test_metrics_avg["test_loss"],
            "per_class": test_metrics_avg["per_class"],
            "wall_clock_minutes": timing_reconciliation["fedavg"]["wall_clock_minutes"]
        },
        "fedprox": {
            "run_name": fedprox_run_name,
            "mu": 0.01,
            "best_validation_round": res_fedprox["best_round"],
            "best_validation_accuracy": res_fedprox["best_validation_accuracy"],
            "best_validation_loss": res_fedprox["best_validation_loss"],
            "test_accuracy": test_metrics_prox["overall_accuracy"],
            "test_macro_f1": test_metrics_prox["macro_f1"],
            "test_balanced_accuracy": test_metrics_prox["balanced_accuracy"],
            "test_loss": test_metrics_prox["test_loss"],
            "per_class": test_metrics_prox["per_class"],
            "wall_clock_minutes": timing_reconciliation["fedprox"]["wall_clock_minutes"]
        },
        "paired_delta": {
            "test_accuracy_delta": float(round(test_metrics_prox["overall_accuracy"] - test_metrics_avg["overall_accuracy"], 5)),
            "test_macro_f1_delta": float(round(test_metrics_prox["macro_f1"] - test_metrics_avg["macro_f1"], 5)),
            "test_balanced_accuracy_delta": float(round(test_metrics_prox["balanced_accuracy"] - test_metrics_avg["balanced_accuracy"], 5))
        },
        "timing_reconciliation": timing_reconciliation
    }
    
    summary_path = os.path.join(results_dir, "seed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(seed_summary, f, indent=2)
        
    print(f"\n{'='*75}")
    print(f"   SEED {seed} EXECUTION SUMMARY")
    print(f"   FedAvg  (Round {res_fedavg['best_round']:02d}): Test Acc = {test_metrics_avg['overall_accuracy']*100:.3f}% | Macro-F1 = {test_metrics_avg['macro_f1']:.4f} | Time: {timing_reconciliation['fedavg']['wall_clock_minutes']}m")
    print(f"   FedProx (Round {res_fedprox['best_round']:02d}): Test Acc = {test_metrics_prox['overall_accuracy']*100:.3f}% | Macro-F1 = {test_metrics_prox['macro_f1']:.4f} | Time: {timing_reconciliation['fedprox']['wall_clock_minutes']}m")
    print(f"   Paired Test Acc Delta (FedProx - FedAvg): {seed_summary['paired_delta']['test_accuracy_delta']*100:+.3f}%")
    print(f"{'='*75}\n")
    return seed_summary


def consolidate_multi_seed_results(completed_summaries: list) -> dict:
    """
    Computes statistical aggregation (mean, std, median, min, max, range, paired deltas)
    across all completed seeds and writes consolidated deliverables.
    """
    if not completed_summaries:
        return {}
        
    global_results_dir = os.path.join(PROJ_ROOT, "experiments", "results")
    os.makedirs(global_results_dir, exist_ok=True)
    
    seeds = [s["seed"] for s in completed_summaries]
    n_seeds = len(completed_summaries)
    
    avg_test_accs = [s["fedavg"]["test_accuracy"] for s in completed_summaries]
    avg_macro_f1s = [s["fedavg"]["test_macro_f1"] for s in completed_summaries]
    avg_bal_accs = [s["fedavg"]["test_balanced_accuracy"] for s in completed_summaries]
    
    prox_test_accs = [s["fedprox"]["test_accuracy"] for s in completed_summaries]
    prox_macro_f1s = [s["fedprox"]["test_macro_f1"] for s in completed_summaries]
    prox_bal_accs = [s["fedprox"]["test_balanced_accuracy"] for s in completed_summaries]
    
    paired_acc_deltas = [p - a for a, p in zip(avg_test_accs, prox_test_accs)]
    paired_f1_deltas = [p - a for a, p in zip(avg_macro_f1s, prox_macro_f1s)]
    paired_bal_deltas = [p - a for a, p in zip(avg_bal_accs, prox_bal_accs)]
    
    def calc_stats(vals):
        arr = np.array(vals)
        return {
            "mean": float(round(float(np.mean(arr)), 5)),
            "std": float(round(float(np.std(arr, ddof=1 if len(arr) > 1 else 0)), 5)),
            "median": float(round(float(np.median(arr)), 5)),
            "min": float(round(float(np.min(arr)), 5)),
            "max": float(round(float(np.max(arr)), 5)),
            "range": float(round(float(np.max(arr) - np.min(arr)), 5))
        }
        
    consolidation = {
        "num_completed_seeds": n_seeds,
        "completed_seeds": seeds,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "per_seed_results": completed_summaries,
        "statistical_summary": {
            "fedavg": {
                "test_accuracy": calc_stats(avg_test_accs),
                "test_macro_f1": calc_stats(avg_macro_f1s),
                "test_balanced_accuracy": calc_stats(avg_bal_accs)
            },
            "fedprox": {
                "test_accuracy": calc_stats(prox_test_accs),
                "test_macro_f1": calc_stats(prox_macro_f1s),
                "test_balanced_accuracy": calc_stats(prox_bal_accs)
            },
            "paired_differences_fedprox_minus_fedavg": {
                "test_accuracy_delta": calc_stats(paired_acc_deltas),
                "test_macro_f1_delta": calc_stats(paired_f1_deltas),
                "test_balanced_accuracy_delta": calc_stats(paired_bal_deltas)
            }
        }
    }
    
    # Save consolidated JSON
    out_json = os.path.join(global_results_dir, "multiseed_consolidated_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(consolidation, f, indent=2)
        
    # Save consolidated CSV
    out_csv = os.path.join(global_results_dir, "multiseed_consolidated_summary.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("seed,algorithm,val_acc,test_acc,test_macro_f1,test_bal_acc,wall_clock_min\n")
        for s in completed_summaries:
            sd = s["seed"]
            f.write(f"{sd},FedAvg,{s['fedavg']['best_validation_accuracy']:.5f},{s['fedavg']['test_accuracy']:.5f},{s['fedavg']['test_macro_f1']:.5f},{s['fedavg']['test_balanced_accuracy']:.5f},{s['fedavg']['wall_clock_minutes']}\n")
            f.write(f"{sd},FedProx,{s['fedprox']['best_validation_accuracy']:.5f},{s['fedprox']['test_accuracy']:.5f},{s['fedprox']['test_macro_f1']:.5f},{s['fedprox']['test_balanced_accuracy']:.5f},{s['fedprox']['wall_clock_minutes']}\n")
            
    print(f"Consolidated multi-seed results saved to {out_json} and {out_csv}")
    return consolidation


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed FL Orchestrator")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of seeds to execute")
    parser.add_argument("--rounds", type=int, default=30, help="Number of communication rounds per algorithm")
    args = parser.parse_args()
    
    print(f"Starting Multi-Seed FL Execution: Seeds = {args.seeds}, Rounds = {args.rounds}")
    
    completed_summaries = []
    for s in args.seeds:
        summary = execute_seed_experiment(seed=s, num_rounds=args.rounds)
        completed_summaries.append(summary)
        
    if len(completed_summaries) > 0:
        consolidate_multi_seed_results(completed_summaries)


if __name__ == "__main__":
    main()
