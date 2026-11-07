"""
Phase 2 CPU Pilot Benchmark Runner.
Executes 2-round pilot runs for FedAvg and FedProx (mu=0.01) on real PathMNIST non-IID partitions,
records batch latency, round time, peak RSS, and calibrates communication horizon.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import sys
import os
import json
import time
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.data.pathmnist_loader import get_training_data, get_validation_data
from research.data.noise_injector import inject_symmetric_label_noise
from research.models.compact_cnn import CompactPathCNN, compute_canonical_model_hash
from research.core.fedavg import run_fedavg
from research.core.fedprox import run_fedprox


def run_cpu_pilot():
    print("======================================================================")
    print("      PHASE 2 CPU PILOT BENCHMARK — REAL PATHMNIST NON-IID DATA")
    print("======================================================================")
    
    # 1. Load real data
    print("\nLoading memory-mapped PathMNIST training and validation splits...")
    train_images, train_labels = get_training_data()
    val_images, val_labels = get_validation_data()
    
    # 2. Load manifests
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    partition_path = os.path.join(proj_root, "experiments", "configs", "partition_manifest.json")
    val_manifest_path = os.path.join(proj_root, "experiments", "configs", "validation_manifest.json")
    noise_manifest_path = os.path.join(proj_root, "experiments", "configs", "noise_manifest.json")
    ckpt_path = os.path.join(proj_root, "experiments", "checkpoints", "initial_model.pt")
    
    with open(partition_path, "r") as f:
        part_data = json.load(f)
    with open(val_manifest_path, "r") as f:
        val_data = json.load(f)
    with open(noise_manifest_path, "r") as f:
        noise_data = json.load(f)
        
    train_partitions = {int(k): np.array(v, dtype=np.int64) for k, v in part_data["client_indices"].items()}
    val_shards = {int(k): np.array(v, dtype=np.int64) for k, v in val_data["client_val_indices"].items()}
    
    # Reconstruct Client 5 noisy labels
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
    
    print(f"Dataset partitions loaded: {sum(len(p) for p in train_partitions.values()):,} train samples")
    print(f"Validation shards loaded: {sum(len(s) for s in val_shards.values()):,} val samples")
    print(f"Client 5 training samples: {len(client_5_indices):,} (corrupted: {np.sum(noise_mask):,})")
    
    # Set PyTorch thread limits
    torch.set_num_threads(2)
    
    # Load initial model
    state_dict = torch.load(ckpt_path)
    
    # ------------------------------------------------------------------
    # PILOT RUN A: FedAvg (2 communication rounds)
    # ------------------------------------------------------------------
    print("\n------------------------------------------------------------------")
    print(" PILOT RUN A: FedAvg Baseline (2 communication rounds)")
    print("------------------------------------------------------------------")
    model_fedavg = CompactPathCNN()
    model_fedavg.load_state_dict(state_dict)
    
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
        num_rounds=2,
        lr=0.01,
        batch_size=32,
        local_epochs=2,
        run_name="fedavg_pilot_real"
    )
    fedavg_wall_time = time.perf_counter() - fedavg_start_wall
    
    # ------------------------------------------------------------------
    # PILOT RUN B: FedProx (mu=0.01, 2 communication rounds)
    # ------------------------------------------------------------------
    print("\n------------------------------------------------------------------")
    print(" PILOT RUN B: FedProx (mu=0.01, 2 communication rounds)")
    print("------------------------------------------------------------------")
    model_fedprox = CompactPathCNN()
    model_fedprox.load_state_dict(state_dict)
    
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
        num_rounds=2,
        lr=0.01,
        mu=0.01,
        batch_size=32,
        local_epochs=2,
        run_name="fedprox_mu0.01_pilot_real"
    )
    fedprox_wall_time = time.perf_counter() - fedprox_start_wall
    
    # ------------------------------------------------------------------
    # METRICS & ROUND CALIBRATION
    # ------------------------------------------------------------------
    # Read metrics records from jsonl
    log_dir = os.path.join(proj_root, "experiments", "logs")
    fedavg_records = []
    with open(os.path.join(log_dir, "fedavg_pilot_real_metrics.jsonl"), "r") as f:
        for line in f:
            fedavg_records.append(json.loads(line))
            
    fedprox_records = []
    with open(os.path.join(log_dir, "fedprox_mu0.01_pilot_real_metrics.jsonl"), "r") as f:
        for line in f:
            fedprox_records.append(json.loads(line))
            
    fedavg_round_times = [r["round_duration_seconds"] for r in fedavg_records]
    fedprox_round_times = [r["round_duration_seconds"] for r in fedprox_records]
    all_round_times = fedavg_round_times + fedprox_round_times
    mean_round_time = float(np.mean(all_round_times))
    
    # Total batches per round across 5 clients: 89,996 / 32 * 2 epochs = ~5,626 batches
    total_batches_per_round = sum(int(np.ceil(len(p) / 32)) * 2 for p in train_partitions.values())
    avg_step_latency_ms = (mean_round_time / total_batches_per_round) * 1000.0
    
    peak_rss_mb = max(
        max(r["process_rss_mb"] for r in fedavg_records),
        max(r["process_rss_mb"] for r in fedprox_records)
    )
    lowest_avail_ram_mb = min(
        min(r["system_available_mb"] for r in fedavg_records),
        min(r["system_available_mb"] for r in fedprox_records)
    )
    
    # Calibrate horizon according to approved mutually exclusive policy
    if mean_round_time < 35.0:
        recommended_horizon = 100
        horizon_reason = "T_round < 35s -> R = 100 rounds (< 58 min runtime)"
    elif 35.0 <= mean_round_time <= 75.0:
        recommended_horizon = 50
        horizon_reason = "35s <= T_round <= 75s -> R = 50 rounds (29 to 62.5 min runtime)"
    elif 75.0 < mean_round_time <= 120.0:
        recommended_horizon = 30
        horizon_reason = "75s < T_round <= 120s -> R = 30 rounds (37.5 to 60 min runtime)"
    else:
        recommended_horizon = 30
        horizon_reason = f"T_round ({mean_round_time:.1f}s) > 120s -> Review recommended; capped at 30 rounds"
        
    benchmark_data = {
        "benchmark_date": "2026-11-07",
        "hardware": {
            "os": "Windows 10 Pro (64-bit)",
            "cpu": "Intel Pentium G3240 @ 3.10GHz (2 Cores, 2 Threads)",
            "physical_ram": "4 GB",
            "device": "CPU"
        },
        "dataset": "PathMNIST (MedMNIST v2)",
        "train_samples": 89996,
        "val_samples": 10004,
        "completed_pilot_rounds": 2,
        "measurements": {
            "fedavg_round_times_sec": fedavg_round_times,
            "fedprox_round_times_sec": fedprox_round_times,
            "mean_round_duration_sec": float(round(mean_round_time, 2)),
            "batches_per_round": total_batches_per_round,
            "avg_step_latency_ms": float(round(avg_step_latency_ms, 2)),
            "peak_process_rss_mb": float(round(peak_rss_mb, 2)),
            "lowest_available_ram_mb": float(round(lowest_avail_ram_mb, 2))
        },
        "performance": {
            "fedavg_final_val_accuracy": fedavg_records[-1]["val/global_accuracy"],
            "fedavg_final_val_loss": fedavg_records[-1]["val/global_loss"],
            "fedprox_final_val_accuracy": fedprox_records[-1]["val/global_accuracy"],
            "fedprox_final_val_loss": fedprox_records[-1]["val/global_loss"],
            "fedavg_mean_clean_noisy_cosine": fedavg_records[-1]["diagnostics/mean_clean_noisy_cosine"],
            "fedprox_mean_clean_noisy_cosine": fedprox_records[-1]["diagnostics/mean_clean_noisy_cosine"]
        },
        "calibration": {
            "mean_round_time_sec": float(round(mean_round_time, 2)),
            "recommended_horizon_rounds": recommended_horizon,
            "calibration_rule_applied": horizon_reason,
            "estimated_full_experiment_minutes_per_algo": float(round(recommended_horizon * mean_round_time / 60.0, 1))
        }
    }
    
    results_dir = os.path.join(proj_root, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    benchmark_file = os.path.join(results_dir, "pilot_benchmark.json")
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)
        
    print("\n======================================================================")
    print("                 PILOT BENCHMARK RESULTS SUMMARY                      ")
    print("======================================================================")
    print(f"Mean Round Duration: {mean_round_time:.2f} s")
    print(f"Average Batch-Step Latency: {avg_step_latency_ms:.2f} ms/batch")
    print(f"Peak Process RSS: {peak_rss_mb:.2f} MB")
    print(f"Lowest Available Physical RAM: {lowest_avail_ram_mb:.2f} MB")
    print(f"FedAvg Val Accuracy (Round 2): {benchmark_data['performance']['fedavg_final_val_accuracy']:.4f}")
    print(f"FedProx Val Accuracy (Round 2): {benchmark_data['performance']['fedprox_final_val_accuracy']:.4f}")
    print(f"Clean-to-Noisy Alignment: FedAvg={benchmark_data['performance']['fedavg_mean_clean_noisy_cosine']:.4f} vs FedProx={benchmark_data['performance']['fedprox_mean_clean_noisy_cosine']:.4f}")
    print(f"\nRecommended Communication Horizon: {recommended_horizon} rounds ({horizon_reason})")
    print(f"Estimated Full Run Duration: ~{benchmark_data['calibration']['estimated_full_experiment_minutes_per_algo']} minutes per algorithm")
    print(f"Results saved to: {benchmark_file}")
    print("======================================================================")


if __name__ == "__main__":
    run_cpu_pilot()
