"""
FedAvg Experiment Runner.
Executes Federated Averaging training across rounds with telemetry logging and checkpoint saving.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import os
import torch
from typing import Dict, Any, List

from research.models.compact_cnn import CompactPathCNN
from research.core.client import FederatedClient
from research.core.server import FederatedServer
from research.diagnostics.tracking import ExperimentTracker
from research.diagnostics.memory_monitor import MemoryMonitor


def run_fedavg(
    global_model: CompactPathCNN,
    train_images,
    train_labels,
    client_train_partitions: Dict[int, Any],
    client_noisy_labels: Dict[int, Any],
    val_images,
    val_labels,
    client_val_indices: Dict[int, Any],
    num_rounds: int = 2,
    lr: float = 0.01,
    batch_size: int = 32,
    local_epochs: int = 2,
    run_name: str = "fedavg_pilot",
    checkpoint_dir: str = None,
    log_dir: str = None,
    resume: bool = False,
    custom_jsonl_name: str = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes a FedAvg experiment for a given number of communication rounds.
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "experiments", "checkpoints"
        )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    tracker = ExperimentTracker(run_name=run_name, log_dir=log_dir, resume=resume, custom_jsonl_name=custom_jsonl_name)
    memory_monitor = MemoryMonitor()
    
    # Check if resuming from previous checkpoint
    latest_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_latest.pt")
    start_round = 1
    best_acc = -1.0
    best_loss = 1e9
    best_round = -1
    
    if resume and len(tracker.records) > 0 and os.path.exists(latest_ckpt_path):
        completed_rounds = len(tracker.records)
        start_round = completed_rounds + 1
        global_model.load_state_dict(torch.load(latest_ckpt_path))
        print(f"Resuming {run_name} from Round {start_round} (loaded checkpoint: {latest_ckpt_path})...")
        for rec in tracker.records:
            acc = rec.get("val/global_accuracy", -1.0)
            loss = rec.get("val/global_loss", 1e9)
            rnd = rec.get("round", -1)
            if acc > best_acc or (acc == best_acc and loss < best_loss):
                best_acc = acc
                best_loss = loss
                best_round = rnd
    
    # Instantiate clients
    clients: List[FederatedClient] = []
    for c in range(5):
        # Use noisy labels for Client 5 (index 4) and clean labels for Clients 1-4
        c_lbls = client_noisy_labels[c] if c in client_noisy_labels else train_labels
        client = FederatedClient(
            client_id=c,
            train_images=train_images,
            train_labels=c_lbls,
            client_indices=client_train_partitions[c],
            algorithm="FedAvg",
            lr=lr,
            batch_size=batch_size,
            local_epochs=local_epochs,
            base_seed=seed
        )
        clients.append(client)
        
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        val_images=val_images,
        val_labels=val_labels,
        client_val_indices=client_val_indices,
        memory_monitor=memory_monitor
    )
    
    print(f"=== Starting FedAvg Run: {run_name} ({start_round} to {num_rounds} rounds) ===")
    
    for r in range(start_round, num_rounds + 1):
        metrics = server.run_communication_round(round_index=r)
        tracker.log_round(metrics)
        
        val_acc = metrics["val/global_accuracy"]
        val_loss = metrics["val/global_loss"]
        duration = metrics["round_duration_seconds"]
        rss = metrics["process_rss_mb"]
        avail_ram = metrics["system_available_mb"]
        
        print(
            f"[Round {r:02d}/{num_rounds:02d}] "
            f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {duration:.1f}s | RSS: {rss:.1f}MB | Avail RAM: {avail_ram:.1f}MB"
        )
        
        # Save latest checkpoint after every completed round
        latest_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_latest.pt")
        torch.save(global_model.state_dict(), latest_ckpt_path)
        
        # Checkpoint selection: peak validation accuracy (tie breaker: lowest loss)
        is_best = False
        if val_acc > best_acc:
            is_best = True
        elif val_acc == best_acc and val_loss < best_loss:
            is_best = True
            
        if is_best:
            best_acc = val_acc
            best_loss = val_loss
            best_round = r
            best_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
            torch.save(global_model.state_dict(), best_ckpt_path)
            
    # Save final model
    final_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
    torch.save(global_model.state_dict(), final_ckpt_path)
    
    tracker.export_csv_summary()
    
    return {
        "algorithm": "FedAvg",
        "run_name": run_name,
        "completed_rounds": num_rounds,
        "best_round": best_round,
        "best_validation_accuracy": best_acc,
        "best_validation_loss": best_loss,
        "latest_metrics": tracker.get_latest_metrics()
    }
