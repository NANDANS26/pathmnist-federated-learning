"""
Weights & Biases Synchronization Script.
Syncs offline durable JSONL experiment logs to a public W&B project: 'pathmnist-fedavg-fedprox'.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import sys
import os
import json

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)


def sync_run_to_wandb(jsonl_path: str, run_name: str, config: dict, project: str = "pathmnist-fedavg-fedprox"):
    try:
        import wandb
    except ImportError:
        print("W&B is not installed in the active environment.")
        print("To install and sync: pip install wandb && wandb login")
        return False
        
    if not os.path.exists(jsonl_path):
        print(f"Log file not found: {jsonl_path}")
        return False
        
    print(f"Initializing W&B run: {run_name} in project {project}...")
    run = wandb.init(project=project, name=run_name, config=config, reinit=True)
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                wandb.log(record)
                
    print(f"Completed syncing {run_name}. Run URL: {run.get_url()}")
    wandb.finish()
    return True


def main():
    log_dir = os.path.join(PROJ_ROOT, "experiments", "logs")
    results_dir = os.path.join(PROJ_ROOT, "experiments", "results")
    final_comp_path = os.path.join(results_dir, "final_comparison.json")
    
    config = {}
    if os.path.exists(final_comp_path):
        with open(final_comp_path, "r", encoding="utf-8") as f:
            config = json.load(f).get("study_metadata", {})
            
    # 1. Sync FedAvg
    avg_log = os.path.join(log_dir, "fedavg_metrics.jsonl")
    sync_run_to_wandb(avg_log, "fedavg_alpha0.3_noise40_seed42", {**config, "algorithm": "FedAvg"})
    
    # 2. Sync FedProx
    prox_log = os.path.join(log_dir, "fedprox_metrics.jsonl")
    sync_run_to_wandb(prox_log, "fedprox_mu0.01_alpha0.3_noise40_seed42", {**config, "algorithm": "FedProx", "mu": 0.01})


if __name__ == "__main__":
    main()
