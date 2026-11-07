"""
Weights & Biases (W&B) Synchronization Engine.
Supports discovered and specified seed experiments (e.g., seeds 42, 123, 2026, 3407, 7777).
Replays offline durable JSONL metrics, update norms, cosine similarities,
and final test evaluation metrics to public W&B runs.
Records actual run URLs in `experiments/results/wandb_sync_status.json`.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import sys
import os
import glob
import json
import argparse
from typing import Dict, Any, List

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)


def sync_single_run(
    jsonl_path: str,
    run_name: str,
    config: Dict[str, Any],
    project: str = "pathmnist-fedavg-fedprox",
    test_metrics: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Synchronizes an offline JSONL run to W&B and records its verified URL.
    """
    try:
        import wandb
    except ImportError:
        print("[ERROR] 'wandb' package is not installed. Run: pip install wandb")
        return {"status": "FAILED", "reason": "wandb not installed"}
        
    if not os.path.exists(jsonl_path):
        print(f"[ERROR] Log file does not exist: {jsonl_path}")
        return {"status": "FAILED", "reason": f"file not found: {jsonl_path}"}
        
    # Check if API key is present
    api_key = os.environ.get("WANDB_API_KEY", None)
    
    try:
        print(f"Initializing W&B run '{run_name}' in project '{project}'...")
        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            reinit=True
        )
        
        # Read and log each round's metrics
        rounds_logged = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    wandb.log(record)
                    rounds_logged += 1
                    
        # Log final held-out test metrics if available
        if test_metrics:
            wandb.log({
                "test/overall_accuracy": test_metrics.get("overall_accuracy", 0.0),
                "test/macro_f1": test_metrics.get("macro_f1", 0.0),
                "test/balanced_accuracy": test_metrics.get("balanced_accuracy", 0.0),
                "test/test_loss": test_metrics.get("test_loss", 0.0)
            })
            
        run_url = run.get_url()
        print(f"[SUCCESS] Synced '{run_name}' ({rounds_logged} rounds). W&B URL: {run_url}")
        wandb.finish()
        
        return {
            "status": "SYNCED",
            "run_name": run_name,
            "rounds_logged": rounds_logged,
            "wandb_url": run_url
        }
    except Exception as e:
        err_msg = str(e)
        print(f"[ERROR] W&B sync failed for '{run_name}': {err_msg}")
        return {
            "status": "FAILED",
            "run_name": run_name,
            "reason": err_msg
        }


def discover_seed_runs() -> List[Dict[str, Any]]:
    """
    Scans `experiments/seed_<id>/` and legacy folders to identify all available runs.
    """
    discovered = []
    
    # 1. Corrected seed runs
    seed_dirs = glob.glob(os.path.join(PROJ_ROOT, "experiments", "seed_*"))
    for sdir in sorted(seed_dirs):
        sname = os.path.basename(sdir)
        try:
            seed_val = int(sname.replace("seed_", ""))
        except ValueError:
            continue
            
        log_dir = os.path.join(sdir, "logs")
        meta_file = os.path.join(sdir, "metadata.json")
        summary_file = os.path.join(sdir, "results", "seed_summary.json")
        
        base_meta = {}
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                base_meta = json.load(f)
                
        test_summary = {}
        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                test_summary = json.load(f)
                
        # Look for FedAvg and FedProx logs
        for algo in ["fedavg", "fedprox"]:
            log_files = glob.glob(os.path.join(log_dir, f"*{algo}*metrics.jsonl"))
            for lf in log_files:
                rname = os.path.basename(lf).replace("_metrics.jsonl", "")
                t_met = test_summary.get(algo, {})
                discovered.append({
                    "seed": seed_val,
                    "algorithm": "FedAvg" if algo == "fedavg" else "FedProx",
                    "run_name": rname,
                    "jsonl_path": lf,
                    "config": {
                        **base_meta.get("hyperparameters", {}),
                        "seed": seed_val,
                        "algorithm": "FedAvg" if algo == "fedavg" else "FedProx",
                        "git_commit_hash": base_meta.get("git_commit_hash", "unknown"),
                        "model_parameters": 36425
                    },
                    "test_metrics": t_met
                })
                
    return discovered


def main():
    parser = argparse.ArgumentParser(description="Sync FL experiments to W&B")
    parser.add_argument("--project", type=str, default="pathmnist-fedavg-fedprox", help="W&B project name")
    parser.add_argument("--dry_run", action="store_true", help="Report discoverable runs without uploading")
    args = parser.parse_args()
    
    print(f"=== Discovering Offline Runs for W&B Sync (Project: {args.project}) ===")
    runs = discover_seed_runs()
    
    if not runs:
        print("No offline seed runs found in experiments/seed_*/")
        return
        
    print(f"Discovered {len(runs)} offline runs across completed seeds:")
    for r in runs:
        print(f"  - [{r['algorithm']}] Seed {r['seed']}: {r['run_name']} ({r['jsonl_path']})")
        
    status_report = {
        "project": args.project,
        "runs_found": len(runs),
        "sync_results": []
    }
    
    if args.dry_run:
        print("\nDry-run mode active. No data uploaded to W&B.")
        return
        
    # Check if wandb has active credentials
    try:
        import wandb
        has_key = bool(os.environ.get("WANDB_API_KEY"))
        if not has_key and not os.path.exists(os.path.expanduser("~/.netrc")):
            print("\n[NOTICE] No active W&B API key found in WANDB_API_KEY or ~/.netrc.")
            print("Preserving local durable JSONL records in offline mode.")
            print("To push to the public dashboard: wandb login <api_key> && python scripts/sync_to_wandb.py")
            status_report["status"] = "PENDING_CREDENTIALS"
            out_path = os.path.join(PROJ_ROOT, "experiments", "results", "wandb_sync_status.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(status_report, f, indent=2)
            return
    except ImportError:
        print("\n[NOTICE] wandb is not installed. Offline JSONL logs preserved.")
        return
        
    for r in runs:
        res = sync_single_run(
            jsonl_path=r["jsonl_path"],
            run_name=r["run_name"],
            config=r["config"],
            project=args.project,
            test_metrics=r["test_metrics"]
        )
        status_report["sync_results"].append(res)
        
    out_path = os.path.join(PROJ_ROOT, "experiments", "results", "wandb_sync_status.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(status_report, f, indent=2)
    print(f"\nW&B sync report saved to {out_path}")


if __name__ == "__main__":
    main()
