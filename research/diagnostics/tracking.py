"""
Durable Local JSONL Logging and CSV Summary Export.
Phase 2 Implementation for IndiaAI Fellowship Screening.
"""

import json
import os
import csv
from typing import Dict, Any, List, Optional


class ExperimentTracker:
    """
    Manages local durable JSONL logging and CSV table generation.
    Supports optional, decoupled Weights & Biases (W&B) syncing.
    """
    def __init__(self, run_name: str, log_dir: str = None, enable_wandb: bool = False, resume: bool = False, custom_jsonl_name: str = None):
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "experiments", "logs"
            )
        self.run_name = run_name
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        filename = custom_jsonl_name if custom_jsonl_name else f"{run_name}_metrics.jsonl"
        self.jsonl_path = os.path.join(self.log_dir, filename)
        self.csv_path = os.path.join(self.log_dir, f"{run_name}_summary.csv")
        self.records: List[Dict[str, Any]] = []
        self.enable_wandb = enable_wandb
        
        # Load existing records if resuming, or truncate for clean run
        if resume and os.path.exists(self.jsonl_path):
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))
        elif not resume and os.path.exists(self.jsonl_path):
            os.remove(self.jsonl_path)
            
    def log_round(self, metrics: Dict[str, Any]):
        """
        Appends metrics dictionary to local JSONL file with synchronous flush.
        """
        self.records.append(metrics)
        
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
            f.flush()
            os.fsync(f.fileno())
            
        if self.enable_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics)
            except ImportError:
                pass

    def export_csv_summary(self):
        """
        Flattens recorded rounds and writes summary CSV table.
        """
        if not self.records:
            return
            
        # Determine all unique fieldnames
        fieldnames = []
        for r in self.records:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
                    
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r)
                
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent round metrics dictionary."""
        return self.records[-1] if self.records else None
