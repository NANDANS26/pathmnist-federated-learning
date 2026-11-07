"""
Federated Server and Orchestration Engine.
Handles sample-weighted aggregation, update diagnostics extraction, and clean validation evaluation.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple

from research.models.compact_cnn import CompactPathCNN, preprocess_batch
from research.core.client import FederatedClient
from research.diagnostics.update_metrics import (
    compute_update_vector,
    compute_update_l2_norm,
    compute_pairwise_cosine_similarity,
    compute_similarity_to_weighted_mean
)
from research.diagnostics.memory_monitor import MemoryMonitor


class FederatedServer:
    """
    Orchestrates synchronous federated learning rounds on CPU.
    Executes sequential client training, sample-weighted aggregation,
    diagnostics extraction, and evaluation on clean validation shards.
    """
    def __init__(
        self,
        global_model: CompactPathCNN,
        clients: List[FederatedClient],
        val_images: np.ndarray,
        val_labels: np.ndarray,
        client_val_indices: Dict[int, np.ndarray],
        memory_monitor: MemoryMonitor = None,
        val_batch_size: int = 128
    ):
        self.global_model = global_model
        self.clients = clients
        self.num_clients = len(clients)
        self.val_images = val_images
        self.val_labels = val_labels.squeeze()
        self.client_val_indices = client_val_indices
        self.val_batch_size = val_batch_size
        self.memory_monitor = memory_monitor or MemoryMonitor()
        
        # Calculate sample weights: p_k = n_k / sum(n_i)
        client_sample_counts = [c.num_samples for c in self.clients]
        total_training_samples = sum(client_sample_counts)
        self.sample_weights = [c / total_training_samples for c in client_sample_counts]
        
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def run_communication_round(self, round_index: int) -> Dict[str, Any]:
        """
        Executes a complete federated communication round.
        
        Steps:
            1. Save pre-training global parameters w_t.
            2. Train clients sequentially.
            3. Compute client update displacement vectors Delta_{k,t}.
            4. Extract L2 norms, pairwise cosine similarities, and weighted-mean alignments.
            5. Perform sample-weighted aggregation: w_{t+1} = sum(p_k * w_{k,t}).
            6. Evaluate newly aggregated model w_{t+1} on clean validation data.
            7. Monitor system memory.
        """
        round_start_time = time.perf_counter()
        
        # Snapshot global model before training
        w_t = CompactPathCNN(in_channels=3, num_classes=9)
        w_t.load_state_dict(self.global_model.state_dict())
        
        local_models: List[CompactPathCNN] = []
        client_metrics_list: List[Dict[str, Any]] = []
        
        # 1. Sequential Client Training
        for client in self.clients:
            self.memory_monitor.check_safety()
            local_model, c_metrics = client.train_round(self.global_model, round_index)
            local_models.append(local_model)
            client_metrics_list.append(c_metrics)
            
        # 2. Extract Client Updates: Delta_{k,t} = w_{k,t} - w_t
        deltas = [compute_update_vector(local_m, w_t) for local_m in local_models]
        
        # 3. Compute Update Diagnostics
        l2_norms = {
            f"diagnostics/client_{k+1}_update_l2_norm": compute_update_l2_norm(deltas[k])
            for k in range(self.num_clients)
        }
        
        pairwise_sims, mean_clean_noisy, mean_clean_clean = compute_pairwise_cosine_similarity(deltas)
        sim_to_mean, weighted_mean_delta = compute_similarity_to_weighted_mean(deltas, self.sample_weights)
        
        # 4. Perform Sample-Weighted Parameter Aggregation: w_{t+1} = sum(p_k * w_{k,t})
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            weighted_param = torch.zeros_like(global_state[key], dtype=torch.float32)
            for k, local_m in enumerate(local_models):
                weighted_param += self.sample_weights[k] * local_m.state_dict()[key].to(torch.float32)
            global_state[key].copy_(weighted_param)
            
        self.global_model.load_state_dict(global_state)
        
        # 5. Evaluate Newly Aggregated Global Model w_{t+1} on Clean Validation Data
        val_metrics = self.evaluate_validation_shards()
        
        # 6. Resource and Timing Telemetry
        round_duration = time.perf_counter() - round_start_time
        mem_stats = self.memory_monitor.check_safety()
        
        # Combine all metrics into unified round record
        round_record = {
            "round": round_index,
            "round_duration_seconds": float(round(round_duration, 3)),
            **val_metrics,
            **l2_norms,
            **pairwise_sims,
            **sim_to_mean,
            "diagnostics/mean_clean_noisy_cosine": mean_clean_noisy,
            "diagnostics/mean_clean_clean_cosine": mean_clean_clean,
            **mem_stats
        }
        
        # Include per-client training losses and training durations
        for k, c_met in enumerate(client_metrics_list):
            round_record[f"train/client_{k+1}_ce_loss"] = c_met["mean_ce_loss"]
            round_record[f"train/client_{k+1}_train_time_sec"] = c_met["duration_seconds"]
            
        return round_record

    @torch.no_grad()
    def evaluate_validation_shards(self) -> Dict[str, float]:
        """
        Evaluates the newly aggregated global model on the official 10,004 clean validation samples.
        Accumulates global validation accuracy and per-client shard validation accuracy efficiently in one pass.
        """
        self.global_model.eval()
        
        # Create map from validation index to client shard index
        idx_to_client = {}
        for c, indices in self.client_val_indices.items():
            for idx in indices:
                idx_to_client[idx] = c
                
        total_val = len(self.val_labels)
        global_loss_accum = 0.0
        global_correct_accum = 0
        
        client_loss_accum = {c: 0.0 for c in range(self.num_clients)}
        client_correct_accum = {c: 0 for c in range(self.num_clients)}
        client_sample_count = {c: len(self.client_val_indices[c]) for c in range(self.num_clients)}
        
        num_batches = int(np.ceil(total_val / self.val_batch_size))
        
        for b in range(num_batches):
            b_indices = np.arange(b * self.val_batch_size, min((b + 1) * self.val_batch_size, total_val))
            b_imgs = self.val_images[b_indices]
            b_lbls = self.val_labels[b_indices]
            
            x, y = preprocess_batch(b_imgs, b_lbls)
            logits = self.global_model(x)
            
            # Loss per sample
            ce_losses = nn.functional.cross_entropy(logits, y, reduction="none")
            preds = torch.argmax(logits, dim=1)
            corrects = (preds == y)
            
            global_loss_accum += float(torch.sum(ce_losses).item())
            global_correct_accum += int(torch.sum(corrects).item())
            
            # Attribute to client shards
            for i, val_idx in enumerate(b_indices):
                c = idx_to_client[val_idx]
                client_loss_accum[c] += float(ce_losses[i].item())
                if corrects[i]:
                    client_correct_accum[c] += 1
                    
        val_results = {
            "val/global_accuracy": float(round(global_correct_accum / total_val, 5)),
            "val/global_loss": float(round(global_loss_accum / total_val, 5))
        }
        
        for c in range(self.num_clients):
            n_c = client_sample_count[c]
            val_results[f"val/client_{c+1}_accuracy"] = float(round(client_correct_accum[c] / n_c, 5)) if n_c > 0 else 0.0
            val_results[f"val/client_{c+1}_loss"] = float(round(client_loss_accum[c] / n_c, 5)) if n_c > 0 else 0.0
            
        return val_results
