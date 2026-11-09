"""
Client Update Diagnostics and Telemetry.
Calculates L2 norms, pairwise cosine similarities, and alignment metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_update_vector(local_model: torch.nn.Module, global_model: torch.nn.Module) -> torch.Tensor:
    """
    Computes parameter displacement vector Delta = w_local - w_global flattened into a 1D tensor.
    """
    diffs = []
    for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
        if p_local.requires_grad:
            diffs.append((p_local.detach().cpu() - p_global.detach().cpu()).flatten())
    return torch.cat(diffs)


def compute_update_l2_norm(delta: torch.Tensor) -> float:
    """Computes Euclidean L2 norm of update vector."""
    return float(torch.norm(delta, p=2).item())


def compute_pairwise_cosine_similarity(
    deltas: List[torch.Tensor],
    eps: float = 1e-12
) -> Tuple[Dict[str, float], float, float]:
    """
    Computes all 10 pairwise cosine similarities among 5 clients,
    as well as mean clean-to-noisy and mean clean-to-clean alignment.
    
    Clients 0..3 are clean (Clients 1..4), Client 4 is noisy (Client 5).
    """
    num_clients = len(deltas)
    pairwise = {}
    
    clean_noisy_sims = []
    clean_clean_sims = []
    
    for i in range(num_clients):
        norm_i = torch.norm(deltas[i], p=2).item()
        for j in range(i + 1, num_clients):
            norm_j = torch.norm(deltas[j], p=2).item()
            denom = (norm_i * norm_j) + eps
            dot = torch.dot(deltas[i], deltas[j]).item()
            sim = float(dot / denom)
            # Clamp between -1.0 and 1.0
            sim = max(-1.0, min(1.0, sim))
            
            key = f"pairwise_cosine/c{i+1}_c{j+1}"
            pairwise[key] = sim
            
            # Check if one is Client 5 (index 4)
            if i == 4 or j == 4:
                clean_noisy_sims.append(sim)
            else:
                clean_clean_sims.append(sim)
                
    mean_clean_noisy = float(np.mean(clean_noisy_sims)) if clean_noisy_sims else 0.0
    mean_clean_clean = float(np.mean(clean_clean_sims)) if clean_clean_sims else 0.0
    
    return pairwise, mean_clean_noisy, mean_clean_clean


def compute_similarity_to_weighted_mean(
    deltas: List[torch.Tensor],
    weights: List[float],
    eps: float = 1e-12
) -> Tuple[Dict[str, float], torch.Tensor]:
    """
    Computes each client's cosine similarity to the sample-weighted mean update vector.
    
    Args:
        deltas: List of client update vectors Delta_k.
        weights: List of normalized client sample weights p_k (sum to 1.0).
    """
    weighted_mean = torch.zeros_like(deltas[0])
    for p_k, delta_k in zip(weights, deltas):
        weighted_mean += float(p_k) * delta_k
        
    mean_norm = torch.norm(weighted_mean, p=2).item()
    
    sim_to_mean = {}
    for i, delta in enumerate(deltas):
        norm_i = torch.norm(delta, p=2).item()
        denom = (norm_i * mean_norm) + eps
        dot = torch.dot(delta, weighted_mean).item()
        sim = float(dot / denom)
        sim = max(-1.0, min(1.0, sim))
        sim_to_mean[f"diagnostics/client_{i+1}_sim_to_weighted_mean"] = sim
        
    return sim_to_mean, weighted_mean
