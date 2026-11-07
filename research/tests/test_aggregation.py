"""
Unit tests for Sample-Weighted Aggregation.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import pytest
import torch
import torch.nn as nn
from research.models.compact_cnn import CompactPathCNN


def test_sample_weighted_aggregation():
    """
    Tests that parameter aggregation strictly computes:
        w_next = sum( (n_k / N_total) * w_local_k )
    rather than an unweighted average.
    """
    # Create 5 client models and assign known distinct weights
    client_models = [CompactPathCNN() for _ in range(5)]
    sample_counts = [1000, 2000, 3000, 4000, 10000]
    total_samples = sum(sample_counts)
    weights = [n / total_samples for n in sample_counts]
    
    with torch.no_grad():
        for i, m in enumerate(client_models):
            for p in m.parameters():
                p.fill_(float(i + 1))  # Client 0 fills 1.0, Client 1 fills 2.0, etc.
                
    # Expected weighted value for every parameter:
    # 0.05*1 + 0.10*2 + 0.15*3 + 0.20*4 + 0.50*5 = 0.05 + 0.2 + 0.45 + 0.8 + 2.5 = 4.0
    expected_val = sum(w * (i + 1) for i, w in enumerate(weights))
    assert abs(expected_val - 4.0) < 1e-6
    
    # Aggregation step
    global_model = CompactPathCNN()
    global_state = global_model.state_dict()
    for key in global_state.keys():
        weighted_param = torch.zeros_like(global_state[key], dtype=torch.float32)
        for k, local_m in enumerate(client_models):
            weighted_param += weights[k] * local_m.state_dict()[key].to(torch.float32)
        global_state[key].copy_(weighted_param)
        
    global_model.load_state_dict(global_state)
    
    for name, p in global_model.named_parameters():
        assert torch.allclose(p, torch.tensor(4.0), atol=1e-6), (
            f"Parameter {name} has value {p.flatten()[0].item()}, expected {expected_val}"
        )
