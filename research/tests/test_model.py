"""
Unit tests for CompactPathCNN Architecture.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import pytest
import torch
from research.models.compact_cnn import CompactPathCNN, count_parameters, compute_canonical_model_hash


def test_parameter_count():
    model = CompactPathCNN(in_channels=3, num_classes=9)
    total_params = count_parameters(model)
    assert total_params == 36425, f"Expected 36,425 parameters, but got {total_params}"


@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_forward_output_shapes(batch_size):
    model = CompactPathCNN(in_channels=3, num_classes=9)
    model.eval()
    x = torch.rand(batch_size, 3, 28, 28, dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (batch_size, 9), f"Expected shape ({batch_size}, 9), got {logits.shape}"
    assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf!"


def test_backward_propagation():
    model = CompactPathCNN(in_channels=3, num_classes=9)
    model.train()
    x = torch.rand(8, 3, 28, 28, dtype=torch.float32)
    y = torch.randint(0, 9, (8,), dtype=torch.int64)
    criterion = torch.nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Gradient has NaN/Inf for {name}"


def test_canonical_hash_determinism():
    torch.manual_seed(42)
    m1 = CompactPathCNN(in_channels=3, num_classes=9)
    h1 = compute_canonical_model_hash(m1)
    
    torch.manual_seed(42)
    m2 = CompactPathCNN(in_channels=3, num_classes=9)
    h2 = compute_canonical_model_hash(m2)
    
    assert h1 == h2, "Canonical state hash is not deterministic for identical initialization!"
