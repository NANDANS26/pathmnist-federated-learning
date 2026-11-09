"""
Unit tests for PerturbedSGD Optimizer (FedProx).
"""

import pytest
import torch
import torch.nn as nn
from research.optimizers.perturbed_sgd import PerturbedSGD


def test_perturbed_sgd_analytical_equivalence():
    """
    Tests that PerturbedSGD mathematically executes:
        w_next = w - lr * (grad + mu * (w - w_global))
    within 1e-7 numerical tolerance.
    """
    torch.manual_seed(42)
    model = nn.Linear(4, 2, bias=True)
    global_model = nn.Linear(4, 2, bias=True)
    
    # Initialize both with known values
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.5)
        global_model.weight.fill_(0.2)
        global_model.bias.fill_(0.1)
        
    lr = 0.01
    mu = 0.1
    optimizer = PerturbedSGD(model.parameters(), lr=lr, mu=mu)
    optimizer.set_global_parameters(global_model)
    
    x = torch.ones(2, 4)
    y = torch.tensor([1, 0], dtype=torch.int64)
    loss_fn = nn.CrossEntropyLoss()
    
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    
    # Capture analytical gradient and expected next weight
    w_orig = model.weight.detach().clone()
    b_orig = model.bias.detach().clone()
    w_grad = model.weight.grad.detach().clone()
    b_grad = model.bias.grad.detach().clone()
    w_glob = global_model.weight.detach().clone()
    b_glob = global_model.bias.detach().clone()
    
    expected_w_next = w_orig - lr * (w_grad + mu * (w_orig - w_glob))
    expected_b_next = b_orig - lr * (b_grad + mu * (b_orig - b_glob))
    
    optimizer.step()
    
    assert torch.allclose(model.weight, expected_w_next, atol=1e-7), "Weight update mismatch with analytical formula!"
    assert torch.allclose(model.bias, expected_b_next, atol=1e-7), "Bias update mismatch with analytical formula!"


def test_global_reference_invariance_during_steps():
    """
    Verifies that multiple optimizer steps do not mutate the cached global parameters.
    """
    model = nn.Linear(2, 2)
    global_model = nn.Linear(2, 2)
    with torch.no_grad():
        global_model.weight.fill_(0.5)
        
    optimizer = PerturbedSGD(model.parameters(), lr=0.01, mu=0.05)
    optimizer.set_global_parameters(global_model)
    
    cached_weight_before = optimizer.global_params[0].clone()
    
    for _ in range(5):
        optimizer.zero_grad()
        dummy_loss = torch.sum(model(torch.ones(2, 2)))
        dummy_loss.backward()
        optimizer.step()
        
    cached_weight_after = optimizer.global_params[0]
    assert torch.equal(cached_weight_before, cached_weight_after), "Global parameter reference was mutated during training!"
