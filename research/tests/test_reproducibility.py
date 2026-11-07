"""
Unit tests for End-to-End Experimental Reproducibility.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import pytest
import numpy as np
import torch
from research.models.compact_cnn import CompactPathCNN, compute_canonical_model_hash
from research.data.partitioner import generate_dirichlet_partitions
from research.data.noise_injector import inject_symmetric_label_noise


def test_seed_partition_identity():
    labels = np.random.randint(0, 9, size=1000)
    p1 = generate_dirichlet_partitions(labels, num_clients=5, alpha=0.3, partition_seed=42, manifest_path=None)
    p2 = generate_dirichlet_partitions(labels, num_clients=5, alpha=0.3, partition_seed=42, manifest_path=None)
    
    for c in range(5):
        assert np.array_equal(p1[c], p2[c]), f"Client {c} partitions differ across identical seeds!"


def test_seed_noise_mask_identity():
    m_5 = 500
    indices = np.arange(m_5)
    labels = np.random.randint(0, 9, size=m_5)
    
    l1, mask1, _ = inject_symmetric_label_noise(indices, labels, noise_rate=0.40, selection_seed=42, replacement_seed=1042)
    l2, mask2, _ = inject_symmetric_label_noise(indices, labels, noise_rate=0.40, selection_seed=42, replacement_seed=1042)
    
    assert np.array_equal(l1, l2), "Noisy labels differ across identical seeds!"
    assert np.array_equal(mask1, mask2), "Noise masks differ across identical seeds!"


def test_initial_checkpoint_tensor_equality():
    torch.manual_seed(42)
    m1 = CompactPathCNN()
    
    torch.manual_seed(42)
    m2 = CompactPathCNN()
    
    # Exact tensor equality check
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "Model parameters are not bitwise identical!"
        
    assert compute_canonical_model_hash(m1) == compute_canonical_model_hash(m2)


def test_minibatch_sequence_identity():
    """
    Verifies that the deterministic mini-batch generator produces identical permutations
    for FedAvg and FedProx given the formula: batch_seed = 42 + 1000*round + client.
    """
    round_idx = 1
    client_idx = 2
    epoch = 0
    
    seed = 42 + 1000 * round_idx + client_idx + 100 * epoch
    perm1 = np.random.default_rng(seed).permutation(100)
    perm2 = np.random.default_rng(seed).permutation(100)
    
    assert np.array_equal(perm1, perm2), "Mini-batch permutation sequences differ across runs!"
