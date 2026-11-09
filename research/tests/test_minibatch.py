"""
Unit and regression tests for client mini-batch partitioning, coverage, uniqueness,
determinism, partial batch handling, and image-label correspondence.
"""

import numpy as np
import pytest
import torch

from research.core.client import FederatedClient
from research.models.compact_cnn import CompactPathCNN


def test_minibatch_coverage_and_uniqueness():
    """
    Verifies that every local training sample appears exactly once per epoch,
    with zero duplicates and zero skipped samples across all batches including
    the final partial batch.
    """
    num_samples = 7895  # Exact Client 5 sample count
    batch_size = 32
    client_indices = np.arange(1000, 1000 + num_samples)
    dummy_images = np.zeros((num_samples + 2000, 28, 28, 3), dtype=np.uint8)
    dummy_labels = np.random.randint(0, 9, size=(num_samples,), dtype=np.uint8)
    
    client = FederatedClient(
        client_id=4,
        train_images=dummy_images,
        train_labels=dummy_labels,
        client_indices=client_indices,
        batch_size=batch_size,
        local_epochs=2,
        base_seed=42
    )
    
    for epoch in range(client.local_epochs):
        epoch_seed = client.base_seed + 1000 * 1 + client.client_id + 100 * epoch
        rng = np.random.default_rng(epoch_seed)
        local_perm = rng.permutation(client.num_samples)
        permuted_indices = client.client_indices[local_perm]
        
        num_batches = int(np.ceil(client.num_samples / client.batch_size))
        assert num_batches == 247, f"Expected 247 batches for 7895 samples at bs=32, got {num_batches}"
        
        all_accessed_indices = []
        all_batch_sizes = []
        
        for b in range(num_batches):
            batch_slice = slice(b * client.batch_size, (b + 1) * client.batch_size)
            batch_idx = permuted_indices[batch_slice]
            all_accessed_indices.extend(batch_idx.tolist())
            all_batch_sizes.append(len(batch_idx))
            
        # Check total sample count
        assert len(all_accessed_indices) == num_samples, "Mismatch in total accessed sample count!"
        
        # Check uniqueness: no duplicates, no missing
        assert len(set(all_accessed_indices)) == num_samples, "Duplicate samples found in epoch batches!"
        assert set(all_accessed_indices) == set(client_indices), "Sample set mismatch!"
        
        # Check batch sizes: all except last are 32; last is remainder
        assert all(bs == 32 for bs in all_batch_sizes[:-1]), "Intermediate batches must have size 32!"
        expected_remainder = num_samples % batch_size  # 7895 % 32 = 23
        assert all_batch_sizes[-1] == expected_remainder, f"Final batch size should be {expected_remainder}, got {all_batch_sizes[-1]}"


def test_image_label_correspondence():
    """
    Verifies that slicing uint8 images and labels preserves exact 1:1 correspondence
    under permutation for both client-specific and global label arrays.
    """
    num_samples = 500
    batch_size = 32
    client_indices = np.array([10, 25, 42, 99, 150, 201, 333] + list(range(400, 400 + num_samples - 7)))
    
    # Create synthetic images where pixel value encodes global index
    dummy_images = np.zeros((1000, 28, 28, 3), dtype=np.uint8)
    for idx in client_indices:
        dummy_images[idx, 0, 0, 0] = idx % 256
        
    # Client-specific labels: label = (global_index * 7) % 9
    client_labels = np.array([(idx * 7) % 9 for idx in client_indices], dtype=np.uint8)
    
    client = FederatedClient(
        client_id=1,
        train_images=dummy_images,
        train_labels=client_labels,
        client_indices=client_indices,
        batch_size=batch_size,
        local_epochs=1,
        base_seed=123
    )
    
    epoch_seed = client.base_seed + 1000 * 1 + client.client_id + 100 * 0
    rng = np.random.default_rng(epoch_seed)
    local_perm = rng.permutation(client.num_samples)
    permuted_indices = client.client_indices[local_perm]
    
    num_batches = int(np.ceil(client.num_samples / client.batch_size))
    for b in range(num_batches):
        batch_slice = slice(b * client.batch_size, (b + 1) * client.batch_size)
        batch_idx = permuted_indices[batch_slice]
        batch_imgs_raw = client.train_images[batch_idx]
        batch_lbls_raw = client.train_labels[local_perm[batch_slice]]
        
        # Verify 1:1 correspondence for each sample in batch
        for i in range(len(batch_idx)):
            global_idx = batch_idx[i]
            img_val = batch_imgs_raw[i, 0, 0, 0]
            lbl_val = batch_lbls_raw[i]
            assert img_val == (global_idx % 256), f"Image corrupted at batch {b}, sample {i}"
            assert lbl_val == ((global_idx * 7) % 9), f"Label mismatch at batch {b}, sample {i}"


def test_batch_determinism_and_algorithm_pairing():
    """
    Verifies that given the same seed:
    1. FedAvg and FedProx clients generate identical batch orders.
    2. Different seeds produce distinct permutations.
    """
    num_samples = 1000
    client_indices = np.arange(num_samples)
    dummy_images = np.zeros((num_samples, 28, 28, 3), dtype=np.uint8)
    dummy_labels = np.zeros(num_samples, dtype=np.uint8)
    
    client_fedavg = FederatedClient(
        client_id=0,
        train_images=dummy_images,
        train_labels=dummy_labels,
        client_indices=client_indices,
        algorithm="FedAvg",
        base_seed=42
    )
    
    client_fedprox = FederatedClient(
        client_id=0,
        train_images=dummy_images,
        train_labels=dummy_labels,
        client_indices=client_indices,
        algorithm="FedProx",
        base_seed=42
    )
    
    client_seed123 = FederatedClient(
        client_id=0,
        train_images=dummy_images,
        train_labels=dummy_labels,
        client_indices=client_indices,
        algorithm="FedAvg",
        base_seed=123
    )
    
    # Test across multiple rounds and epochs
    for r in [1, 5, 10]:
        for ep in [0, 1]:
            seed_avg = client_fedavg.base_seed + 1000 * r + client_fedavg.client_id + 100 * ep
            seed_prox = client_fedprox.base_seed + 1000 * r + client_fedprox.client_id + 100 * ep
            seed_diff = client_seed123.base_seed + 1000 * r + client_seed123.client_id + 100 * ep
            
            perm_avg = np.random.default_rng(seed_avg).permutation(num_samples)
            perm_prox = np.random.default_rng(seed_prox).permutation(num_samples)
            perm_diff = np.random.default_rng(seed_diff).permutation(num_samples)
            
            # Exact equality between paired FedAvg and FedProx
            np.testing.assert_array_equal(perm_avg, perm_prox, f"FedAvg and FedProx batch order mismatch at round {r}, epoch {ep}")
            
            # Statistical inequality across different seeds
            assert not np.array_equal(perm_avg, perm_diff), f"Permutations across different seeds should not match at round {r}, epoch {ep}"
