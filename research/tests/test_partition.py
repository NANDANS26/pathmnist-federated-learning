"""
Unit tests for Dirichlet Partitioning and Validation Apportionment.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import pytest
import numpy as np
from research.data.partitioner import generate_dirichlet_partitions
from research.data.validation_splitter import generate_client_validation_shards


def test_dirichlet_partitioning_synthetic():
    # Synthetic dataset with 9 classes, 4500 samples
    num_samples = 4500
    num_clients = 5
    labels = np.random.randint(0, 9, size=num_samples)
    
    partitions = generate_dirichlet_partitions(
        labels=labels,
        num_clients=num_clients,
        alpha=0.3,
        partition_seed=42,
        manifest_path=None
    )
    
    # 1. Total sample conservation
    total_assigned = sum(len(p) for p in partitions.values())
    assert total_assigned == num_samples, f"Sample count {total_assigned} != {num_samples}"
    
    # 2. Mutual exclusivity
    all_indices = np.concatenate([partitions[c] for c in range(num_clients)])
    assert len(np.unique(all_indices)) == num_samples, "Duplicate sample indices detected!"
    
    # 3. Non-empty client total partitions
    for c in range(num_clients):
        assert len(partitions[c]) > 0, f"Client {c} has empty partition!"
        
    # 4. Class histograms integrity
    for c in range(num_clients):
        c_lbls = labels[partitions[c]]
        assert len(c_lbls) == len(partitions[c])


def test_validation_shards_apportionment_synthetic():
    num_val = 1000
    num_train = 4500
    num_clients = 5
    
    train_labels = np.random.randint(0, 9, size=num_train)
    val_labels = np.random.randint(0, 9, size=num_val)
    
    train_partitions = generate_dirichlet_partitions(
        labels=train_labels,
        num_clients=num_clients,
        alpha=0.3,
        partition_seed=42,
        manifest_path=None
    )
    
    val_shards = generate_client_validation_shards(
        val_labels=val_labels,
        train_partitions=train_partitions,
        train_labels=train_labels,
        num_clients=num_clients,
        val_seed=42,
        manifest_path=None
    )
    
    # 1. Exact conservation
    total_assigned = sum(len(s) for s in val_shards.values())
    assert total_assigned == num_val, f"Validation count {total_assigned} != {num_val}"
    
    # 2. Mutual exclusivity
    all_val = np.concatenate([val_shards[c] for c in range(num_clients)])
    assert len(np.unique(all_val)) == num_val, "Duplicate validation indices detected!"
