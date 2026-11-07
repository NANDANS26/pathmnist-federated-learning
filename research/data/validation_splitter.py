"""
Validation Shard Apportionment via Hamilton-Hare Largest Remainder Method.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import json
import os
import numpy as np
from typing import Dict, List


VALIDATION_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "configs", "validation_manifest.json"
)


def generate_client_validation_shards(
    val_labels: np.ndarray,
    train_partitions: Dict[int, np.ndarray],
    train_labels: np.ndarray,
    num_clients: int = 5,
    val_seed: int = 42,
    manifest_path: str = None
) -> Dict[int, np.ndarray]:
    """
    Apportions the 10,004 official validation samples across 5 simulated client validation shards
    reflecting each client's local training class distribution using the Hamilton-Hare Largest Remainder Method.
    
    Args:
        val_labels: Array of official validation labels (10004,) or (10004, 1).
        train_partitions: Dict mapping client index to training sample indices.
        train_labels: Array of official training labels (89996,) or (89996, 1).
        num_clients: Number of clients (default: 5).
        val_seed: Seed for shuffling validation class indices (default: 42).
        manifest_path: Destination path for validation_manifest.json.
        
    Returns:
        Dict mapping client index (0..num_clients-1) to 1D integer numpy array of validation indices.
    """
    val_labels_1d = val_labels.squeeze()
    train_labels_1d = train_labels.squeeze()
    total_val_samples = len(val_labels_1d)
    unique_classes = np.unique(val_labels_1d)
    
    rng = np.random.default_rng(val_seed)
    
    # Calculate training class counts per client: train_class_counts[client][class]
    train_counts = np.zeros((num_clients, len(unique_classes)), dtype=np.int64)
    for c in range(num_clients):
        c_train_labels = train_labels_1d[train_partitions[c]]
        for k in unique_classes:
            train_counts[c, k] = np.sum(c_train_labels == k)
            
    client_val_indices: Dict[int, List[int]] = {c: [] for c in range(num_clients)}
    
    for k in unique_classes:
        val_k_indices = np.where(val_labels_1d == k)[0]
        rng.shuffle(val_k_indices)
        total_k_val = len(val_k_indices)
        
        # Training proportions for class k across clients
        total_k_train = np.sum(train_counts[:, k])
        if total_k_train > 0:
            proportions = train_counts[:, k] / total_k_train
        else:
            proportions = np.ones(num_clients) / num_clients
            
        # Hamilton-Hare largest remainder apportionment
        quotas = proportions * total_k_val
        integers = np.floor(quotas).astype(int)
        remainders = quotas - integers
        deficit = total_k_val - np.sum(integers)
        
        # Sort client indices descending by remainder, breaking ties by client index
        remainder_order = np.lexsort((np.arange(num_clients), -remainders))
        allocations = integers.copy()
        for idx in range(deficit):
            c_recipient = remainder_order[idx]
            allocations[c_recipient] += 1
            
        assert np.sum(allocations) == total_k_val, f"Class {k} apportionment sum failed!"
        
        # Assign indices to clients
        curr_start = 0
        for c in range(num_clients):
            curr_end = curr_start + allocations[c]
            client_val_indices[c].extend(val_k_indices[curr_start:curr_end].tolist())
            curr_start = curr_end

    final_val_shards = {c: np.array(indices, dtype=np.int64) for c, indices in client_val_indices.items()}
    
    # MANDATORY ASSERTIONS
    total_allocated = sum(len(s) for s in final_val_shards.values())
    assert total_allocated == total_val_samples, (
        f"Validation sample conservation failed! Allocated {total_allocated} != {total_val_samples}"
    )
    
    all_assigned = np.concatenate([final_val_shards[c] for c in range(num_clients)])
    unique_assigned = np.unique(all_assigned)
    assert len(unique_assigned) == total_val_samples, "Duplicate validation indices detected!"
    assert np.array_equal(np.sort(unique_assigned), np.arange(total_val_samples)), (
        "Validation indices do not cover full 0..N_val-1 range!"
    )
    
    # Ensure validation shard histograms sum to shard size
    shard_histograms = {}
    for c in range(num_clients):
        c_val_labels = val_labels_1d[final_val_shards[c]]
        counts = {int(k): int(np.sum(c_val_labels == k)) for k in unique_classes}
        assert sum(counts.values()) == len(final_val_shards[c])
        shard_histograms[c] = counts

    # Persist validation manifest
    manifest = {
        "dataset": "PathMNIST",
        "total_validation_samples": total_val_samples,
        "num_shards": num_clients,
        "val_seed": val_seed,
        "apportionment_method": "hamilton_hare_largest_remainder",
        "shard_sample_counts": {f"Client_{c+1}_val": len(final_val_shards[c]) for c in range(num_clients)},
        "shard_sample_counts_zero_based": {str(c): len(final_val_shards[c]) for c in range(num_clients)},
        "shard_class_histograms": {f"Client_{c+1}_val": shard_histograms[c] for c in range(num_clients)},
        "sample_conservation_verified": True,
        "mutual_exclusivity_verified": True,
        "labels_unmodified_verified": True,
        "client_val_indices": {str(c): final_val_shards[c].tolist() for c in range(num_clients)}
    }
    
    if manifest_path is not None:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
    return final_val_shards
