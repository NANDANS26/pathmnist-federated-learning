"""
Dirichlet Non-IID Client Partitioning for Federated Learning.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple


PARTITION_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "configs", "partition_manifest.json"
)


def generate_dirichlet_partitions(
    labels: np.ndarray,
    num_clients: int = 5,
    alpha: float = 0.3,
    partition_seed: int = 42,
    manifest_path: str = None
) -> Dict[int, np.ndarray]:
    """
    Partitions dataset training indices across clients using class-wise Dirichlet sampling.
    
    Args:
        labels: 1D or 2D array of class labels (N,) or (N, 1).
        num_clients: Number of federated clients (default: 5).
        alpha: Dirichlet concentration parameter (default: 0.3).
        partition_seed: Deterministic seed for NumPy generator (default: 42).
        manifest_path: Destination path for partition_manifest.json.
        
    Returns:
        Dict mapping client index (0..num_clients-1) to 1D integer numpy array of sample indices.
    """
    labels_1d = labels.squeeze()
    total_samples = len(labels_1d)
    unique_classes = np.unique(labels_1d)
    num_classes = len(unique_classes)
    
    rng = np.random.default_rng(partition_seed)
    
    client_indices: Dict[int, List[int]] = {c: [] for c in range(num_clients)}
    
    # Class-wise Dirichlet partition
    for cls in unique_classes:
        cls_indices = np.where(labels_1d == cls)[0]
        rng.shuffle(cls_indices)
        
        # Draw client proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        
        # Cumulative sample counts
        splits = (np.cumsum(proportions) * len(cls_indices)).astype(int)
        
        # Assign slices to clients
        start_idx = 0
        for c in range(num_clients):
            end_idx = splits[c] if c < num_clients - 1 else len(cls_indices)
            client_indices[c].extend(cls_indices[start_idx:end_idx].tolist())
            start_idx = end_idx
            
    # Convert lists to numpy arrays
    final_partitions = {c: np.array(indices, dtype=np.int64) for c, indices in client_indices.items()}
    
    # MANDATORY ASSERTIONS
    allocated_count = sum(len(p) for p in final_partitions.values())
    assert allocated_count == total_samples, (
        f"Sample conservation failed! Allocated {allocated_count} != total {total_samples}"
    )
    
    all_assigned_indices = np.concatenate([final_partitions[c] for c in range(num_clients)])
    unique_assigned = np.unique(all_assigned_indices)
    assert len(unique_assigned) == total_samples, "Duplicate indices detected across client partitions!"
    assert np.array_equal(np.sort(unique_assigned), np.arange(total_samples)), (
        "Partition indices do not match contiguous 0..N-1 original indices!"
    )
    
    # Empty client check: every client MUST have at least one sample
    client_counts = {c: len(final_partitions[c]) for c in range(num_clients)}
    for c, count in client_counts.items():
        assert count > 0, f"Client {c} has zero training samples! Empty clients are not accepted."
        
    # Calculate class histograms per client
    class_histograms = {}
    for c in range(num_clients):
        c_labels = labels_1d[final_partitions[c]]
        counts = {int(k): int(np.sum(c_labels == k)) for k in unique_classes}
        assert sum(counts.values()) == len(final_partitions[c]), "Class histogram sum does not match sample count!"
        class_histograms[c] = counts

    # Persist partition manifest
    manifest = {
        "dataset": "PathMNIST",
        "num_clients": num_clients,
        "alpha": alpha,
        "partition_seed": partition_seed,
        "total_training_samples": total_samples,
        "client_sample_counts": {f"Client_{c+1}": int(client_counts[c]) for c in range(num_clients)},
        "client_sample_counts_zero_based": {str(c): int(client_counts[c]) for c in range(num_clients)},
        "class_histograms": {f"Client_{c+1}": class_histograms[c] for c in range(num_clients)},
        "sample_conservation_verified": True,
        "mutual_exclusivity_verified": True,
        "no_empty_client_verified": True,
        "client_indices": {str(c): final_partitions[c].tolist() for c in range(num_clients)}
    }
    
    if manifest_path is not None:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
    return final_partitions
