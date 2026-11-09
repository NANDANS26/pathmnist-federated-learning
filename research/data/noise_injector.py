"""
Exact 40% Symmetric Label Noise Injection for Client 5.
"""

import json
import os
import numpy as np
from typing import Dict, Tuple


NOISE_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "configs", "noise_manifest.json"
)


def inject_symmetric_label_noise(
    client_train_indices: np.ndarray,
    train_labels: np.ndarray,
    target_client_human: str = "Client 5",
    target_client_index: int = 4,
    noise_rate: float = 0.40,
    num_classes: int = 9,
    selection_seed: int = 42,
    replacement_seed: int = 1042,
    manifest_path: str = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Applies exact symmetric multiclass label flipping to 40% of Client 5's training partition.
    
    Formula:
        y_noisy = (y_clean + r) % num_classes, where r ~ Uniform(1..num_classes-1)
        
    Args:
        client_train_indices: 1D array of original training indices assigned to Client 5.
        train_labels: Complete official training labels array (89996,) or (89996, 1).
        target_client_human: Human-readable client name (default: "Client 5").
        target_client_index: Zero-based client index (default: 4).
        noise_rate: Proportion of samples to corrupt (default: 0.40).
        num_classes: Total number of classes (default: 9).
        selection_seed: Seed for choosing corrupted indices without replacement (default: 42).
        replacement_seed: Seed for drawing class offsets r (default: 1042).
        manifest_path: Destination path for noise_manifest.json.
        
    Returns:
        noisy_labels: 1D array of labels for Client 5 with exactly round(0.40 * M_5) corruptions.
        noise_mask: Boolean array of shape (M_5,) where True indicates a corrupted label.
        manifest: Dictionary of metadata and verification metrics.
    """
    labels_1d = train_labels.squeeze()
    client_clean_labels = labels_1d[client_train_indices].copy()
    m_5 = len(client_train_indices)
    
    num_noisy = int(round(noise_rate * m_5))
    
    # 1. Select sample positions within Client 5 without replacement
    selection_rng = np.random.default_rng(selection_seed)
    chosen_positions = selection_rng.choice(m_5, size=num_noisy, replace=False)
    
    # Boolean noise mask
    noise_mask = np.zeros(m_5, dtype=bool)
    noise_mask[chosen_positions] = True
    
    # 2. Draw class replacement offsets r in [1, num_classes-1]
    replacement_rng = np.random.default_rng(replacement_seed)
    offsets = replacement_rng.integers(1, num_classes, size=num_noisy)
    
    client_noisy_labels = client_clean_labels.copy()
    for pos, r in zip(chosen_positions, offsets):
        orig_y = client_clean_labels[pos]
        flipped_y = (orig_y + r) % num_classes
        assert flipped_y != orig_y, f"Flipped label must differ from original! orig={orig_y}, flipped={flipped_y}"
        assert 0 <= flipped_y < num_classes, f"Flipped label {flipped_y} out of bounds!"
        client_noisy_labels[pos] = flipped_y
        
    # MANDATORY ASSERTIONS
    actual_corruptions = np.sum(client_clean_labels != client_noisy_labels)
    assert actual_corruptions == num_noisy, (
        f"Realized corruption count {actual_corruptions} != expected {num_noisy}"
    )
    assert np.all((client_noisy_labels >= 0) & (client_noisy_labels < num_classes))
    assert np.sum(noise_mask) == num_noisy
    
    # Class distribution before and after noise
    orig_dist = {int(k): int(np.sum(client_clean_labels == k)) for k in range(num_classes)}
    noisy_dist = {int(k): int(np.sum(client_noisy_labels == k)) for k in range(num_classes)}
    
    manifest = {
        "dataset": "PathMNIST",
        "target_client_human": target_client_human,
        "target_client_index": target_client_index,
        "total_client_samples": m_5,
        "nominal_noise_rate": noise_rate,
        "exact_corruptions_count": num_noisy,
        "realized_noise_percentage": float(num_noisy / m_5 * 100.0),
        "selection_seed": selection_seed,
        "replacement_seed": replacement_seed,
        "num_classes": num_classes,
        "noise_type": "symmetric_multiclass_flipping",
        "formula": "y_noisy = (y_clean + r) % 9 where r in Uniform(1..8)",
        "all_flips_differ_verified": True,
        "other_clients_clean_guaranteed": True,
        "validation_clean_guaranteed": True,
        "original_class_distribution": orig_dist,
        "noisy_class_distribution": noisy_dist,
        "corrupted_client_positions": chosen_positions.tolist(),
        "noise_mask": noise_mask.tolist()
    }
    
    if manifest_path is not None:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
    return client_noisy_labels, noise_mask, manifest
