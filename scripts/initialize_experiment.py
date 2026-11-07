"""
Experiment Artifact Initializer.
Generates deterministic partition manifest, validation manifest, noise manifest, and initial model checkpoint.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import sys
import os
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.data.pathmnist_loader import get_training_data, get_validation_data
from research.data.partitioner import generate_dirichlet_partitions, PARTITION_MANIFEST_PATH
from research.data.validation_splitter import generate_client_validation_shards, VALIDATION_MANIFEST_PATH
from research.data.noise_injector import inject_symmetric_label_noise, NOISE_MANIFEST_PATH
from research.models.compact_cnn import CompactPathCNN, compute_canonical_model_hash


def initialize_all():
    print("=== INITIALIZING EXPERIMENTAL DATA ARTIFACTS ===")
    
    # 1. Load data
    train_images, train_labels = get_training_data()
    val_images, val_labels = get_validation_data()
    
    # 2. Partition training set (89,996) across 5 clients using Dirichlet(alpha=0.3, seed=42)
    print("\n1. Generating Dirichlet partitions (alpha=0.3, 5 clients, seed=42)...")
    train_partitions = generate_dirichlet_partitions(
        labels=train_labels,
        num_clients=5,
        alpha=0.3,
        partition_seed=42,
        manifest_path=PARTITION_MANIFEST_PATH
    )
    for c in range(5):
        print(f"   Client {c+1}: {len(train_partitions[c]):,} training samples")
        
    # 3. Apportion validation set (10,004) across 5 shards using Hamilton-Hare largest remainder
    print("\n2. Apportioning validation shards (Hamilton-Hare, seed=42)...")
    val_shards = generate_client_validation_shards(
        val_labels=val_labels,
        train_partitions=train_partitions,
        train_labels=train_labels,
        num_clients=5,
        val_seed=42,
        manifest_path=VALIDATION_MANIFEST_PATH
    )
    for c in range(5):
        print(f"   Client {c+1} Validation Shard: {len(val_shards[c]):,} samples")
        
    # 4. Inject 40% symmetric label noise on Client 5 (zero-based index 4)
    print("\n3. Injecting 40% symmetric multiclass label noise on Client 5...")
    client_5_indices = train_partitions[4]
    noisy_labels_c4, noise_mask_c4, noise_manifest = inject_symmetric_label_noise(
        client_train_indices=client_5_indices,
        train_labels=train_labels,
        target_client_human="Client 5",
        target_client_index=4,
        noise_rate=0.40,
        selection_seed=42,
        replacement_seed=1042,
        manifest_path=NOISE_MANIFEST_PATH
    )
    print(f"   Client 5 total samples: {len(client_5_indices):,}")
    print(f"   Client 5 corrupted count: {np.sum(noise_mask_c4):,} ({noise_manifest['realized_noise_percentage']:.2f}%)")
    
    # 5. Create canonical initial model checkpoint
    print("\n4. Initializing CompactPathCNN initial model (seed=42)...")
    torch.manual_seed(42)
    initial_model = CompactPathCNN(in_channels=3, num_classes=9)
    canonical_hash = compute_canonical_model_hash(initial_model)
    print(f"   Trainable parameter count: {sum(p.numel() for p in initial_model.parameters()):,}")
    print(f"   Canonical Model Hash (SHA-256): {canonical_hash}")
    
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "initial_model.pt")
    torch.save(initial_model.state_dict(), ckpt_path)
    print(f"   Saved initial model to: {ckpt_path}")
    
    print("\n=== EXPERIMENTAL INITIALIZATION COMPLETE ===")


def main():
    initialize_all()


if __name__ == "__main__":
    main()
