"""
1-Round Functional Smoke Test on Synthetic Dataset.
Verifies client training, FedAvg/FedProx optimization, metrics, JSONL logging, and checkpointing.
"""

import sys
import os
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.models.compact_cnn import CompactPathCNN
from research.core.fedavg import run_fedavg
from research.core.fedprox import run_fedprox


def run_smoke_test():
    print("=== STARTING 1-ROUND FUNCTIONAL SMOKE TEST ===")
    
    # Generate tiny synthetic dataset: 500 training images, 100 validation images
    np.random.seed(42)
    torch.manual_seed(42)
    
    synth_train_images = np.random.randint(0, 256, size=(500, 28, 28, 3), dtype=np.uint8)
    synth_train_labels = np.random.randint(0, 9, size=(500,), dtype=np.uint8)
    
    synth_val_images = np.random.randint(0, 256, size=(100, 28, 28, 3), dtype=np.uint8)
    synth_val_labels = np.random.randint(0, 9, size=(100,), dtype=np.uint8)
    
    # 5 clients with 100 samples each
    train_partitions = {c: np.arange(c * 100, (c + 1) * 100) for c in range(5)}
    val_shards = {c: np.arange(c * 20, (c + 1) * 20) for c in range(5)}
    
    # Client 5 has noisy labels
    noisy_labels_c4 = synth_train_labels[train_partitions[4]].copy()
    noisy_labels_c4[:40] = (noisy_labels_c4[:40] + 1) % 9
    client_noisy_labels = {4: noisy_labels_c4}
    
    # 1. Test FedAvg 1-round smoke test
    print("\n1. Testing FedAvg (1 round)...")
    m_avg = CompactPathCNN()
    res_avg = run_fedavg(
        global_model=m_avg,
        train_images=synth_train_images,
        train_labels=synth_train_labels,
        client_train_partitions=train_partitions,
        client_noisy_labels=client_noisy_labels,
        val_images=synth_val_images,
        val_labels=synth_val_labels,
        client_val_indices=val_shards,
        num_rounds=1,
        lr=0.01,
        batch_size=32,
        local_epochs=1,
        run_name="smoke_test_fedavg"
    )
    assert res_avg["completed_rounds"] == 1, "FedAvg smoke test failed to complete round 1!"
    print("FedAvg smoke test PASSED!")
    
    # 2. Test FedProx 1-round smoke test
    print("\n2. Testing FedProx (mu=0.01, 1 round)...")
    m_prox = CompactPathCNN()
    res_prox = run_fedprox(
        global_model=m_prox,
        train_images=synth_train_images,
        train_labels=synth_train_labels,
        client_train_partitions=train_partitions,
        client_noisy_labels=client_noisy_labels,
        val_images=synth_val_images,
        val_labels=synth_val_labels,
        client_val_indices=val_shards,
        num_rounds=1,
        lr=0.01,
        mu=0.01,
        batch_size=32,
        local_epochs=1,
        run_name="smoke_test_fedprox"
    )
    assert res_prox["completed_rounds"] == 1, "FedProx smoke test failed to complete round 1!"
    print("FedProx smoke test PASSED!")
    
    # Check that logs were written
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "logs")
    assert os.path.exists(os.path.join(log_dir, "smoke_test_fedavg_metrics.jsonl"))
    assert os.path.exists(os.path.join(log_dir, "smoke_test_fedprox_metrics.jsonl"))
    print("\n=== FUNCTIONAL SMOKE TEST COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    run_smoke_test()
