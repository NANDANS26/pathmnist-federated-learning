"""
Unit tests for 40% Symmetric Label Noise Injection on Client 5.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import pytest
import numpy as np
from research.data.noise_injector import inject_symmetric_label_noise


def test_exact_40_percent_noise():
    # Synthetic client training set with known size M_5 = 1000
    m_5 = 1000
    client_indices = np.arange(m_5)
    train_labels = np.random.randint(0, 9, size=m_5)
    
    noisy_labels, noise_mask, manifest = inject_symmetric_label_noise(
        client_train_indices=client_indices,
        train_labels=train_labels,
        noise_rate=0.40,
        selection_seed=42,
        replacement_seed=1042
    )
    
    expected_noisy_count = int(round(0.40 * m_5))
    assert expected_noisy_count == 400
    
    actual_corruptions = np.sum(train_labels != noisy_labels)
    assert actual_corruptions == expected_noisy_count, (
        f"Realized corruptions {actual_corruptions} != {expected_noisy_count}"
    )
    assert np.sum(noise_mask) == expected_noisy_count
    
    # Check that all corrupted positions differ from original
    corrupted_positions = np.where(noise_mask)[0]
    for pos in corrupted_positions:
        assert noisy_labels[pos] != train_labels[pos]
        assert 0 <= noisy_labels[pos] < 9
        
    # Check that clean positions are preserved
    clean_positions = np.where(~noise_mask)[0]
    assert np.array_equal(noisy_labels[clean_positions], train_labels[clean_positions])


def test_noise_determinism():
    m_5 = 500
    client_indices = np.arange(m_5)
    train_labels = np.random.randint(0, 9, size=m_5)
    
    l1, m1, _ = inject_symmetric_label_noise(client_indices, train_labels, noise_rate=0.40, selection_seed=42, replacement_seed=1042)
    l2, m2, _ = inject_symmetric_label_noise(client_indices, train_labels, noise_rate=0.40, selection_seed=42, replacement_seed=1042)
    
    assert np.array_equal(l1, l2), "Noise injection is not deterministic!"
    assert np.array_equal(m1, m2), "Noise mask is not deterministic!"
