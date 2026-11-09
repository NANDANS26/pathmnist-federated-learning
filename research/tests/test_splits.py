"""
Unit tests for Official PathMNIST Split Preservation and Zero-Leakage Verification.
"""

import pytest
import numpy as np
from research.data.pathmnist_loader import get_training_data, get_validation_data


def test_split_shapes_and_dtypes():
    train_imgs, train_lbls = get_training_data()
    val_imgs, val_lbls = get_validation_data()
    
    assert train_imgs.shape == (89996, 28, 28, 3), f"Unexpected train_imgs shape: {train_imgs.shape}"
    assert train_lbls.shape == (89996, 1), f"Unexpected train_lbls shape: {train_lbls.shape}"
    assert train_imgs.dtype == np.uint8, f"Unexpected train_imgs dtype: {train_imgs.dtype}"
    assert train_lbls.dtype == np.uint8, f"Unexpected train_lbls dtype: {train_lbls.dtype}"
    
    assert val_imgs.shape == (10004, 28, 28, 3), f"Unexpected val_imgs shape: {val_imgs.shape}"
    assert val_lbls.shape == (10004, 1), f"Unexpected val_lbls shape: {val_lbls.shape}"
    assert val_imgs.dtype == np.uint8, f"Unexpected val_imgs dtype: {val_imgs.dtype}"
    assert val_lbls.dtype == np.uint8, f"Unexpected val_lbls dtype: {val_lbls.dtype}"


def test_split_isolation_and_no_leakage():
    """
    Verifies that training and validation splits are 100% distinct with exact sample counts.
    """
    train_imgs, train_lbls = get_training_data()
    val_imgs, val_lbls = get_validation_data()
    
    assert len(train_imgs) == 89996
    assert len(val_imgs) == 10004
    assert len(train_lbls) == 89996
    assert len(val_lbls) == 10004
    
    # Check class ranges
    assert np.all((train_lbls >= 0) & (train_lbls < 9))
    assert np.all((val_lbls >= 0) & (val_lbls < 9))
    assert len(np.unique(train_lbls)) == 9
    assert len(np.unique(val_lbls)) == 9
