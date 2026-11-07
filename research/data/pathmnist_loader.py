"""
PathMNIST Dataset Acquisition, Split Preservation, and Memory-Mapped Storage.
Phase 2 Implementation for IndiaAI Fellowship Screening.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import os
import hashlib
import json
import numpy as np
import medmnist
from medmnist import PathMNIST


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
MMAP_DIR = os.path.join(DATA_DIR, "pathmnist_mmap")
MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "manifests", "dataset_manifest.json"
)


def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_file_md5(filepath: str, chunk_size: int = 65536) -> str:
    """Computes MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def acquire_and_prepare_pathmnist(root_dir: str = DATA_DIR, mmap_dir: str = MMAP_DIR) -> dict:
    """
    Downloads PathMNIST via official MedMNIST API, verifies integrity,
    extracts train and val splits to uncompressed .npy files for memory-mapped access,
    and writes the provenance manifest.
    
    Test split is preserved in the archive but NOT memory-mapped or materialized during Phase 2.
    """
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(mmap_dir, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    
    # Trigger download of official pathmnist.npz via MedMNIST dataset class
    train_dataset = PathMNIST(split="train", download=True, root=root_dir)
    val_dataset = PathMNIST(split="val", download=True, root=root_dir)
    
    npz_path = os.path.join(root_dir, "pathmnist.npz")
    assert os.path.exists(npz_path), f"Expected {npz_path} to exist after MedMNIST download."
    
    # Verify MD5 against official MedMNIST specification
    expected_md5 = medmnist.INFO["pathmnist"]["MD5"]
    actual_md5 = compute_file_md5(npz_path)
    actual_sha256 = compute_file_sha256(npz_path)
    assert actual_md5 == expected_md5, f"MD5 mismatch! Expected {expected_md5}, got {actual_md5}"
    
    # Load raw npz archive
    with np.load(npz_path) as data:
        train_imgs = data["train_images"]
        train_lbls = data["train_labels"]
        val_imgs = data["val_images"]
        val_lbls = data["val_labels"]
        test_imgs = data["test_images"]
        test_lbls = data["test_labels"]
        
        # Verify official counts and dimensions
        assert train_imgs.shape == (89996, 28, 28, 3), f"Unexpected train_imgs shape: {train_imgs.shape}"
        assert train_lbls.shape == (89996, 1), f"Unexpected train_lbls shape: {train_lbls.shape}"
        assert val_imgs.shape == (10004, 28, 28, 3), f"Unexpected val_imgs shape: {val_imgs.shape}"
        assert val_lbls.shape == (10004, 1), f"Unexpected val_lbls shape: {val_lbls.shape}"
        assert test_imgs.shape == (7180, 28, 28, 3), f"Unexpected test_imgs shape: {test_imgs.shape}"
        assert test_lbls.shape == (7180, 1), f"Unexpected test_lbls shape: {test_lbls.shape}"
        
        assert train_imgs.dtype == np.uint8, f"Unexpected train_imgs dtype: {train_imgs.dtype}"
        assert train_lbls.dtype == np.uint8, f"Unexpected train_lbls dtype: {train_lbls.dtype}"
        
        # Save uncompressed .npy arrays for memory-mapping
        train_imgs_path = os.path.join(mmap_dir, "train_images.npy")
        train_lbls_path = os.path.join(mmap_dir, "train_labels.npy")
        val_imgs_path = os.path.join(mmap_dir, "val_images.npy")
        val_lbls_path = os.path.join(mmap_dir, "val_labels.npy")
        
        if not os.path.exists(train_imgs_path):
            np.save(train_imgs_path, train_imgs)
        if not os.path.exists(train_lbls_path):
            np.save(train_lbls_path, train_lbls)
        if not os.path.exists(val_imgs_path):
            np.save(val_imgs_path, val_imgs)
        if not os.path.exists(val_lbls_path):
            np.save(val_lbls_path, val_lbls)

    # Compile dataset manifest
    manifest = {
        "dataset_name": "PathMNIST",
        "benchmark_suite": "MedMNIST v2",
        "official_url": medmnist.INFO["pathmnist"]["url"],
        "npz_path": os.path.abspath(npz_path),
        "npz_file_size_bytes": os.path.getsize(npz_path),
        "npz_md5": actual_md5,
        "npz_sha256": actual_sha256,
        "classes": medmnist.INFO["pathmnist"]["label"],
        "num_classes": 9,
        "image_resolution": [28, 28, 3],
        "splits": {
            "train": {
                "sample_count": 89996,
                "image_shape": [89996, 28, 28, 3],
                "image_dtype": "uint8",
                "label_shape": [89996, 1],
                "label_dtype": "uint8",
                "mmap_image_file": os.path.abspath(train_imgs_path),
                "label_file": os.path.abspath(train_lbls_path)
            },
            "val": {
                "sample_count": 10004,
                "image_shape": [10004, 28, 28, 3],
                "image_dtype": "uint8",
                "label_shape": [10004, 1],
                "label_dtype": "uint8",
                "mmap_image_file": os.path.abspath(val_imgs_path),
                "label_file": os.path.abspath(val_lbls_path)
            },
            "test": {
                "sample_count": 7180,
                "image_shape": [7180, 28, 28, 3],
                "image_dtype": "uint8",
                "label_shape": [7180, 1],
                "label_dtype": "uint8",
                "status": "HELD_OUT_NOT_LOADED_IN_PHASE_2"
            }
        },
        "storage_mode": "memory_mapped_uint8_npy",
        "phase_2_status": "DATASET_ACQUIRED_AND_SPLIT_PRESERVED"
    }
    
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        
    return manifest


def get_training_data(mmap_dir: str = MMAP_DIR):
    """Returns memory-mapped training images and in-memory label array."""
    images_path = os.path.join(mmap_dir, "train_images.npy")
    labels_path = os.path.join(mmap_dir, "train_labels.npy")
    if not os.path.exists(images_path):
        acquire_and_prepare_pathmnist()
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path)
    return images, labels


def get_validation_data(mmap_dir: str = MMAP_DIR):
    """Returns memory-mapped validation images and in-memory label array."""
    images_path = os.path.join(mmap_dir, "val_images.npy")
    labels_path = os.path.join(mmap_dir, "val_labels.npy")
    if not os.path.exists(images_path):
        acquire_and_prepare_pathmnist()
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path)
    return images, labels


def get_test_data(root_dir: str = DATA_DIR):
    """
    Returns official PathMNIST test images and labels from pathmnist.npz.
    Strictly held-out: must only be called ONCE per algorithm after training is completed.
    """
    npz_path = os.path.join(root_dir, "pathmnist.npz")
    assert os.path.exists(npz_path), f"Expected {npz_path} to exist."
    with np.load(npz_path) as data:
        images = data["test_images"].copy()
        labels = data["test_labels"].copy()
    assert images.shape == (7180, 28, 28, 3), f"Unexpected test images shape: {images.shape}"
    assert labels.shape == (7180, 1), f"Unexpected test labels shape: {labels.shape}"
    return images, labels
