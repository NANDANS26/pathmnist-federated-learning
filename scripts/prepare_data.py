"""
Data preparation script for PathMNIST.
Downloads PathMNIST and sets up memory-mapped arrays.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.data.pathmnist_loader import acquire_and_prepare_pathmnist

if __name__ == "__main__":
    print("Beginning official PathMNIST data acquisition and preparation...")
    manifest = acquire_and_prepare_pathmnist()
    print("PathMNIST successfully acquired and verified!")
    print(f"NPZ MD5: {manifest['npz_md5']}")
    print(f"NPZ SHA256: {manifest['npz_sha256']}")
    print(f"Train samples: {manifest['splits']['train']['sample_count']}")
    print(f"Val samples: {manifest['splits']['val']['sample_count']}")
    print(f"Test samples (held out): {manifest['splits']['test']['sample_count']}")
