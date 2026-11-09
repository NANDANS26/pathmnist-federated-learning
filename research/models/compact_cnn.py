"""
CompactPathCNN Architecture and Canonical State Hashing.
Phase 2 Implementation for IndiaAI Fellowship Screening.
"""

import hashlib
import torch
import torch.nn as nn
from typing import Dict, Tuple


class CompactPathCNN(nn.Module):
    """
    Lightweight 2-stage CNN with Adaptive Average Pooling and 2 Linear layers.
    Optimized for dual-core CPU federated learning on PathMNIST (28x28x3 RGB).
    
    Total Trainable Parameters: Exactly 36,425.
    Complexity: 4,307,008 MACs (~8.61 MFLOPs) per image.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 9):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 3, 28, 28), float32 in range [0.0, 1.0].
        Returns:
            Logits tensor of shape (B, 9).
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        logits = self.fc2(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_canonical_model_hash(model: nn.Module) -> str:
    """
    Computes deterministic SHA-256 hash over sorted parameter names, dtypes, shapes,
    and contiguous raw parameter byte buffers.
    """
    hasher = hashlib.sha256()
    state_dict = model.state_dict()
    for key in sorted(state_dict.keys()):
        param = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(param.dtype).encode("utf-8"))
        hasher.update(str(list(param.shape)).encode("utf-8"))
        hasher.update(param.numpy().tobytes())
    return hasher.hexdigest()


def preprocess_batch(batch_uint8, batch_labels) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transforms raw batch from numpy storage into PyTorch CNN tensors.
    
    Transforms:
        batch_uint8: (B, 28, 28, 3) NHWC uint8 -> (B, 3, 28, 28) NCHW float32 [0.0, 1.0]
        batch_labels: (B, 1) or (B,) uint8 -> (B,) int64
    """
    if not isinstance(batch_uint8, torch.Tensor):
        x = torch.from_numpy(batch_uint8)
    else:
        x = batch_uint8
    x = x.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    
    if not isinstance(batch_labels, torch.Tensor):
        y = torch.from_numpy(batch_labels)
    else:
        y = batch_labels
    if y.ndim > 1:
        y = y.squeeze(1)
    y = y.to(torch.int64)
    return x, y
