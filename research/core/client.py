"""
Federated Client Implementation.
Executes local training loops for FedAvg and FedProx with deterministic mini-batch shuffling.
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

from research.models.compact_cnn import CompactPathCNN, preprocess_batch
from research.optimizers.perturbed_sgd import PerturbedSGD


class FederatedClient:
    """
    Simulates a federated client with memory-mapped data access.
    Executes sequential CPU training for FedAvg or FedProx.
    """
    def __init__(
        self,
        client_id: int,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        client_indices: np.ndarray,
        algorithm: str = "FedAvg",
        lr: float = 0.01,
        mu: float = 0.01,
        batch_size: int = 32,
        local_epochs: int = 2,
        base_seed: int = 42
    ):
        """
        Args:
            client_id: Zero-based client index (0..4).
            train_images: Memory-mapped or array of all training images (89996, 28, 28, 3), uint8.
            train_labels: Array of labels for this client (or complete training labels), uint8.
            client_indices: Assigned original indices for this client.
            algorithm: 'FedAvg' or 'FedProx'.
            lr: Local SGD learning rate.
            mu: FedProx proximal coefficient.
            batch_size: Local batch size (default: 32).
            local_epochs: Local training epochs (default: 2).
            base_seed: Base random seed for reproducible mini-batch sampling (default: 42).
        """
        self.client_id = client_id
        self.human_id = f"Client {client_id + 1}"
        self.train_images = train_images
        self.train_labels = train_labels
        self.client_indices = client_indices
        self.num_samples = len(client_indices)
        
        self.algorithm = algorithm
        self.lr = lr
        self.mu = mu
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.base_seed = base_seed
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def train_round(
        self,
        global_model: CompactPathCNN,
        round_index: int
    ) -> Tuple[CompactPathCNN, Dict[str, Any]]:
        """
        Executes local training for the specified communication round.
        
        Returns:
            local_model: Trained local model instance.
            metrics: Dictionary of local training metrics and latency.
        """
        start_time = time.perf_counter()
        
        # 1. Initialize local model with received global parameters
        local_model = CompactPathCNN(in_channels=3, num_classes=9)
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()
        
        # 2. Setup optimizer
        if self.algorithm.lower() == "fedprox":
            optimizer = PerturbedSGD(local_model.parameters(), lr=self.lr, mu=self.mu)
            optimizer.set_global_parameters(global_model)
        else:
            optimizer = torch.optim.SGD(local_model.parameters(), lr=self.lr, momentum=0.0, weight_decay=0.0)
            
        epoch_losses = []
        total_batches = 0
        
        # 3. Local training loop
        for epoch in range(self.local_epochs):
            # Deterministic batch generator per client, round, and epoch
            epoch_seed = self.base_seed + 1000 * round_index + self.client_id + 100 * epoch
            rng = np.random.default_rng(epoch_seed)
            
            # Permute client sample positions deterministically
            local_perm = rng.permutation(self.num_samples)
            permuted_indices = self.client_indices[local_perm]
            
            num_batches = int(np.ceil(self.num_samples / self.batch_size))
            epoch_loss_accum = 0.0
            
            for b in range(num_batches):
                batch_slice = slice(b * self.batch_size, (b + 1) * self.batch_size)
                batch_idx = permuted_indices[batch_slice]
                
                # Fetch uint8 images on-the-fly from mmap array
                batch_imgs_raw = self.train_images[batch_idx]
                
                # If train_labels is client-specific (length == num_samples) or global
                if len(self.train_labels) == self.num_samples:
                    batch_lbls_raw = self.train_labels[local_perm[batch_slice]]
                else:
                    batch_lbls_raw = self.train_labels[batch_idx]
                    
                # Preprocess batch into PyTorch float32 tensors
                x, y = preprocess_batch(batch_imgs_raw, batch_lbls_raw)
                
                optimizer.zero_grad(set_to_none=True)
                logits = local_model(x)
                ce_loss = self.loss_fn(logits, y)
                ce_loss.backward()
                optimizer.step()
                
                epoch_loss_accum += ce_loss.item() * len(y)
                total_batches += 1
                
            epoch_losses.append(epoch_loss_accum / self.num_samples)
            
        duration = time.perf_counter() - start_time
        
        metrics = {
            "client_id": self.client_id,
            "human_id": self.human_id,
            "train_samples": self.num_samples,
            "mean_ce_loss": float(np.mean(epoch_losses)),
            "final_epoch_loss": float(epoch_losses[-1]),
            "duration_seconds": float(round(duration, 3)),
            "batches_processed": total_batches
        }
        
        return local_model, metrics
