"""
Exact FedProx Proximal SGD Optimizer (Perturbed SGD).
Supervised by Dr. Sachin D N, IIIT Bhopal.
"""

import torch
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional


class PerturbedSGD(Optimizer):
    """
    Implements Proximal Stochastic Gradient Descent for FedProx.
    
    Update rule:
        d_p = grad_F + mu * (w - w_global)
        w <- w - lr * d_p
        
    Args:
        params: Model parameters.
        lr: Learning rate (default: 0.01).
        mu: Proximal regularization coefficient (default: 0.01).
    """
    def __init__(self, params, lr: float = required, mu: float = 0.01):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid proximal coefficient mu: {mu}")
            
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)
        self.global_params: Optional[List[torch.Tensor]] = None
        
    def set_global_parameters(self, global_model: torch.nn.Module):
        """
        Caches a detached, frozen clone of global model parameters for proximal penalty computation.
        Must be called at the beginning of each client's local training round.
        """
        self.global_params = [
            p.detach().clone() for p in global_model.parameters() if p.requires_grad
        ]
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single proximal optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        if self.global_params is None:
            raise RuntimeError("Global model parameters not set! Call optimizer.set_global_parameters() before step().")
            
        param_idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                d_p = p.grad.data
                w_global = self.global_params[param_idx]
                
                if mu != 0.0:
                    # In-place proximal gradient addition: d_p = grad + mu * (w - w_global)
                    d_p = d_p.add(p.data - w_global.data, alpha=mu)
                    
                p.data.add_(d_p, alpha=-lr)
                param_idx += 1
                
        return loss

    def compute_proximal_penalty(self, model: torch.nn.Module) -> float:
        """
        Computes scalar proximal penalty: 0.5 * mu * sum(||w - w_global||^2)
        using squared L2 norm for accurate diagnostic reporting.
        """
        if self.global_params is None:
            return 0.0
            
        total_sq_norm = 0.0
        param_idx = 0
        for p in model.parameters():
            if p.requires_grad:
                w_global = self.global_params[param_idx]
                diff = p.data - w_global.data
                total_sq_norm += torch.sum(diff ** 2).item()
                param_idx += 1
                
        mu = self.param_groups[0]["mu"]
        return 0.5 * mu * total_sq_norm
