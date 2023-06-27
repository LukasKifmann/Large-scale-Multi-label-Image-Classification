from .linreg import linreg
import torch


def logreg(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(linreg(X, W))
