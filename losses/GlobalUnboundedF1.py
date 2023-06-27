import torch
from torch import nn


class GlobalUnboundedF1(nn.Module):
    def __repr__(self) -> str:
        return self.__class__.__name__

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tp = torch.sum(y_hat * y)
        fp = torch.sum(y_hat * (1 - y))
        fn = torch.sum((1 - y_hat) * y)
        return torch.mean(1 - (2 * tp / (2 * tp + fn + fp + 1e-16)))
