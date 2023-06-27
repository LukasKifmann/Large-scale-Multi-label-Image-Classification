import torch
from torch import nn


class SigmoidF1(nn.Module):
    @property
    def beta(self) -> float:
        return self.__beta

    @property
    def eta(self) -> float:
        return self.__eta

    def __init__(
        self,
        beta: float = 1,
        eta: float = 0,
    ):
        super(SigmoidF1, self).__init__()
        self.__beta = beta
        self.__eta = eta

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(beta={self.beta}, eta={self.eta})"

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(self.beta * (y_hat + self.eta))
        tp = torch.sum(sig * y, dim=0)
        fp = torch.sum(sig * (1 - y), dim=0)
        fn = torch.sum((1 - sig) * y, dim=0)
        return torch.mean(1 - (2 * tp / (2 * tp + fn + fp + 1e-16)))
