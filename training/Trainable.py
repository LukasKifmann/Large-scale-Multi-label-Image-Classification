from typing import Protocol, Iterator
from abc import abstractmethod
import torch


class Trainable(Protocol):
    @abstractmethod
    def logits(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        pass

    @abstractmethod
    def requires_grad_(self, requires_grad: bool = True):
        pass

    @abstractmethod
    def train(self, train: bool = True):
        pass

    @abstractmethod
    def eval(self):
        pass
