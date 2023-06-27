import torch
from abc import ABC, abstractmethod
from typing import Iterable


class Metric(ABC):
    def __repr__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, labels: Iterable[set[int]]) -> float:
        pass
