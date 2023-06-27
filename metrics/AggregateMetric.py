from typing import Protocol, Iterable
from abc import abstractmethod
import torch


class AggregateMetric(Protocol):
    @abstractmethod
    def __call__(
        self, y_pred: torch.Tensor, labels: Iterable[set[int]], verbose: bool = False
    ) -> dict[str, float]:
        pass
