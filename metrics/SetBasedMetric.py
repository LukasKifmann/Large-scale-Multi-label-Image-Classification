import torch
from abc import abstractmethod
from .ExampleBasedMetric import ExampleBasedMetric


class SetBasedMetric(ExampleBasedMetric):
    @property
    def threshold(self) -> float:
        return self.__threshold

    def __init__(self, threshold: float):
        self.__threshold = threshold

    def __repr__(self):
        return super().__repr__() + f"({self.threshold})"

    def compute_for_example(self, row: torch.Tensor, labels: set[int]) -> float:
        labels_pred = set(i.item() for i in (row >= self.threshold).nonzero())
        return self.compute_from_sets(labels_pred, labels)  # type: ignore

    @abstractmethod
    def compute_from_sets(self, labels_pred: set[int], labels: set[int]) -> float:
        pass
