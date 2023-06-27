import torch
from .ExampleBasedMetric import ExampleBasedMetric
from abc import abstractmethod
from typing import Iterable, Optional
from ..util import prog


class SortingBasedMetric(ExampleBasedMetric):
    def compute_for_example(self, row: torch.Tensor, labels: set[int]) -> float:
        sorted_idcs = row.sort(descending=True).indices
        return self.compute_from_sorting_order(row, sorted_idcs, labels)

    @abstractmethod
    def compute_from_sorting_order(
        self, row: torch.Tensor, sorted_idcs: torch.Tensor, labels: set[int]
    ) -> float:
        pass
