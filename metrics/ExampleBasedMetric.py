from .Metric import Metric
import torch
from abc import abstractmethod
from typing import Iterable, Optional
from ..util import prog


class ExampleBasedMetric(Metric):
    def __call__(
        self,
        y_pred: torch.Tensor,
        labels: Iterable[set[int]],
        verbose: Optional[bool] = None,
    ) -> float:
        count = y_pred.shape[0]
        total = 0
        for i, row_labels in (
            prog(f"computing metric {repr(self)}", enumerate(labels), count)
            if verbose
            else enumerate(labels)
        ):
            total += self.compute_for_example(y_pred[i, :], row_labels)
        return total / count

    @abstractmethod
    def compute_for_example(self, row: torch.Tensor, labels: set[int]) -> float:
        pass
