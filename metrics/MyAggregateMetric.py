from .AggregateMetricImpl import AggregateMetricImpl
from .Accuracy import Accuracy
from .Precision import Precision
from .Recall import Recall
from .f_score import f_score
from .MNDCG import MNDCG
import torch
from typing import Iterable


class MyAggregateMetric:
    @property
    def threshold(self) -> float:
        return self.__threshold

    @property
    def beta(self) -> int:
        return self.__beta

    @property
    def k(self) -> int:
        return self.__k

    def __init__(self, threshold: float = 0.5, beta: int = 1, k: int = 30):
        self.__threshold = threshold
        self.__beta = beta
        self.__k = k
        self.__accuracy = Accuracy(threshold)
        self.__precision = Precision(threshold)
        self.__recall = Recall(threshold)
        self.__mndcg = MNDCG(k)

    def __call__(
        self, y_pred: torch.Tensor, labels: Iterable[set[int]], verbose: bool = False
    ) -> dict[str, float]:
        temp = AggregateMetricImpl(
            self.__accuracy, self.__precision, self.__recall, self.__mndcg
        )(y_pred, labels, verbose)
        return {
            "Accuracy": temp[repr(self.__accuracy)],
            "Precision": temp[repr(self.__precision)],
            "Recall": temp[repr(self.__recall)],
            "F1": f_score(
                temp[repr(self.__precision)], temp[repr(self.__recall)], self.beta
            ),
            f"MNDCG@{self.k}": temp[repr(self.__mndcg)],
        }
