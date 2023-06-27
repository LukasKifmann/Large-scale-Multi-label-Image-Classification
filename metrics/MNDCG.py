import torch
from math import log2
from .SortingBasedMetric import SortingBasedMetric


class MNDCG(SortingBasedMetric):
    @property
    def k(self) -> int:
        return self.__k

    def __init__(self, k: int = -1):
        self.__k = k

    def __repr__(self) -> str:
        return super().__repr__() + f"({self.k})"

    def compute_from_sorting_order(
        self, row: torch.Tensor, sorted_idcs: torch.Tensor, labels: set[int]
    ) -> float:
        DCG = 0
        IDCG = 0
        label_count = len(labels)
        for i in range(min(self.__k, row.shape[0]) if self.__k != -1 else row.shape[0]):
            gain = 1 / log2(2 + i)
            if i < label_count:
                IDCG += gain
            if sorted_idcs[i].item() in labels:
                DCG += gain
        if IDCG == 0:
            return 1
        else:
            return DCG / IDCG
