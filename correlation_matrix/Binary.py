from ..graph import EdgeWeightedDigraph
import torch
from typing import Tuple
from ..util import prog
from .CorrelationMatrixBuilder import CorrelationMatrixBuilder


class Binary(CorrelationMatrixBuilder):
    @property
    def threshold(self) -> float:
        return self.__threshold

    def __init__(self, threshold: float):
        self.__threshold = threshold

    def __repr__(self):
        return super().__repr__() + f"(threshold={self.threshold})"

    def __call__(self, g: EdgeWeightedDigraph) -> Tuple[torch.Tensor, torch.Tensor]:
        ii = []
        jj = []
        weights = []
        for v, w in prog(
            "computing binary correlation matrix: discovering neighbours",
            g.arches,
            g.arch_count,
        ):
            if g.edge_weight(v, w) >= self.threshold:
                ii.append(w)
                jj.append(v)
                weights.append(1)
        for v in prog(
            "computing binary correlation matrix: adding self-connections",
            range(g.node_count),
        ):
            ii.append(v)
            jj.append(v)
            weights.append(1)
        return torch.tensor([ii, jj], dtype=torch.long), torch.tensor(
            weights, dtype=torch.float
        )
