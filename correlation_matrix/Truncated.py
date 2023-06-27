from ..graph import EdgeWeightedDigraph
import torch
from typing import Tuple
from ..util import prog
from .CorrelationMatrixBuilder import CorrelationMatrixBuilder


class Truncated(CorrelationMatrixBuilder):
    @property
    def threshold(self) -> float:
        return self.__threshold

    def __init__(self, threshold):
        self.__threshold = threshold

    def __repr__(self) -> str:
        return super().__repr__() + f"(threshold={self.threshold})"

    def __call__(self, g: EdgeWeightedDigraph) -> Tuple[torch.Tensor, torch.Tensor]:
        ii = []
        jj = []
        weights = []
        buckets = [[] for _ in g.nodes]
        for v, w in prog(
            "computing truncated correlation matrix: discovering child nodes",
            g.arches,
            g.arch_count,
        ):
            if g.edge_weight(v, w) >= self.threshold:
                buckets[v].append(w)
        for v in prog(
            "computing truncated correlation matrix: computing coefficients",
            g.nodes,
            g.node_count,
        ):
            weight_sum = 1 + sum(map(lambda w: g.edge_weight(v, w), buckets[v]))
            ii.append(v)
            jj.append(v)
            weights.append(1 / weight_sum)
            for w in buckets[v]:
                ii.append(v)
                jj.append(w)
                weights.append(g.edge_weight(v, w) / weight_sum)
        return torch.tensor([ii, jj], dtype=torch.long), torch.tensor(
            weights, dtype=torch.float
        )
