from ..graph import EdgeWeightedDigraph
import torch
from typing import Tuple
from ..util import prog
from .CorrelationMatrixBuilder import CorrelationMatrixBuilder


class TruncatedInverse(CorrelationMatrixBuilder):
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
            "computing truncated inverse correlation matrix: discovering parent nodes",
            g.arches,
            g.arch_count,
        ):
            if g.edge_weight(v, w) >= self.threshold:
                buckets[w].append(v)
        for w in prog(
            "computing truncated inverse correlation matrix: computing coefficients",
            g.nodes,
            g.node_count,
        ):
            weight_sum = 1 + sum(map(lambda v: g.edge_weight(v, w), buckets[w]))
            ii.append(w)
            jj.append(w)
            weights.append(1 / weight_sum)
            for v in buckets[w]:
                ii.append(w)
                jj.append(v)
                weights.append(g.edge_weight(v, w) / weight_sum)
        return torch.tensor([ii, jj], dtype=torch.long), torch.tensor(
            weights, dtype=torch.float
        )
