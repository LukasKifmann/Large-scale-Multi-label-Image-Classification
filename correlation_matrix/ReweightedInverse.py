from ..graph import EdgeWeightedDigraph
import torch
from typing import Tuple
from ..util import prog
from .CorrelationMatrixBuilder import CorrelationMatrixBuilder


class ReweightedInverse(CorrelationMatrixBuilder):
    @property
    def threshold(self) -> float:
        return self.__threshold

    @property
    def neighbourhood_weight(self) -> float:
        return self.__neighbourhood_weight

    def __init__(self, threshold, neighbourhood_weight):
        self.__threshold = threshold
        self.__neighbourhood_weight = neighbourhood_weight

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(threshold={self.threshold}, neighbourhood_weight={self.neighbourhood_weight})"
        )

    def __call__(self, g: EdgeWeightedDigraph) -> Tuple[torch.Tensor, torch.Tensor]:
        ii = []
        jj = []
        weights = []
        buckets = [[] for _ in g.nodes]
        for v, w in prog(
            "computing reweighted inverse correlation matrix: discovering parent nodes",
            g.arches,
            g.arch_count,
        ):
            if g.edge_weight(v, w) >= self.threshold:
                buckets[w].append(v)
        node_weight = 1 - self.neighbourhood_weight
        for w in prog(
            "computing reweighted inverse correlation matrix: computing coefficients",
            g.nodes,
            g.node_count,
        ):
            ii.append(w)
            jj.append(w)
            weights.append(node_weight)
            parent_count = len(buckets[w])
            if parent_count:
                parent_weight = self.neighbourhood_weight / parent_count
                for v in buckets[w]:
                    ii.append(w)
                    jj.append(v)
                    weights.append(parent_weight)
        return torch.tensor([ii, jj], dtype=torch.long), torch.tensor(
            weights, dtype=torch.float
        )
