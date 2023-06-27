from ..graph import EdgeWeightedDigraph
import torch
from typing import Tuple
from ..util import prog
from .CorrelationMatrixBuilder import CorrelationMatrixBuilder


class Raw(CorrelationMatrixBuilder):
    def __call__(self, g: EdgeWeightedDigraph) -> Tuple[torch.Tensor, torch.Tensor]:
        length = g.arch_count + g.node_count
        edge_index = torch.empty((2, length), dtype=torch.long)
        edge_weight = torch.empty(length, dtype=torch.float)
        for i, (v, w) in prog(
            "computing raw correlation matrix: computing correlations",
            enumerate(g.arches),
            g.arch_count,
        ):
            edge_index[0, i] = v
            edge_index[1, i] = w
            edge_weight[i] = g.edge_weight(v, w)
        for v in prog(
            "computing raw correlation matrix: adding self-connections",
            range(g.node_count),
        ):
            i = v + g.arch_count
            edge_index[0, i] = v
            edge_index[1, i] = v
            edge_weight[i] = 1
        return edge_index, edge_weight
