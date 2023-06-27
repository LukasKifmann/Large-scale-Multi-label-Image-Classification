from .EdgeWeightedDigraph import EdgeWeightedDigraph
from .EdgeAndNodeWeighteGraph import EdgeAndNodeWeightedGraph
from typing import Iterator, Tuple


class LazyCorrelationDigraph(EdgeWeightedDigraph):
    @property
    def node_count(self) -> int:
        return self.__cooccurrence_graph.node_count

    @property
    def arch_count(self) -> int:
        return self.__cooccurrence_graph.arch_count

    @property
    def arches(self) -> Iterator[Tuple[int, int]]:
        return self.__cooccurrence_graph.arches

    def __init__(self, cooccurrence_graph: EdgeAndNodeWeightedGraph):
        self.__cooccurrence_graph = cooccurrence_graph

    def has_arch(self, v: int, w: int) -> bool:
        return self.__cooccurrence_graph.has_arch(v, w)

    def edge_weight(self, v: int, w: int) -> float:
        cooccurrence = self.__cooccurrence_graph.edge_weight(v, w)
        if cooccurrence:
            return cooccurrence / self.__cooccurrence_graph.node_weight(v)
        else:
            return 0

    def in_degree(self, v: int) -> int:
        return self.__cooccurrence_graph.degree(v)

    def out_degree(self, v: int) -> int:
        return self.__cooccurrence_graph.degree(v)
