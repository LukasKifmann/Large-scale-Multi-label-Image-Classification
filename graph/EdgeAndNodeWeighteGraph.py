from .Graph import Graph
from .EdgeWeightMixin import EdgeWeightMixin
from .NodeWeightMixin import NodeWeightMixin


class EdgeAndNodeWeightedGraph(EdgeWeightMixin, NodeWeightMixin, Graph):
    pass
