from abc import abstractmethod
from .Digraph import Digraph
from typing import Iterator, Tuple


class Graph(Digraph):
    @property
    @abstractmethod
    def edge_count(self) -> int:
        pass

    @property
    @abstractmethod
    def edges(self) -> Iterator[Tuple[int, int]]:
        pass

    @abstractmethod
    def has_edge(self, v, w) -> bool:
        pass

    @abstractmethod
    def degree(self, v: int) -> int:
        pass

    @property
    def arches(self) -> Iterator[Tuple[int, int]]:
        for v, w in self.edges:
            yield v, w
            yield w, v

    @property
    def arch_count(self) -> int:
        return 2 * self.edge_count

    def has_arch(self, v, w) -> bool:
        return self.has_edge(v, w)

    def in_degree(self, v: int) -> int:
        return self.degree(v)

    def out_degree(self, v: int) -> int:
        return self.degree(v)
