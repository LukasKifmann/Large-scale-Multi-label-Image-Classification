from abc import ABC, abstractmethod
from typing import Iterator, Tuple


class Digraph(ABC):
    @property
    @abstractmethod
    def node_count(self) -> int:
        pass

    @property
    def nodes(self) -> Iterator[int]:
        return range(self.node_count).__iter__()

    @property
    @abstractmethod
    def arch_count(self) -> int:
        pass

    @property
    @abstractmethod
    def arches(self) -> Iterator[Tuple[int, int]]:
        pass

    @abstractmethod
    def has_arch(self, v: int, w: int) -> bool:
        pass

    @abstractmethod
    def in_degree(self, v: int) -> int:
        pass

    @abstractmethod
    def out_degree(self, v: int) -> int:
        pass
