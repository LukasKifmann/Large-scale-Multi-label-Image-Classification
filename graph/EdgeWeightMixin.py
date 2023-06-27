from abc import ABC, abstractmethod


class EdgeWeightMixin(ABC):
    @abstractmethod
    def edge_weight(self, v: int, w: int) -> float:
        pass
