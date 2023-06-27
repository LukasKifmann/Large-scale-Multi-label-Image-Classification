from abc import ABC, abstractmethod


class NodeWeightMixin(ABC):
    @abstractmethod
    def node_weight(self, v: int) -> float:
        pass
