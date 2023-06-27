from typing import Tuple
from abc import ABC, abstractmethod
from ..graph import EdgeWeightedDigraph
import torch


class CorrelationMatrixBuilder(ABC):
    def __repr__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, g: EdgeWeightedDigraph) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
