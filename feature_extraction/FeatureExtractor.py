from abc import ABC, abstractmethod
import torch
from typing import Iterable, Generic, TypeVar
from ..util import prog

T = TypeVar("T")


class FeatureExtractor(Generic[T], ABC):
    def __call__(self, input: T | Iterable[T]) -> torch.Tensor:
        if self._is_multi_input(input):
            if hasattr(input, "__len__"):
                list = []
                for x in prog(f"feature extraction with extractor {self.__class__.__name__}", input, verbose=True):  # type: ignore
                    list.append(self._extract(x))
                return torch.concat(list)
            else:
                return torch.concat([self._extract(x) for x in input])  # type: ignore
        else:
            return self._extract(input)  # type: ignore

    @property
    @abstractmethod
    def output_dimension(self) -> int:
        pass

    @abstractmethod
    def _is_multi_input(self, input: T | Iterable[T]) -> bool:
        pass

    @abstractmethod
    def _extract(self, input: T) -> torch.Tensor:
        pass
