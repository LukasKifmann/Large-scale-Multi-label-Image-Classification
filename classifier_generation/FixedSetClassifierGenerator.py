from .ClassifierGenerator import ClassifierGenerator
from abc import abstractmethod
import torch
from typing import Iterable, Optional
from ..functional import linreg


class FixedSetClassifierGenerator(ClassifierGenerator):
    @property
    def labels(self) -> list[str]:
        return [l for l in self._labels]

    @labels.setter
    def labels(self, value: list[str]):
        self._labels = value
        self._reverse_lookup = {}
        for i, l in enumerate(value):
            self._reverse_lookup[l] = i

    def __init__(self):
        self._labels = []
        self._reverse_lookup = {}

    def __item__(self, input: int | torch.Tensor | str | Iterable[str]) -> torch.Tensor:
        if isinstance(input, int):
            return self.from_index(input)
        elif isinstance(input, torch.Tensor):
            return self.from_tensor(input)
        else:
            return super().__item__(input)

    def from_string(self, label):
        try:
            return self.from_index(self._reverse_lookup[label])
        except KeyError as e:
            raise LookupError(f"unknown label {label}") from e

    def from_tensor(self, idcs: torch.Tensor) -> torch.Tensor:
        if len(idcs.shape) == 1:
            return torch.concat([self.from_index(x) for x in idcs], 1)  # type: ignore
        else:
            raise IndexError("label index tensor must be one-dimensional")

    def label(self, i: int) -> str:
        return self._labels[i]

    def logits(self, X: torch.Tensor) -> torch.Tensor:
        return linreg(X, self.all)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(X))

    def logit_for_index(self, i: int, X: torch.Tensor) -> torch.Tensor:
        return linreg(X, self.from_index(i))

    def predict_for_index(self, i: int, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit_for_index(i, X))

    def logits_for_tensor(self, ii: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return linreg(X, self.from_tensor(ii))

    def predict_for_tensor(self, ii: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_for_tensor(ii, X))

    @property
    @abstractmethod
    def all(self) -> torch.Tensor:
        pass

    @abstractmethod
    def from_index(self, i: int) -> torch.Tensor:
        pass
