from abc import ABC, abstractmethod
import torch
from typing import Iterable
from ..functional import linreg


class ClassifierGenerator(ABC):
    def __item__(self, input: str | Iterable[str]) -> torch.Tensor:
        if isinstance(input, str):
            return self.from_string(input)
        else:
            return self.from_strings(input)

    def from_strings(self, labels: Iterable[str]) -> torch.Tensor:
        return torch.concat([self.from_string(l) for l in labels], 1)

    def logit_for_label(self, label: str, X: torch.Tensor) -> torch.Tensor:
        return linreg(X, self.from_string(label))

    def predict_for_label(self, label: str, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit_for_label(label, X))

    def logits_for_labels(self, labels: list[str], X: torch.Tensor) -> torch.Tensor:
        return linreg(X, self.from_strings(labels))

    def predict_for_labels(self, labels: list[str], X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_for_labels(labels, X))

    @abstractmethod
    def from_string(self, label: str) -> torch.Tensor:
        pass
