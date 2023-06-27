from abc import ABC, abstractmethod
from typing import Protocol


class Callback(Protocol):
    @abstractmethod
    def __call__(
        self,
        training_losses: list[float],
        validation_losses: list[float],
        metric_values: dict[int, float | dict[str, float]],
    ) -> bool:
        pass


class AbstractCallback(ABC, Callback):
    def __repr__(self) -> str:
        return self.__class__.__name__

    def __and__(self, other: Callback):
        return AndCallback(self, other)

    def __or__(self, other: Callback):
        return OrCallback(self, other)


class ComposedCallback(AbstractCallback):
    def __init__(self, left: Callback, right: Callback):
        self._left = left
        self._right = right

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._left)}, {repr(self._right)})"


class AndCallback(ComposedCallback):
    def __call__(
        self,
        training_losses: list[float],
        validation_losses: list[float],
        metric_values: dict[int, float | dict[str, float]],
    ) -> bool:
        return self._left(
            training_losses, validation_losses, metric_values
        ) and self._right(training_losses, validation_losses, metric_values)


class OrCallback(ComposedCallback):
    def __call__(
        self,
        training_losses: list[float],
        validation_losses: list[float],
        metric_values: dict[int, float | dict[str, float]],
    ) -> bool:
        return self._left(
            training_losses, validation_losses, metric_values
        ) or self._right(training_losses, validation_losses, metric_values)
