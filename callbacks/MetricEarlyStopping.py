from .Callback import AbstractCallback
from typing import Optional


class MetricEarlyStopping(AbstractCallback):
    @property
    def patience(self) -> int:
        return self.__patience

    @property
    def metric_name(self) -> Optional[str]:
        return self.__metric_name

    @property
    def maximize(self) -> bool:
        return self.__maximize

    def __init__(
        self, patience: int, metric_name: Optional[str] = None, maximize: bool = True
    ):
        self.__patience = patience
        self.__metric_name = metric_name
        self.__maximize = maximize

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(patience={self.patience}, metric_name={self.metric_name}, maximize={self.maximize})"
        )

    def __call__(
        self,
        training_losses: list[float],
        validation_losses: list[float],
        metric_values: dict[int, float | dict[str, float]],
    ) -> bool:
        indices = list(metric_values.keys())
        if indices:
            indices.sort()
            best_i = indices[0]
            best = self.__get_value(metric_values, best_i)
            for i in (indices[k] for k in range(1, len(indices))):
                value = self.__get_value(metric_values, i)
                if self.maximize and value > best or not self.maximize and value < best:
                    best_i = i
                    best = value
            return indices[-1] - best_i <= self.patience
        else:
            return True

    def __get_value(
        self, metric_values: dict[int, float | dict[str, float]], i: int
    ) -> float:
        if not self.metric_name:
            return metric_values[i]  # type: ignore
        else:
            return metric_values[i][self.metric_name]  # type: ignore
