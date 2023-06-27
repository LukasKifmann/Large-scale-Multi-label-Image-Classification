from .Callback import AbstractCallback


class LossEarlyStopping(AbstractCallback):
    @property
    def patience(self) -> int:
        return self.__patience

    @property
    def maximize(self) -> bool:
        return self.__maximize

    def __init__(self, patience: int, maximize: bool = False):
        self.__patience = patience
        self.__maximize = maximize

    def __repr__(self) -> str:
        return super().__repr__() + f"(patience={self.__patience}, maximize={self.maximize})"

    def __call__(
        self,
        training_losses: list[float],
        validation_losses: list[float],
        metric_values: dict[int, float | dict[str, float]],
    ) -> bool:
        if validation_losses:
            best_i = 0
            best = validation_losses[0]
            for i in range(1, len(validation_losses)):
                value = validation_losses[i]
                if self.maximize and value > best or not self.maximize and value < best:
                    best_i = i
                    best = value
            return len(validation_losses) - 1 - best_i <= self.patience
        else:
            return True