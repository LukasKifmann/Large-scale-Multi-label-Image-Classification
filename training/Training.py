from .Trainable import Trainable
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from typing import Tuple, Optional, Collection, Iterator, Callable, Iterable
from ..util import prog
from ..metrics import Metric, AggregateMetric
from functools import reduce


class Training:
    @property
    def training_data(self) -> Collection[Tuple[torch.Tensor, torch.Tensor]]:
        return self._training_data

    @training_data.setter
    def training_data(
        self, training_data: Collection[Tuple[torch.Tensor, torch.Tensor]]
    ):
        self._training_data = training_data

    @property
    def validation_data(
        self,
    ) -> Optional[Collection[Tuple[torch.Tensor, torch.Tensor]]]:
        return self._validation_data

    @validation_data.setter
    def validation_data(
        self, validation_data: Optional[Collection[Tuple[torch.Tensor, torch.Tensor]]]
    ):
        self._validation_data = validation_data

    @property
    def validation_label_sets(self) -> Optional[Iterable[set[int]]]:
        return self._validation_label_sets

    @validation_label_sets.setter
    def validation_label_sets(self, validation_label_sets):
        self._validation_label_sets = validation_label_sets

    @property
    def input_dropout(self) -> float:
        return self._input_dropout

    @property
    def use_logits(self) -> bool:
        return self._use_logits

    def __init__(
        self,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        get_optimizer: Callable[[Iterator[torch.nn.Parameter]], Optimizer],
        validation_metric: Optional[Metric | AggregateMetric] = None,
        metric_interval: int = 1,
        get_scheduler: Optional[Callable[[Optimizer], LRScheduler]] = None,
        input_dropout: float = 0,
        callback: Callable[
            [list[float], list[float], dict[int, float | dict[str, float]]], bool
        ] = (lambda t, v, m: True),
        use_logits: bool = True,
    ):
        self._loss = loss
        self._get_optimizer = get_optimizer
        self._training_data = []
        self._validation_data = None
        self._validation_label_sets = None
        self._validation_metric = validation_metric
        self._metric_interval = metric_interval
        self._input_dropout = input_dropout
        self._get_scheduler = get_scheduler
        self._dropout = nn.Dropout(input_dropout) if input_dropout else lambda x: x
        self._callback = callback
        self._use_logits = use_logits

    def __call__(
        self, model: Trainable, epoch_count: int
    ) -> (
        list[float]
        | Tuple[list[float], list[float]]
        | Tuple[
            list[float], list[float], dict[int, float] | dict[int, dict[str, float]]
        ]
    ):
        optimizer = self._get_optimizer(model.parameters())
        scheduler = self._get_scheduler(optimizer) if self._get_scheduler else None
        training_losses, validation_losses = [], []
        metric_scores = {}
        for i in range(1, epoch_count + 1):
            training_loss, validation_loss, metric_score = self._epoch(
                i, model, optimizer
            )
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)
            if scheduler:
                scheduler.step()
            if metric_score:
                metric_scores[i] = metric_score
            if not self._callback(training_losses, validation_losses, metric_scores):
                break
        if self._validation_data:
            if self._validation_label_sets and self._validation_metric:
                return (
                    training_losses,
                    validation_losses,
                    metric_scores,
                )
            else:
                return training_losses, validation_losses
        else:
            return training_losses

    def _epoch(
        self,
        i: int,
        model: Trainable,
        optimizer: Optimizer,
    ) -> Tuple[float, float, Optional[float | dict[str, float]]]:
        training_loss = 0.0
        model.requires_grad_()
        model.train()
        for X, y in prog(
            f"  training epoch no. {i:3d}", self.training_data, verbose=True
        ):
            X = self._dropout(X)
            optimizer.zero_grad()
            loss = self._loss(
                model.logits(X) if self.use_logits else model.predict(X), y
            )
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(self.training_data)
        model.requires_grad_(False)
        model.eval()
        validation_loss = 0.0
        metric_score = None
        if self._validation_data:
            print("    validating...", end="")
            validation_loss = self._validation_loss(model)  # type: ignore
            if (
                self._validation_label_sets
                and self._validation_metric
                and not i % self._metric_interval
            ):
                print("\r    computing metric(s)...", end="")
                try:
                    metric_score = self._compute_validation_metric(model)
                except Exception as e:
                    print(
                        "computation of validation metric failed due to an exception: "
                        + repr(e)
                    )
            if metric_score != None:
                print(
                    f"\r    average training loss: {training_loss}, average validation loss: {validation_loss}, {self._show_validation_metric(metric_score)}"
                )
            else:
                print(
                    f"\r    average training loss: {training_loss}, average validation loss: {validation_loss}"
                )
        else:
            print(f"  average training loss: {training_loss}")
        return training_loss, validation_loss, metric_score

    def _validation_loss(self, model: Trainable) -> float:
        loss = 0.0
        for X, y in self.validation_data:  # type: ignore
            loss += self._loss(
                model.logits(X) if self.use_logits else model.predict(X), y
            ).item()
        return loss / len(self.validation_data)  # type: ignore

    def _compute_validation_metric(self, model: Trainable) -> float | dict[str, float]:
        pred_list = []
        for X, y in self.validation_data:  # type: ignore
            pred_list.append(model.predict(X).to("cpu"))
        return self._validation_metric(torch.concat(pred_list), self._validation_label_sets)  # type: ignore

    def _show_validation_metric(self, x: float | dict[str, float]) -> str:
        if isinstance(x, float):
            return f"{repr(self._validation_metric)}: {x}"
        else:
            return reduce(
                (lambda a, b: f"{a}, {b}"),
                map((lambda name: f"{name}: {x[name]}"), x.keys()),
            )
