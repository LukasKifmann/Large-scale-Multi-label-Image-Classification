from .ExampleBasedMetric import ExampleBasedMetric
from .SetBasedMetric import SetBasedMetric
from .SortingBasedMetric import SortingBasedMetric
from .Metric import Metric
import torch
from typing import Iterable, Iterator
from ..util import prog
from functools import cached_property


class AggregateMetricImpl:
    @cached_property
    def __metrics_str(self) -> str:
        metrics = list(self.metrics)
        if not metrics:
            return ""
        else:
            s = repr(metrics[0])
            for i in range(1, len(metrics)):
                s += ", " + repr(metrics[i])
        return s

    @property
    def metrics(self) -> Iterator[Metric]:
        for l in (
            self.__sorting_based_metrics,
            self.__set_based_metrics,
            self.__example_based_metrics,
            self.__other_metrics,
        ):
            for m in l:
                yield m

    def __init__(self, *metrics: Metric):
        self.__sorting_based_metrics: list[SortingBasedMetric] = []
        self.__set_based_metrics: list[SetBasedMetric] = []
        self.__example_based_metrics: list[ExampleBasedMetric] = []
        self.__other_metrics: list[Metric] = []
        self.__thresholds = set()
        reprs = set()
        for metric in metrics:
            r = repr(metric)
            if not r in reprs:
                reprs.add(r)
                if isinstance(metric, SortingBasedMetric):
                    self.__sorting_based_metrics.append(metric)
                elif isinstance(metric, SetBasedMetric):
                    self.__thresholds.add(metric.threshold)
                    self.__set_based_metrics.append(metric)
                elif isinstance(metric, ExampleBasedMetric):
                    self.__example_based_metrics.append(metric)
                else:
                    self.__other_metrics.append(metric)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__metrics_str})"

    def __call__(
        self, y_pred: torch.Tensor, labels: Iterable[set[int]], verbose: bool = False
    ) -> dict[str, float]:
        count = y_pred.shape[0]
        totals = {repr(metric): 0.0 for metric in self.metrics}
        for i, row_labels in (
            prog(f"computing example based metrics", enumerate(labels), count)
            if verbose
            else enumerate(labels)
        ):
            self.__compute_example_based_for_example(y_pred[i, :], row_labels, totals)
        for metric, value in totals.items():
            totals[metric] = value / count
        for metric in (
            prog(f"computing other metrics", self.__other_metrics)
            if verbose
            else self.__other_metrics
        ):
            totals[repr(metric)] += metric(y_pred, labels)
        return totals

    def __compute_sorting_based_for_example(
        self,
        row: torch.Tensor,
        labels: set[int],
        totals: dict[str, float],
    ):
        sorted_idcs = row.sort(descending=True).indices
        for metric in self.__sorting_based_metrics:
            totals[repr(metric)] += metric.compute_from_sorting_order(
                row, sorted_idcs, labels
            )

    def __compute_set_based_for_example(
        self, row: torch.Tensor, labels: set[int], totals: dict[str, float]
    ):
        sets = {
            threshold: set((i.item() for i in (row >= threshold).nonzero()))
            for threshold in self.__thresholds
        }
        for metric in self.__set_based_metrics:
            totals[repr(metric)] += metric.compute_from_sets(
                sets[metric.threshold], labels
            )

    def __compute_example_based_for_example(
        self, row: torch.Tensor, labels: set[int], totals: dict[str, float]
    ):
        self.__compute_set_based_for_example(row, labels, totals)
        self.__compute_sorting_based_for_example(row, labels, totals)
        for metric in self.__example_based_metrics:
            totals[repr(metric)] += metric.compute_for_example(row, labels)
