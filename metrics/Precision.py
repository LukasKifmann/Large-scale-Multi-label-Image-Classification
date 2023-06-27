from .SetBasedMetric import SetBasedMetric


class Precision(SetBasedMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)

    def compute_from_sets(self, labels_pred: set[int], labels: set[int]) -> float:
        if not labels_pred:
            return 0
        else:
            return len(labels.intersection(labels_pred)) / len(labels_pred)
