from .SetBasedMetric import SetBasedMetric


class Accuracy(SetBasedMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)

    def compute_from_sets(self, labels_pred: set[int], labels: set[int]) -> float:
        I = len(labels.intersection(labels_pred))
        U = len(labels.union(labels_pred))
        if not U:
            return 1
        else:
            return I / U
