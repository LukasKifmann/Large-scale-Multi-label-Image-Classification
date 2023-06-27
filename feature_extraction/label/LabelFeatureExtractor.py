from ..FeatureExtractor import FeatureExtractor
from typing import Iterable


class LabelFeatureExtractor(FeatureExtractor[str]):
    def _is_multi_input(self, input: str | Iterable[str]) -> bool:
        return not isinstance(input, str)
