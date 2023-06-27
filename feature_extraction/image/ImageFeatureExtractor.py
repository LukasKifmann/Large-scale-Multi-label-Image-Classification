from typing import Iterable
from ..FeatureExtractor import FeatureExtractor
from ...ImageProcessorInput import ImageProcessorInput, isImageProcessorInput


class ImageFeatureExtractor(FeatureExtractor[ImageProcessorInput]):
    def _is_multi_input(
        self, input: ImageProcessorInput | Iterable[ImageProcessorInput]
    ) -> bool:
        return not isImageProcessorInput(input)
