import torch
from ... import ImageProcessorInput
from .ImageFeatureExtractor import ImageFeatureExtractor
from PIL import Image
from ..InvalidInputError import InvalidInputError


class TorchProxyFeatureExtractor(ImageFeatureExtractor):
    @property
    def output_dimension(self) -> int:
        return self.__model.num_labels

    def __init__(self, preproc, model):
        self.__preproc = preproc
        self.__model = model

    def _extract(self, X: ImageProcessorInput) -> torch.Tensor:
        self._assert_input_valid(X)
        with torch.no_grad():
            return self.__model(**self.__preproc(X, return_tensors="pt")).logits

    def _assert_input_valid(self, X: ImageProcessorInput):
        if isinstance(X, Image.Image):
            if X.mode != "RGB":
                raise InvalidInputError(
                    f"image mode was expected to be 'RGB' was '{X.mode}'"
                )
