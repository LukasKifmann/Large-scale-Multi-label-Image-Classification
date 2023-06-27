from .TorchProxyFeatureExtractor import TorchProxyFeatureExtractor
from transformers import AutoFeatureExtractor, ResNetForImageClassification


class ResNetFeatureExtractor(TorchProxyFeatureExtractor):
    def __init__(self):
        ResNetFeatureExtractor.__load_pretrained()
        super().__init__(
            ResNetFeatureExtractor.__PREPROC, ResNetFeatureExtractor.__PRETRAINED
        )

    @staticmethod
    def __load_pretrained():
        if not ResNetFeatureExtractor.__PREPROC:
            ResNetFeatureExtractor.__PREPROC = AutoFeatureExtractor.from_pretrained(
                ResNetFeatureExtractor.__PRETRAINED_REPO
            )
        if not ResNetFeatureExtractor.__PRETRAINED:
            ResNetFeatureExtractor.__PRETRAINED = (
                ResNetForImageClassification.from_pretrained(
                    ResNetFeatureExtractor.__PRETRAINED_REPO
                )
            )

    __PRETRAINED_REPO = "microsoft/resnet-101"
    __PREPROC = None
    __PRETRAINED = None
