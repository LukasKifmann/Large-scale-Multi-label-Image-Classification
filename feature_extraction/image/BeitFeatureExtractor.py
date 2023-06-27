from .TorchProxyFeatureExtractor import TorchProxyFeatureExtractor
from transformers import BeitImageProcessor, BeitForImageClassification


class BeitFeatureExtractor(TorchProxyFeatureExtractor):
    def __init__(self):
        BeitFeatureExtractor.__load_pretrained()
        super().__init__(
            BeitFeatureExtractor.__PREPROC, BeitFeatureExtractor.__PRETRAINED
        )

    @staticmethod
    def __load_pretrained():
        if not BeitFeatureExtractor.__PREPROC:
            BeitFeatureExtractor.__PREPROC = BeitImageProcessor.from_pretrained(
                BeitFeatureExtractor.__PRETRAINED_REPO
            )
        if not BeitFeatureExtractor.__PRETRAINED:
            BeitFeatureExtractor.__PRETRAINED = (
                BeitForImageClassification.from_pretrained(
                    BeitFeatureExtractor.__PRETRAINED_REPO
                )
            )

    __PRETRAINED_REPO = "microsoft/beit-base-patch16-224"
    __PREPROC = None
    __PRETRAINED = None
