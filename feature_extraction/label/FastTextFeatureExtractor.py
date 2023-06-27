import torch
from .LabelFeatureExtractor import LabelFeatureExtractor
import os
from os import path


class FastTextFeatureExtractor(LabelFeatureExtractor):
    @property
    def output_dimension(self) -> int:
        return self.__ft.get_dimension()

    @property
    def lang(self) -> str:
        return self.__lang

    def __init__(self, lang: str):
        FastTextFeatureExtractor.__download_language(lang)
        self.__lang = lang
        self.__ft = FastTextFeatureExtractor.__cache[lang]

    def _extract(self, input: str) -> torch.Tensor:
        return torch.from_numpy(self.__ft.get_word_vector(input).reshape(1, -1))

    PATH = path.join(path.expanduser("~"), ".fasttext")

    __cache = {}

    @staticmethod
    def __download_language(lang: str):
        import fasttext
        import fasttext.util

        if not lang in FastTextFeatureExtractor.__cache:
            cwd = os.getcwd()
            if not path.isdir(FastTextFeatureExtractor.PATH):
                os.mkdir(FastTextFeatureExtractor.PATH)
            os.chdir(FastTextFeatureExtractor.PATH)
            FastTextFeatureExtractor.__cache[lang] = fasttext.load_model(
                fasttext.util.download_model(lang, if_exists="ignore")
            )
            os.chdir(cwd)
