import numpy as np
import torch
from PIL.Image import Image

types = [np.ndarray, torch.Tensor, Image]

ImageProcessorInput = np.ndarray | torch.Tensor | Image


def isImageProcessorInput(x) -> bool:
    return any((isinstance(x, t) for t in types))
