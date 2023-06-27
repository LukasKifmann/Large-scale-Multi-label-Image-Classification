import torch
from typing import Collection, Iterator, Tuple, Optional
import math
from ..util import load


class LabelListDataset(Collection[Tuple[torch.Tensor, torch.Tensor]]):
    @property
    def X(self) -> torch.Tensor:
        return self.__X

    @property
    def y(self) -> list[list[int]]:
        return self.__y

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def label_count(self) -> int:
        return self.__label_count

    @property
    def sample_size(self) -> int:
        return self.__X.shape[0]

    @property
    def device(self) -> Optional[torch.device]:
        return self.__device

    def __init__(
        self,
        X: torch.Tensor,
        y: list[list[int]],
        label_count: int,
        batch_size: int,
        device: Optional[torch.device],
    ):
        if X.shape[0] != len(y):
            raise IndexError("length of X and y did not match")
        self.__X = X
        self.__y = y
        self.__label_count = label_count
        self.__batch_size = batch_size
        self.__device = device

    def __len__(self) -> int:
        return math.ceil(self.sample_size / self.batch_size)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.device and self.X.device != self.device:
            for i, j in self.__batch_idcs():
                yield self.X[i:j].to(self.device), load.tensor_from_lists(
                    (self.y[k] for k in range(i, j)),
                    self.label_count,
                    j - i,
                    self.device,
                )
        else:
            for i, j in self.__batch_idcs():
                yield self.X[i:j], load.tensor_from_lists(
                    (self.y[k] for k in range(i, j)),
                    self.label_count,
                    j - i,
                    self.X.device,
                )

    def __contains__(self, __x: object) -> bool:
        return False

    def __batch_idcs(self) -> Iterator[Tuple[int, int]]:
        i = 0
        j = self.batch_size
        while j < self.sample_size:
            yield i, j
            i = j
            j += self.batch_size
        yield i, self.sample_size
