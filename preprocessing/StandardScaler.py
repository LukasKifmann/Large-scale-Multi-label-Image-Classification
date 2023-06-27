import torch
from typing import Optional


class StandardScaler:
    @property
    def is_initialized(self) -> bool:
        return self.__std.shape[0] != 0

    @property
    def std(self) -> torch.Tensor:
        return self.__std

    @property
    def mean(self) -> torch.Tensor:
        return self.__mean

    @property
    def device(self) -> Optional[torch.device]:
        return self.__device

    @device.setter
    def device(self, device: str | torch.device):
        if isinstance(device, str):
            self.__device = torch.device(device)
        else:
            self.__device = device
        self.__reallocate()

    def __init__(self, device: Optional[str | torch.device] = None):
        self.__std = torch.tensor([])
        self.__mean = torch.tensor([])
        if device:
            self.device = device
        else:
            self.__device = None

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if not self.is_initialized:
            self.fit(data)
        return (data - self.mean) / self.std

    def fit(self, data: torch.Tensor) -> "StandardScaler":
        self.__std, self.__mean = torch.std_mean(data, 0)
        if self.device:
            self.__reallocate()
        else:
            self.__device = data.device
        return self

    def to(self, device: str | torch.device) -> "StandardScaler":
        return StandardScaler.load(self.mean, self.std, device)

    def __reallocate(self):
        if self.is_initialized:
            self.__std = self.__std.to(self.device)
            self.__mean = self.__mean.to(self.device)

    @staticmethod
    def fit_transform(data: torch.Tensor) -> torch.Tensor:
        return StandardScaler()(data)

    @staticmethod
    def load(
        mean: torch.Tensor,
        std: torch.Tensor,
        device: Optional[torch.device | str] = None,
    ) -> "StandardScaler":
        r = StandardScaler()
        r.__mean = mean
        r.__std = std
        if device:
            r.device = device
        else:
            r.device = mean.device
        return r
