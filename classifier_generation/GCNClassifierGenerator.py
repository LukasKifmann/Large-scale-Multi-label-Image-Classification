from .FixedSetClassifierGenerator import FixedSetClassifierGenerator
from ..feature_extraction.label import LabelFeatureExtractor
from torch_geometric.nn.models import GCN
import torch
from typing import Callable, Optional, Tuple


class GCNClassifierGenerator(FixedSetClassifierGenerator):
    @property
    def device(self) -> torch.device:
        return self.__device

    @FixedSetClassifierGenerator.labels.setter
    def labels(self, value: list[str]):
        FixedSetClassifierGenerator.labels.fset(self, value)
        self.__features_dirty = True
        self.__dirty = True

    @property
    def correlation_matrix(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.__edge_index, self.__edge_weight

    @correlation_matrix.setter
    def correlation_matrix(self, value: Tuple[torch.Tensor, Optional[torch.Tensor]]):
        self.__dirty = True
        self.__edge_index = value[0]
        self.__edge_weight = value[1]

    @property
    def all(self) -> torch.Tensor:
        self.update()
        return self.__weights

    @property
    def requires_grad(self) -> bool:
        return self.__requires_grad

    def __init__(
        self,
        extractor: LabelFeatureExtractor,
        gcn: GCN,
        device: torch.device = torch.device("cpu"),
        label_embedding_preprocessing: Callable[
            [torch.Tensor], torch.Tensor
        ] = lambda x: x,
        input_dropout: float = 0,
    ):
        super().__init__()
        self.__extractor = extractor
        self.labels = []
        self.__gcn = gcn
        self.__device = device
        self.__label_embedding_preprocessing = label_embedding_preprocessing
        self.__features_dirty = False
        self.__dirty = False
        self.__features = torch.tensor([], device=self.device)
        self.__edge_index = torch.tensor([], device=self.device)
        self.__edge_weight = None
        self.__weights = torch.tensor([], device=self.device)
        self.__requires_grad = False
        self.__input_dropout = (
            torch.nn.Dropout(input_dropout) if input_dropout else lambda x: x
        )
        self.__train = False

    def from_index(self, i):
        return self.all[:, i].reshape(-1, 1)  # type: ignore

    def update(self):
        if self.__train or self.requires_grad:
            self.__update_features()
            self.__compute_weights()
            self.__dirty = True
        elif self.__dirty:
            self.__update_features()
            with torch.no_grad():
                self.__compute_weights()
                self.__dirty = False

    def parameters(self):
        return self.__gcn.parameters()

    def requires_grad_(self, requires_grad: bool = True):
        self.__requires_grad = requires_grad
        self.__gcn.requires_grad_(requires_grad)

    def train(self, train: bool = True):
        self.__train = train
        self.__gcn.train(train)

    def eval(self):
        self.train(False)

    def __update_features(self):
        if self.__features_dirty:
            self.__features = self.__label_embedding_preprocessing(
                self.__extractor(self._labels).to(self.device)
            )
            self.__features_dirty = False

    def __compute_weights(self):
        self.__weights = self.__gcn.forward(
            self.__input_dropout(self.__features) if self.__train else self.__features,
            self.__edge_index,
            edge_weight=self.__edge_weight,
        ).T
