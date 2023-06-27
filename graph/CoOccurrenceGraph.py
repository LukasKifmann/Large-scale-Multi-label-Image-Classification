from .EdgeAndNodeWeighteGraph import EdgeAndNodeWeightedGraph
import torch
from typing import Iterator, Tuple, Dict, Optional, Iterable
from ..util import prog
import os


class CoOccurrenceGraph(EdgeAndNodeWeightedGraph):
    @property
    def node_count(self) -> int:
        return self.__node_weights.shape[0]

    @property
    def edge_count(self) -> int:
        return self.__edge_count

    @property
    def edges(self) -> Iterator[Tuple[int, int]]:
        for v in self.__edge_weights:
            for w in self.__edge_weights[v]:
                yield v, w

    def __init__(self):
        self.__node_weights = torch.zeros(0, dtype=torch.int32)
        self.__degrees = torch.zeros(0, dtype=torch.int32)
        self.__edge_weights: Dict[int, Dict[int, int]] = {}
        self.__edge_count = 0

    def has_edge(self, v: int, w: int) -> bool:
        return self.edge_weight(v, w) != 0

    def node_weight(self, v: int) -> float:
        return self.__node_weights[v].item()

    def edge_weight(self, v: int, w: int) -> float:
        if v > w:
            v, w = w, v
        if v in self.__edge_weights and w in self.__edge_weights[v]:
            return self.__edge_weights[v][w]
        else:
            return 0

    def degree(self, v: int) -> int:
        return self.__degrees[v].item()  # type: ignore

    def save(self, path: str):
        if not os.path.isdir(path):
            os.mkdir(path)
        edge_counts = torch.empty(self.node_count, dtype=torch.int32)
        edges = torch.empty((self.edge_count, 2), dtype=torch.int32)
        j = 0
        for v in self.nodes:
            count = len(self.__edge_weights[v]) if v in self.__edge_weights else 0
            edge_counts[v] = count
            if count:
                for w, weight in self.__edge_weights[v].items():
                    edges[j, 0] = w
                    edges[j, 1] = weight
                    j += 1
        torch.save(self.__node_weights, path + "/node_weights.pt")
        torch.save(self.__degrees, path + "/degrees.pt")
        torch.save(edge_counts, path + "/edge_counts.pt")
        torch.save(edges, path + "/edges.pt")

    def __load(self, path: str):
        self.__node_weights = torch.load(path + "/node_weights.pt")
        self.__degrees = torch.load(path + "/degrees.pt")
        edge_counts = torch.load(path + "/edge_counts.pt")
        edges = torch.load(path + "/edges.pt")
        self.__edge_count = 0
        j = 0
        for v in range(edge_counts.shape[0]):
            count = edge_counts[v].item()
            if count:
                self.__edge_weights[v] = {}
                self.__edge_count += count
                for _ in range(count):
                    w = edges[j, 0].item()
                    self.__edge_weights[v][w] = edges[j, 1].item()
                    j += 1

    def __count_tensor(self, y: torch.Tensor):
        (sample_count, label_count) = y.shape
        self.__node_weights = torch.zeros(label_count, dtype=torch.int32)
        self.__degrees = torch.zeros((label_count), dtype=torch.int32)
        self.__edge_count = 0
        for k in prog("computing co-occurrence graph", range(sample_count)):
            idcs = []
            for v in range(label_count):
                if y[k, v]:
                    idcs.append(v)
            for i, v in enumerate(idcs):
                self.__node_weights[v] += 1
                for j in range(i + 1, len(idcs)):
                    w = idcs[j]
                    self.__increment_edge_weight(v, w)

    def __count_lists(
        self, ll: Iterable[list[int]], sample_size: int, label_count: int
    ):
        self.__node_weights = torch.zeros(label_count, dtype=torch.int32)
        self.__degrees = torch.zeros((label_count), dtype=torch.int32)
        self.__edge_count = 0
        for l in prog("computing co-occurrence graph", ll, sample_size):
            l = list(set(l))
            for i, v in enumerate(l):
                self.__node_weights[v] += 1
                for j in range(i + 1, len(l)):
                    w = l[j]
                    self.__increment_edge_weight(v, w)

    def __increment_edge_weight(self, v: int, w: int):
        if v >= w:
            v, w = w, v
        if not v in self.__edge_weights:
            self.__handle_new_edge(v, w)
            self.__edge_weights[v] = {w: 1}
        elif not w in self.__edge_weights[v]:
            self.__handle_new_edge(v, w)
            self.__edge_weights[v][w] = 1
        else:
            self.__edge_weights[v][w] += 1

    def __handle_new_edge(self, v: int, w: int):
        self.__edge_count += 1
        self.__degrees[v] += 1
        self.__degrees[w] += 1

    @staticmethod
    def from_tensor(y: torch.Tensor) -> "CoOccurrenceGraph":
        g = CoOccurrenceGraph()
        g.__count_tensor(y)
        return g

    @staticmethod
    def from_lists(
        ll: Iterable[list[int]], sample_size: int, label_count: int
    ) -> "CoOccurrenceGraph":
        g = CoOccurrenceGraph()
        g.__count_lists(ll, sample_size, label_count)
        return g

    @staticmethod
    def load(path: str) -> "CoOccurrenceGraph":
        g = CoOccurrenceGraph()
        g.__load(path)
        return g
