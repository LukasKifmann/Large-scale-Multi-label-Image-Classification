import torch
from typing import Tuple, Iterable


def lists_from_csv(path: str) -> Tuple[list[list[int]], int]:
    lists = []
    max_label = -1
    with open(path) as fp:
        while True:
            line = fp.readline()
            if line:
                line = line[:-1]
                l = []
                if line:
                    for i in (int(s) for s in line.split(";")):
                        l.append(i)
                        if i > max_label:
                            max_label = i
                lists.append(l)
            else:
                break
    return lists, max_label + 1


def tensor_from_lists(
    lists: Iterable[list[int]], label_count: int, length: int, device=None, sparse=False
) -> torch.Tensor:
    if sparse:
        ii = []
        jj = []
        vv = []
        for i, l in enumerate(lists):
            for j in l:
                ii.append(i)
                jj.append(j)
                vv.append(1)
        return torch.sparse_coo_tensor(
            torch.tensor([ii, jj], dtype=torch.int),
            vv,
            (length, label_count),
            dtype=torch.float,
            device=device,
        )
    else:
        y = torch.zeros((length, label_count), dtype=torch.float)
        for i, l in enumerate(lists):
            for j in l:
                y[i, j] = 1
        return y.to(device)


def tensor_from_list_of_lists(lists: list[list[int]], label_count: int, device=None):
    return tensor_from_lists(lists, label_count, len(lists), device)


def tensor_from_csv(path: str, device=None) -> torch.Tensor:
    lists, label_count = lists_from_csv(path)
    return tensor_from_list_of_lists(lists, label_count, device)
