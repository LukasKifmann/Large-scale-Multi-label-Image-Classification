import torch


def linreg(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return (
        torch.concat(
            [X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)], 1
        )
        @ W
    )
