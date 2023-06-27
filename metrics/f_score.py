def f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision == 0 or recall == 0:
        return 0
    else:
        beta_squared = beta * beta
        return (
            (1 + beta_squared)
            * precision
            * recall
            / (beta_squared * precision + recall)
        )
