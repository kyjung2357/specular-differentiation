import math


def _num_steps(t_0: float, T: float, h: float) -> int:
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    interval = T - t_0
    if interval < 0:
        raise ValueError(
            f"Final time T must be greater than or equal to t_0. "
            f"Got t_0={t_0}, T={T}"
        )

    return int(math.floor(interval / h + 1e-12))
