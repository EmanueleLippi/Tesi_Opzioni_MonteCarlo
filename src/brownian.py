import numpy as np
from typing import Tuple

def simulate_standard_brownian_motion(
    T: float,
    N: int,
    M: int,
    seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula M traiettorie di un moto browniano standard (W_t) su [0, T].
    Ritorna (t, W) con:
      - t.shape == (N+1,)
      - W.shape == (M, N+1)
      - W[:, 0] == 0

    Parametri:
        T   : orizzonte temporale (T > 0)
        N   : numero di passi (N > 0)
        M   : numero di traiettorie (M > 0)
        seed: seme RNG opzionale
    """
    assert T > 0, "T deve essere > 0"
    assert N > 0, "N deve essere > 0"
    assert M > 0, "M deve essere > 0"

    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t = np.linspace(0.0, T, N + 1)

    dW = np.sqrt(dt) * np.random.randn(M, N)   # incrementi ~ N(0, dt)
    W = np.zeros((M, N + 1), dtype=float)
    W[:, 1:] = np.cumsum(dW, axis=1)

    return t, W


def simulate_brownian_with_drift(
    mu: float,
    sigma: float,
    T: float,
    N: int,
    M: int,
    seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula M traiettorie del processo:
        dX_t = mu * dt + sigma * dW_t,   con X_0 = 0.
    Ritorna (t, X) con X.shape == (M, N+1).

    Parametri:
        mu   : drift costante
        sigma: volatilità costante (sigma >= 0)
        T, N, M, seed: come sopra
    """
    assert sigma >= 0, "sigma deve essere >= 0"

    t, W = simulate_standard_brownian_motion(T, N, M, seed)
    # X_t = mu * t + sigma * W_t (broadcasting sul batch M)
    X = mu * t + sigma * W
    return t, X
