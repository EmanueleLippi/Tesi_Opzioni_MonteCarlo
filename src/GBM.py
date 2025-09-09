import numpy as np
from typing import Tuple

# Se "src" è un package (cioè contiene __init__.py), l'import relativo è consigliato:
from .brownian import simulate_standard_brownian_motion
# In alternativa, senza package: from brownian import simulate_standard_brownian_motion

def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    M: int,
    seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula GBM esatto:
        S_t = S0 * exp( (mu - 0.5*sigma^2)*t + sigma * W_t )
    su [0, T] con N passi e M traiettorie.

    Ritorna (t, S) con:
      - t.shape == (N+1,)
      - S.shape == (M, N+1)
      - S[:, 0] == S0

    Parametri:
        S0   : valore iniziale (S0 > 0)
        mu   : drift
        sigma: volatilità (sigma >= 0)
        T, N, M, seed: come sopra
    """
    assert S0 > 0, "S0 deve essere > 0"
    assert sigma >= 0, "sigma deve essere >= 0"
    assert T > 0 and N > 0 and M > 0, "T, N, M devono essere positivi"

    if seed is not None:
        np.random.seed(seed)

    t, W = simulate_standard_brownian_motion(T, N, M, seed)
    drift = (mu - 0.5 * sigma**2) * t  # shape: (N+1,)
    S = S0 * np.exp(drift + sigma * W) # broadcasting su (M, N+1)
    return t, S
