import numpy as np
from typing import Tuple
from brownian import simulate_standard_brownian_motion

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
    Simula GBM esatto: S_t = S0 * exp((μ - 0.5σ^2)t + σ W_t).
    Ritorna (t, S) con S.shape=(M, N+1) e S[:,0]=S0.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    t = np.linspace(0.0, T, N + 1) # griglia temporale
    S = np.zeros((M, N + 1))
    S[:, 0] = S0
    # genera W_t coerente con t
    _, W = simulate_standard_brownian_motion(T, N, M) # W.shape=(M, N+1)
    # calcola S_t
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W) # broadcasting su (M, N+1)
    return t, S
