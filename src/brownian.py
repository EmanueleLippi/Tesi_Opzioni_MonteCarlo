import numpy as np
from typing import Tuple

def simulate_standard_brownian_motion(T: float, N: int, M: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula SBM (W_t): ritorna (time_grid, W) con W.shape=(M, N+1) e W[:,0]=0.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N # passo temporale
    t = np.linspace(0.0, T, N + 1) # griglia temporale
    dW = np.sqrt(dt) * np.random.randn(M, N) # incrementi del moto browniano
    W = np.zeros((M, N + 1)) # matrice del moto browniano
    W[:, 1:] = np.cumsum(dW, axis=1) # calcolo del moto browniano
    return t, W 

def simulate_brownian_with_drift(mu: float, sigma: float, T: float, N: int, M: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    dX_t = μ dt + σ dW_t con X0=0: ritorna (t, X) con X.shape=(M, N+1).
    """
    t, W = simulate_standard_brownian_motion(T, N, M, seed) # W.shape=(M, N+1)
    X = mu * t + sigma * W # broadcasting su (M, N+1)
    return t, X