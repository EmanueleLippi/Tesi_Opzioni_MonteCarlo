import numpy as np
from brownian import simulation_brownian_motion

def simulate_gbm(S0, mu, sigma, T, N, M, seed=None):
    """
    Simula M traiettorie di un processo di diffusione geometrica (GBM).
    S0: prezzo iniziale
    mu: drift
    sigma: volatilità
    T: orizzonte temporale
    N: passi
    M: numero simulazioni
    seed: seme per la generazione di numeri casuali (opzionale)
    Ritorna una matrice MxN con le traiettorie simulate.

    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N # passo temporale
    W = simulation_brownian_motion(T, mu, sigma, N, M, seed) # moto browniano
    time_grid = np.linspace(dt, T, N) # griglia temporale
    exponent = (mu - 0.5 * sigma**2) * time_grid + sigma * W # esponente della GBM
    S = S0 * np.exp(exponent) # processo GBM
    return S
