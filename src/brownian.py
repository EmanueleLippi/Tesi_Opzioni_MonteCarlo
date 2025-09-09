import numpy as np

def simulate_generalized_brownian_motion(T, mu, sigma, N, M, seed=None):
    """
    Simula M traiettorie di un processo di diffusione generalizzato (GBM).
    T: orizzonte temporale
    mu: drift
    sigma: volatilità
    N: passi
    M: numero simulazioni
    seed: seme per la generazione di numeri casuali (opzionale)
    Ritorna una matrice MxN con le traiettorie simulate.

    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N # passo temporale
    dW = np.random.normal(0, np.sqrt(dt), (M, N)) # incrementi del moto browniano
    W = np.cumsum(dW, axis=1) # moto browniano

    return W
