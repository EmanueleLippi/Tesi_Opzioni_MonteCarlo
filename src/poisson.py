#codice per simulare il processo di Poisson
import numpy as np
from typing import Tuple, Callable #importiamo Callable per il tipo di funzione e Tuple per il tipo di ritorno
def simulate_poisson_counts(lam: float, T: float, N: int, M: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula i conteggi di un processo di Poisson con intensità λ su [0, T].
    restituisce (t, N) con N.shape == (M, N+1) e N[:, 0] == 0

    Parametri:
        lam : intensità del processo di Poisson (λ > 0)
        T   : orizzonte temporale (T > 0)
        N   : numero di passi (N > 0)
        M   : numero di traiettorie (M > 0)
        seed: seme RNG opzionale

    Ritorna:
        t: array dei tempi con shape (N+1,)
        N: array dei conteggi cumulativi N_t per ogni path con shape (M, N+1)
    """

    if lam < 0:
        raise ValueError("λ deve essere >= 0")
    
    rng = np.random.default_rng(seed)
    dt = T / N # passo temporale
    t = np.linspace(0.0, T, N + 1) # array dei tempi

    N_traiettorie = np.empty((M, N + 1), dtype=int) # array per i conteggi cumulativi
    N_traiettorie[:, 0] = 0 # inizializziamo il primo conteggio a 0

    # ciclo per simulare i conteggi incrementali e cumulativi per ogni traiettoria
    for i in range(1, N + 1):
        dN = rng.poisson(lam * dt, size=M) # incrementi indipendenti di conteggio ~ Poisson(λ*dt)
        N_traiettorie[:, i] = N_traiettorie[:, i - 1] + dN # conteggio cumulativo
    return t, N_traiettorie

def simulate_compound_poisson(lam: float, T: float, N: int, M: int, jump_sampler: Callable[[int], np.ndarray], seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula un processo di Poisson composto: X_t = Σ_{i=1}^{N_t} Y_i
    dove (Y_i) sono i salti indipendenti e identicamente distribuiti generati da jump_sampler.

    restituisce (t, N_traiettorie, X_traiettorie)

    Parametri:
        lam, T, N, M : come sopra
        jump_sampler : Callable[[int], np.ndarray]
        Funzione che, dato un intero n, restituisce un array (n,) di salti IID Y_k.
        Esempio: per salti normali: lambda n: rng.normal(mu, sigma, size=n)
        seed: seme RNG opzionale
    """

    rng = np.random.default_rng(seed)
    dt = T / N # passo temporale
    t = np.linspace(0.0, T, N + 1) # array dei tempi

    N_traiettorie = np.empty((M, N + 1), dtype=int) # array per i conteggi cumulativi
    N_traiettorie[:, 0] = 0 # inizializziamo il primo conteggio a 0
    X_traiettorie = np.empty((M, N + 1), dtype=float) # array per i valori del processo composto
    X_traiettorie[:, 0] = 0.0 # inizializziiamo il primo valore a 0

    # ciclo per simulare i conteggi incrementali e cumulativi per ogni traiettoria
    for i in range(1, N + 1):
        dN = rng.poisson(lam * dt, size=M) # incrementi indipendenti di conteggio ~ Poisson(λ*dt)
        N_traiettorie[:, i] = N_traiettorie[:, i - 1] + dN # conteggio cumulativo

        # generiamo i salti e sommiamo agli incrementi del processo composto
        for j in range(M):
            n_jumps = dN[j]
            if n_jumps > 0:
                Y = jump_sampler(n_jumps) # campioniamo i salti --> Y.shape == (n_jumps,)
                X_traiettorie[j, i] = X_traiettorie[j, i - 1] + float(np.sum(Y)) # sommiamo i salti al processo composto
            else:
                X_traiettorie[j, i] = X_traiettorie[j, i - 1] # nessun salto, il valore rimane lo stesso
    return t, N_traiettorie, X_traiettorie



    



