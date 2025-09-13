"""""
merton_jd.py
Modello di Merton Jump-Diffusion per il valore azionario/dell'impresa e prezzatura di opzioni call europee tramite Monte Carlo.
Dinamica di S_t sotto Q (risk-neutral):
dS_t / S_{t-} = (r - q - lam*kappa) dt + sigma dW_t + d( prodotto_j J_j - 1 ),
con log J ~ N(mu_J, sigma_J^2)  e  kappa = E[J - 1] = exp(mu_J + 0.5*sigma_J^2) - 1.
Passo di Eulero esatto (su dt):
S_{t+dt} = S_t * exp((r - q - lam*kappa - 0.5*sigma^2) dt + sigma sqrt(dt) Z) * exp( Somma_{k=1}^{dN} Y_k ),
con dN ~ Poisson(lam*dt), Y_k ~ N(mu_J, sigma_J^2).
Autore: Emanuele Lippi (Tesi – Monte Carlo per opzioni)
"""

from dataclasses import dataclass # per definire classi di dati
from typing import Tuple # per tipi di ritorno
import numpy as np # per array e funzioni numeriche
from bsm import bs_price # per il prezzo di BS come controllo

@dataclass
class JDParams: # classe per i parametri del modello
    sigma: float # volatilità del processo di diffusione
    lam: float   # intensità del processo di Poisson (λ > 0)
    mu_J: float  # media del log dei salti Y = log J ~ N(mu_j, sigma_J^2)
    sigma_J: float # deviazione standard del log dei salti


"""
Genera i log-fattori di salto per ciascuna traiettoria in un modello di Merton Jump Diffusion.

    Questa funzione calcola, per ogni traiettoria simulata, la somma dei logaritmi dei fattori di salto (L),
    dove ciascun salto è modellato come una variabile casuale normale con media `mu_J` e deviazione standard `sigma_J`.
    Il numero di salti per ciascuna traiettoria è specificato dall'array `dN`, che rappresenta il numero di salti
    avvenuti in ciascuna traiettoria durante un intervallo di tempo `dt`.

    Per ogni traiettoria:
    - Se `dN[i] == 0`, non ci sono salti e il log-fattore di salto è 0.
    - Se `dN[i] > 0`, la somma dei salti è distribuita come una normale con media `dN[i] * mu_J` e deviazione standard
        `sqrt(dN[i]) * sigma_J`, sfruttando la proprietà della somma di variabili normali indipendenti.

    Parametri:
            rng (np.random.Generator): Generatore di numeri casuali per la simulazione.
            dN (np.ndarray): Array di interi (shape: (M,)) che indica il numero di salti per ciascuna delle M traiettorie.
            mu_J (float): Media della distribuzione normale dei salti.
            sigma_J (float): Deviazione standard della distribuzione normale dei salti.

    Restituisce:
            np.ndarray: Array di float (shape: (M,)) contenente, per ciascuna traiettoria, la somma dei log-fattori di salto.
"""


def _sample_jumps_log_factors(rng: np.random.Generator, dN: np.ndarray, mu_J: float, sigma_J: float) -> np.ndarray:
    """
    Ritorna un array (M, ) con log-fattore di salto per ciascuna traiettoria nel passo dt:
    L = Σ_{k=1}^{dN} Y_k  con Y_k ~ N(mu_J, sigma_J^2)
    Se dN = 0, allora L = 0 (nessun salto).
    """
    M = dN.shape[0] # assegniamo il numero di traiettorie M dalla shape di dN
    L = np.zeros(M) # inizializziamo l'array dei log-fattori di salto a 0
    idx = dN > 0 # indici delle traiettorie con almeno un salto

    # Ciclo per sommare i dN salti normali: Normal(dN*mu_J, sqrt(dN)*sigma_J^2)
    if np.any(idx): # finché ci sono traiettorie con dN > 0
        mu = dN[idx] * mu_J # media = dN * mu_J
        sd = np.sqrt(dN[idx]) * sigma_J # deviazione standard = sqrt(dN) * sigma_J
        L[idx] = rng.normal(loc=mu, scale=sd) # somma dei salti per le traiettorie con dN > 0
    return L

def simulate_merton_jd(S0: float, r: float, q: float, params: JDParams, T: float, N: int, M: int, seed: int | None = None, antithetic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula M traiettorie del modello di Merton Jump Diffusion per N passi.

    Ritorna:
    t: array dei tempi con shape (N+1,)
    S: array dei valori del processo S_t per ogni path con shape (M, N+1)
    """

    if S0 <= 0:
        raise ValueError("S0 deve essere > 0")
    if params.sigma < 0 or params.lam < 0 or params.sigma_J < 0:
        raise ValueError("sigma, lam, sigma_J devono essere >= 0")
    
    rng = np.random.default_rng(seed)
    dt = T / N # passo temporale
    t = np.linspace(0.0, T, N + 1) # array dei tempi

    # Antithetic: generiamo Z e -Z; i salti (dN e loro log-prod) sono condivisi per la coppia
    traiettorie_M = M if not antithetic else (M // 2 + M % 2) # numero di traiettorie da simulare con Z (la metà se antithetic)
    S = np.empty((M, N + 1), dtype=float) # array per i valori del processo
    S[:, 0] = S0 # inizializziamo il primo valore a S

    kappa = np.exp(params.mu_J + 0.5 * params.sigma_J**2) - 1.0 # kappa = E[J-1] = exp(mu_J + 0.5*sigma_J^2) - 1
    drift_dt = (r - q - params.lam * kappa - 0.5 * params.sigma**2) * dt # drift del processo di diffusione su dt
    sig_sqrt_dt = params.sigma * np.sqrt(dt) # volatilità del processo di diffusione su sqrt(dt)

    # organizzo le traiettorie in coppie (Z, -Z) antithetic
    # per ogni traiettoria i, genero un salto Z e il suo opposto -Z

    for i in range(1, N + 1):
        # salti condivisi per la coppia (Z, -Z)
        dN_base = rng.poisson(params.lam * dt, size=traiettorie_M) # incrementi indipendenti di conteggio ~ Poisson(λ*dt)
        L_base = _sample_jumps_log_factors(rng, dN_base, params.mu_J, params.sigma_J) # log-fattori di salto per la coppia

        if not antithetic: # caso standard senza antithetic
            Z = rng.standard_normal(M) # M salti normali standard N(0,1)
            # per portare i salti su M replico o faccio trim
            if traiettorie_M != M:
                # adatta (edge case teorico se M dispari e antithetic True/False cambia)
                dN = dN_base[:M] # adatta a M
                L = L_base[:M] # adatta a M
            else:
                dN = dN_base
                L = L_base
            S[:, i] = S[:, i - 1] * np.exp(drift_dt + sig_sqrt_dt * Z + L) # passo esatto di Merton JD
        else: # caso antithetic
            # costruisco Z e -Z
            Zp = rng.standard_normal(traiettorie_M) # traiettorie_M salti normali standard N(0,1)
            Zn = -Zp # antithetic

            # espando i salti per la coppia (Z, -Z)
            dN_pair = np.repeat(dN_base, 2) # ripeto ogni dN_base due volte
            L_pair = np.repeat(L_base, 2) # ripeto ogni L_base due
            Z_pair = np.concatenate([Zp, Zn]) # array per i salti Z e -Z

            # se M è dispari, tolgo l'ultimo elemento per adattare a M
            if M % 2 == 1:
                dN_pair = dN_pair[:M]
                L_pair = L_pair[:M]
                Z_pair = Z_pair[:M]
    
            S[:, i] = S[:, i - 1] * np.exp(drift_dt + sig_sqrt_dt * Z_pair + L_pair) # passo esatto di Merton JD

    return t, S

def price_euro_call_mc_jd(S0: float, K: float, r: float, q: float, params: JDParams, T: float, N: int, M: int, seed: int | None = None, antithetic: bool = True, control_bs_sigma: float | None = None) -> Tuple[float,float]:
    """
    Funzione per il prezzo di una call europea con payoff (S_T - K)^+ nel modello di Merton Jump Diffusion tramite Monte Carlo.
    Ritorna il prezzo stimato e l'errore standard della stima.
    Se control_bs_sigma è fornito (non None), calcola anche il prezzo BS con tale volatilità come controllo.
    """
    t, S = simulate_merton_jd(S0, r, q, params, T, N, M, seed, antithetic) # simulo le traiettorie di Merton JD
    ST = S[:, -1] # prendo i valori finali S_T delle M traiettorie simulate (ultima colonna)
    payoff = np.maximum(ST - K, 0.0) # calcolo il payoff della call europea (S_T - K)^+
    discount = np.exp(-r * T) # fattore di sconto
    if control_bs_sigma is None:
        price = discount * payoff.mean() # prezzo stimato come media scontata dei payoff
        stderr = discount * payoff.std(ddof=1) / np.sqrt(M) # errore standard della stima del prezzo ddf=1 significa campione (non popolazione)
        return price, stderr
    
    # Control variate: rigenero il GBM con stesse Z (stesso seed + schema antitetico)
    rng = np.random.default_rng(seed) # generatore di numeri casuali
    dt = T / N # passo temporale
    S_bs = np.empty((M, N + 1), dtype=float) # array per i valori del processo BS
    S_bs[:, 0] = S0 # inizializzo il primo valore a S0
    drift_dt = (r - q - 0.5 * control_bs_sigma**2) * dt # drift del processo di diffusione su dt
    sig_sqrt_dt = control_bs_sigma * np.sqrt(dt) # volatilità del processo di diffusione su sqrt(dt)
    base_M = M if not antithetic else (M // 2 + M % 2) # numero di traiettorie da simulare con Z (la metà se antithetic)
    # organizzo le traiettorie in coppie (Z, -Z) antithetic
    for i in range(1, N + 1):
        if not antithetic: # caso standard senza antithetic
            Z = rng.standard_normal(M) # M salti normali standard N(0,1)
        else: # caso antithetic
            Zp = rng.standard_normal(base_M) # base_M salti normali standard N(0,1)
            Zn = -Zp # antithetic
            Z = np.concatenate([Zp, Zn]) # array per i salti Z e -Z
        S_bs[:, i] = S_bs[:, i - 1] * np.exp(drift_dt + sig_sqrt_dt * Z) # passo esatto di GBM con Z
    V = np.maximum(S_bs[:, -1] - K, 0.0) # payoff della call europea nel GBM con volatilità di controllo
    C_bs = bs_price(S0, K, T, r, control_bs_sigma, q, kind="call") # prezzo BS con volatilità di controllo

    U, Vc = payoff, V - V.mean() # variabili per il control variate
    b = (np.dot(U - U.mean(), Vc) / (M - 1) / (np.dot(Vc, Vc) / (M - 1))) if np.dot(Vc, Vc) > 0 else 0.0 # coefficiente di controllo ottimale
    U_star = U - b * (V - C_bs) # variabile controllata
    price = discount * U_star.mean() # prezzo stimato come media scontata della variabile controllata
    stderr = discount * U_star.std(ddof=1) / np.sqrt(M) # errore standard della stima del prezzo ddf=1 significa campione (non popolazione)
    return price, stderr

def discounted_martingale_error(S0: float, r: float, q: float, params: JDParams, T: float, N: int, M: int, seed: int | None = None) -> float:
    """
    Calcola l'errore di martingala scontato per il prezzo di una call europea nel modello di Merton Jump Diffusion.
    L'errore di martingala è dato da | E_Q[e^{-rT} S_T] - S_0 |.
    Ritorna l'errore assoluto.
    """
    _, S = simulate_merton_jd(S0, r, q, params, T, N, M, seed, antithetic=True) # simulo le traiettorie di Merton JD (antithetic per ridurre varianza)
    est = np.exp(-(r-q)*T) * S[:, -1].mean() # stima di E_Q[e^{-rT} S_T]
    return (est - S0) / S0 # errore relativo |E_Q[e^{-rT} S_T] - S_0| / S0