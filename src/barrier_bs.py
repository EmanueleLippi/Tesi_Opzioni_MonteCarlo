from scipy.stats import norm
import numpy as np, math
from __future__ import annotations
from typing import Union
from bsm import bs_price

ArrayLike = Union[float, np.ndarray]

try:
    # Funzione per la densità della normale standard usando scipy
    def _norm_pdf(x):
        x = np.asarray(x, dtype=float)
        return norm.pdf(x)
    # Funzione per la funzione di ripartizione della normale standard usando scipy
    def _norm_cdf(x):
        x = np.asarray(x, dtype=float)
        return norm.cdf(x)
    
    _NORM_BACKEND = 'scipy'  # Indica che si sta usando scipy come backend

except Exception:
    # Funzione per la densità della normale standard usando solo numpy
    def _norm_pdf(x):
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * x**2) / np.sqrt(2 * math.pi)
    # Funzione per la funzione di ripartizione della normale standard usando solo numpy
    def _norm_cdf(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (1 + np.erf(x / math.sqrt(2)))
    
    _NORM_BACKEND = 'erf'  # Indica che si sta usando il backend basato su erf


def _validate_inputs(S: ArrayLike, K: ArrayLike, H: float, T: float, r: float, sigma: float, q: float):
    if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0) or H <= 0:
        raise ValueError("S, K, H must be > 0.")
    if T <= 0: 
        raise ValueError("T must be > 0.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    
def _rho_plus(z, T, r, q, sigma):
    z = np.asarray(z, dtype=float)
    return (np.log(z) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def _rho_minus(z, T, r, q, sigma):
    z = np.asarray(z, dtype=float)
    return (np.log(z) + (r - q - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def call_up_and_out(S: ArrayLike, K: ArrayLike, H: float, T: float, r: float, sigma: float, q: float = 0.0) -> ArrayLike:
    """
    Up-and-Out Call (monitoraggio continuo, rebate = 0).
    Implementa la Eq. (11.5) nello schema con ρ⁺/ρ⁻ (valido per H > K). 
    Se S >= H (barriera già toccata) -> prezzo 0.
    """
    _validate_inputs(S, K, H, T, r, sigma, q)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)

    # Knock-out immediato se già sopra la barriera
    out = np.zeros_like(np.broadcast_to(S, np.broadcast(S, K)[0].shape), dtype=float)
    S_b, K_b = np.broadcast_arrays(S, K)

    mask = S_b < H # Solo per S < H calcoliamo il prezzo
    if not np.any(mask): # Tutti i prezzi sono 0
        return out if out.ndim > 0 else float(out) # Ritorna uno scalare se l'input era scalare
    
    Sm = S_b[mask]
    Km = K_b[mask]

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    beta = 2 * (r - q) / sigma**2

    # I quattro argomenti che compaiono in (11.5)
    # 1) S/K  2) S/H  3) H^2/(K S)  4) H/S
    rk_SK_p = _rho_plus (Sm / Km, T, r, q, sigma)
    rk_SB_p = _rho_plus (Sm / H , T, r, q, sigma)
    rk_H2_p = _rho_plus (H*H / (Km*Sm), T, r, q, sigma)
    rk_BS_p = _rho_plus (H / Sm, T, r, q, sigma)

    rk_SK_m = _rho_minus(Sm / Km, T, r, q, sigma)
    rk_SB_m = _rho_minus(Sm / H , T, r, q, sigma)
    rk_H2_m = _rho_minus(H*H / (Km*Sm), T, r, q, sigma)
    rk_BS_m = _rho_minus(H / Sm, T, r, q, sigma)

    # Blocchi come in (11.5)
    A = _N(rk_SK_p) - _N(rk_SB_p) - (H/Sm)**(1.0 + beta) * (_N(rk_H2_p) - _N(rk_BS_p))
    B = _N(rk_SK_m) - _N(rk_SB_m) - (Sm/H)**(1.0 - beta) * (_N(rk_H2_m) - _N(rk_BS_m))

    price = Sm * disc_q * A - Km * disc_r * B
    out[mask] = price
    return out if out.ndim > 0 else float(out)

def call_up_in(S: ArrayLike, K: ArrayLike, H: float, T: float, r: float, sigma: float, q: float = 0.0) -> ArrayLike:
    # In–Out parity: C_in = C_vanilla − C_out
    return bs_price(S, K, T, r, sigma, q, kind='call') - call_up_and_out(S, K, H, T, r, sigma, q)


