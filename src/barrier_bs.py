from __future__ import annotations
from scipy.stats import norm
import numpy as np, math
from typing import Union
from bsm import bs_price

__all__ = [
    "call_up_and_out", "call_up_and_in",
    "call_down_and_out", "call_down_and_in",
    "barrier_price",
    "put_down_and_out", "put_down_and_in",
    "put_up_and_out", "put_up_and_in",
]

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

def call_up_and_out(
    S: ArrayLike, K: ArrayLike, H: float, T: float, r: float, sigma: float, q: float = 0.0
) -> ArrayLike:
    """
    Up-and-Out Call (monitoraggio continuo, rebate = 0).
    Implementa la Eq. (11.5) nello schema con ρ⁺/ρ⁻ (valido per H > K).
    Se S >= H (barriera già toccata) -> prezzo 0.
    Per H <= K, il prezzo è 0 (se S < H), perché ogni payoff S_T > K avrebbe necessariamente superato H prima.
    """
    _validate_inputs(S, K, H, T, r, sigma, q)

    # normalizza a ndarray e allinea le shape
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    S_b, K_b = np.broadcast_arrays(S, K)

    # array di output; per up-barrier, se S >= H è già knock-out: 0
    out = np.zeros_like(S_b, dtype=float)

    # calcola solo dove NON è già knockout (S < H) E dove K < H (altrimenti prezzo 0)
    mask = (S_b < H) & (K_b < H)
    if not np.any(mask):
        return out.item() if out.ndim == 0 else out

    Sm = S_b[mask]
    Km = K_b[mask]

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    # beta = 2(r - q)/σ^2
    beta = 2.0 * (r - q) / (sigma * sigma)

    # Argomenti di ρ± come da formula:
    # 1) S/K  2) S/H  3) H^2/(K S)  4) H/S
    rk_SK_p = _rho_plus (Sm / Km,            T, r, q, sigma)
    rk_SB_p = _rho_plus (Sm / H,             T, r, q, sigma)
    rk_H2_p = _rho_plus ((H*H) / (Km * Sm),  T, r, q, sigma)
    rk_BS_p = _rho_plus (H / Sm,             T, r, q, sigma)

    rk_SK_m = _rho_minus(Sm / Km,            T, r, q, sigma)
    rk_SB_m = _rho_minus(Sm / H,             T, r, q, sigma)
    rk_H2_m = _rho_minus((H*H) / (Km * Sm),  T, r, q, sigma)
    rk_BS_m = _rho_minus(H / Sm,             T, r, q, sigma)

    # Blocchi A e B (Eq. 11.5)
    A = _norm_cdf(rk_SK_p) - _norm_cdf(rk_SB_p) \
        - (H / Sm)**(1.0 + beta) * (_norm_cdf(rk_H2_p) - _norm_cdf(rk_BS_p))

    B = _norm_cdf(rk_SK_m) - _norm_cdf(rk_SB_m) \
        - (Sm / H)**(1.0 - beta) * (_norm_cdf(rk_H2_m) - _norm_cdf(rk_BS_m))

    price = Sm * disc_q * A - Km * disc_r * B
    out[mask] = price

    return out.item() if out.ndim == 0 else out


def call_up_and_in(S: ArrayLike, K: ArrayLike, H: float, T: float, r: float, sigma: float, q: float = 0.0) -> ArrayLike:
    # In–Out parity: C_in = C_vanilla − C_out
    return bs_price(S, K, T, r, sigma, q, kind='call') - call_up_and_out(S, K, H, T, r, sigma, q)


def call_down_and_out(
    S: ArrayLike, K: ArrayLike, H: float, T: float, r: float, sigma: float, q: float = 0.0
) -> ArrayLike:
    """
    Down-and-Out Call (monitoraggio continuo, rebate = 0).
    Casi corretti secondo le formule chiuse:
      - H < K:  C_DO = C_vanilla - S*(H/S)^{1+β} N(d7) + K*e^{-rT}*(H/S)^{β-1} N(d8)
      - H > K:  C_DO = S N(d3) - K e^{-rT} N(d4) + S*(H/S)^{1+β} N(d5) - K e^{-rT}*(H/S)^{β-1} N(d6)
    con β = 2(r−q)/σ² e d_i come in main.pdf.
    """
    _validate_inputs(S, K, H, T, r, sigma, q)

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    S_b, K_b = np.broadcast_arrays(S, K)
    out = np.zeros_like(S_b, dtype=float)

    # KO immediato se S <= H
    mask_alive = (S_b > H)
    if not np.any(mask_alive):
        return out.item() if out.ndim == 0 else out

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    beta   = 2.0 * (r - q) / (sigma * sigma)

    # ----- Caso A: H < K -----
    mask_A = mask_alive & (H < K_b)
    if np.any(mask_A):
        Sm = S_b[mask_A]
        Km = K_b[mask_A]
        # d1,d2,d7,d8 (in forma ρ±)
        d1p = _rho_plus (Sm / Km,            T, r, q, sigma)
        d2m = _rho_minus(Sm / Km,            T, r, q, sigma)
        d7p = _rho_plus ((H*H) / (Sm*Km),    T, r, q, sigma)
        d8m = _rho_minus((H*H) / (Sm*Km),    T, r, q, sigma)
        out[mask_A] = (
            Sm * disc_q * _norm_cdf(d1p)
            - Km * disc_r * _norm_cdf(d2m)
            - Sm * disc_q * (H/Sm)**(1.0 + beta) * _norm_cdf(d7p)
            + Km * disc_r * (H/Sm)**(beta - 1.0) * _norm_cdf(d8m)
        )

    # ----- Caso B: H > K -----
    mask_B = mask_alive & (H > K_b)
    if np.any(mask_B):
        Sm = S_b[mask_B]
        Km = K_b[mask_B]
        # d3,d4,d5,d6 (in forma ρ±)
        d3p = _rho_plus (Sm / H,  T, r, q, sigma)
        d4m = _rho_minus(Sm / H,  T, r, q, sigma)
        d5p = _rho_plus (H  / Sm, T, r, q, sigma)
        d6m = _rho_minus(H  / Sm, T, r, q, sigma)
        out[mask_B] = (
            Sm * disc_q * _norm_cdf(d3p)
            - Km * disc_r * _norm_cdf(d4m)
            + Sm * disc_q * (H/Sm)**(1.0 + beta) * _norm_cdf(d5p)
            - Km * disc_r * (H/Sm)**(beta - 1.0) * _norm_cdf(d6m)
        )

    return out.item() if out.ndim == 0 else out


def call_down_and_in(
    S: ArrayLike, K: ArrayLike, H: float, T: float, r: float, sigma: float, q: float = 0.0
) -> ArrayLike:
    # In–Out parity (down): C_in = C_vanilla − C_out
    return bs_price(S, K, T, r, sigma, q, kind="call") - call_down_and_out(S, K, H, T, r, sigma, q)

def put_down_and_in(
    S: ArrayLike, K: ArrayLike, H: float, T: float, r: float, sigma: float, q: float = 0.0
) -> ArrayLike:
    """
    Down-and-In Put (monitoraggio continuo, rebate = 0).
    - Se S <= H all'origine: già "knocked-in" → put vanilla.
    - Se S > H: formula chiusa (d3..d6 riscritti con ρ±), con β = 2(r−q)/σ².
    """
    _validate_inputs(S, K, H, T, r, sigma, q)

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    S_b, K_b = np.broadcast_arrays(S, K)

    out = np.zeros_like(S_b, dtype=float)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    beta   = 2.0 * (r - q) / (sigma * sigma)

    # 1) Già "in" all'origine
    mask_in0 = (S_b <= H)
    if np.any(mask_in0):
        out[mask_in0] = bs_price(S_b[mask_in0], K_b[mask_in0], T, r, sigma, q, kind="put")

    # 2) Caso standard S > H
    mask = (S_b > H)
    if np.any(mask):
        Sm = S_b[mask]
        Km = K_b[mask]

        d3p = _rho_plus (Sm / H, T, r, q, sigma)
        d4m = _rho_minus(Sm / H, T, r, q, sigma)
        d5p = _rho_plus (H / Sm, T, r, q, sigma)
        d6m = _rho_minus(H / Sm, T, r, q, sigma)

        price = (
            - Sm * disc_q * _norm_cdf(-d3p)
            + Km * disc_r * _norm_cdf(-d4m)
            + Sm * disc_q * (H / Sm) ** (1.0 + beta) * _norm_cdf(d5p)
            - Km * disc_r * (H / Sm) ** (beta - 1.0) * _norm_cdf(d6m)
        )
        out[mask] = price

    return out.item() if out.ndim == 0 else out


def put_down_and_out(
    S: ArrayLike, K: ArrayLike, H: float, T: float, r: float, sigma: float, q: float = 0.0
) -> ArrayLike:
    # In–Out parity (put, down): P_out = P_vanilla − P_in
    return bs_price(S, K, T, r, sigma, q, kind="put") - put_down_and_in(S, K, H, T, r, sigma, q)

def put_up_and_out(S, K, H, T, r, sigma, q=0.0):
    """
    Up-and-Out Put (monitoraggio continuo, rebate = 0), S < H.
    Formula Reiner–Rubinstein in forma ρ±.
    """
    _validate_inputs(S, K, H, T, r, sigma, q)
    S = np.asarray(S, float); K = np.asarray(K, float)
    S_b, K_b = np.broadcast_arrays(S, K)
    out = np.zeros_like(S_b, float)

    # knockout immediato (up): S >= H -> 0
    alive = (S_b < H)
    if not np.any(alive): 
        return out.item() if out.ndim == 0 else out

    Sm = S_b[alive]; Km = K_b[alive]
    disc_q = math.exp(-q*T); disc_r = math.exp(-r*T)
    beta = 2.0*(r - q)/(sigma*sigma)

    # x1 = ρ+(S/K), x2 = ρ−(S/K), y1 = ρ+(H^2/(S K)), y2 = ρ−(H^2/(S K))
    x1 = _rho_plus (Sm / Km,           T, r, q, sigma)
    x2 = _rho_minus(Sm / Km,           T, r, q, sigma)
    y1 = _rho_plus ((H*H)/(Sm*Km),     T, r, q, sigma)
    y2 = _rho_minus((H*H)/(Sm*Km),     T, r, q, sigma)

    out[alive] = (
        Km*disc_r*( _norm_cdf(-x2) - (H/Sm)**(beta-1.0)*_norm_cdf(-y2) )
        - Sm*disc_q*( _norm_cdf(-x1) - (H/Sm)**(beta+1.0)*_norm_cdf(-y1) )
    )
    return out.item() if out.ndim == 0 else out


def put_up_and_in(S, K, H, T, r, sigma, q=0.0):
    # Parità in–out (put, up): P_in = P_vanilla − P_out
    return bs_price(S, K, T, r, sigma, q, kind="put") - put_up_and_out(S, K, H, T, r, sigma, q)



def barrier_price(option_type: str, barrier: str, knock: str,
                  S: ArrayLike, K: ArrayLike, H: float, T: float,
                  r: float, sigma: float, q: float = 0.0) -> ArrayLike:
    option_type = option_type.lower()
    barrier = barrier.lower()
    knock = knock.lower()

    if option_type == "call" and barrier == "up" and knock == "out":
        return call_up_and_out(S, K, H, T, r, sigma, q)
    if option_type == "call" and barrier == "up" and knock == "in":
        return call_up_and_in(S, K, H, T, r, sigma, q)

    if option_type == "call" and barrier == "down" and knock == "out":
        return call_down_and_out(S, K, H, T, r, sigma, q)
    if option_type == "call" and barrier == "down" and knock == "in":
        return call_down_and_in(S, K, H, T, r, sigma, q)
    if option_type == "put" and barrier == "up" and knock == "in":
        return put_up_and_in(S, K, H, T, r, sigma, q)
    if option_type == "put" and barrier == "up" and knock == "out":
        return put_up_and_out(S, K, H, T, r, sigma, q)
    if option_type == "put" and barrier == "down" and knock == "in":
        return put_down_and_in(S, K, H, T, r, sigma, q)
    if option_type == "put" and barrier == "down" and knock == "out":
        return put_down_and_out(S, K, H, T, r, sigma, q)
    
    


    raise NotImplementedError("Combinazione non ancora implementata.")
