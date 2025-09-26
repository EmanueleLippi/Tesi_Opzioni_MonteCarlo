"""
charfun_jd.py
Funzione caratteristica e cumulant exponent per il modello di Merton Jump-Diffusion.

Sotto misura risk-neutral Q:
    dS_t / S_{t-} = (r - q - λ κ) dt + σ dW_t + d( Π_j J_j - 1 ),
con log J ~ N(mu_J, sigma_J^2) e κ = E[J - 1] = exp(mu_J + 0.5*sigma_J^2) - 1.

La funzione caratteristica di log(S_T) è:
    φ(u;T) = exp( T * ψ(u) ),
dove ψ(u) è l’esponente cumulante:
    ψ(u) = i u (r - q - 0.5 σ^2 - λ κ) - 0.5 σ^2 u^2
           + λ ( exp(i u mu_J - 0.5 σ_J^2 u^2) - 1 ).
"""

import numpy as np
from typing import Union

__all__ = ["cumulant_exponent", "charfun_jd"]

def cumulant_exponent(
    u: Union[float, np.ndarray],
    r: float,
    q: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float
) -> np.ndarray:
    """
    Restituisce l’esponente cumulante ψ(u) del modello JD di Merton.
    """
    u = np.asarray(u, dtype=complex)
    kappa = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0
    drift = (r - q - lam * kappa - 0.5 * sigma**2)
    diffusion = -0.5 * sigma**2 * u**2
    jump_term = lam * (np.exp(1j * u * mu_J - 0.5 * (sigma_J**2) * u**2) - 1.0)
    return 1j * u * drift + diffusion + jump_term

def charfun_jd(
    u: Union[float, np.ndarray],
    T: float,
    r: float,
    q: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float
) -> np.ndarray:
    """
    Funzione caratteristica φ(u;T) di log(S_T) nel modello JD di Merton.
    """
    psi = cumulant_exponent(u, r, q, sigma, lam, mu_J, sigma_J)
    return np.exp(T * psi)
