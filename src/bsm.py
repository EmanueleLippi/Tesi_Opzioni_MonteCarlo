import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
from scipy.optimize import brentq
from math import isfinite

__all__ = ["bs_price", "bs_greeks", "implied_vol"]

EPS_SIGMA = 1e-12  # evita divisioni per zero su sigma
EPS_T = 1e-12      # evita T=0 in formule d1/d2


def _broadcast(*args) -> tuple[np.ndarray, ...]:
    """Converte in ndarray e fa broadcast al medesimo shape."""
    return np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])


def _d1_d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike):
    S, K, T, r, q, sigma = _broadcast(S, K, T, r, q, sigma)
    # protezioni numeriche (senza alterare T==0 per i branch che lo usano)
    sigma_safe = np.maximum(sigma, EPS_SIGMA)
    T_safe = np.maximum(T, EPS_T)
    vsqrt = sigma_safe * np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe * sigma_safe) * T_safe) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


def bs_price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
             q: ArrayLike = 0.0, kind: str = "call") -> np.ndarray | float:
    """
    Prezzo Black–Scholes–Merton con dividendo continuo q.
    Accetta scalari o array (broadcastabili).
    """
    S, K, T, r, q, sigma = _broadcast(S, K, T, r, q, sigma)
    price = np.empty_like(S, dtype=float)

    # Caso T = 0: payoff intrinseco scontato al "momento 0" (equivalente)
    mask_T0 = T <= 0.0
    if np.any(mask_T0):
        if kind == "call":
            price[mask_T0] = np.maximum(S[mask_T0] - K[mask_T0], 0.0)
        else:
            price[mask_T0] = np.maximum(K[mask_T0] - S[mask_T0], 0.0)

    # Caso T > 0: formula chiusa
    mask_pos = ~mask_T0
    if np.any(mask_pos):
        d1, d2 = _d1_d2(S[mask_pos], K[mask_pos], T[mask_pos], r[mask_pos], q[mask_pos], sigma[mask_pos])
        disc_r = np.exp(-r[mask_pos] * T[mask_pos])
        disc_q = np.exp(-q[mask_pos] * T[mask_pos])

        if kind == "call":
            price[mask_pos] = S[mask_pos] * disc_q * norm.cdf(d1) - K[mask_pos] * disc_r * norm.cdf(d2)
        else:
            price[mask_pos] = K[mask_pos] * disc_r * norm.cdf(-d2) - S[mask_pos] * disc_q * norm.cdf(-d1)

    return float(price) if price.shape == () else price


def bs_greeks(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
              q: ArrayLike = 0.0, kind: str = "call") -> dict[str, np.ndarray | float]:
    """
    Greche BS (delta, gamma, vega, theta, rho). Vega = ∂Price/∂sigma (non per 'vol point').
    Ritorna array con lo stesso shape (o scalari).
    """
    S, K, T, r, q, sigma = _broadcast(S, K, T, r, q, sigma)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    pdf = norm.pdf(d1)

    delta_call = disc_q * norm.cdf(d1)
    delta_put = delta_call - disc_q
    gamma = disc_q * pdf / (S * np.maximum(sigma, EPS_SIGMA) * np.sqrt(np.maximum(T, EPS_T)))
    vega = S * disc_q * pdf * np.sqrt(np.maximum(T, EPS_T))
    theta_call = (-S * disc_q * pdf * np.maximum(sigma, EPS_SIGMA) / (2.0 * np.sqrt(np.maximum(T, EPS_T)))
                  - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1))
    theta_put = (-S * disc_q * pdf * np.maximum(sigma, EPS_SIGMA) / (2.0 * np.sqrt(np.maximum(T, EPS_T)))
                 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))
    rho_call = K * T * disc_r * norm.cdf(d2)
    rho_put = -K * T * disc_r * norm.cdf(-d2)

    if kind == "call":
        out = {"delta": delta_call, "gamma": gamma, "vega": vega, "theta": theta_call, "rho": rho_call}
    else:
        out = {"delta": delta_put, "gamma": gamma, "vega": vega, "theta": theta_put, "rho": rho_put}

    # cast a float se shape scalare
    return {k: (float(v) if np.asarray(v).shape == () else v) for k, v in out.items()}


def _no_arb_bounds(S, K, T, r, q, kind: str):
    """Limiti no-arbitrage per controlli su IV."""
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if kind == "call":
        lower = np.maximum(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = np.maximum(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    return lower, upper


def implied_vol(target_price: ArrayLike, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
                q: ArrayLike = 0.0, kind: str = "call",
                lo: float = 1e-9, hi: float = 5.0, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray | float:
    """
    Implied volatility via Brent (robusto). Supporta input scalari o array.
    Restituisce la IV con lo stesso shape dell'input broadcastato.
    """
    S, K, T, r, q, target_price = _broadcast(S, K, T, r, q, target_price)
    out = np.full_like(S, np.nan, dtype=float)

    # maschere: solo elementi con T>0 e prezzo nei limiti
    mask_valid_T = T > 0.0
    lower, upper = _no_arb_bounds(S, K, T, r, q, kind)
    mask_in_bounds = (target_price >= lower - 1e-12) & (target_price <= upper + 1e-12)
    mask = mask_valid_T & mask_in_bounds

    if not np.any(mask):
        return float(out) if out.shape == () else out

    # funzione per brentq su scalare
    def f_sigma(sig, s, k, t, rr, qq, tp):
        return bs_price(s, k, t, rr, sig, qq, kind) - tp

    # loop sugli elementi validi (brentq non è vettoriale; data la robustezza va bene così)
    idxs = np.argwhere(mask)
    for (i, j, *rest) in (idxs if idxs.ndim > 1 else [idxs]):
        # supporta qualunque dimensionalità
        inds = tuple(int(v) for v in ([i, j] + rest)) if idxs.ndim > 1 else (int(idxs),)

        s = S[inds]; k = K[inds]; t = T[inds]; rr = r[inds]; qq = q[inds]; tp = target_price[inds]

        # verifica bracket; se necessario allarga hi una volta
        flo = f_sigma(lo, s, k, t, rr, qq, tp)
        fhi = f_sigma(hi, s, k, t, rr, qq, tp)
        if flo * fhi > 0:
            hi2 = 10.0
            fhi = f_sigma(hi2, s, k, t, rr, qq, tp)
            if flo * fhi > 0:
                # non converge: lascia NaN
                continue
            hi_local = hi2
        else:
            hi_local = hi

        try:
            iv = brentq(f_sigma, lo, hi_local, args=(s, k, t, rr, qq, tp), xtol=tol, maxiter=max_iter)
            out[inds] = iv if isfinite(iv) else np.nan
        except Exception:
            # fall-back: NaN
            out[inds] = np.nan

    return float(out) if out.shape == () else out
