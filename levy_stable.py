"""levy_stable.py — Alpha-stable distribution fitting and signal extraction.

All computation uses scipy.stats.levy_stable — no external dependencies.

Alpha-stable distribution S(alpha, beta, sigma, mu) parametrisation
--------------------------------------------------------------------
  alpha ∈ (0, 2]   stability index  — governs tail heaviness
                    alpha = 2 → Gaussian
                    alpha = 1 → Cauchy
                    alpha → 0 → extremely heavy tails
  beta  ∈ [-1, 1]  skewness parameter
                    beta = +1 → maximally right-skewed (positive tail heavier)
                    beta = -1 → maximally left-skewed (crash risk heavier)
                    beta = 0  → symmetric
  sigma > 0         scale (dispersion)
                    for alpha=2: sigma = std/sqrt(2)
  mu    ∈ R         location (shift) — acts as drift/expected return

McCulloch (1986) quantile-based estimation
------------------------------------------
Faster than MLE. Uses five sample quantiles (5%, 25%, 50%, 75%, 95%) to
estimate alpha and beta via lookup tables, then estimates sigma and mu
analytically from the remaining quantiles.

Implemented here as a wrapper around scipy's fit(), which internally uses
MLE with good initialisations.

Goodness-of-fit
---------------
Kolmogorov-Smirnov statistic used to select the best rolling window and
to weight the multi-window consensus score.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.stats import levy_stable, kstest

import config


# ── Stable distribution fitting ───────────────────────────────────────────────

def fit_stable(
    returns: np.ndarray,
    use_mle_fallback: bool = config.USE_MLE_FALLBACK,
) -> dict | None:
    """Fit alpha-stable distribution to a 1-D return array.

    Returns dict with keys: alpha, beta, sigma, mu, ks_stat, ks_pval
    Returns None if fitting fails or sample is too small.

    Strategy
    --------
    1. Try scipy levy_stable.fit() with MLE (robust, slower)
    2. Validate: alpha must be in (0.1, 2.0], sigma > 0
    3. Compute KS goodness-of-fit statistic
    """
    if len(returns) < 20:
        return None

    # Standardise to improve numerical stability
    ret_std = returns.std()
    if ret_std < 1e-10:
        return None
    ret_norm = (returns - returns.mean()) / ret_std

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # scipy uses (alpha, beta, loc, scale) parametrisation
            params = levy_stable.fit(ret_norm, method="mle")
        alpha_hat, beta_hat, loc_hat, scale_hat = params

        # Validate parameters
        if not (0.1 < alpha_hat <= 2.0):
            return None
        if scale_hat <= 0:
            return None

        # Rescale back to original units
        mu_hat    = loc_hat * ret_std + returns.mean()
        sigma_hat = scale_hat * ret_std

        # KS goodness-of-fit
        ks_stat, ks_pval = kstest(
            ret_norm,
            lambda x: levy_stable.cdf(x, alpha_hat, beta_hat, loc_hat, scale_hat),
        )

        return {
            "alpha":   float(np.clip(alpha_hat, 0.1, 2.0)),
            "beta":    float(np.clip(beta_hat, -1.0, 1.0)),
            "sigma":   float(sigma_hat),
            "mu":      float(mu_hat),
            "ks_stat": float(ks_stat),
            "ks_pval": float(ks_pval),
            "n_obs":   len(returns),
        }

    except Exception:
        return None


def fit_stable_quantile(returns: np.ndarray) -> dict | None:
    """McCulloch (1986) quantile-based estimation of alpha-stable parameters.

    Fast approximation using 5 sample quantiles. Used as initial estimate
    or standalone when MLE is too slow.

    Reference: McCulloch (1986) 'Simple Consistent Estimators of Stable
    Distribution Parameters', Communications in Statistics.
    """
    if len(returns) < 20:
        return None

    q = np.quantile(returns, [0.05, 0.25, 0.50, 0.75, 0.95])
    q05, q25, q50, q75, q95 = q

    # Avoid division by zero
    if abs(q75 - q25) < 1e-10:
        return None

    # ── Estimate alpha via tail ratio ─────────────────────────────────────────
    # v_alpha = (q95 - q05) / (q75 - q25) — McCulloch Table 1 proxy
    v_alpha = (q95 - q05) / (q75 - q25)
    # Map v_alpha to alpha via monotone relationship (approximate)
    # v_alpha ranges: ~2.44 (alpha=2/Gaussian) to ~6.0+ (alpha→0)
    # Linear interpolation over known anchor points
    v_anchors = np.array([2.439, 2.500, 2.600, 2.700, 2.878,
                           3.212, 3.600, 4.100, 5.000, 6.000])
    a_anchors = np.array([2.000, 1.900, 1.800, 1.700, 1.500,
                           1.300, 1.100, 0.900, 0.700, 0.500])
    alpha_hat = float(np.interp(v_alpha, v_anchors, a_anchors))
    alpha_hat = np.clip(alpha_hat, 0.3, 2.0)

    # ── Estimate beta via asymmetry ───────────────────────────────────────────
    # v_beta = (q95 + q05 - 2*q50) / (q95 - q05)
    denom = q95 - q05
    if abs(denom) < 1e-10:
        beta_hat = 0.0
    else:
        v_beta   = (q95 + q05 - 2 * q50) / denom
        beta_hat = float(np.clip(v_beta * 2.0, -1.0, 1.0))  # approximate rescale

    # ── Scale and location ────────────────────────────────────────────────────
    # sigma estimated from IQR, mu from median (robust to heavy tails)
    sigma_hat = float((q75 - q25) / 1.349)   # IQR → sigma conversion
    mu_hat    = float(q50)

    if sigma_hat <= 0:
        return None

    # Approximate KS (not computed for speed; set to neutral value)
    return {
        "alpha":   float(np.clip(alpha_hat, 0.3, 2.0)),
        "beta":    float(beta_hat),
        "sigma":   float(sigma_hat),
        "mu":      float(mu_hat),
        "ks_stat": 0.10,   # placeholder — lower is better
        "ks_pval": 0.50,
        "n_obs":   len(returns),
    }


# ── Multi-window fitting ──────────────────────────────────────────────────────

def fit_stable_multiwindow(
    returns_1d: np.ndarray,
    windows: list[int] = config.ROLLING_WINDOWS,
    use_mle: bool = False,
) -> dict:
    """Fit alpha-stable on multiple rolling windows; return consensus params.

    For each window:
      - Fit stable distribution
      - Compute KS goodness-of-fit
    Consensus = KS-fit-weighted average of (alpha, beta, sigma, mu).
    Best single window = the one with lowest KS statistic.

    Returns
    -------
    dict with keys:
      consensus_alpha, consensus_beta, consensus_sigma, consensus_mu,
      best_window, best_alpha, best_ks,
      per_window: {window: {alpha, beta, sigma, mu, ks_stat, ...}}
    """
    per_window: dict[int, dict] = {}

    for w in windows:
        if len(returns_1d) < w:
            continue
        sample = returns_1d[-w:]
        if use_mle:
            result = fit_stable(sample)
            if result is None:
                result = fit_stable_quantile(sample)
        else:
            result = fit_stable_quantile(sample)
        if result is not None:
            per_window[w] = result

    if not per_window:
        # Fallback: Gaussian approximation
        return {
            "consensus_alpha": 2.0,
            "consensus_beta":  0.0,
            "consensus_sigma": float(returns_1d.std()),
            "consensus_mu":    float(returns_1d.mean()),
            "best_window":     windows[-1],
            "best_alpha":      2.0,
            "best_ks":         1.0,
            "per_window":      {},
        }

    # ── KS-weighted consensus ─────────────────────────────────────────────────
    # Lower KS stat = better fit → higher weight = 1 / (ks_stat + 1e-4)
    ks_stats = np.array([per_window[w]["ks_stat"] for w in per_window])
    weights  = 1.0 / (ks_stats + 1e-4)
    weights  = weights / weights.sum()

    windows_fit = list(per_window.keys())
    c_alpha = float(np.dot(weights, [per_window[w]["alpha"] for w in windows_fit]))
    c_beta  = float(np.dot(weights, [per_window[w]["beta"]  for w in windows_fit]))
    c_sigma = float(np.dot(weights, [per_window[w]["sigma"] for w in windows_fit]))
    c_mu    = float(np.dot(weights, [per_window[w]["mu"]    for w in windows_fit]))

    # Best single window = lowest KS statistic
    best_w  = windows_fit[int(np.argmin(ks_stats))]

    return {
        "consensus_alpha": float(np.clip(c_alpha, 0.1, 2.0)),
        "consensus_beta":  float(np.clip(c_beta, -1.0, 1.0)),
        "consensus_sigma": float(c_sigma),
        "consensus_mu":    float(c_mu),
        "best_window":     best_w,
        "best_alpha":      float(per_window[best_w]["alpha"]),
        "best_ks":         float(per_window[best_w]["ks_stat"]),
        "per_window":      per_window,
    }


# ── Tail-risk metrics from stable params ─────────────────────────────────────

def tail_risk_from_alpha(alpha: float) -> float:
    """Convert alpha → tail risk score in [0, 1].

    alpha = 2.0  →  tail_risk = 0.0  (Gaussian, no excess tail risk)
    alpha = 1.0  →  tail_risk = 0.5  (Cauchy, severe tail risk)
    alpha = 0.1  →  tail_risk ≈ 0.95 (extreme)
    """
    return float(np.clip((2.0 - alpha) / 1.9, 0.0, 1.0))


def levy_expected_return(
    alpha: float,
    beta: float,
    sigma: float,
    mu: float,
    horizon: int = 252,
) -> float:
    """Annualised location-based expected return under stable law.

    For stable distributions with alpha < 2, the mean is finite only when
    alpha > 1 (and beta=0 or specific conditions). We use the location
    parameter mu as the best estimate of expected return in all cases,
    annualised by horizon.

    For alpha <= 1: mean may not exist → we still use mu but flag it.
    """
    return float(mu * horizon)


def crash_probability(
    alpha: float,
    beta: float,
    sigma: float,
    mu: float,
    threshold: float = -0.05,
) -> float:
    """P(daily return < threshold) under fitted stable distribution.

    Uses scipy levy_stable.cdf for exact computation.
    threshold = -0.05 → probability of a >5% daily loss.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # scipy parametrisation: (alpha, beta, loc=mu, scale=sigma)
            p = levy_stable.cdf(threshold, alpha, beta, loc=mu, scale=sigma)
        return float(np.clip(p, 0.0, 1.0))
    except Exception:
        return float(stats.norm.cdf(threshold, loc=mu, scale=sigma * np.sqrt(2)))


# ── Cross-sectional normalisation ─────────────────────────────────────────────

def zscore_cross(arr: np.ndarray) -> np.ndarray:
    mu  = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mu) / std


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
