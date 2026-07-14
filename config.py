"""config.py — Lévy Process / Alpha-Stable Distribution Engine configuration.

Core idea
---------
Fit an alpha-stable distribution S(alpha, beta, sigma, mu) to each ETF's
rolling return series. The stability index alpha ∈ (0, 2] is the key signal:

  alpha = 2.0  →  Gaussian (no heavy tails, finite variance)
  alpha < 2.0  →  heavy tails, power-law decay, potentially infinite variance
  alpha < 1.5  →  very heavy tails → crash risk → penalise
  alpha → 0    →  extreme Cauchy-like tails → avoid entirely

Secondary signals from the stable distribution:
  beta  ∈ [-1, 1]  skewness parameter (+1 = right-skewed, -1 = left-skewed)
  sigma > 0         scale (dispersion) — equivalent to volatility for Gaussian
  mu    ∈ R         location (drift) — expected return under stable law

Lévy exponent advantage over Merton Jump-Diffusion
---------------------------------------------------
Merton uses Gaussian jump sizes → finite variance, Poisson jump arrival.
Alpha-stable Lévy processes allow:
  - Infinite variance (alpha < 2)
  - Power-law tail decay (vs Gaussian exponential)
  - Jump clustering captured by the stable index
  - Skewness in jump distribution via beta

Scoring formula
---------------
  tail_risk(i)   = 2 - alpha(i)          # 0=Gaussian, 2=Cauchy
  tail_norm(i)   = minmax(tail_risk)     # normalise to [0, 1]
  skew_signal(i) = beta(i)               # +1 = right tail heavier (bullish)
  drift_z(i)     = zscore(mu_annualised) # location parameter z-score
  scale_z(i)     = zscore(-sigma)        # lower scale = more stable = better

  composite(i)   = DRIFT_WT   * drift_z(i)
                 + SKEW_WT    * zscore(skew_signal)
                 + SCALE_WT   * scale_z(i)
                 - TAIL_WT    * tail_norm(i)  ← heavy tails penalised

  All cross-sectionally z-scored → final rank.
"""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE   = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-levy-stable-results"
HF_TOKEN       = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "SMH", "SOXX", "XLB", "IWD", "IWO",
    "IWF", "XSD", "XBI", "IWM",
]
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
COMBINED_TICKERS       = sorted(set(EQUITY_SECTORS_TICKERS + FI_COMMODITIES_TICKERS))

UNIVERSES = {
    "EQUITY_SECTORS":  EQUITY_SECTORS_TICKERS,
    "FI_COMMODITIES":  FI_COMMODITIES_TICKERS,
    "COMBINED":        COMBINED_TICKERS,
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# ── Fitting windows ───────────────────────────────────────────────────────────
# We test three rolling windows and select the one that gives the best
# Kolmogorov-Smirnov goodness-of-fit against the empirical CDF.
ROLLING_WINDOWS    = [63, 126, 252]   # 1-quarter, 2-quarter, 1-year
MIN_FIT_WINDOW     = 63               # minimum days required to fit stable dist

# ── McCulloch estimation method ───────────────────────────────────────────────
# scipy.stats.levy_stable.fit() uses MLE which can be slow.
# We use the faster McCulloch (1986) quantile-based method as primary,
# with MLE as fallback for robustness on short windows.
USE_MLE_FALLBACK   = True             # fallback to MLE if McCulloch fails
MLE_MAX_ITER       = 500             # max iterations for MLE optimiser

# ── Scoring weights ───────────────────────────────────────────────────────────
TAIL_WT            = 0.40   # weight on tail-risk penalty  (2 - alpha)
DRIFT_WT           = 0.35   # weight on mu (location/drift) signal
SKEW_WT            = 0.15   # weight on beta (skewness) signal
SCALE_WT           = 0.10   # weight on -sigma (lower scale = more stable)

# ── Regime conditioning via VIX ───────────────────────────────────────────────
# In high-VIX regimes, tail risk matters more → boost TAIL_WT
VIX_HIGH_THRESHOLD = 25.0            # VIX above this → stressed regime
VIX_TAIL_BOOST     = 0.10            # extra weight on tail penalty in stress

# ── Multi-window consensus ────────────────────────────────────────────────────
# Score each window's stable fit, then blend by KS goodness-of-fit weight
# (better KS fit → higher weight in consensus)
USE_CONSENSUS      = True            # blend across all ROLLING_WINDOWS
CONSENSUS_WINDOWS  = [63, 126, 252]  # must match ROLLING_WINDOWS

# ── CASH threshold ────────────────────────────────────────────────────────────
CASH_THRESHOLD     = -0.60           # composite z-score below → recommend CASH

# ── Top N ─────────────────────────────────────────────────────────────────────
TOP_N              = 6

# ── OOS start ─────────────────────────────────────────────────────────────────
# First date we publish scores (needs MIN_FIT_WINDOW of history)
OOS_START          = "2009-06-01"    # ~252 days after dataset start

# ── Output ────────────────────────────────────────────────────────────────────
TODAY = datetime.now().strftime("%Y-%m-%d")
