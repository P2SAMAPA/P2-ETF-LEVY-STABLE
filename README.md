# ⚡ P2-ETF-LEVY-STABLE

**P2Quant Engine** · Lévy Process / Alpha-Stable Distribution · ETF Ranking

[![Daily Lévy Stable Engine](https://github.com/P2SAMAPA/P2-ETF-LEVY-STABLE/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-LEVY-STABLE/actions/workflows/daily_run.yml)

---

## What Is This?

This engine fits an **alpha-stable distribution S(α, β, σ, μ)** to each ETF's
rolling return series and ranks ETFs by the signals extracted from the four
distribution parameters. It is a direct upgrade of the Merton Jump-Diffusion
engine, replacing Gaussian jump sizes with the full family of stable distributions.

### Why alpha-stable over Gaussian / Merton?

| Property | Gaussian (α=2) | Merton Jump-Diffusion | Alpha-Stable (α < 2) |
|---|---|---|---|
| Tail decay | Exponential | Exponential (Gaussian jumps) | **Power-law** |
| Variance | Finite | Finite | **May be infinite (α < 2)** |
| Jump clustering | ✗ | i.i.d. Poisson | **Captured by stable index** |
| Skewness | Fixed (0) | Fixed (0) | **β ∈ [−1, +1]** |
| Crash probability | Underestimated | Underestimated | **Exact heavy-tail mass** |

---

## Stable Distribution Parameters

| Parameter | Range | Meaning |
|---|---|---|
| **α (stability index)** | (0, 2] | Tail heaviness. α=2 → Gaussian. α=1 → Cauchy. α<1.5 → heavy tails. |
| **β (skewness)** | [−1, +1] | Tail asymmetry. β>0 → right tail heavier (upside). β<0 → left tail heavier (crash risk). |
| **σ (scale)** | > 0 | Dispersion. Equivalent to std for Gaussian; generalises volatility. |
| **μ (location)** | ℝ | Drift / expected return. Used as annualised return signal. |

---

## Scoring Formula

```
tail_risk(i)   = (2 − α(i)) / 1.9          → [0, 1]  (0=Gaussian, 1=extreme)
drift_ann(i)   = μ(i) × 252                → annualised location drift
skew_signal(i) = β(i)                      → +1 bullish, −1 bearish
scale_signal(i)= −σ(i)                     → lower scale = more stable

composite(i)   = 0.35 × drift_z(i)
               + 0.15 × skew_z(i)
               + 0.10 × scale_z(i)
               − 0.40 × tail_risk(i)
```

**VIX regime conditioning:**  
If VIX > 25 (stress regime) → tail_weight = 0.40 + 0.10 = 0.50  
Heavy tails matter more when macro is already stressed.

If `top_composite_score < CASH_THRESHOLD (−0.60)` → recommend CASH.

---

## Estimation Method

**Primary:** `scipy.stats.levy_stable.fit()` — Maximum Likelihood Estimation.

**Multi-window consensus:** Fit on three rolling windows (63d / 126d / 252d).
Weight each window's parameters by KS goodness-of-fit:

```
weight(w) = 1 / (KS_stat(w) + ε)
consensus_α = Σ weight(w) × α(w) / Σ weight(w)
```

**Best single window** = the one with lowest KS statistic (best fit).  
**Refit frequency:** every 5 trading days (balance accuracy vs runtime).

---

## Data Split (2008 → 2026 YTD)

No train/test split required — stable distribution fitting is non-parametric
maximum likelihood. Scores are published from `OOS_START = 2009-06-01`
(~252 days after dataset start, ensuring sufficient history for 252d window fit).

---

## Universes

| Universe | Tickers |
|---|---|
| EQUITY_SECTORS | SPY QQQ XLK XLF XLE XLV XLI XLY XLP XLU GDX XME IWF XSD XBI IWM |
| FI_COMMODITIES | TLT VCIT LQD HYG VNQ GLD SLV |
| COMBINED | All above |

---

## Output Files (per universe)

| File | Content |
|---|---|
| `levy_YYYY-MM-DD_{universe}.json` | Latest params, scores, rankings, config |
| `daily_{universe}.csv` | Top pick, CASH flag, VIX, regime, mean α, mean crash prob |
| `scores_{universe}.csv` | Full daily composite score history |
| `alpha_{universe}.csv` | Full daily stability index α history |
| `beta_{universe}.csv` | Full daily skewness β history |
| `sigma_{universe}.csv` | Full daily scale σ history |
| `mu_{universe}.csv` | Full daily location μ history |
| `crash_{universe}.csv` | Full daily P(r < −5%) history |
| `rankings_{universe}.csv` | Full daily rank history |

**Results repo:** [P2SAMAPA/p2-etf-levy-stable-results](https://huggingface.co/datasets/P2SAMAPA/p2-etf-levy-stable-results)

---

## Streamlit Dashboard — 5 Tabs

1. **Rankings & Scores** — composite score bar, α bar (green=Gaussian / red=heavy tail), top-N cards with all 4 parameters
2. **Stability Index (α)** — α time-series with Gaussian/Cauchy reference lines, universe systemic tail-risk gauge, α heatmap, rolling β skewness
3. **Crash Probability** — P(r < −5%) snapshot bar, time-series, universe mean crash probability gauge
4. **Score & Alpha History** — composite score time-series + heatmap, VIX regime overlay, top-pick frequency
5. **Full Parameters Table** — all α, β, σ, μ, tail risk, crash prob per ETF + engine config

---

## Runtime Note

`scipy.stats.levy_stable.fit()` with MLE is the most computationally expensive
fitting step in the suite (~0.5–2s per ETF per window depending on sample size).
The GitHub Actions timeout is set to 90 minutes. The engine refits every 5 days
(not daily) to manage this — parameters are stable enough over 5-day periods.

---

## References

- Samorodnitsky, G. & Taqqu, M.S. (1994). *Stable Non-Gaussian Random Processes.* Chapman & Hall.
- McCulloch, J.H. (1986). *Simple Consistent Estimators of Stable Distribution Parameters.* Communications in Statistics.
- Nolan, J.P. (2020). *Univariate Stable Distributions.* Springer.
- Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes.* CRC Press.
- Rachev, S.T. & Mittnik, S. (2000). *Stable Paretian Models in Finance.* Wiley.

---

*P2Quant Engine Suite · Built by P2SAMAPA*
