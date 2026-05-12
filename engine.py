"""engine.py — Lévy / Alpha-Stable walk-forward engine.

Daily pipeline per universe
---------------------------
For each day t >= OOS_START:
  1. For each ETF: fit alpha-stable S(alpha, beta, sigma, mu) on all
     configured rolling windows (63 / 126 / 252 days)
  2. Compute KS-weighted consensus parameters
  3. Derive signals:
       tail_risk    = (2 - alpha) / 1.9         → [0,1], high = bad
       drift_ann    = mu * 252                   → annualised location
       skew_signal  = beta                       → +1 bullish, -1 bearish
       scale_signal = -sigma                     → lower dispersion = better
       crash_prob   = P(r < -5%) under stable   → downside tail mass
  4. VIX regime conditioning: in high-VIX regime boost tail_risk weight
  5. Composite z-score → cross-sectional rank
  6. Store daily: params, scores, crash probs, rankings

Performance note
----------------
scipy levy_stable.fit() (MLE) is O(n) per ETF per window.
For 23 ETFs × 3 windows × ~4000 days = ~276,000 fits.
We cache: if parameters change by < 1e-4 and n_obs unchanged, reuse.
Refit frequency: every REFIT_FREQ days (default 5) to manage runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from levy_stable import (
    fit_stable_multiwindow,
    tail_risk_from_alpha,
    levy_expected_return,
    crash_probability,
    zscore_cross,
    minmax_norm,
)

REFIT_FREQ = 21  # refit stable params every N days (balance accuracy vs speed)


def run_engine(
    log_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    universe_tickers: list[str],
    universe_name: str,
) -> dict:
    """Run Lévy / Alpha-Stable engine for one universe."""

    avail = [t for t in universe_tickers if t in log_returns.columns]
    n_etf = len(avail)

    print(
        f"\n{'='*60}\n"
        f"Universe: {universe_name}  ({n_etf} ETFs)\n"
        f"Period: {log_returns.index[0].date()} → {log_returns.index[-1].date()}"
        f"  ({len(log_returns)} days)\n"
        f"{'='*60}"
    )

    ret_arr  = log_returns[avail].values    # (T, n_etf)
    dates    = log_returns.index
    vix_col  = "VIX" if "VIX" in macro_df.columns else None
    vix_arr  = macro_df["VIX"].values if vix_col else np.zeros(len(ret_arr))

    oos_start = pd.Timestamp(config.OOS_START)

    # ── Storage ───────────────────────────────────────────────────────────────
    param_records    : list[dict] = []   # alpha, beta, sigma, mu per ETF
    score_records    : list[dict] = []   # composite score per ETF
    crash_records    : list[dict] = []   # crash probability per ETF
    ranking_records  : list[dict] = []   # rank per ETF
    daily_records    : list[dict] = []   # top pick, CASH flag, regime

    # Cache: last fitted params per ETF to avoid redundant refits
    param_cache : dict[int, dict] = {}   # etf_idx → latest fit result
    last_refit_t = -REFIT_FREQ           # force refit on first iteration

    n_scored = 0

    for t in range(config.MIN_FIT_WINDOW, len(ret_arr)):
        date = dates[t]
        if date < oos_start:
            continue

        do_refit = (t - last_refit_t) >= REFIT_FREQ

        # ── VIX regime ────────────────────────────────────────────────────────
        vix_now     = float(vix_arr[t]) if t < len(vix_arr) else 15.0
        high_stress = vix_now > config.VIX_HIGH_THRESHOLD

        tail_wt  = config.TAIL_WT  + (config.VIX_TAIL_BOOST if high_stress else 0.0)
        drift_wt = config.DRIFT_WT
        skew_wt  = config.SKEW_WT
        scale_wt = config.SCALE_WT

        # ── Fit stable distributions ──────────────────────────────────────────
        alphas      = np.zeros(n_etf)
        betas       = np.zeros(n_etf)
        sigmas      = np.zeros(n_etf)
        mus         = np.zeros(n_etf)
        ks_stats    = np.ones(n_etf)
        best_wins   = np.zeros(n_etf, dtype=int)

        for i in range(n_etf):
            if do_refit or i not in param_cache:
                series  = ret_arr[:t, i]
                result  = fit_stable_multiwindow(
                    series,
                    windows=config.ROLLING_WINDOWS,
                    use_mle=(t == len(ret_arr) - 1),  # MLE only on last day
                )
                param_cache[i] = result

            r = param_cache[i]
            alphas[i]    = r["consensus_alpha"]
            betas[i]     = r["consensus_beta"]
            sigmas[i]    = r["consensus_sigma"]
            mus[i]       = r["consensus_mu"]
            ks_stats[i]  = r["best_ks"]
            best_wins[i] = r["best_window"]

        if do_refit:
            last_refit_t = t

        # ── Derive signals ────────────────────────────────────────────────────
        tail_risk   = np.array([tail_risk_from_alpha(a) for a in alphas])
        drift_ann   = mus * 252
        skew_signal = betas
        scale_signal = -sigmas   # lower scale → more stable → positive signal

        crash_probs = np.array([
            crash_probability(alphas[i], betas[i], sigmas[i], mus[i],
                              threshold=-0.05)
            for i in range(n_etf)
        ])

        # ── Composite score ───────────────────────────────────────────────────
        drift_z  = zscore_cross(drift_ann)
        skew_z   = zscore_cross(skew_signal)
        scale_z  = zscore_cross(scale_signal)
        tail_pen = tail_risk   # already in [0,1], no zscore needed

        composite = (
            drift_wt * drift_z
            + skew_wt  * skew_z
            + scale_wt * scale_z
            - tail_wt  * tail_pen
        )
        composite_z = zscore_cross(composite)

        # ── Rank ──────────────────────────────────────────────────────────────
        ranked_idx = np.argsort(composite_z)[::-1]
        top_ticker = avail[ranked_idx[0]]
        top_score  = float(composite_z[ranked_idx[0]])
        cash_flag  = top_score < config.CASH_THRESHOLD

        ds = date.strftime("%Y-%m-%d")
        n_scored += 1

        param_records.append({"date": ds, **{
            avail[i]: {
                "alpha": round(float(alphas[i]), 5),
                "beta":  round(float(betas[i]), 5),
                "sigma": round(float(sigmas[i]), 8),
                "mu":    round(float(mus[i]), 8),
                "ks":    round(float(ks_stats[i]), 5),
                "win":   int(best_wins[i]),
            }
            for i in range(n_etf)
        }})

        score_records.append({"date": ds,
            **{avail[i]: round(float(composite_z[i]), 6) for i in range(n_etf)}
        })

        crash_records.append({"date": ds,
            **{avail[i]: round(float(crash_probs[i]), 6) for i in range(n_etf)}
        })

        ranking_records.append({"date": ds,
            **{avail[ranked_idx[r]]: r + 1 for r in range(n_etf)}
        })

        regime = "STRESS" if high_stress else "NORMAL"
        daily_records.append({
            "date":         ds,
            "top_ticker":   "CASH" if cash_flag else top_ticker,
            "top_score":    round(top_score, 6),
            "cash_flag":    cash_flag,
            "vix":          round(vix_now, 2),
            "regime":       regime,
            "mean_alpha":   round(float(alphas.mean()), 5),
            "min_alpha":    round(float(alphas.min()), 5),
            "mean_crash_p": round(float(crash_probs.mean()), 6),
        })

        if n_scored % 252 == 0 or t == len(ret_arr) - 1:
            top5 = [
                (avail[ranked_idx[r]],
                 round(float(composite_z[ranked_idx[r]]), 3),
                 round(float(alphas[ranked_idx[r]]), 3))
                for r in range(min(5, n_etf))
            ]
            print(
                f"  {ds} [{regime} VIX={vix_now:.1f}] | "
                + "  ".join(f"{tk}(z={sc:+.2f} α={al:.2f})" for tk, sc, al in top5)
            )

    # ── Latest snapshot ───────────────────────────────────────────────────────
    latest_params  = param_records[-1]
    latest_scores  = score_records[-1]
    latest_crash   = crash_records[-1]
    latest_ranking = ranking_records[-1]
    latest_date    = daily_records[-1]["date"]

    latest_out: dict[str, dict] = {}
    for i, tkr in enumerate(avail):
        latest_out[tkr] = {
            "composite_score": latest_scores[tkr],
            "alpha":           latest_params[tkr]["alpha"],
            "beta":            latest_params[tkr]["beta"],
            "sigma":           latest_params[tkr]["sigma"],
            "mu_annual":       round(latest_params[tkr]["mu"] * 252, 6),
            "tail_risk":       round(tail_risk_from_alpha(latest_params[tkr]["alpha"]), 5),
            "crash_prob_5pct": latest_crash[tkr],
            "best_window":     latest_params[tkr]["win"],
            "rank":            int(latest_ranking[tkr]),
        }

    latest_ranked = sorted(
        latest_out.items(),
        key=lambda x: x[1]["composite_score"],
        reverse=True,
    )

    # ── Flatten param records for CSV ─────────────────────────────────────────
    # Separate CSV per param: alpha_df, beta_df, sigma_df, mu_df
    alpha_rows, beta_rows, sigma_rows, mu_rows = [], [], [], []
    for rec in param_records:
        d = rec["date"]
        alpha_rows.append({"date": d,
            **{t: rec[t]["alpha"] for t in avail}})
        beta_rows.append({"date": d,
            **{t: rec[t]["beta"]  for t in avail}})
        sigma_rows.append({"date": d,
            **{t: rec[t]["sigma"] for t in avail}})
        mu_rows.append({"date": d,
            **{t: rec[t]["mu"]    for t in avail}})

    alpha_df   = pd.DataFrame(alpha_rows).set_index("date")
    beta_df    = pd.DataFrame(beta_rows).set_index("date")
    sigma_df   = pd.DataFrame(sigma_rows).set_index("date")
    mu_df      = pd.DataFrame(mu_rows).set_index("date")
    score_df   = pd.DataFrame(score_records).set_index("date")
    crash_df   = pd.DataFrame(crash_records).set_index("date")
    ranking_df = pd.DataFrame(ranking_records).set_index("date")
    daily_df   = pd.DataFrame(daily_records).set_index("date")

    print(
        f"\n  Latest ({latest_date}) top-{config.TOP_N}: "
        + "  ".join(
            f"{t}(z={v['composite_score']:+.3f} α={v['alpha']:.3f})"
            for t, v in latest_ranked[: config.TOP_N]
        )
    )
    print(f"  Days scored (OOS): {n_scored}")

    return {
        "latest_date":   latest_date,
        "latest_scores": latest_out,
        "latest_ranked": latest_ranked,
        "daily_df":      daily_df,
        "score_df":      score_df,
        "alpha_df":      alpha_df,
        "beta_df":       beta_df,
        "sigma_df":      sigma_df,
        "mu_df":         mu_df,
        "crash_df":      crash_df,
        "ranking_df":    ranking_df,
        "universe":      universe_name,
        "n_etf":         n_etf,
    }
