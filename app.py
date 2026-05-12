"""app.py — Lévy Process / Alpha-Stable Distribution Engine · Streamlit Dashboard."""

from __future__ import annotations

import os
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from us_calendar import next_trading_day

st.set_page_config(
    page_title="Lévy Stable · P2Quant",
    layout="wide",
    page_icon="⚡",
)

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

PALETTE = [
    "#1B4F8A", "#27AE60", "#E74C3C", "#F39C12",
    "#8E44AD", "#148F77", "#CA6F1E", "#2471A3",
    "#CB4335", "#1A5276", "#117A65", "#B7950B",
    "#884EA0", "#1F618D", "#B9770E", "#148F77",
    "#922B21", "#1A5276",
]

def score_colour(v: float) -> str:
    if v >= 0.5:  return "#1D9E75"
    if v >= 0.0:  return "#82C3A9"
    if v >= -0.5: return "#F0A07A"
    return "#E74C3C"

def alpha_colour(a: float) -> str:
    """Colour for stability index: green=Gaussian, red=heavy-tail."""
    if a >= 1.8:  return "#1D9E75"
    if a >= 1.5:  return "#82C3A9"
    if a >= 1.2:  return "#F39C12"
    return "#E74C3C"

def fmt(v: float, d: int = 4) -> str:
    return f"{v:+.{d}f}"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading Lévy results…")
def load_json(universe: str) -> dict | None:
    slug = universe.lower().replace("_", "-")
    try:
        r = requests.get(BASE_API, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        files   = sorted(f["path"] for f in r.json() if f["path"].endswith(".json"))
        matches = [f for f in files if f"_{slug}.json" in f]
        if not matches:
            return None
        resp = requests.get(f"{BASE_RAW}/{matches[-1]}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading history…")
def load_csv(filename: str) -> pd.DataFrame | None:
    try:
        r = requests.get(f"{BASE_RAW}/{filename}", headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    universe = st.selectbox("Universe", list(config.UNIVERSES.keys()))
    st.divider()
    st.markdown(f"**ETFs:** {len(config.UNIVERSES[universe])}")
    st.markdown(f"**Rolling windows:** {config.ROLLING_WINDOWS}")
    st.markdown(f"**Tail weight:** {config.TAIL_WT}")
    st.markdown(f"**Drift weight:** {config.DRIFT_WT}")
    st.markdown(f"**Skew weight:** {config.SKEW_WT}")
    st.markdown(f"**Scale weight:** {config.SCALE_WT}")
    st.markdown(f"**VIX stress threshold:** {config.VIX_HIGH_THRESHOLD}")
    st.markdown(f"**VIX tail boost:** {config.VIX_TAIL_BOOST}")
    st.markdown(f"**OOS from:** {config.OOS_START}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    st.divider()
    st.markdown("**Score formula:**")
    st.code(
        "score = 0.35*drift_z\n"
        "      + 0.15*skew_z\n"
        "      + 0.10*scale_z\n"
        "      - 0.40*tail_risk\n"
        "(+0.10 tail boost in stress)",
        language="python",
    )
    st.divider()
    st.markdown("**Alpha interpretation:**")
    st.markdown("🟢 α ≥ 1.8 — near-Gaussian, low tail risk")
    st.markdown("🟡 α 1.5–1.8 — moderate heavy tails")
    st.markdown("🟠 α 1.2–1.5 — significant tail risk")
    st.markdown("🔴 α < 1.2 — severe tail risk → penalise")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Lévy Process / Alpha-Stable Distribution Engine")
st.caption(
    "Fits S(α, β, σ, μ) per ETF on 63/126/252d rolling windows · "
    "KS-weighted consensus · α < 2 = heavy tails → penalised · "
    "VIX-regime conditioned tail weight · scipy only"
)

slug       = universe.lower().replace("_", "-")
data       = load_json(universe)
daily_df   = load_csv(f"daily_{slug}.csv")
score_df   = load_csv(f"scores_{slug}.csv")
alpha_df   = load_csv(f"alpha_{slug}.csv")
beta_df    = load_csv(f"beta_{slug}.csv")
crash_df   = load_csv(f"crash_{slug}.csv")
ranking_df = load_csv(f"rankings_{slug}.csv")

if data is None:
    st.warning("⚠️ No results found. Run `python trainer.py` first.")
    st.stop()

latest_scores = data.get("latest_scores", {})
latest_ranked = data.get("latest_ranked", [])
latest_date   = data.get("latest_date", "?")
run_date      = data.get("run_date", "?")
cfg           = data.get("config", {})

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Run Date",    run_date)
k2.metric("Latest Date", latest_date)
k3.metric("Universe",    universe)
k4.metric("ETFs Scored", len(latest_scores))

if latest_ranked:
    top       = latest_ranked[0]
    cash_flag = top.get("composite_score", 0) < config.CASH_THRESHOLD
    alphas    = [v.get("alpha", 2.0) for v in latest_scores.values()]
    mean_alpha = float(np.mean(alphas)) if alphas else 2.0

    # Regime from latest daily
    regime = "?"
    if daily_df is not None and "regime" in daily_df.columns:
        regime = str(daily_df["regime"].iloc[-1])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Top Pick",
              "CASH" if cash_flag else top["ticker"])
    m2.metric("Top Score",     fmt(top.get("composite_score", 0)))
    m3.metric("Mean α (universe)", f"{mean_alpha:.4f}",
              help="Mean stability index; closer to 2.0 = more Gaussian")
    m4.metric("VIX Regime",   regime)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Rankings & Scores",
    "⚡ Stability Index (α)",
    "📉 Crash Probability",
    "📈 Score & Alpha History",
    "📋 Full Parameters Table",
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Rankings & Scores
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Lévy Rankings as of {latest_date}")

    tickers_r    = [r["ticker"] for r in latest_ranked]
    scores_r     = [r.get("composite_score", 0)    for r in latest_ranked]
    alphas_r     = [r.get("alpha", 2.0)             for r in latest_ranked]
    tail_risks_r = [r.get("tail_risk", 0)           for r in latest_ranked]
    colours_r    = [score_colour(s) for s in scores_r]
    alpha_cols_r = [alpha_colour(a) for a in alphas_r]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Composite Score**")
        fig = go.Figure(go.Bar(
            y=tickers_r, x=scores_r, orientation="h",
            marker_color=colours_r,
            text=[fmt(s) for s in scores_r],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Score = 0.35×drift + 0.15×skew + 0.10×scale − 0.40×tail_risk",
            xaxis_title="Composite z-score",
            yaxis=dict(autorange="reversed"),
            height=max(300, len(tickers_r) * 30),
            margin=dict(t=50, b=20, l=60, r=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key="rank_bar")

    with col_r:
        st.markdown("**Stability Index α (green=safe, red=heavy tails)**")
        fig2 = go.Figure(go.Bar(
            y=tickers_r, x=alphas_r, orientation="h",
            marker_color=alpha_cols_r,
            text=[f"{a:.4f}" for a in alphas_r],
            textposition="outside",
        ))
        fig2.add_vline(x=2.0, line_dash="dot", line_color="#1D9E75",
                       annotation_text="Gaussian (α=2)")
        fig2.add_vline(x=1.5, line_dash="dash", line_color="#F39C12",
                       annotation_text="Lévy threshold")
        fig2.update_layout(
            title="α → 2 = Gaussian | α < 1.5 = heavy tails | α = 1 = Cauchy",
            xaxis_title="Stability index α",
            xaxis=dict(range=[0, 2.1]),
            yaxis=dict(autorange="reversed"),
            height=max(300, len(tickers_r) * 30),
            margin=dict(t=50, b=20, l=60, r=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True, key="alpha_bar")

    # Top-N recommendation cards
    st.markdown(f"### 🎯 Top {config.TOP_N} for {next_trading_day()}")
    cols = st.columns(config.TOP_N)
    for i, row in enumerate(latest_ranked[: config.TOP_N]):
        with cols[i]:
            sc  = row.get("composite_score", 0)
            al  = row.get("alpha", 2.0)
            bet = row.get("beta", 0.0)
            cp  = row.get("crash_prob_5pct", 0.0)
            bg  = score_colour(sc)
            ac  = alpha_colour(al)
            st.markdown(
                f"**#{i+1} {row['ticker']}**\n\n"
                f"Score: `{fmt(sc)}`\n\n"
                f'α: <span style="color:{ac}">**{al:.4f}**</span>\n\n'
                f"β: `{bet:+.4f}`\n\n"
                f"P(r<−5%): `{cp:.4f}`\n\n"
                f'<span style="background:{bg};color:white;padding:2px 8px;'
                f'border-radius:8px;font-size:11px">Rank #{row.get("rank", i+1)}</span>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Stability Index (α)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Stability Index α Over Time")
    st.caption(
        "α = 2.0 → Gaussian (finite variance). α < 2 → power-law tails, "
        "potentially infinite variance. **α spikes downward = tail-risk regime shift.**"
    )

    if alpha_df is not None:
        etf_cols = [c for c in alpha_df.columns if c in config.UNIVERSES[universe]]
        selected = st.multiselect(
            "Select ETFs", etf_cols, default=etf_cols[:6], key="alpha_sel"
        )
        period = st.radio(
            "Period", ["Last 2 years", "Last 5 years", "Full OOS"],
            horizontal=True, key="alpha_period"
        )
        df_a = alpha_df.copy()
        if period == "Last 2 years":
            df_a = df_a[df_a.index >= "2024-01-01"]
        elif period == "Last 5 years":
            df_a = df_a[df_a.index >= "2021-01-01"]

        if selected:
            fig_a = go.Figure()
            for i, tkr in enumerate(selected):
                if tkr in df_a.columns:
                    fig_a.add_trace(go.Scatter(
                        x=df_a.index, y=df_a[tkr],
                        mode="lines", name=tkr,
                        line=dict(width=1.4, color=PALETTE[i % len(PALETTE)]),
                    ))
            # Reference lines
            fig_a.add_hline(y=2.0, line_dash="dot", line_color="#1D9E75",
                            annotation_text="Gaussian (α=2.0)")
            fig_a.add_hline(y=1.5, line_dash="dash", line_color="#F39C12",
                            annotation_text="Heavy tail threshold")
            fig_a.add_hline(y=1.0, line_dash="dash", line_color="#E74C3C",
                            annotation_text="Cauchy (α=1.0)")
            fig_a.update_layout(
                title="Rolling stability index α per ETF",
                yaxis_title="α (stability index)",
                yaxis=dict(range=[0.3, 2.1]),
                height=420,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_a, use_container_width=True, key="alpha_ts")

        # Universe mean α — systemic tail-risk gauge
        st.markdown("**Universe mean α — systemic tail-risk gauge**")
        df_all = alpha_df[[c for c in alpha_df.columns
                           if c in config.UNIVERSES[universe]]]
        mean_a = df_all.mean(axis=1)
        min_a  = df_all.min(axis=1)

        fig_sys = go.Figure()
        fig_sys.add_trace(go.Scatter(
            x=mean_a.index, y=mean_a.values,
            mode="lines", name="Mean α",
            line=dict(color="#1B4F8A", width=1.5),
            fill="tozeroy", fillcolor="rgba(27,79,138,0.06)",
        ))
        fig_sys.add_trace(go.Scatter(
            x=min_a.index, y=min_a.values,
            mode="lines", name="Min α (most heavy-tailed ETF)",
            line=dict(color="#E74C3C", width=1.2, dash="dot"),
        ))
        fig_sys.add_hline(y=1.5, line_dash="dash", line_color="#F39C12",
                          annotation_text="Heavy tail threshold")
        fig_sys.update_layout(
            title="Universe-wide stability index α — systemic tail risk gauge",
            yaxis_title="α",
            yaxis=dict(range=[0.3, 2.1]),
            height=340,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_sys, use_container_width=True, key="alpha_sys")

        # α heatmap — last 126 days
        recent_a = df_all.tail(126)
        fig_ah = go.Figure(go.Heatmap(
            z=recent_a.values.T,
            x=recent_a.index.strftime("%Y-%m-%d"),
            y=list(recent_a.columns),
            colorscale="RdYlGn",
            zmin=0.5, zmax=2.0,
            colorbar=dict(title="α"),
            hoverongaps=False,
        ))
        fig_ah.update_layout(
            title="α Heatmap — last 126 days (green=Gaussian, red=heavy tails)",
            height=max(300, len(recent_a.columns) * 22 + 80),
            margin=dict(t=40, b=60, l=60, r=20),
            xaxis=dict(tickangle=-45, nticks=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ah, use_container_width=True, key="alpha_heat")

        # Skewness β
        if beta_df is not None:
            st.markdown("**Skewness parameter β over time (β > 0 = right-skewed = bullish)**")
            df_b = beta_df[[c for c in beta_df.columns
                            if c in config.UNIVERSES[universe]]].tail(252)
            fig_b = go.Figure()
            for i, tkr in enumerate(etf_cols[:8]):
                if tkr in df_b.columns:
                    fig_b.add_trace(go.Scatter(
                        x=df_b.index, y=df_b[tkr],
                        mode="lines", name=tkr,
                        line=dict(width=1.2, color=PALETTE[i % len(PALETTE)]),
                    ))
            fig_b.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_b.update_layout(
                title="Rolling β — last 252 days (positive = right-skewed, bullish)",
                yaxis_title="β (skewness)",
                yaxis=dict(range=[-1.1, 1.1]),
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_b, use_container_width=True, key="beta_ts")
    else:
        st.info("No α history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Crash Probability
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Crash Probability P(r < −5%) under Stable Distribution")
    st.caption(
        "Exact tail probability computed from the fitted α-stable CDF. "
        "For Gaussian ETFs (α≈2), this is very small. "
        "For heavy-tailed ETFs (α < 1.5), crash probability can be 10-100× higher."
    )

    if crash_df is not None:
        etf_cols_c = [c for c in crash_df.columns
                      if c in config.UNIVERSES[universe]]

        # Latest crash prob bar
        crash_latest = {
            tkr: latest_scores[tkr].get("crash_prob_5pct", 0.0)
            for tkr in tickers_r if tkr in latest_scores
        } if latest_ranked else {}

        if crash_latest:
            tkr_c = list(crash_latest.keys())
            val_c = list(crash_latest.values())
            col_c = [
                "#E74C3C" if v > np.percentile(list(val_c), 75) else
                "#F39C12" if v > np.percentile(list(val_c), 50) else "#27AE60"
                for v in val_c
            ]
            fig_cp = go.Figure(go.Bar(
                x=tkr_c, y=val_c,
                marker_color=col_c,
                text=[f"{v:.4f}" for v in val_c],
                textposition="outside",
            ))
            fig_cp.update_layout(
                title=f"P(daily return < −5%) — {latest_date}",
                yaxis_title="Crash probability",
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cp, use_container_width=True, key="crash_snap")

        # Crash prob time-series
        selected_cp = st.multiselect(
            "Select ETFs", etf_cols_c, default=etf_cols_c[:5], key="crash_sel"
        )
        period_cp = st.radio(
            "Period", ["Last 2 years", "Last 5 years", "Full OOS"],
            horizontal=True, key="crash_period"
        )
        df_cp = crash_df.copy()
        if period_cp == "Last 2 years":
            df_cp = df_cp[df_cp.index >= "2024-01-01"]
        elif period_cp == "Last 5 years":
            df_cp = df_cp[df_cp.index >= "2021-01-01"]

        if selected_cp:
            fig_cpts = go.Figure()
            for i, tkr in enumerate(selected_cp):
                if tkr in df_cp.columns:
                    fig_cpts.add_trace(go.Scatter(
                        x=df_cp.index, y=df_cp[tkr],
                        mode="lines", name=tkr,
                        line=dict(width=1.3, color=PALETTE[i % len(PALETTE)]),
                    ))
            fig_cpts.update_layout(
                title="Rolling crash probability P(r < −5%) over time",
                yaxis_title="Probability",
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cpts, use_container_width=True, key="crash_ts")

        # Universe mean crash prob — systemic risk gauge
        mean_cp = crash_df[etf_cols_c].mean(axis=1)
        fig_mcp = go.Figure(go.Scatter(
            x=mean_cp.index, y=mean_cp.values,
            mode="lines", name="Mean crash prob",
            line=dict(color="#E74C3C", width=1.5),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
        ))
        fig_mcp.update_layout(
            title="Universe mean crash probability — systemic tail risk over time",
            yaxis_title="P(r < −5%)",
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mcp, use_container_width=True, key="crash_mean")
    else:
        st.info("No crash probability history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Score & Alpha History
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Composite Score & Alpha History")

    if score_df is not None:
        etf_cols_s = [c for c in score_df.columns
                      if c in config.UNIVERSES[universe]]
        selected_s = st.multiselect(
            "Select ETFs", etf_cols_s, default=etf_cols_s[:6], key="score_sel"
        )
        period_s = st.radio(
            "Period", ["Last 2 years", "Last 5 years", "Full OOS"],
            horizontal=True, key="score_period"
        )
        df_s = score_df.copy()
        if period_s == "Last 2 years":
            df_s = df_s[df_s.index >= "2024-01-01"]
        elif period_s == "Last 5 years":
            df_s = df_s[df_s.index >= "2021-01-01"]

        if selected_s:
            fig_s = go.Figure()
            for i, tkr in enumerate(selected_s):
                if tkr in df_s.columns:
                    fig_s.add_trace(go.Scatter(
                        x=df_s.index, y=df_s[tkr],
                        mode="lines", name=tkr,
                        line=dict(width=1.4, color=PALETTE[i % len(PALETTE)]),
                    ))
            fig_s.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_s.update_layout(
                title="Composite Lévy score (cross-sectional z-score)",
                yaxis_title="Score",
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_s, use_container_width=True, key="score_ts")

        # Score heatmap
        recent_s = score_df[etf_cols_s].tail(252)
        fig_sh = go.Figure(go.Heatmap(
            z=recent_s.values.T,
            x=recent_s.index.strftime("%Y-%m-%d"),
            y=list(recent_s.columns),
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="Score"),
            hoverongaps=False,
        ))
        fig_sh.update_layout(
            title="Score Heatmap — last 252 days",
            height=max(300, len(recent_s.columns) * 22 + 80),
            margin=dict(t=40, b=60, l=60, r=20),
            xaxis=dict(tickangle=-45, nticks=12),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sh, use_container_width=True, key="score_heat")

        # VIX regime overlay on daily summary
        if daily_df is not None and "vix" in daily_df.columns:
            st.markdown("**VIX Regime & Top Score over time**")
            stress = daily_df[daily_df["regime"] == "STRESS"]
            normal = daily_df[daily_df["regime"] == "NORMAL"]
            fig_vx = go.Figure()
            if not normal.empty:
                fig_vx.add_trace(go.Scatter(
                    x=normal.index, y=normal["top_score"],
                    mode="lines", name="Top score (NORMAL)",
                    line=dict(color="#1B4F8A", width=1.2),
                ))
            if not stress.empty:
                fig_vx.add_trace(go.Scatter(
                    x=stress.index, y=stress["top_score"],
                    mode="markers", name="Top score (STRESS VIX>25)",
                    marker=dict(color="#E74C3C", size=4),
                ))
            fig_vx.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_vx.update_layout(
                title="Top-1 Lévy Score: normal vs stress VIX regime",
                yaxis_title="Composite z-score",
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_vx, use_container_width=True, key="vix_regime")

        # Top-pick frequency
        if daily_df is not None and "top_ticker" in daily_df.columns:
            picks = daily_df["top_ticker"].value_counts()
            fig_freq = go.Figure(go.Bar(
                x=picks.index, y=picks.values,
                marker_color="#1B4F8A",
                text=picks.values, textposition="outside",
            ))
            fig_freq.update_layout(
                title="Top-Pick Frequency (full OOS)",
                yaxis_title="Days as #1 Lévy pick",
                height=280,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_freq, use_container_width=True, key="pick_freq")
    else:
        st.info("No score history found.")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Full Parameters Table
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"Full Lévy Parameters — {latest_date}")

    if latest_ranked:
        rows = []
        for i, row in enumerate(latest_ranked):
            rows.append({
                "Rank":           i + 1,
                "Ticker":         row["ticker"],
                "Composite Score":fmt(row.get("composite_score", 0)),
                "α (stability)":  f"{row.get('alpha', 2.0):.5f}",
                "β (skewness)":   f"{row.get('beta', 0.0):+.5f}",
                "σ (scale)":      f"{row.get('sigma', 0.0):.6f}",
                "μ annual":       f"{row.get('mu_annual', 0.0):+.4f}",
                "Tail Risk":      f"{row.get('tail_risk', 0.0):.5f}",
                "P(r<−5%)":       f"{row.get('crash_prob_5pct', 0.0):.5f}",
                "Best Window":    row.get("best_window", "?"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True, height=600)

    st.divider()
    st.markdown("**Engine Configuration**")
    cfg_rows = [{"Parameter": k, "Value": str(v)} for k, v in cfg.items()]
    st.dataframe(pd.DataFrame(cfg_rows), use_container_width=True,
                 hide_index=True, height=300)

    if daily_df is not None:
        st.divider()
        st.markdown("**Daily summary (last 20 days)**")
        st.dataframe(daily_df.tail(20), use_container_width=True)

    st.divider()
    st.caption(
        f"P2Quant Lévy Engine · Run: {run_date} · "
        f"Alpha-Stable S(α,β,σ,μ) · KS-weighted multi-window consensus · "
        f"scipy.stats.levy_stable · Data: {config.HF_DATA_REPO}"
    )
