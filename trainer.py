"""trainer.py — Lévy / Alpha-Stable engine orchestrator."""

from __future__ import annotations

import io
import json
import os

from huggingface_hub import HfApi

import config
import data_manager
from engine import run_engine


def push_results(result: dict, universe: str, token: str) -> None:
    slug = universe.lower().replace("_", "-")
    api  = HfApi(token=token)

    api.create_repo(
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    output = {
        "run_date":      config.TODAY,
        "universe":      universe,
        "latest_date":   result["latest_date"],
        "latest_scores": result["latest_scores"],
        "latest_ranked": [
            {"ticker": t, **v} for t, v in result["latest_ranked"]
        ],
        "config": {
            "rolling_windows":    config.ROLLING_WINDOWS,
            "min_fit_window":     config.MIN_FIT_WINDOW,
            "tail_wt":            config.TAIL_WT,
            "drift_wt":           config.DRIFT_WT,
            "skew_wt":            config.SKEW_WT,
            "scale_wt":           config.SCALE_WT,
            "vix_high_threshold": config.VIX_HIGH_THRESHOLD,
            "vix_tail_boost":     config.VIX_TAIL_BOOST,
            "use_consensus":      config.USE_CONSENSUS,
            "cash_threshold":     config.CASH_THRESHOLD,
            "top_n":              config.TOP_N,
            "oos_start":          config.OOS_START,
        },
    }

    def _push(bytesio: io.BytesIO, path: str, msg: str) -> None:
        api.upload_file(
            path_or_fileobj=bytesio,
            path_in_repo=path,
            repo_id=config.HF_OUTPUT_REPO,
            repo_type="dataset",
            commit_message=msg,
        )

    _push(io.BytesIO(json.dumps(output, indent=2, default=str).encode()),
          f"levy_{config.TODAY}_{slug}.json",
          f"Lévy results {config.TODAY} — {slug}")

    for name, df in [
        ("daily",    result["daily_df"]),
        ("scores",   result["score_df"]),
        ("alpha",    result["alpha_df"]),
        ("beta",     result["beta_df"]),
        ("sigma",    result["sigma_df"]),
        ("mu",       result["mu_df"]),
        ("crash",    result["crash_df"]),
        ("rankings", result["ranking_df"]),
    ]:
        _push(io.BytesIO(df.to_csv().encode()),
              f"{name}_{slug}.csv",
              f"{name} history {config.TODAY} — {slug}")

    print(f"  ✅ Pushed → {config.HF_OUTPUT_REPO}/levy_{config.TODAY}_{slug}.json")


def main() -> None:
    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting.")
        return

    target = os.environ.get("LEVY_UNIVERSE", "ALL").upper()
    log_returns, macro_df = data_manager.load_data(token=token)

    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue
        result = run_engine(
            log_returns=log_returns,
            macro_df=macro_df,
            universe_tickers=tickers,
            universe_name=universe_name,
        )
        push_results(result, universe_name, token)

    print("\n✅ Lévy / Alpha-Stable engine complete.")


if __name__ == "__main__":
    main()
