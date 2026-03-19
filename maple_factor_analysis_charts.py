"""
Generate ENA-style factor analysis charts for MAPLE / SYRUP.

Outputs:
  - charts/figures/maple_factor_cumulative_vs_factors.png
  - charts/figures/maple_factor_ols_log_summary.png
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from artemis import Artemis


START_DATE = "2022-01-01"
END_DATE = None
MC_THRESHOLD = 1_000_000_000
MIN_CROSS_SECTION = 20
TOP_N_MARKET = 20
MOM_LOOKBACK_WEEKS = 12
MAPLE_CANDIDATE_SYMBOLS = ["syrup", "maple"]
CHART_DIR = Path("charts/figures")
CHART_DIR.mkdir(parents=True, exist_ok=True)


def get_api_key() -> str:
    api_key = os.getenv("ARTEMIS_API_KEY")
    if not api_key:
        env_path = Path(".env")
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "ARTEMIS_API_KEY":
                    api_key = v.strip().strip("'").strip('"')
                    break
    if not api_key:
        raise EnvironmentError("ARTEMIS_API_KEY is missing in environment.")
    return api_key


def extract_symbol_metrics(symbol_payload) -> Dict[str, Dict[str, List]]:
    if isinstance(symbol_payload, dict):
        return symbol_payload
    data_obj = getattr(symbol_payload, "data", None)
    symbols = getattr(data_obj, "symbols", None)
    if symbols:
        return symbols
    return {}


def fetch_panel(client: Artemis, api_key: str) -> pd.DataFrame:
    all_assets = client.asset.list_asset_symbols()
    if isinstance(all_assets, list):
        asset_rows = all_assets
    elif isinstance(all_assets, dict) and "assets" in all_assets:
        asset_rows = all_assets["assets"]
    else:
        asset_rows = []

    symbols = [a.get("symbol") for a in asset_rows if isinstance(a, dict) and a.get("symbol")]
    if not symbols:
        raise RuntimeError("No symbols from list_asset_symbols().")

    end_date = END_DATE or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    metrics = "price,mc,mc_fees_ratio,fees,revenue"

    full_data_dict: Dict[str, Dict[str, List]] = {}
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        ok = False
        for _ in range(3):
            try:
                result = client.fetch_metrics(
                    metric_names=metrics,
                    symbols=",".join(batch),
                    start_date=START_DATE,
                    end_date=end_date,
                    api_key=api_key,
                )
                full_data_dict.update(extract_symbol_metrics(result.data.symbols))
                ok = True
                time.sleep(0.3)
                break
            except Exception:
                time.sleep(1.5)
        if not ok:
            continue

    records = []
    for asset, metric_dict in full_data_dict.items():
        for metric, values in metric_dict.items():
            for item in values:
                if isinstance(item, dict):
                    dt = item.get("date")
                    val = item.get("val")
                else:
                    dt = getattr(item, "date", None)
                    val = getattr(item, "val", None)
                if dt is not None and val is not None:
                    records.append({"date": dt, "asset": asset, "metric": metric, "value": val})

    raw_df = pd.DataFrame(records)
    if raw_df.empty:
        raise RuntimeError("No metric records returned.")

    pivoted_df = (
        raw_df.pivot(index=["date", "asset"], columns="metric", values="value")
        .reset_index()
        .copy()
    )
    pivoted_df["date"] = pd.to_datetime(pivoted_df["date"])
    pivoted_df = pivoted_df.sort_values(["asset", "date"]).reset_index(drop=True)
    return pivoted_df


def build_weekly_and_factors(pivoted_df: pd.DataFrame):
    agg_cols = {c: "last" for c in ["mc", "mc_fees_ratio", "price", "fees", "revenue"] if c in pivoted_df.columns}
    weekly_df = (
        pivoted_df.set_index("date")
        .groupby("asset")
        .resample("W")
        .agg(agg_cols)
        .reset_index()
        .sort_values(["date", "asset"])
    )

    weekly_df["price_weekly_pct_change"] = weekly_df.groupby("asset")["price"].pct_change()
    weekly_df["mc_t_minus_1"] = weekly_df.groupby("asset")["mc"].shift(1)
    weekly_df["mc_fees_ratio_t_minus_1"] = weekly_df.groupby("asset")["mc_fees_ratio"].shift(1)
    weekly_df["price_12wk_pct_change"] = (
        weekly_df.groupby("asset")["price"].transform(lambda x: x.shift(1) / x.shift(1 + MOM_LOOKBACK_WEEKS) - 1)
    )

    weekly_df = weekly_df[~weekly_df["asset"].str.contains("usd", case=False, na=False)]
    weekly_df = weekly_df[weekly_df["date"] >= pd.to_datetime(START_DATE)]
    weekly_df = weekly_df.sort_values(["asset", "date"])
    lead_valid = weekly_df[["mc", "price"]].notna().any(axis=1)
    weekly_df = weekly_df[lead_valid.groupby(weekly_df["asset"]).cummax()].reset_index(drop=True)

    assets_available = set(weekly_df["asset"].dropna().unique())
    matched_candidates = [s for s in MAPLE_CANDIDATE_SYMBOLS if s in assets_available]
    if not matched_candidates:
        raise RuntimeError("No Maple/SYRUP symbol in weekly dataset.")
    maple_symbol = "syrup" if "syrup" in matched_candidates else matched_candidates[0]

    factor_returns = []
    for date, group in weekly_df.groupby("date"):
        group_1b = group[group["mc_t_minus_1"] >= MC_THRESHOLD].dropna(subset=["mc_t_minus_1", "price_weekly_pct_change"])
        if len(group_1b) < MIN_CROSS_SECTION:
            continue

        row = {"date": date}

        def wret(df):
            w = df["mc_t_minus_1"]
            r = df["price_weekly_pct_change"]
            return np.nan if w.sum() == 0 else (w * r).sum() / w.sum()

        top = group_1b.nlargest(TOP_N_MARKET, "mc_t_minus_1")
        row["market"] = wret(top)

        med = group_1b["mc_t_minus_1"].median()
        small = group_1b[group_1b["mc_t_minus_1"] <= med]
        big = group_1b[group_1b["mc_t_minus_1"] > med]
        if len(small) > 0 and len(big) > 0:
            row["smb"] = wret(small) - wret(big)

        mom_g = group_1b.dropna(subset=["price_12wk_pct_change"])
        if len(mom_g) >= MIN_CROSS_SECTION:
            k = max(1, int(len(mom_g) * 0.30))
            row["mom"] = wret(mom_g.nlargest(k, "price_12wk_pct_change")) - wret(mom_g.nsmallest(k, "price_12wk_pct_change"))

        val_g = group_1b.dropna(subset=["mc_fees_ratio_t_minus_1"])
        if len(val_g) >= MIN_CROSS_SECTION:
            k = max(1, int(len(val_g) * 0.30))
            row["value"] = wret(val_g.nsmallest(k, "mc_fees_ratio_t_minus_1")) - wret(val_g.nlargest(k, "mc_fees_ratio_t_minus_1"))

        factor_returns.append(row)

    factor_df = pd.DataFrame(factor_returns).sort_values("date").dropna(subset=["market"]).reset_index(drop=True)
    maple_df = weekly_df[weekly_df["asset"] == maple_symbol].copy().sort_values("date")
    maple_df["maple_ret"] = maple_df["price"].pct_change()
    maple_df["maple_log_ret"] = np.log(maple_df["price"]).diff()
    return weekly_df, factor_df, maple_df, maple_symbol


def fit_maple_regression(factor_df: pd.DataFrame, maple_df: pd.DataFrame):
    merged = (
        maple_df.set_index("date")[["maple_ret", "maple_log_ret"]]
        .join(factor_df.set_index("date")[["market", "smb", "mom", "value"]], how="inner")
        .dropna()
    )
    X = sm.add_constant(merged[["market", "smb", "mom", "value"]])
    model_log = sm.OLS(merged["maple_log_ret"], X).fit()
    return merged, model_log


def chart_cumulative(merged: pd.DataFrame) -> Path:
    cum_df = pd.DataFrame(index=merged.index)
    cum_df["MAPLE"] = (1 + merged["maple_ret"]).cumprod()
    cum_df["MARKET"] = (1 + merged["market"]).cumprod()
    cum_df["SMB"] = (1 + merged["smb"]).cumprod()
    cum_df["MOM"] = (1 + merged["mom"]).cumprod()
    cum_df["VALUE"] = (1 + merged["value"]).cumprod()

    plt.figure(figsize=(11, 5))
    plt.plot(cum_df.index, cum_df["MAPLE"], label="MAPLE", linewidth=2.5, color="tab:blue")
    plt.plot(cum_df.index, cum_df["MARKET"], label="MARKET", linewidth=1.5, alpha=0.8)
    plt.plot(cum_df.index, cum_df["SMB"], label="SMB", linewidth=1.2, alpha=0.8)
    plt.plot(cum_df.index, cum_df["MOM"], label="MOM", linewidth=1.2, alpha=0.8)
    plt.plot(cum_df.index, cum_df["VALUE"], label="VALUE", linewidth=1.2, alpha=0.8)
    plt.title("Cumulative Value: MAPLE vs Factors (Starting at 1.0)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = CHART_DIR / "maple_factor_cumulative_vs_factors.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def chart_ols_summary(model_log) -> Path:
    txt = "=== MAPLE LOG RETURNS ~ market + SMB + MOM + VALUE ===\n\n" + model_log.summary().as_text()
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.axis("off")
    ax.text(
        0.01,
        0.99,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="white",
        family="monospace",
    )
    fig.tight_layout()
    path = CHART_DIR / "maple_factor_ols_log_summary.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> None:
    api_key = get_api_key()
    client = Artemis(api_key=api_key)
    pivoted = fetch_panel(client, api_key)
    _, factor_df, maple_df, maple_symbol = build_weekly_and_factors(pivoted)
    merged, model_log = fit_maple_regression(factor_df, maple_df)
    p1 = chart_cumulative(merged)
    p2 = chart_ols_summary(model_log)

    print(f"Using Maple symbol: {maple_symbol}")
    print(f"Regression observations: {len(merged)}")
    print("Saved files:")
    print(f"  - {p1}")
    print(f"  - {p2}")


if __name__ == "__main__":
    main()
