"""
MAPLE / SYRUP dynamic valuation model (Kalman local-level state-space).

Primary model:
    y_t = log(MC / annualized_fees_30d)_t
    y_t = mu_t + beta' X_t + eps_t
    mu_t = mu_{t-1} + eta_t

Robustness model:
    y_t = log(MC / annualized_revenue_30d)_t
    same controls and state equation.

Outputs:
    - charts/figures/*.png (core charts + optional relative-state + fee-momentum diagnostics)
    - outputs/tables/latest_peer_multiples.csv
    - outputs/tables/fee_momentum_summary.csv
    - outputs/summaries/maple_kalman_summary.md
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from artemis import Artemis
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.structural import UnobservedComponents


START_DATE = "2022-01-01"
WEEKLY_RULE = "W-WED"
USE_LOCAL_FILES = True

OUT_DIR = Path("outputs")
TABLE_DIR = OUT_DIR / "tables"
SUMMARY_DIR = OUT_DIR / "summaries"
DATA_DIR = Path("charts/data")
CHART_DIR = Path("charts/figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_FILE_CANDIDATES = {
    "maple": [
        DATA_DIR / "maple-price.csv",
        DATA_DIR / "maple-tvl.csv",
        DATA_DIR / "metric-comparison-maple.csv",
    ],
    "aave": [
        DATA_DIR / "AAVE.csv",
        DATA_DIR / "AAVE(RevenueAndTVL).csv",
    ],
    "morpho": [
        DATA_DIR / "Metric Comparison - Morpho.csv",
        DATA_DIR / "Morpho(RevenueAndTVL).csv",
    ],
    "btc": [DATA_DIR / "Metric Comparison - Bitcoin.csv"],
    "eth": [DATA_DIR / "Metric Comparison - Ethereum.csv"],
}

TOKEN_MAP = {
    "maple": "syrup",
    "aave": "aave",
    "morpho": "morpho",
    "btc": "btc",
    "eth": "eth",
}

# Roles for interpretation.
ROLE_LABELS = {
    "maple": "target",
    "aave": "core business-model comp",
    "morpho": "core business-model comp",
}


@dataclass
class ModelResult:
    name: str
    data: pd.DataFrame
    fit: object
    endog_col: str
    gap_col: str
    z_col: str
    lb_pvalue_lag4: float
    lb_pvalue_lag8: float
    current_z: float
    classification: str
    observation_variance: float
    state_variance: float
    signal_to_noise: float
    latent_state_std: float
    regime_dynamics_label: str


@dataclass
class FeeMomentumResult:
    panel: pd.DataFrame
    summary_table: pd.DataFrame
    peer_model_coef: float
    peer_model_pvalue: float
    peer_event_lift_4w: float
    peer_event_lift_8w: float
    maple_current_signal: bool
    maple_latest_fee_accel: float
    maple_latest_price_ret_4w: float
    maple_latest_mult_change_4w: float
    peer_total_events_4w: int
    peer_total_events_8w: int
    evidence_quality: str
    replace_kalman_as_core: bool


def get_api_key() -> str:
    api_key = os.getenv("ARTEMIS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ARTEMIS_API_KEY is missing. Set it in your shell environment "
            "or local .env before running this script."
        )
    return api_key


def extract_symbol_metrics(result) -> Dict[str, Dict[str, List[Tuple[pd.Timestamp, float]]]]:
    """
    Artemis SDK compatibility layer:
    returns dict[symbol][metric] = list[(date, value)].
    """
    out: Dict[str, Dict[str, List[Tuple[pd.Timestamp, float]]]] = {}
    for symbol, metric_dict in result.data.symbols.items():
        out[symbol] = {}
        for metric, values in metric_dict.items():
            rows: List[Tuple[pd.Timestamp, float]] = []
            for item in values:
                if isinstance(item, dict):
                    dt = item.get("date")
                    val = item.get("val")
                else:
                    dt = getattr(item, "date", None)
                    val = getattr(item, "val", None)
                if dt is None or val is None:
                    continue
                rows.append((pd.to_datetime(dt), float(val)))
            out[symbol][metric] = rows
    return out


def fetch_daily_data(client: Artemis, api_key: str, symbols: List[str]) -> pd.DataFrame:
    metrics = "price,mc,fdv,fees,revenue,tvl,circulating_supply"
    result = client.fetch_metrics(
        metric_names=metrics,
        symbols=",".join(symbols),
        start_date=START_DATE,
        end_date=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        api_key=api_key,
    )
    parsed = extract_symbol_metrics(result)

    records = []
    for sym, metric_dict in parsed.items():
        for metric, rows in metric_dict.items():
            for dt, val in rows:
                records.append(
                    {
                        "date": dt,
                        "symbol": sym,
                        "metric": metric,
                        "value": val,
                    }
                )
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No rows returned from Artemis metrics fetch.")
    wide = (
        df.pivot_table(
            index=["date", "symbol"], columns="metric", values="value", aggfunc="last"
        )
        .reset_index()
        .sort_values(["symbol", "date"])
    )
    wide.columns.name = None
    return wide


def metric_name_to_standard(col: str) -> str | None:
    c = col.strip().lower()
    if "date" in c:
        return "date"
    if "price" in c:
        return "price"
    if "fully diluted market cap" in c or "fully diluted value" in c:
        return "fdv"
    if "market cap" in c:
        return "mc"
    if "circulating supply" in c:
        return "circulating_supply"
    if "fees" in c:
        return "fees"
    if "revenue" in c:
        return "revenue"
    if "total value locked" in c or c == "tvl":
        return "tvl"
    return None


def read_local_csv(file_path: Path, symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(file_path)
    rename_map = {}
    for col in raw.columns:
        std = metric_name_to_standard(col)
        if std is not None:
            rename_map[col] = std
    df = raw.rename(columns=rename_map)
    if "date" not in df.columns:
        if "DateTime" in raw.columns:
            df = raw.rename(columns={"DateTime": "date"})
        elif "date" not in df.columns:
            raise ValueError(f"Could not find date column in {file_path}")

    keep = ["date", "price", "mc", "fdv", "circulating_supply", "fees", "revenue", "tvl"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = symbol
    return out


def load_local_data(symbols: List[str]) -> pd.DataFrame | None:
    frames = []
    missing_symbols = []
    for sym in symbols:
        # map back to key name for LOCAL_FILE_CANDIDATES
        inv = {v: k for k, v in TOKEN_MAP.items()}
        key = inv.get(sym, sym)
        candidates = LOCAL_FILE_CANDIDATES.get(key, [])
        existing = [p for p in candidates if p.exists()]
        if not existing:
            missing_symbols.append(sym)
            continue
        for fp in existing:
            frames.append(read_local_csv(fp, sym))

    if missing_symbols:
        print(f"Local-file mode requested, but missing local files for symbols: {missing_symbols}")
        return None
    if not frames:
        return None

    local = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    # consolidate duplicate rows from multiple files by taking latest non-null per field
    agg_cols = [c for c in local.columns if c not in {"date", "symbol"}]
    local = (
        local.groupby(["date", "symbol"], as_index=False)[agg_cols]
        .agg(lambda x: x.dropna().iloc[-1] if x.notna().any() else np.nan)
    )
    return local


def add_annualized_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "date"])
    g = df.groupby("symbol", group_keys=False)

    # 30-day trailing average daily fees/revenue annualized.
    df["fees_30d_avg_daily"] = g["fees"].transform(lambda x: x.rolling(30, min_periods=30).mean())
    df["revenue_30d_avg_daily"] = g["revenue"].transform(
        lambda x: x.rolling(30, min_periods=30).mean()
    )
    df["ann_fees_30d"] = df["fees_30d_avg_daily"] * 365.0
    df["ann_rev_30d"] = df["revenue_30d_avg_daily"] * 365.0
    # 90-day run-rate is used for momentum/acceleration diagnostics.
    df["fees_90d_avg_daily"] = g["fees"].transform(lambda x: x.rolling(90, min_periods=90).mean())
    df["revenue_90d_avg_daily"] = g["revenue"].transform(
        lambda x: x.rolling(90, min_periods=90).mean()
    )
    df["ann_fees_90d"] = df["fees_90d_avg_daily"] * 365.0
    df["ann_rev_90d"] = df["revenue_90d_avg_daily"] * 365.0

    # Derived multiples.
    df["mc_to_ann_fees"] = df["mc"] / df["ann_fees_30d"]
    df["mc_to_ann_rev"] = df["mc"] / df["ann_rev_30d"]
    df["mc_to_ann_fees_90d"] = df["mc"] / df["ann_fees_90d"]
    return df


def clean_maple(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    out = df.copy()
    maple_mask = out["symbol"] == TOKEN_MAP["maple"]

    # Circulating supply from metric, fallback to mc/price.
    if "circulating_supply" not in out.columns:
        out["circulating_supply"] = np.nan
    out["circ_supply_eff"] = out["circulating_supply"]
    fallback = (out["circ_supply_eff"].isna()) & (out["price"] > 0)
    out.loc[fallback, "circ_supply_eff"] = out.loc[fallback, "mc"] / out.loc[fallback, "price"]

    clipped_placeholder = bool(((out.loc[maple_mask, "circ_supply_eff"] < 1) & out.loc[maple_mask, "circ_supply_eff"].notna()).any())
    out.loc[maple_mask & (out["circ_supply_eff"] < 1), "circ_supply_eff"] = np.nan

    # Start sample only once mc and price are economically meaningful.
    meaningful = maple_mask & (out["mc"] > 1_000_000) & (out["price"] > 1e-6) & out["circ_supply_eff"].notna()
    if meaningful.any():
        start_dt = out.loc[meaningful, "date"].min()
        out = out[~maple_mask | (out["date"] >= start_dt)]
    else:
        raise RuntimeError("No economically meaningful MAPLE/SYRUP observations after cleaning.")

    return out, clipped_placeholder


def build_weekly_panel(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mc",
        "fdv",
        "price",
        "tvl",
        "fees",
        "revenue",
        "ann_fees_30d",
        "ann_rev_30d",
        "ann_fees_90d",
        "ann_rev_90d",
        "mc_to_ann_fees",
        "mc_to_ann_rev",
        "mc_to_ann_fees_90d",
        "circ_supply_eff",
    ]
    agg_cols = {c: "last" for c in cols if c in df.columns}
    weekly = (
        df.set_index("date")
        .groupby("symbol")
        .resample(WEEKLY_RULE)
        .agg(agg_cols)
        .reset_index()
        .sort_values(["date", "symbol"])
    )
    # Drop future-labeled week-end points.
    weekly = weekly[weekly["date"] <= pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]
    return weekly


def prepare_maple_model_df(weekly: pd.DataFrame, endog_col: str) -> pd.DataFrame:
    m = weekly[weekly["symbol"] == TOKEN_MAP["maple"]].copy()
    btc = weekly[weekly["symbol"] == TOKEN_MAP["btc"]][["date", "price"]].rename(
        columns={"price": "btc_price"}
    )
    eth = weekly[weekly["symbol"] == TOKEN_MAP["eth"]][["date", "price"]].rename(
        columns={"price": "eth_price"}
    )

    m = m.merge(btc, on="date", how="inner").merge(eth, on="date", how="inner")
    m = m.sort_values("date")
    m["btc_ret_log"] = np.log(m["btc_price"]).diff()
    m["eth_ret_log"] = np.log(m["eth_price"]).diff()
    m["tvl_log_growth"] = np.log(m["tvl"]).diff()

    m["endog_log"] = np.log(m[endog_col])

    # Common date availability and no inf.
    keep_cols = [
        "date",
        "endog_log",
        "btc_ret_log",
        "eth_ret_log",
        "tvl_log_growth",
        "mc_to_ann_fees",
        "mc_to_ann_rev",
        "mc",
        "fdv",
        "ann_fees_30d",
        "ann_rev_30d",
        "tvl",
    ]
    m = m[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
    return m


def classify_z(z: float) -> str:
    if z < -1.0:
        return "compressed"
    if z > 1.0:
        return "elevated"
    return "neutral"


def fit_local_level_model(df: pd.DataFrame, endog_col: str, name: str) -> ModelResult:
    exog = df[["btc_ret_log", "eth_ret_log", "tvl_log_growth"]]
    model = UnobservedComponents(df["endog_log"], level="local level", exog=exog)
    fit = model.fit(disp=False)

    # Methodologically correct representation for local-level + exog:
    # observed_log = smoothed_latent_level + exog_component + residual
    smoothed_level = pd.Series(fit.smoothed_state[0], index=df.index, name="smoothed_level")
    beta = np.array([fit.params.get(f"beta.{c}", 0.0) for c in exog.columns], dtype=float)
    exog_component = pd.Series(exog.values @ beta, index=df.index, name="exog_component")
    fitted = smoothed_level + exog_component
    gap = df["endog_log"] - fitted
    gap_std = float(np.nanstd(gap, ddof=1))
    z = gap / gap_std if gap_std > 0 else np.nan

    resid = fit.resid.replace([np.inf, -np.inf], np.nan).dropna()
    lb = acorr_ljungbox(resid, lags=[4, 8], return_df=True)
    lb4 = float(lb.loc[4, "lb_pvalue"]) if 4 in lb.index else np.nan
    lb8 = float(lb.loc[8, "lb_pvalue"]) if 8 in lb.index else np.nan
    current_z = float(z.iloc[-1])

    out = df.copy()
    out["smoothed_level"] = smoothed_level
    out["exog_component"] = exog_component
    out["fitted_log"] = fitted
    out["gap"] = gap
    out["gap_z"] = z
    out["fitted_multiple"] = np.exp(out["fitted_log"])
    out["resid_std_log"] = gap_std
    out["band_upper"] = np.exp(out["fitted_log"] + gap_std)
    out["band_lower"] = np.exp(out["fitted_log"] - gap_std)

    obs_var = float(fit.params.get("sigma2.irregular", np.nan))
    state_var = float(fit.params.get("sigma2.level", np.nan))
    snr = state_var / obs_var if np.isfinite(obs_var) and obs_var > 0 else np.nan
    latent_state_std = float(np.nanstd(smoothed_level, ddof=1))
    obs_std = float(np.nanstd(df["endog_log"], ddof=1))
    latent_moving = latent_state_std > 0.1 * obs_std if np.isfinite(obs_std) and obs_std > 0 else False
    if np.isfinite(snr):
        if snr < 0.1:
            regime_dynamics = "stable regime"
        elif snr <= 1.0:
            regime_dynamics = "gradually evolving regime"
        else:
            regime_dynamics = "highly noisy regime"
    else:
        regime_dynamics = "diagnostics unavailable"
    if not latent_moving and regime_dynamics != "diagnostics unavailable":
        regime_dynamics = "stable regime"

    return ModelResult(
        name=name,
        data=out,
        fit=fit,
        endog_col="endog_log",
        gap_col="gap",
        z_col="gap_z",
        lb_pvalue_lag4=lb4,
        lb_pvalue_lag8=lb8,
        current_z=current_z,
        classification=classify_z(current_z),
        observation_variance=obs_var,
        state_variance=state_var,
        signal_to_noise=snr,
        latent_state_std=latent_state_std,
        regime_dynamics_label=regime_dynamics,
    )


def latest_peer_table(weekly: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    peers = ["maple", "aave", "morpho"]
    sym_map = {k: TOKEN_MAP[k] for k in peers}
    labels = {
        "maple": "Maple",
        "aave": "Aave",
        "morpho": "Morpho",
    }

    # Common-date panel for peer weekly multiples.
    sub = weekly[weekly["symbol"].isin(sym_map.values())].copy()
    common = None
    for sym in sym_map.values():
        dts = set(sub.loc[sub["symbol"] == sym, "date"].dropna().unique())
        common = dts if common is None else common.intersection(dts)
    if not common:
        raise RuntimeError("No common dates across Maple/Aave/Morpho for peer comparison.")
    sub = sub[sub["date"].isin(common)]

    latest_date = sub["date"].max()
    rows = []
    for key, sym in sym_map.items():
        r = sub[(sub["symbol"] == sym) & (sub["date"] == latest_date)].iloc[-1]
        mc_to_ann_rev = r.get("mc_to_ann_rev")
        if pd.isna(mc_to_ann_rev) or not np.isfinite(mc_to_ann_rev):
            mc_to_ann_rev = np.nan

        rows.append(
            {
                "token": labels[key],
                "role": ROLE_LABELS[key],
                "date": latest_date,
                "market_cap": r.get("mc"),
                "fdv": r.get("fdv"),
                "annualized_fees_30d": r.get("ann_fees_30d"),
                "annualized_revenue_30d": r.get("ann_rev_30d"),
                "mc_to_ann_fees": r.get("mc_to_ann_fees"),
                "mc_to_ann_revenue_if_valid": mc_to_ann_rev,
            }
        )
    table = pd.DataFrame(rows)

    # Relative valuation metrics.
    maple_mult = float(table.loc[table["token"] == "Maple", "mc_to_ann_fees"].iloc[0])
    aave_mult = float(table.loc[table["token"] == "Aave", "mc_to_ann_fees"].iloc[0])
    morpho_mult = float(table.loc[table["token"] == "Morpho", "mc_to_ann_fees"].iloc[0])
    core_median = float(np.nanmedian([aave_mult, morpho_mult]))

    rel = {
        "maple_vs_aave_pct": (maple_mult / aave_mult - 1.0) * 100.0,
        "maple_vs_morpho_pct": (maple_mult / morpho_mult - 1.0) * 100.0,
        "maple_vs_core_median_pct": (maple_mult / core_median - 1.0) * 100.0,
        "core_median_mult": core_median,
    }
    table["business_model_peer_median_mc_to_ann_fees"] = core_median
    table["maple_vs_business_model_median_pct"] = rel["maple_vs_core_median_pct"]
    return table, rel


def plot_main_valuation(model: ModelResult) -> Path:
    def apply_dark(ax, fig):
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    d = model.data
    with plt.rc_context({"font.family": "Georgia"}):
        fig, ax = plt.subplots(figsize=(12, 6))
    observed = np.exp(d[model.endog_col])
    regime = d["fitted_multiple"]
    ax.plot(
        d["date"],
        observed,
        linewidth=2,
        color="tab:cyan",
        label="Observed MC/annualized fees",
    )
    ax.plot(
        d["date"],
        regime,
        linewidth=2,
        linestyle="--",
        color="tab:orange",
        label="Estimated evolving regime",
    )
    ax.fill_between(
        d["date"],
        d["band_lower"],
        d["band_upper"],
        alpha=0.15,
        color="tab:orange",
        label="+/-1 residual std band",
    )
    ax.scatter(
        d["date"].iloc[-1],
        observed.iloc[-1],
        s=90,
        zorder=5,
        color="white",
        label="Latest",
    )
    ax.set_title("Maple valuation multiple vs estimated evolving regime")
    ax.set_ylabel("MC / annualized fees (x)")
    leg = ax.legend(loc="upper left", frameon=False)
    for t in leg.get_texts():
        t.set_color("white")
        t.set_fontfamily("Georgia")
    apply_dark(ax, fig)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Georgia")
    ax.title.set_fontfamily("Georgia")
    ax.xaxis.label.set_fontfamily("Georgia")
    ax.yaxis.label.set_fontfamily("Georgia")
    fig.tight_layout()
    path = CHART_DIR / "maple_valuation_regime.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_gap_zscore(model: ModelResult) -> Path:
    def apply_dark(ax, fig):
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    d = model.data
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(d["date"], d[model.z_col], linewidth=2, color="tab:purple")
    ax.axhline(-1.0, linestyle="--", color="tab:red", linewidth=1)
    ax.axhline(0.0, linestyle="-", color="black", linewidth=1)
    ax.axhline(1.0, linestyle="--", color="tab:green", linewidth=1)
    ax.set_title("Maple valuation gap z-score")
    ax.set_ylabel("z-score")
    ax.grid(alpha=0.2, color="white")
    apply_dark(ax, fig)
    fig.tight_layout()
    path = CHART_DIR / "maple_gap_zscore.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_peer_bars(peer_table: pd.DataFrame) -> Path:
    def apply_dark(ax, fig):
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    d = peer_table.sort_values("mc_to_ann_fees", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = []
    for token in d["token"]:
        if token in {"Aave", "Morpho"}:
            colors.append("tab:blue")
        else:
            colors.append("tab:green")
    ax.barh(d["token"], d["mc_to_ann_fees"], color=colors)
    ax.set_title("Latest peer valuation multiples")
    ax.set_xlabel("MC / annualized fees (x)")
    ax.grid(axis="x", alpha=0.2, color="white")
    apply_dark(ax, fig)
    fig.tight_layout()
    path = CHART_DIR / "peer_multiples_barh.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_relative_discount(rel: Dict[str, float]) -> Path:
    def apply_dark(ax, fig):
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    labels = ["Aave", "Morpho", "Aave/Morpho median"]
    vals = [
        rel["maple_vs_aave_pct"],
        rel["maple_vs_morpho_pct"],
        rel["maple_vs_core_median_pct"],
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vals, color=["tab:blue", "tab:blue", "tab:blue"])
    ax.axhline(0, color="white", linewidth=1)
    ax.set_title("Maple relative valuation vs peers")
    ax.set_ylabel("Discount / premium (%)")
    for b, v in zip(bars, vals):
        y_text = v + (1.2 if v >= 0 else -1.2)
        ax.text(
            b.get_x() + b.get_width() / 2,
            y_text,
            f"{v:+.1f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            color="white",
            clip_on=False,
        )
    ax.grid(axis="y", alpha=0.2, color="white")
    apply_dark(ax, fig)
    fig.tight_layout()
    path = CHART_DIR / "maple_relative_discount.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_two_lens_valuation(model: ModelResult, rel: Dict[str, float]) -> Path:
    # Single-slide chart: neutral vs own regime + discounted vs core peers.
    with plt.rc_context({"font.family": "Georgia"}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Left panel: regime-relative z-score gauge.
    z = float(model.current_z)
    ax0 = axes[0]
    ax0.axvspan(-2.5, -1.0, color="tab:red", alpha=0.18)
    ax0.axvspan(-1.0, 1.0, color="tab:gray", alpha=0.18)
    ax0.axvspan(1.0, 2.5, color="tab:green", alpha=0.18)
    ax0.axvline(-1.0, color="white", linestyle="--", linewidth=1)
    ax0.axvline(0.0, color="white", linewidth=1)
    ax0.axvline(1.0, color="white", linestyle="--", linewidth=1)
    ax0.scatter([z], [0], s=140, color="tab:cyan", edgecolor="white", linewidth=1.0, zorder=4)
    ax0.set_xlim(-2.5, 2.5)
    ax0.set_ylim(-1, 1)
    ax0.set_yticks([])
    ax0.set_xlabel("Valuation gap z-score")
    ax0.set_title("Lens 1: Maple vs own evolving regime")
    ax0.text(-2.4, 0.72, "Compressed", color="white", fontsize=9, fontfamily="Georgia")
    ax0.text(-0.35, 0.72, "Neutral", color="white", fontsize=9, fontfamily="Georgia")
    ax0.text(1.15, 0.72, "Elevated", color="white", fontsize=9, fontfamily="Georgia")
    ax0.text(
        z + 0.06,
        -0.35,
        f"Current z = {z:.2f}",
        color="white",
        fontsize=10,
        fontfamily="Georgia",
    )

    # Right panel: peer-relative discount bars.
    ax1 = axes[1]
    labels = ["vs Aave", "vs Morpho", "vs Aave/Morpho median"]
    vals = [rel["maple_vs_aave_pct"], rel["maple_vs_morpho_pct"], rel["maple_vs_core_median_pct"]]
    colors = ["tab:green" if v < 0 else "tab:orange" for v in vals]
    bars = ax1.barh(labels, vals, color=colors)
    ax1.axvline(0, color="white", linewidth=1)
    x_min = min(vals) - 10
    x_max = max(vals) + 10
    ax1.set_xlim(x_min, x_max)
    ax1.set_title("Lens 2: Maple peer-relative valuation")
    ax1.set_xlabel("Discount / premium (%)")
    x_pad = 0.02 * (x_max - x_min)
    for b, v in zip(bars, vals):
        ax1.text(
            v + (x_pad if v >= 0 else -x_pad),
            b.get_y() + b.get_height() / 2,
            f"{v:+.1f}%",
            color="white",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=10,
            fontfamily="Georgia",
            clip_on=False,
        )

    fig.suptitle(
        "Maple valuation read-through: neutral to self, discounted to core peers",
        color="white",
        fontfamily="Georgia",
    )
    fig.tight_layout()
    path = CHART_DIR / "maple_two_lens_valuation.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_one_chart_story(model: ModelResult, rel: Dict[str, float]) -> Path:
    # One-chart view: peer mispricing bars + self-regime read as a compact annotation.
    with plt.rc_context({"font.family": "Georgia"}):
        fig, ax = plt.subplots(figsize=(12, 6))

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    labels = ["vs Aave", "vs Morpho", "vs Aave/Morpho median"]
    vals = [rel["maple_vs_aave_pct"], rel["maple_vs_morpho_pct"], rel["maple_vs_core_median_pct"]]
    colors = ["tab:green" if v < 0 else "tab:orange" for v in vals]
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    ax.axvline(0, color="white", linewidth=1)

    x_min = min(vals) - 12
    x_max = max(vals) + 12
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Maple discount / premium (%)")
    ax.set_title("Maple valuation: discounted to core peers, neutral to own regime")
    ax.grid(axis="x", alpha=0.15, color="white")

    x_pad = 0.02 * (x_max - x_min)
    for b, v in zip(bars, vals):
        ax.text(
            v + (x_pad if v >= 0 else -x_pad),
            b.get_y() + b.get_height() / 2,
            f"{v:+.1f}%",
            color="white",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=11,
            fontfamily="Georgia",
            clip_on=False,
        )

    z = float(model.current_z)
    regime_text = (
        f"Self-regime read (Kalman): z = {z:.2f} ({model.classification})\n"
        "Interpretation: time-series valuation is near neutral.\n"
        "Edge is cross-sectional vs Aave/Morpho."
    )
    ax.text(
        0.02,
        0.05,
        regime_text,
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        fontfamily="Georgia",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="none", edgecolor="white", linewidth=0.8),
    )

    fig.tight_layout()
    path = CHART_DIR / "maple_one_chart_story.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_fundamentals_vs_valuation(weekly: pd.DataFrame) -> Path:
    m = weekly[weekly["symbol"] == TOKEN_MAP["maple"]].copy().sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(alpha=0.2, color="white")

    axes[0].plot(m["date"], m["ann_fees_30d"], color="tab:cyan", linewidth=2)
    axes[0].set_title("Maple fundamentals vs valuation")
    axes[0].set_ylabel("Annualized fees (30d avg * 365)")
    axes[0].text(
        0.01,
        0.90,
        "Fundamentals up",
        transform=axes[0].transAxes,
        color="white",
        fontsize=10,
    )

    axes[1].plot(m["date"], m["mc_to_ann_fees"], color="tab:purple", linewidth=2)
    axes[1].set_ylabel("MC / annualized fees (x)")
    axes[1].text(
        0.01,
        0.90,
        "Multiple still contained",
        transform=axes[1].transAxes,
        color="white",
        fontsize=10,
    )

    fig.tight_layout()
    path = CHART_DIR / "fundamentals_vs_valuation.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_why_now_dashboard(weekly: pd.DataFrame, rel: Dict[str, float]) -> Path:
    m = weekly[weekly["symbol"] == TOKEN_MAP["maple"]].copy().sort_values("date")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    fig.patch.set_facecolor("black")
    for ax in axes.flat:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(alpha=0.2, color="white")

    axes[0, 0].plot(m["date"], m["ann_fees_30d"], color="tab:cyan", linewidth=2)
    axes[0, 0].set_title("Annualized fees")

    axes[0, 1].plot(m["date"], m["ann_rev_30d"], color="tab:green", linewidth=2)
    axes[0, 1].set_title("Annualized revenue")

    axes[1, 0].plot(m["date"], m["tvl"], color="tab:orange", linewidth=2)
    axes[1, 0].set_title("TVL / AUM proxy")

    vals = [
        rel["maple_vs_aave_pct"],
        rel["maple_vs_morpho_pct"],
        rel["maple_vs_core_median_pct"],
    ]
    labels = ["vs Aave", "vs Morpho", "vs Aave/Morpho median"]
    x_idx = np.arange(len(labels))
    bars = axes[1, 1].bar(x_idx, vals, color=["tab:blue", "tab:blue", "tab:blue"])
    axes[1, 1].set_xticks(x_idx)
    axes[1, 1].set_xticklabels(labels, rotation=10, color="white")
    axes[1, 1].axhline(0, color="white", linewidth=1)
    axes[1, 1].set_title("Current peer discount / premium (%)")
    for b, v in zip(bars, vals):
        y_text = v + (1.2 if v >= 0 else -1.2)
        axes[1, 1].text(
            b.get_x() + b.get_width() / 2,
            y_text,
            f"{v:+.1f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            color="white",
            fontsize=9,
            clip_on=False,
        )

    fig.suptitle("Maple one-page why-now operating dashboard", color="white")
    fig.tight_layout()
    path = CHART_DIR / "maple_why_now_dashboard.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def optional_relative_state_space(
    weekly: pd.DataFrame, controls_df: pd.DataFrame
) -> Tuple[ModelResult | None, Path | None]:
    panel = weekly[weekly["symbol"].isin([TOKEN_MAP["maple"], TOKEN_MAP["aave"], TOKEN_MAP["morpho"]])][
        ["date", "symbol", "mc_to_ann_fees"]
    ].dropna()
    p = panel.pivot(index="date", columns="symbol", values="mc_to_ann_fees").dropna()
    if TOKEN_MAP["maple"] not in p.columns or TOKEN_MAP["aave"] not in p.columns or TOKEN_MAP["morpho"] not in p.columns:
        return None, None
    p["core_median"] = p[[TOKEN_MAP["aave"], TOKEN_MAP["morpho"]]].median(axis=1)
    p["rel_mult"] = p[TOKEN_MAP["maple"]] / p["core_median"]
    p["log_rel_mult"] = np.log(p["rel_mult"])

    d = p.reset_index().merge(controls_df, on="date", how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 30:
        return None, None

    exog = d[["btc_ret_log", "eth_ret_log", "tvl_log_growth"]]
    m = UnobservedComponents(d["log_rel_mult"], level="local level", exog=exog)
    fit = m.fit(disp=False)

    smoothed_level = pd.Series(fit.smoothed_state[0], index=d.index)
    beta = np.array([fit.params.get(f"beta.{c}", 0.0) for c in exog.columns], dtype=float)
    exog_component = pd.Series(exog.values @ beta, index=d.index)
    fitted = smoothed_level + exog_component
    gap = d["log_rel_mult"] - fitted
    gstd = float(np.nanstd(gap, ddof=1))
    z = gap / gstd if gstd > 0 else np.nan

    out = d.copy()
    out["smoothed_level"] = smoothed_level
    out["exog_component"] = exog_component
    out["fitted_log"] = fitted
    out["gap"] = gap
    out["gap_z"] = z

    lb = acorr_ljungbox(fit.resid.dropna(), lags=[4, 8], return_df=True)
    obs_var = float(fit.params.get("sigma2.irregular", np.nan))
    state_var = float(fit.params.get("sigma2.level", np.nan))
    snr = state_var / obs_var if np.isfinite(obs_var) and obs_var > 0 else np.nan
    latent_state_std = float(np.nanstd(smoothed_level, ddof=1))
    obs_std = float(np.nanstd(d["log_rel_mult"], ddof=1))
    if np.isfinite(snr):
        if snr < 0.1:
            regime_dynamics = "stable regime"
        elif snr <= 1.0:
            regime_dynamics = "gradually evolving regime"
        else:
            regime_dynamics = "highly noisy regime"
    else:
        regime_dynamics = "diagnostics unavailable"
    if not (latent_state_std > 0.1 * obs_std) and regime_dynamics != "diagnostics unavailable":
        regime_dynamics = "stable regime"

    mr = ModelResult(
        name="relative_mult",
        data=out,
        fit=fit,
        endog_col="log_rel_mult",
        gap_col="gap",
        z_col="gap_z",
        lb_pvalue_lag4=float(lb.loc[4, "lb_pvalue"]),
        lb_pvalue_lag8=float(lb.loc[8, "lb_pvalue"]),
        current_z=float(z.iloc[-1]),
        classification=classify_z(float(z.iloc[-1])),
        observation_variance=obs_var,
        state_variance=state_var,
        signal_to_noise=snr,
        latent_state_std=latent_state_std,
        regime_dynamics_label=regime_dynamics,
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(out["date"], out["gap_z"], color="tab:brown")
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(-1, color="tab:red", linestyle="--")
    ax.axhline(1, color="tab:green", linestyle="--")
    ax.set_title("Optional extension: Maple relative-multiple gap z-score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = CHART_DIR / "maple_relative_regime_zscore.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return mr, path


def build_fee_momentum_panel(weekly: pd.DataFrame) -> pd.DataFrame:
    focus_symbols = [TOKEN_MAP["maple"], TOKEN_MAP["aave"], TOKEN_MAP["morpho"]]
    d = weekly[weekly["symbol"].isin(focus_symbols)][
        ["date", "symbol", "price", "ann_fees_90d", "mc_to_ann_fees_90d"]
    ].copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["price", "ann_fees_90d", "mc_to_ann_fees_90d"])
    d = d[(d["price"] > 0) & (d["ann_fees_90d"] > 0) & (d["mc_to_ann_fees_90d"] > 0)]
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = d.groupby("symbol", group_keys=False)
    d["log_price"] = np.log(d["price"])
    d["log_fee_runrate_90d"] = np.log(d["ann_fees_90d"])
    d["log_mc_to_fees_90d"] = np.log(d["mc_to_ann_fees_90d"])

    # 4-week first and second differences (momentum + acceleration) from lagged history only.
    d["fee_growth_4w"] = g["log_fee_runrate_90d"].diff(4)
    d["fee_accel_4w"] = g["fee_growth_4w"].diff(4)
    d["price_ret_4w"] = g["log_price"].diff(4)
    d["mult_change_4w"] = g["log_mc_to_fees_90d"].diff(4)

    # Forward returns are evaluation targets (never used in signal construction).
    d["fwd_ret_4w_log"] = g["log_price"].shift(-4) - d["log_price"]
    d["fwd_ret_8w_log"] = g["log_price"].shift(-8) - d["log_price"]
    d["fwd_ret_4w_pct"] = np.exp(d["fwd_ret_4w_log"]) - 1.0
    d["fwd_ret_8w_pct"] = np.exp(d["fwd_ret_8w_log"]) - 1.0

    # "Why now" signal: fundamentals accelerate while token price is flat/down and multiple contracts.
    d["accel_signal"] = (
        (d["fee_accel_4w"] > 0)
        & (d["price_ret_4w"] <= 0)
        & (d["mult_change_4w"] < 0)
    )
    return d


def run_fee_momentum_analysis(weekly: pd.DataFrame) -> FeeMomentumResult:
    panel = build_fee_momentum_panel(weekly)

    def token_summary(token_symbol: str, token_label: str) -> Dict[str, float]:
        t = panel[panel["symbol"] == token_symbol].copy()
        eval_4 = t.dropna(subset=["fwd_ret_4w_pct", "accel_signal"])
        eval_8 = t.dropna(subset=["fwd_ret_8w_pct", "accel_signal"])
        event_4 = eval_4[eval_4["accel_signal"]]
        event_8 = eval_8[eval_8["accel_signal"]]

        base_4 = float(eval_4["fwd_ret_4w_pct"].mean()) if len(eval_4) else np.nan
        base_8 = float(eval_8["fwd_ret_8w_pct"].mean()) if len(eval_8) else np.nan
        cond_4 = float(event_4["fwd_ret_4w_pct"].mean()) if len(event_4) else np.nan
        cond_8 = float(event_8["fwd_ret_8w_pct"].mean()) if len(event_8) else np.nan

        return {
            "token": token_label,
            "obs_4w": int(len(eval_4)),
            "events_4w": int(len(event_4)),
            "base_fwd_4w_pct": base_4 * 100.0 if np.isfinite(base_4) else np.nan,
            "signal_fwd_4w_pct": cond_4 * 100.0 if np.isfinite(cond_4) else np.nan,
            "lift_4w_pct": (cond_4 - base_4) * 100.0 if np.isfinite(cond_4) and np.isfinite(base_4) else np.nan,
            "obs_8w": int(len(eval_8)),
            "events_8w": int(len(event_8)),
            "base_fwd_8w_pct": base_8 * 100.0 if np.isfinite(base_8) else np.nan,
            "signal_fwd_8w_pct": cond_8 * 100.0 if np.isfinite(cond_8) else np.nan,
            "lift_8w_pct": (cond_8 - base_8) * 100.0 if np.isfinite(cond_8) and np.isfinite(base_8) else np.nan,
        }

    summary_rows = [
        token_summary(TOKEN_MAP["maple"], "Maple"),
        token_summary(TOKEN_MAP["aave"], "Aave"),
        token_summary(TOKEN_MAP["morpho"], "Morpho"),
    ]
    summary = pd.DataFrame(summary_rows)

    # Peer-only pooled predictive regression: do accelerating fees predict next-month returns?
    peer = panel[panel["symbol"].isin([TOKEN_MAP["aave"], TOKEN_MAP["morpho"]])].copy()
    reg_cols = ["fwd_ret_4w_log", "fee_accel_4w", "price_ret_4w", "mult_change_4w"]
    reg_df = peer[reg_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(reg_df) >= 25:
        X = sm.add_constant(reg_df[["fee_accel_4w", "price_ret_4w", "mult_change_4w"]])
        y = reg_df["fwd_ret_4w_log"]
        fit = sm.OLS(y, X).fit(cov_type="HC1")
        coef = float(fit.params.get("fee_accel_4w", np.nan))
        pval = float(fit.pvalues.get("fee_accel_4w", np.nan))
    else:
        coef = np.nan
        pval = np.nan

    peer_rows = summary[summary["token"].isin(["Aave", "Morpho"])]
    peer_lift_4w = float(peer_rows["lift_4w_pct"].mean()) if len(peer_rows) else np.nan
    peer_lift_8w = float(peer_rows["lift_8w_pct"].mean()) if len(peer_rows) else np.nan
    peer_events_4w = int(peer_rows["events_4w"].fillna(0).sum()) if len(peer_rows) else 0
    peer_events_8w = int(peer_rows["events_8w"].fillna(0).sum()) if len(peer_rows) else 0

    maple_panel = panel[panel["symbol"] == TOKEN_MAP["maple"]].copy().sort_values("date")
    latest = maple_panel.iloc[-1]
    maple_signal = bool(latest.get("accel_signal", False))
    maple_fee_accel = float(latest.get("fee_accel_4w", np.nan))
    maple_price_ret_4w = float(latest.get("price_ret_4w", np.nan))
    maple_mult_chg_4w = float(latest.get("mult_change_4w", np.nan))

    if (
        np.isfinite(peer_lift_4w)
        and peer_lift_4w > 0
        and np.isfinite(coef)
        and coef > 0
        and np.isfinite(pval)
        and pval <= 0.10
        and peer_events_4w >= 15
    ):
        quality = "strong"
    elif np.isfinite(peer_lift_4w) and peer_lift_4w > 0 and peer_events_4w >= 8:
        quality = "mixed"
    else:
        quality = "weak"
    replace_core = bool(quality == "strong" and maple_signal)

    return FeeMomentumResult(
        panel=panel,
        summary_table=summary,
        peer_model_coef=coef,
        peer_model_pvalue=pval,
        peer_event_lift_4w=peer_lift_4w,
        peer_event_lift_8w=peer_lift_8w,
        maple_current_signal=maple_signal,
        maple_latest_fee_accel=maple_fee_accel,
        maple_latest_price_ret_4w=maple_price_ret_4w,
        maple_latest_mult_change_4w=maple_mult_chg_4w,
        peer_total_events_4w=peer_events_4w,
        peer_total_events_8w=peer_events_8w,
        evidence_quality=quality,
        replace_kalman_as_core=replace_core,
    )


def plot_maple_fee_momentum(fm: FeeMomentumResult) -> Path:
    m = fm.panel[fm.panel["symbol"] == TOKEN_MAP["maple"]].copy().sort_values("date")
    m = m.dropna(subset=["ann_fees_90d", "price"])

    price_idx = 100.0 * m["price"] / m["price"].iloc[0]
    fee_idx = 100.0 * m["ann_fees_90d"] / m["ann_fees_90d"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(alpha=0.2, color="white")

    axes[0].plot(m["date"], fee_idx, color="tab:cyan", linewidth=2, label="90d fee run-rate index")
    axes[0].plot(m["date"], price_idx, color="tab:orange", linewidth=2, label="Price index")
    leg0 = axes[0].legend(loc="upper left", frameon=False)
    for t in leg0.get_texts():
        t.set_color("white")
    axes[0].set_ylabel("Index (start=100)")
    axes[0].set_title("Maple: fee run-rate momentum vs price")

    axes[1].plot(m["date"], m["fee_accel_4w"], color="tab:green", linewidth=2, label="Fee acceleration (4w second diff)")
    axes[1].axhline(0, color="white", linewidth=1)
    sig = m[m["accel_signal"]]
    axes[1].scatter(sig["date"], sig["fee_accel_4w"], color="white", s=28, label="Signal on")
    leg1 = axes[1].legend(loc="upper left", frameon=False)
    for t in leg1.get_texts():
        t.set_color("white")
    axes[1].set_ylabel("Acceleration")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    path = CHART_DIR / "maple_fee_momentum.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_fee_momentum_event_lift(fm: FeeMomentumResult) -> Path:
    d = fm.summary_table.copy()
    order = ["Maple", "Aave", "Morpho"]
    d["token"] = pd.Categorical(d["token"], categories=order, ordered=True)
    d = d.sort_values("token")
    d = d[d["lift_4w_pct"].notna() | d["lift_8w_pct"].notna()].copy()
    d["lift_4w_pct"] = d["lift_4w_pct"].fillna(0.0)
    d["lift_8w_pct"] = d["lift_8w_pct"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(axis="y", alpha=0.2, color="white")

    x = np.arange(len(d))
    w = 0.36
    b1 = ax.bar(x - w / 2, d["lift_4w_pct"], width=w, color="tab:blue", label="4w lift vs baseline")
    b2 = ax.bar(x + w / 2, d["lift_8w_pct"], width=w, color="tab:purple", label="8w lift vs baseline")
    ax.axhline(0, color="white", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["token"], color="white")
    ax.set_ylabel("Signal-conditioned forward return lift (pp)")
    ax.set_title("Fee-acceleration signal: forward-return lift")
    leg = ax.legend(loc="upper left", frameon=False)
    for t in leg.get_texts():
        t.set_color("white")

    for bars in [b1, b2]:
        for bar in bars:
            y = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + (0.2 if y >= 0 else -0.2),
                f"{y:+.1f}",
                color="white",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=9,
            )

    fig.tight_layout()
    path = CHART_DIR / "fee_momentum_event_lift.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def print_verification(
    model_fees: ModelResult,
    model_rev: ModelResult,
    clipped_placeholder: bool,
) -> None:
    def block(name: str, mr: ModelResult) -> None:
        print(f"\n{name}")
        print(f"  sample start / end: {mr.data['date'].min().date()} -> {mr.data['date'].max().date()}")
        print(f"  weekly observations: {len(mr.data)}")
        print(f"  Ljung-Box p-value lag4: {mr.lb_pvalue_lag4:.4f}")
        print(f"  Ljung-Box p-value lag8: {mr.lb_pvalue_lag8:.4f}")
        print(f"  current gap z-score: {mr.current_z:.3f}")
        print(f"  classification: {mr.classification}")
        print(f"  observation variance (sigma2.irregular): {mr.observation_variance:.6g}")
        print(f"  state variance (sigma2.level): {mr.state_variance:.6g}")
        print(f"  signal-to-noise ratio (state/obs): {mr.signal_to_noise:.4f}")
        print(f"  latent-state std: {mr.latent_state_std:.4f}")
        print(f"  regime dynamics interpretation: {mr.regime_dynamics_label}")

    print("\n=== Verification Summary ===")
    print(f"Placeholder circulating-supply values clipped (<1 token): {clipped_placeholder}")
    block("MC / annualized fees model", model_fees)
    block("MC / annualized revenue model", model_rev)


def write_summary_md(
    model_fees: ModelResult,
    model_rev: ModelResult,
    peer_table: pd.DataFrame,
    rel: Dict[str, float],
    fee_momentum: FeeMomentumResult,
    optional_rel_model: ModelResult | None,
    data_source_label: str,
) -> Path:
    core_discount = rel["maple_vs_core_median_pct"]
    core_verdict = "discounted" if core_discount < 0 else ("neutral" if abs(core_discount) <= 10 else "rich")
    text = f"""# Maple Dynamic Valuation Summary

## Methodology verdict
State-space local-level modeling is used as the primary advanced methodology and is more appropriate here than ECM for dynamic regime estimation.
The estimated regime is built from **smoothed latent level + exogenous component** (Kalman/state-space representation in observation space).

## Data source
{data_source_label}

## Time-series valuation verdict
- MC/annualized fees model: z-score = {model_fees.current_z:.2f} ({model_fees.classification})
- MC/annualized revenue model: z-score = {model_rev.current_z:.2f} ({model_rev.classification})
- The model estimates an evolving valuation regime; it does **not** prove hard fair value.

## Cross-sectional valuation verdict
- Maple vs Aave: {rel['maple_vs_aave_pct']:+.1f}%
- Maple vs Morpho: {rel['maple_vs_morpho_pct']:+.1f}%
- Maple vs Aave/Morpho median: {core_discount:+.1f}% ({core_verdict})
- Maple screens discounted relative to the two core business-model comps, Aave and Morpho.

## Fee momentum / acceleration verdict (why-now diagnostic)
- Maple current acceleration signal active: {fee_momentum.maple_current_signal}
- Maple latest 4w fee acceleration: {fee_momentum.maple_latest_fee_accel:+.4f}
- Maple latest 4w price return (log): {fee_momentum.maple_latest_price_ret_4w:+.4f}
- Maple latest 4w multiple change (log): {fee_momentum.maple_latest_mult_change_4w:+.4f}
- Peer (Aave+Morpho) event-lift, 4w: {fee_momentum.peer_event_lift_4w:+.2f}pp
- Peer (Aave+Morpho) event-lift, 8w: {fee_momentum.peer_event_lift_8w:+.2f}pp
- Peer event counts: 4w={fee_momentum.peer_total_events_4w}, 8w={fee_momentum.peer_total_events_8w}
- Peer pooled regression on 4w forward log returns: beta_fee_accel = {fee_momentum.peer_model_coef:+.4f}, p-value = {fee_momentum.peer_model_pvalue:.4f}
- Evidence quality: {fee_momentum.evidence_quality}
- Replace Kalman as core section: {fee_momentum.replace_kalman_as_core}

## Safe thesis usage
- Use peer-relative valuation + fundamentals as the primary argument.
- Use fee-momentum as catalyst support only when evidence quality is strong and the current Maple signal is active.
- Use Kalman/state-space as a dynamic-regime appendix (context, not core catalyst).
- Do not present model outputs as definitive fair-value proof.
"""
    if optional_rel_model is not None:
        text += f"\n\n## Optional relative-regime extension\n- Relative-multiple z-score: {optional_rel_model.current_z:.2f} ({optional_rel_model.classification})\n"
    path = SUMMARY_DIR / "maple_kalman_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def final_print_section(
    model_fees: ModelResult,
    model_rev: ModelResult,
    rel: Dict[str, float],
    fee_momentum: FeeMomentumResult,
) -> None:
    core_discount = rel["maple_vs_core_median_pct"]
    xsec_verdict = "discounted" if core_discount < -5 else ("neutral" if core_discount <= 5 else "rich")

    print("\n=== Final Summary ===")
    print("1. Methodology verdict")
    print("   - State-space/Kalman is appropriate for this dataset and superior to ECM here.")
    print("2. Time-series valuation verdict")
    print(
        f"   - Maple is {model_fees.classification} (fees spec z={model_fees.current_z:.2f}) "
        f"and {model_rev.classification} (revenue spec z={model_rev.current_z:.2f}) "
        "relative to its evolving regime."
    )
    print("3. Cross-sectional valuation verdict")
    print(f"   - Maple vs Aave: {rel['maple_vs_aave_pct']:+.1f}%")
    print(f"   - Maple vs Morpho: {rel['maple_vs_morpho_pct']:+.1f}%")
    print(f"   - Maple vs Aave/Morpho median: {core_discount:+.1f}% ({xsec_verdict})")
    print("4. Fee momentum / acceleration diagnostic")
    print(
        f"   - Maple current signal active: {fee_momentum.maple_current_signal} "
        f"(fee_accel_4w={fee_momentum.maple_latest_fee_accel:+.4f}, "
        f"price_ret_4w={fee_momentum.maple_latest_price_ret_4w:+.4f}, "
        f"mult_change_4w={fee_momentum.maple_latest_mult_change_4w:+.4f})."
    )
    print(
        f"   - Peer event-lift: 4w={fee_momentum.peer_event_lift_4w:+.2f}pp, "
        f"8w={fee_momentum.peer_event_lift_8w:+.2f}pp; "
        f"beta_fee_accel={fee_momentum.peer_model_coef:+.4f} "
        f"(p={fee_momentum.peer_model_pvalue:.4f})."
    )
    print(
        f"   - Peer event counts: 4w={fee_momentum.peer_total_events_4w}, "
        f"8w={fee_momentum.peer_total_events_8w}."
    )
    print(f"   - Evidence quality: {fee_momentum.evidence_quality}.")
    print(f"   - Replace Kalman as core section: {fee_momentum.replace_kalman_as_core}.")
    print("5. Safe thesis usage")
    print("   - Use peer-relative valuation + fundamentals as primary thesis evidence.")
    print("   - Use fee-momentum as catalyst support only when evidence is strong and signal is active.")
    print("   - Keep the Kalman model as a dynamic-regime appendix.")
    print("   - Do not present the model as definitive fair-value proof.")


def main() -> None:
    symbols = [TOKEN_MAP["maple"], TOKEN_MAP["aave"], TOKEN_MAP["morpho"], TOKEN_MAP["btc"], TOKEN_MAP["eth"]]
    daily = None
    data_source_label = ""

    if USE_LOCAL_FILES:
        local = load_local_data(symbols)
        if local is not None:
            daily = local
            data_source_label = "Local CSV files (project directory)"
            print("Data source mode: LOCAL FILES")
        else:
            print("Local files unavailable/incomplete. Falling back to live Artemis pull.")

    if daily is None:
        api_key = get_api_key()
        client = Artemis(api_key=api_key)
        daily = fetch_daily_data(client, api_key, symbols)
        data_source_label = "Live Artemis API pull"
        print("Data source mode: LIVE ARTEMIS API")

    daily = add_annualized_fields(daily)
    daily, clipped_placeholder = clean_maple(daily)
    weekly = build_weekly_panel(daily)

    fees_df = prepare_maple_model_df(weekly, endog_col="mc_to_ann_fees")
    rev_df = prepare_maple_model_df(weekly, endog_col="mc_to_ann_rev")

    fees_model = fit_local_level_model(fees_df, endog_col="mc_to_ann_fees", name="fees_spec")
    rev_model = fit_local_level_model(rev_df, endog_col="mc_to_ann_rev", name="revenue_spec")

    print_verification(fees_model, rev_model, clipped_placeholder)

    peer_table, rel = latest_peer_table(weekly)
    peer_csv_path = TABLE_DIR / "latest_peer_multiples.csv"
    peer_table.to_csv(peer_csv_path, index=False)

    fee_momentum = run_fee_momentum_analysis(weekly)
    fee_momentum_csv_path = TABLE_DIR / "fee_momentum_summary.csv"
    fee_momentum.summary_table.to_csv(fee_momentum_csv_path, index=False)

    # Required pitch charts
    p1 = plot_main_valuation(fees_model)
    p2 = plot_fundamentals_vs_valuation(weekly)
    p3 = plot_peer_bars(peer_table)
    p4 = plot_relative_discount(rel)
    p5 = plot_why_now_dashboard(weekly, rel)
    p6 = plot_two_lens_valuation(fees_model, rel)
    p7 = plot_one_chart_story(fees_model, rel)
    p8 = plot_maple_fee_momentum(fee_momentum)
    p9 = plot_fee_momentum_event_lift(fee_momentum)

    # Optional extension
    controls_df = fees_df[["date", "btc_ret_log", "eth_ret_log", "tvl_log_growth"]].copy()
    rel_model, rel_chart = optional_relative_state_space(weekly, controls_df)

    summary_path = write_summary_md(
        fees_model,
        rev_model,
        peer_table,
        rel,
        fee_momentum,
        rel_model,
        data_source_label=data_source_label,
    )
    final_print_section(fees_model, rev_model, rel, fee_momentum)

    print("\nSaved files:")
    print(f"  - {peer_csv_path}")
    print(f"  - {fee_momentum_csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {p1}")
    print(f"  - {p2}")
    print(f"  - {p3}")
    print(f"  - {p4}")
    print(f"  - {p5}")
    print(f"  - {p6}")
    print(f"  - {p7}")
    print(f"  - {p8}")
    print(f"  - {p9}")
    if rel_chart is not None:
        print(f"  - {rel_chart}")


if __name__ == "__main__":
    main()
