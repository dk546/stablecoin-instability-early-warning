"""Prediction-time feature construction from prepared hourly observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stablecoin_ews.contracts import (
    CONTEXT_ASSETS,
    M0_FEATURES,
    M1_FEATURES,
    STUDY_ASSETS,
)


class FeatureError(ValueError):
    """Raised when required history or temporal ordering is ambiguous."""


MARKET_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "timestamp_utc",
    "price_observed",
    "price_effective",
    "market_cap_effective",
    "volume_24h_usd_effective",
)
MACRO_SOURCE_FEATURES: tuple[str, ...] = (
    "vix_close",
    "us_3m_yield",
    "us_10y_yield",
    "effective_federal_funds_rate",
)


def _normalize_market(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(MARKET_COLUMNS).difference(frame.columns)
    if missing:
        raise FeatureError(f"market data are missing columns: {sorted(missing)}")
    output = frame.loc[:, MARKET_COLUMNS].copy()
    output["timestamp_utc"] = pd.to_datetime(
        output["timestamp_utc"], utc=True, errors="raise"
    )
    if (output["timestamp_utc"].dt.floor("h") != output["timestamp_utc"]).any():
        raise FeatureError("market timestamps must be exact UTC hours")
    if output.duplicated(["asset_id", "timestamp_utc"]).any():
        raise FeatureError("market data contain duplicate asset-hours")
    expected_assets = set(STUDY_ASSETS + CONTEXT_ASSETS)
    missing_assets = expected_assets.difference(output["asset_id"].unique())
    if missing_assets:
        raise FeatureError(f"market data are missing assets: {sorted(missing_assets)}")
    if output["price_observed"].dtype != bool:
        raise FeatureError("price_observed must be boolean")
    for name in MARKET_COLUMNS[3:]:
        output[name] = pd.to_numeric(output[name], errors="coerce")
    observed_price = output.loc[output["price_observed"], "price_effective"]
    if observed_price.isna().any() or observed_price.le(0).any():
        raise FeatureError("observed prices must be positive finite values")
    output.loc[output["market_cap_effective"].le(0), "market_cap_effective"] = np.nan
    output.loc[output["volume_24h_usd_effective"].lt(0), "volume_24h_usd_effective"] = (
        np.nan
    )
    return output.sort_values(["asset_id", "timestamp_utc"], kind="mergesort")


def _normalize_macro(frame: pd.DataFrame | None) -> pd.DataFrame:
    required = {"feature_name", "available_at_utc", "value"}
    if frame is None:
        return pd.DataFrame(columns=sorted(required))
    missing = required.difference(frame.columns)
    if missing:
        raise FeatureError(f"macro data are missing columns: {sorted(missing)}")
    output = frame.loc[:, ["feature_name", "available_at_utc", "value"]].copy()
    unknown = set(output["feature_name"]).difference(MACRO_SOURCE_FEATURES)
    if unknown:
        raise FeatureError(f"macro data contain unknown features: {sorted(unknown)}")
    output["available_at_utc"] = pd.to_datetime(
        output["available_at_utc"], utc=True, errors="raise"
    )
    output["value"] = pd.to_numeric(output["value"], errors="coerce")
    if not np.isfinite(output["value"].to_numpy(dtype=float)).all():
        raise FeatureError("macro values must be finite")
    if output.duplicated(["feature_name", "available_at_utc"]).any():
        raise FeatureError("macro data contain duplicate availability timestamps")
    return output.sort_values(["feature_name", "available_at_utc"], kind="mergesort")


def _complete_hourly_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp_utc", kind="mergesort").set_index(
        "timestamp_utc"
    )
    full_index = pd.date_range(group.index.min(), group.index.max(), freq="h", tz="UTC")
    output = group.reindex(full_index)
    output.index.name = "timestamp_utc"
    output["asset_id"] = group["asset_id"].iloc[0]
    output["price_observed"] = output["price_observed"].fillna(False).astype(bool)
    return output


def _complete_window(series: pd.Series, window: int) -> pd.Series:
    return series.notna().rolling(window, min_periods=window).sum().eq(window)


def _asset_features(group: pd.DataFrame) -> pd.DataFrame:
    group = _complete_hourly_group(group)
    price = group["price_effective"].where(group["price_observed"])
    market_cap = group["market_cap_effective"]
    volume = group["volume_24h_usd_effective"]
    output = pd.DataFrame(index=group.index)
    output["asset_id"] = group["asset_id"]
    output["timestamp_utc"] = group.index
    output["price_usd"] = price
    output["downside_peg_deviation"] = (1.0 - price).clip(lower=0.0)
    for hours in (1, 6, 24):
        ready = _complete_window(price, hours + 1)
        output[f"log_return_{hours}h"] = np.log(price / price.shift(hours)).where(ready)
    output["rolling_min_price_24h"] = price.rolling(24, min_periods=24).min()
    returns = np.log(price / price.shift(1))
    output["realized_vol_24h"] = returns.rolling(24, min_periods=24).std(ddof=1)
    output["realized_vol_168h"] = returns.rolling(168, min_periods=168).std(ddof=1)
    output["downside_semivol_24h"] = (
        returns.clip(upper=0).pow(2).rolling(24, min_periods=24).mean().pow(0.5)
    )
    output["log_market_cap"] = np.log(market_cap)
    output["market_cap_change_24h"] = (market_cap / market_cap.shift(24) - 1.0).where(
        _complete_window(market_cap, 25)
    )
    output["volume_24h_usd"] = volume
    output["volume_24h_change_24h"] = (volume / volume.shift(24) - 1.0).where(
        _complete_window(volume, 25) & volume.shift(24).gt(0)
    )
    mean = volume.rolling(168, min_periods=168).mean()
    standard_deviation = volume.rolling(168, min_periods=168).std(ddof=1)
    output["volume_24h_z_168h"] = ((volume - mean) / standard_deviation).where(
        standard_deviation.gt(0)
    )
    output["turnover_24h"] = volume / market_cap
    return output


def _macro_asof(macro: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    base = pd.DataFrame({"timestamp_utc": times}).sort_values("timestamp_utc")
    output = base.set_index("timestamp_utc")
    for name in MACRO_SOURCE_FEATURES:
        source = macro.loc[macro["feature_name"].eq(name)].rename(
            columns={"available_at_utc": "selected_available_at_utc"}
        )
        if source.empty:
            output[name] = np.nan
            continue
        selected = pd.merge_asof(
            base,
            source.loc[:, ["selected_available_at_utc", "value"]].sort_values(
                "selected_available_at_utc"
            ),
            left_on="timestamp_utc",
            right_on="selected_available_at_utc",
            direction="backward",
            allow_exact_matches=True,
        )
        output[name] = selected.set_index("timestamp_utc")["value"]
    return output


def build_feature_frames(
    market_frame: pd.DataFrame, macro_frame: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned M0 and M1 hourly frames without feature imputation."""

    market = _normalize_market(market_frame)
    macro = _normalize_macro(macro_frame)
    own = {
        str(asset): _asset_features(group)
        for asset, group in market.groupby("asset_id", sort=True)
    }
    all_times = pd.DatetimeIndex(
        sorted(
            market.loc[market["asset_id"].isin(STUDY_ASSETS), "timestamp_utc"].unique()
        )
    )
    target_panels = [own[asset].reindex(all_times) for asset in STUDY_ASSETS]
    cap_panel = pd.concat(
        [
            panel["log_market_cap"].apply(np.exp).rename(asset)
            for asset, panel in zip(STUDY_ASSETS, target_panels, strict=True)
        ],
        axis=1,
    )
    price_panel = pd.concat(
        [
            panel["price_usd"].rename(asset)
            for asset, panel in zip(STUDY_ASSETS, target_panels, strict=True)
        ],
        axis=1,
    )
    cap_ready = cap_panel.notna().sum(axis=1).eq(len(STUDY_ASSETS))
    price_ready = price_panel.notna().sum(axis=1).eq(len(STUDY_ASSETS))
    total_cap = cap_panel.sum(axis=1, min_count=len(STUDY_ASSETS))
    hhi = (
        cap_panel.div(total_cap, axis=0).pow(2).sum(axis=1, min_count=len(STUDY_ASSETS))
    )
    macro_values = _macro_asof(macro, all_times)
    target_frames: list[pd.DataFrame] = []
    for asset in STUDY_ASSETS:
        frame = own[asset].reindex(all_times).copy()
        frame["stablecoin_mcap_share"] = (cap_panel[asset] / total_cap).where(cap_ready)
        frame["stablecoin_mcap_hhi"] = hhi.where(cap_ready)
        peers = price_panel.drop(columns=asset)
        frame["peer_price_dispersion_1h"] = (
            peers.sub(1.0).std(axis=1, ddof=1).where(price_ready)
        )
        frame["peer_downside_stress_1h"] = (
            (1.0 - peers).clip(lower=0).mean(axis=1).where(price_ready)
        )
        for prefix, context_asset in (("btc", "bitcoin"), ("eth", "ethereum")):
            context = own[context_asset].reindex(all_times)
            frame[f"{prefix}_log_return_24h"] = context["log_return_24h"]
            frame[f"{prefix}_realized_vol_24h"] = context["realized_vol_24h"]
        for name in MACRO_SOURCE_FEATURES:
            frame[name] = macro_values[name]
        frame["term_spread_10y_3m"] = frame["us_10y_yield"] - frame["us_3m_yield"]
        target_frames.append(frame.reset_index(drop=True))
    features = pd.concat(target_frames, ignore_index=True).sort_values(
        ["asset_id", "timestamp_utc"], kind="mergesort"
    )
    metadata = features.loc[:, ["asset_id", "timestamp_utc", "price_usd"]].reset_index(
        drop=True
    )
    m0 = pd.concat(
        [metadata, features.loc[:, M0_FEATURES].astype(float).reset_index(drop=True)],
        axis=1,
    )
    m1 = pd.concat(
        [metadata, features.loc[:, M1_FEATURES].astype(float).reset_index(drop=True)],
        axis=1,
    )
    for frame, columns, family in (
        (m0, M0_FEATURES, "M0_market"),
        (m1, M1_FEATURES, "M1_market_macro"),
    ):
        values = frame.loc[:, columns].to_numpy(dtype=float)
        frame["source_family"] = family
        frame["feature_eligibility"] = np.isfinite(values).all(axis=1)
        frame["missing_feature_count"] = (~np.isfinite(values)).sum(axis=1)
    return m0.reset_index(drop=True), m1.reset_index(drop=True)
