"""Small, publication-oriented plots for prepared study data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from stablecoin_ews.contracts import ASSET_LABELS, STUDY_ASSETS


def plot_price_and_macro_context(
    market: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    *,
    macro_column: str = "vix_close",
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """Plot all stablecoin prices and optional macro context on aligned panels."""

    required = {"asset_id", "timestamp_utc", "price_usd"}
    missing = required.difference(market.columns)
    if missing:
        raise ValueError(f"market data are missing columns: {sorted(missing)}")
    use_macro = macro is not None and macro_column in macro.columns
    figure, axes = plt.subplots(
        2 if use_macro else 1,
        1,
        figsize=(11, 6.5 if use_macro else 4.5),
        sharex=use_macro,
        constrained_layout=True,
    )
    axis_tuple = tuple(axes) if use_macro else (axes,)
    prepared = market.copy()
    prepared["timestamp_utc"] = pd.to_datetime(
        prepared["timestamp_utc"], utc=True, errors="raise"
    )
    for asset in STUDY_ASSETS:
        selected = prepared.loc[prepared["asset_id"].eq(asset)].sort_values(
            "timestamp_utc"
        )
        axis_tuple[0].plot(
            selected["timestamp_utc"],
            selected["price_usd"],
            label=ASSET_LABELS[asset],
            linewidth=0.9,
        )
    axis_tuple[0].axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    axis_tuple[0].axhline(0.99, color="black", linewidth=0.7, linestyle="--")
    axis_tuple[0].set_ylabel("Price (USD)")
    axis_tuple[0].legend(ncol=3, frameon=False)
    if use_macro and macro is not None:
        context = macro.copy()
        context["timestamp_utc"] = pd.to_datetime(
            context["timestamp_utc"], utc=True, errors="raise"
        )
        axis_tuple[1].plot(
            context["timestamp_utc"],
            context[macro_column],
            color="black",
            linewidth=0.8,
        )
        axis_tuple[1].set_ylabel(macro_column.replace("_", " ").title())
    axis_tuple[-1].set_xlabel("UTC date")
    return figure, axis_tuple
