"""Prespecified study scope and ordered scientific interfaces."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real


class ContractError(ValueError):
    """Raised when input differs from a prespecified scientific interface."""


class SourceFamily(StrEnum):
    """Feature families compared in the study."""

    M0_MARKET = "M0_market"
    M1_MARKET_MACRO = "M1_market_macro"


STUDY_ASSETS: tuple[str, ...] = ("dai", "tether", "usd-coin")
CONTEXT_ASSETS: tuple[str, ...] = ("bitcoin", "ethereum")
ASSET_LABELS: Mapping[str, str] = {
    "dai": "DAI",
    "tether": "USDT",
    "usd-coin": "USDC",
}

M0_FEATURES: tuple[str, ...] = (
    "downside_peg_deviation",
    "log_return_1h",
    "log_return_6h",
    "log_return_24h",
    "rolling_min_price_24h",
    "realized_vol_24h",
    "realized_vol_168h",
    "downside_semivol_24h",
    "log_market_cap",
    "market_cap_change_24h",
    "volume_24h_usd",
    "volume_24h_change_24h",
    "volume_24h_z_168h",
    "turnover_24h",
    "stablecoin_mcap_share",
    "stablecoin_mcap_hhi",
    "peer_price_dispersion_1h",
    "peer_downside_stress_1h",
    "btc_log_return_24h",
    "btc_realized_vol_24h",
    "eth_log_return_24h",
    "eth_realized_vol_24h",
)

MACRO_FEATURES: tuple[str, ...] = (
    "vix_close",
    "us_3m_yield",
    "term_spread_10y_3m",
    "effective_federal_funds_rate",
)

M1_FEATURES: tuple[str, ...] = M0_FEATURES + MACRO_FEATURES

PROBABILITY_THRESHOLDS: tuple[float, ...] = (
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
)
PERSISTENCE_HOURS: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class StudyDesign:
    """Central parameters fixed before model evaluation."""

    event_threshold_usd: float = 0.99
    event_duration_hours: int = 2
    warning_start_hours: int = 6
    warning_end_hours: int = 24
    confirmation_through_hours: int = 25
    test_block_hours: int = 24
    test_step_hours: int = 24
    training_embargo_hours: int = 24
    minimum_training_hours: int = 720
    minimum_training_rows: int = 720
    minimum_training_positives: int = 1
    minimum_training_negatives: int = 1
    calibration_embargo_hours: int = 24
    minimum_calibration_negatives: int = 2_000
    minimum_calibration_positives: int = 30
    minimum_calibration_days: int = 90
    minimum_calibration_rows_per_asset: int = 250
    reference_probability_threshold: float = 0.20
    reference_persistence_hours: int = 2
    bootstrap_replicates: int = 2_000
    bootstrap_seed: int = 42

    def __post_init__(self) -> None:
        if not 0 < self.event_threshold_usd < 1:
            raise ContractError("event threshold must lie between zero and one")
        if self.event_duration_hours < 1:
            raise ContractError("event duration must be positive")
        if not 0 <= self.warning_start_hours <= self.warning_end_hours:
            raise ContractError("warning horizon is invalid")
        if self.confirmation_through_hours <= self.warning_end_hours:
            raise ContractError("confirmation must extend beyond the warning horizon")


def feature_columns(source_family: SourceFamily | str) -> tuple[str, ...]:
    """Return the exact ordered columns for one model specification."""

    try:
        family = SourceFamily(source_family)
    except (TypeError, ValueError) as exc:
        raise ContractError("unknown source family") from exc
    return M0_FEATURES if family is SourceFamily.M0_MARKET else M1_FEATURES


def assess_feature_record(
    record: Mapping[str, object], source_family: SourceFamily | str
) -> tuple[bool, tuple[str, ...]]:
    """Check completeness and finiteness without imputing unavailable values."""

    invalid: list[str] = []
    for name in feature_columns(source_family):
        value = record.get(name)
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            invalid.append(name)
    return not invalid, tuple(invalid)


def validate_feature_order(
    columns: Sequence[str], source_family: SourceFamily | str
) -> None:
    """Require an exact feature order so models cannot reinterpret matrices."""

    expected = feature_columns(source_family)
    provided = tuple(columns)
    if len(provided) != len(set(provided)):
        raise ContractError("feature columns contain duplicates")
    if provided != expected:
        raise ContractError("feature columns differ from the prespecified order")


def study_summary() -> dict[str, object]:
    """Return a compact, serializable description of the empirical design."""

    design = StudyDesign()
    return {
        "assets": list(STUDY_ASSETS),
        "event": {
            "threshold_usd": design.event_threshold_usd,
            "duration_hours": design.event_duration_hours,
        },
        "warning_horizon_hours": [
            design.warning_start_hours,
            design.warning_end_hours,
        ],
        "source_families": {
            SourceFamily.M0_MARKET.value: list(M0_FEATURES),
            SourceFamily.M1_MARKET_MACRO.value: list(M1_FEATURES),
        },
        "probability_thresholds": list(PROBABILITY_THRESHOLDS),
        "persistence_hours": list(PERSISTENCE_HOURS),
    }
