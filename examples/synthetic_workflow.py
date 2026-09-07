"""Run the data-preparation stages on deterministic synthetic observations."""

from stablecoin_ews.features import build_feature_frames
from stablecoin_ews.labels import build_warning_labels
from stablecoin_ews.synthetic import synthetic_hourly_inputs
from stablecoin_ews.validation import expanding_daily_folds, model_population


def main() -> None:
    market, macro = synthetic_hourly_inputs()
    label_input = market.loc[
        market["asset_id"].isin(("dai", "tether", "usd-coin")),
        ["asset_id", "timestamp_utc", "price_observed", "price_effective"],
    ].rename(columns={"price_effective": "price_usd"})
    labels, episodes = build_warning_labels(label_input)
    m0, m1 = build_feature_frames(market, macro)
    m0_population = model_population(labels, m0)
    m1_population = model_population(labels, m1)
    m0_folds = expanding_daily_folds(m0_population)
    m1_folds = expanding_daily_folds(m1_population)
    print(
        {
            "event_episodes": len(episodes),
            "m0_eligible_rows": len(m0_population),
            "m1_eligible_rows": len(m1_population),
            "m0_daily_folds": len(m0_folds),
            "m1_daily_folds": len(m1_folds),
        }
    )


if __name__ == "__main__":
    main()
