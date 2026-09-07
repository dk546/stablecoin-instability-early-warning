# Stablecoin Instability Early Warning

Research code accompanying the master's thesis *From Signals to Warnings: An Explainable Event-Window Framework for Stablecoin Instability*.

The project studies whether elevated instability risk can be identified before a sustained downside peg breach begins. It treats the task as risk ranking and decision support, not as prediction of an exact de-peg time.

## Study design

- Assets: DAI, Tether (USDT), and USD Coin (USDC)
- Outcome: the onset of at least two consecutive observed hourly prices below USD 0.99
- Warning target: an onset 6 to 24 hours ahead
- Validation: daily expanding windows, with outcomes admitted to training only after their observation period and a 24-hour embargo
- Models: L2-regularized logistic regression and pooled LightGBM
- Source families: a 22-feature market specification (M0) and a 26-feature market-plus-macro specification (M1)
- Calibration: fold-causal Platt calibration based only on admissible earlier out-of-sample predictions
- Evaluation: discrimination, calibration, operational alert burden, event detection, lead time, paired uncertainty, and fold-local SHAP summaries

The reference alert rule uses a probability threshold of 0.20 and two consecutive eligible hours. Results across the complete prespecified threshold and persistence grid should also be reported.

## Repository scope

This repository contains the reusable scientific methods. It begins from caller-supplied hourly market and release-aware macro data. Provider acquisition code, raw source data, generated results, and the thesis document are not distributed here.

The implementation is deliberately separated into small stages:

1. validate the study scope and feature schemas;
2. detect persistent downside events and construct censored warning labels;
3. calculate market and macro features using information available at prediction time;
4. build a common daily expanding-window fold plan;
5. fit logistic and LightGBM models, calibrate them causally, and compute fold-local SHAP values;
6. evaluate predictive and operational performance.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Quick check

The example uses synthetic observations and does not download data.

```bash
python examples/synthetic_workflow.py
python -m unittest discover -s tests -v
python tools/check_publication.py
```

## Input expectations

Market observations are hourly and keyed by `asset_id` and `timestamp_utc`. The target stablecoins are `dai`, `tether`, and `usd-coin`; `bitcoin` and `ethereum` provide market context. Prepared market data should distinguish observed price values from admissible carried market-cap and volume values.

Macro observations are keyed by `feature_name` and `available_at_utc`. Their availability timestamp, rather than the period to which the observation refers, controls as-of selection. This prevents revised or not-yet-released values from entering earlier predictions.

No imputation is used to make a row eligible. A model row is included only when every required feature for its source family is finite and its outcome is observable.

## Reproducibility notes

The code fixes the feature order, daily fold construction, estimator settings, calibration readiness rules, alert grid, and random seeds. A complete empirical reproduction still requires appropriately licensed source data prepared to the documented interfaces.

## Limitations

The framework was evaluated on three large USD-referenced stablecoins. Rare events, structural market changes, provider revisions, incomplete asset histories, and dependence between observations limit generalization. SHAP values describe model associations and do not establish causation. The outputs should be interpreted as monitoring evidence, not as autonomous trading or risk decisions.

## Citation

Citation metadata is provided in `CITATION.cff`.

## License

The code is available under the MIT License. Data and third-party materials retain their respective terms.
