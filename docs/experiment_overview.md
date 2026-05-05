# Experiment Overview

## Forward and Sensitivity Layer

Primary script:

- `scripts/test_arts_era5_profile.py`

Main outputs:

- `multi_profile_observations.csv`
- temperature sensitivity tables
- layer sensitivity tables
- humidity sensitivity tables

Purpose:

- validate the clear-sky forward bridge from ERA5 to ARTS
- quantify target-dependent atmospheric sensitivity before retrieval

## Retrieval Baseline Layer

Primary scripts:

- `scripts/run_linear_retrieval_experiment.py`
- `scripts/run_cross_regime_retrieval_experiment.py`
- `scripts/run_regime_aware_retrieval_experiment.py`
- `scripts/run_retrieval_by_target_experiment.py`

Purpose:

- compare simple linear retrievals against a mean baseline
- test regime dependence
- compare lower-layer and upper-layer retrieval behavior

## Instrument Trade-Off Layer

Primary scripts:

- `scripts/run_noise_sweep_retrieval_experiment.py`
- `scripts/run_channel_noise_sensitivity_experiment.py`
- `scripts/run_channel_ablation_experiment.py`

Purpose:

- connect retrieval error to increasing instrument noise,
- isolate the effect of degrading individual channels,
- identify the minimal strong channel subset.

## Reporting Layer

Primary scripts:

- `scripts/plot_experiment_diagnostics.py`
- `scripts/plot_retrieval_diagnostics.py`

Primary written summaries:

- `reports/noise_sweep_retrieval_interpretation.md`
- `reports/channel_noise_sensitivity_interpretation.md`
- `reports/channel_ablation_interpretation.md`
- `reports/final_retrieval_system_analysis.md`

Purpose:

- turn experiment outputs into interpretable figures and narrative findings
- preserve a reproducible scientific record of the completed work
