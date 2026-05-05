# mw-sat-atmos-retrieval

## Overview

`mw-sat-atmos-retrieval` is a research-oriented Python project for simulating microwave satellite observations of the atmosphere and studying how instrument assumptions influence those observations.

The current validated workflow is a forward-model and sensitivity-analysis pipeline built around:

- ERA5 pressure-level atmospheric profiles,
- clear-sky PyARTS forward simulation,
- simple instrument channel and noise assumptions, and
- small controlled perturbation experiments.

The longer-term goal is to support retrieval and instrument trade-off studies, but the repository should not currently be interpreted as a complete retrieval system.

## Project Goal

The project is intended to support reproducible studies of:

- microwave satellite observation simulation,
- instrument-aware sensitivity analysis, and
- eventual retrieval and instrument trade-off experiments.

At the current stage, the emphasis is on building a scientifically cautious forward-modeling foundation before adding more advanced retrieval components.

## High-Level Workflow

The project is organized around the following conceptual pipeline:

1. Atmospheric data preparation
2. Forward simulation with ARTS/PyARTS
3. Instrument modeling
4. Sensitivity diagnostics
5. Later retrieval experiments
6. Evaluation and visualization

Only the forward-model and sensitivity-analysis parts are currently validated in a meaningful way.

## Repository Structure

```text
mw-sat-atmos-retrieval/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── reports/
│   └── figures/
├── scripts/
├── src/
│   └── mwsat/
│       ├── evaluation/
│       ├── experiments/
│       ├── forward/
│       ├── instrument/
│       ├── io/
│       ├── pipeline/
│       ├── plotting/
│       ├── profiles/
│       ├── retrieval/
│       └── utils/
├── tests/
├── environment.yml
├── pyproject.toml
└── README.md
```

## Environment Setup

Create and activate the Conda environment with:

```bash
conda env create -f environment.yml
conda activate mwsat
```

## Current Capabilities

The validated project state currently includes:

- ERA5 pressure-level profile ingestion for temperature, specific humidity, and geopotential.
- PyARTS clear-sky forward simulation with `O2-PWR98` and `H2O-PWR98`.
- Conversion of ERA5 specific humidity into H2O volume mixing ratio for ARTS input.
- A reusable `InstrumentConfig` representation with channels and per-channel NEΔT values.
- Synthetic noisy observations generated from brightness temperatures with deterministic, profile-specific random seeds.
- Uniform atmospheric temperature sensitivity diagnostics.
- Lower-atmosphere and upper-atmosphere temperature sensitivity diagnostics based on simple pressure-threshold perturbations.
- Controlled humidity sensitivity diagnostics based on `+10%` and `-10%` H2O VMR perturbations.
- Observation-space humidity sensitivity ratios relative to channel NEΔT.
- Small multi-profile experiment tables stored as CSV outputs.
- Basic diagnostic plotting from those CSV outputs.

## PyARTS Validation Scripts

The repository includes standalone PyARTS smoke tests for the current clear-sky forward-model work.

Run the controlled synthetic validation:

```bash
conda run -n mwsat python scripts/test_arts_minimal.py
```

Run the ERA5 profile bridge smoke test:

```bash
conda run -n mwsat python scripts/test_arts_era5_profile.py
```

The ERA5 smoke-test script currently exercises:

- clear-sky O2 + H2O forward simulation,
- instrument-aware noisy observations,
- temperature perturbation diagnostics,
- lower-versus-upper atmosphere sensitivity diagnostics, and
- controlled humidity perturbation diagnostics, and
- small multi-profile experiment output generation.

## Current Stress-Test Profiles

To probe the stability of the diagnostic conclusions beyond the narrow regional ERA5 sample, the current workflow also includes a small set of controlled synthetic profiles built from a baseline ERA5 case:

- warm/moist synthetic profile: temperature `+15 K` throughout the atmosphere and H2O VMR scaled by `x2`,
- cold/dry synthetic profile: temperature `-15 K` throughout the atmosphere and H2O VMR scaled by `x0.3`, and
- strong-lapse synthetic profile: an increased vertical temperature gradient with a warmer lower atmosphere and colder upper atmosphere.

These profiles use the same pressure grid and geopotential-derived altitude as the baseline case and are treated as additional clear-sky sensitivity-analysis samples rather than as realistic global climatological cases.

## ERA5 Sample Data

Small ERA5 sample data can be downloaded with:

```bash
conda run -n mwsat python scripts/download_era5_arts_sample.py
```

The output defaults to `data/raw/era5/arts_era5_sample.nc`.

Downloaded NetCDF files under `data/` are ignored by git and should not be committed.

## Example Outputs

Current experiment outputs are written to:

- `data/processed/experiments/`

Typical CSV outputs include:

- `multi_profile_observations.csv`
- `multi_profile_temperature_sensitivity.csv`
- `multi_profile_layer_sensitivity.csv`
- `multi_profile_humidity_sensitivity.csv`

Current diagnostic figures are written to:

- `reports/figures/experiments/`

Typical diagnostic figures include:

- `temperature_sensitivity_vs_frequency.png`
- `sensitivity_to_noise_vs_frequency.png`
- `layer_sensitivity_ratio_vs_frequency.png`
- `humidity_sensitivity_ratio_vs_frequency.png`

In the current 3-channel oxygen-band setup, the clear-sky humidity perturbation responses are present but small relative to channel measurement noise in the current regional sample. That is a useful forward-model sensitivity result, but it should not be interpreted as evidence of an operational water-vapor retrieval capability.

## Current Limitations

The present workflow is intentionally limited:

- clear-sky only,
- no scattering or clouds,
- no retrievals yet,
- limited ERA5 sample diversity in the current validation dataset,
- no formal Jacobians yet, and
- no full information-content analysis yet.

The current diagnostics are useful for controlled forward-model checks, but they should not be interpreted as complete retrieval-value metrics.

## Current Project Status

The project is now in a validated early forward-model stage.

The core clear-sky PyARTS setup has been exercised against a small ERA5 sample with O2 + H2O absorption, simple instrument noise, perturbation-based temperature sensitivity diagnostics, multi-profile CSV outputs, and basic plots. This is enough to support controlled forward experiments and small sensitivity studies, but it is still short of a complete retrieval framework.

Within the current regional ERA5 sample, the main channel-sensitivity patterns are fairly stable. The synthetic stress tests show that some regime dependence is already visible, however: warm/moist conditions can shift the strongest uniform temperature sensitivity toward `54.4 GHz`, and cold/dry conditions can slightly shift lower-layer sensitivity toward `50.3 GHz`. The humidity response also increases in the moist synthetic case, but it remains below the measurement-noise scale in this simple 3-channel setup. These are useful sensitivity-analysis results, but they should not be interpreted as evidence of global robustness or retrieval capability.

## Next Steps

Reasonable near-term development steps include:

- expand ERA5 profile diversity,
- introduce simple retrieval experiments later, and
- evaluate instrument trade-offs more systematically.

## Planned Roadmap

Planned development areas still include:

- more complete package-level experiment workflows,
- broader atmospheric input handling,
- promotion of validated standalone logic into reusable package components where appropriate,
- initial retrieval methods after the forward and sensitivity pieces are mature enough, and
- improved evaluation and visualization utilities.

This project should currently be understood as a forward-model and sensitivity-analysis pipeline under active development, not as a complete retrieval system.
