# mw-sat-atmos-retrieval

## Overview

`mw-sat-atmos-retrieval` is a research-oriented Python project focused on simulating microwave satellite observations of the atmosphere and investigating how instrument characteristics influence atmospheric temperature retrieval accuracy.

The project is centered on three connected components: forward simulation of microwave radiances, representation of instrument characteristics, and evaluation of how those characteristics influence retrieval performance. The overall aim is to support controlled, reproducible studies of temperature sounding in a modular scientific workflow designed for controlled numerical experimentation.

ARTS/PyARTS is being introduced as the main radiative transfer framework. The current validated ARTS work lives in standalone scripts while the package-level backend remains a scaffold. The initial scope is clear-sky oxygen-band temperature sounding, with the codebase structured so that later extensions can be added without redesigning the project.

## Main Objective

The main objective of this project is to investigate how instrument characteristics affect atmospheric temperature retrieval accuracy.

This will be done through controlled numerical experiments in which atmospheric states, forward-model assumptions, and retrieval settings can be held fixed while instrument parameters are varied systematically. Examples of such parameters may include channel selection, spectral configuration, noise assumptions, and vertical sensitivity characteristics.

The project is intended to provide a reproducible environment for testing how sensor design choices influence retrieval quality under clearly defined conditions.

## High-Level Workflow

The project is organized around the following high-level pipeline:

1. **Atmospheric data preparation**  
   Prepare atmospheric profiles and supporting inputs from external datasets and intermediate preprocessing steps.

2. **Forward simulation with ARTS/PyARTS**  
   Use ARTS/PyARTS to simulate microwave observations from prescribed atmospheric states under controlled conditions.

3. **Instrument modeling**  
   Apply instrument-related assumptions such as channel configuration, noise, and simplified sensor response characteristics.

4. **Retrieval**  
   Estimate atmospheric temperature profiles from simulated observations using retrieval methods implemented within the project.

5. **Evaluation**  
   Compare retrieved profiles against reference states and quantify retrieval accuracy, sensitivity, and error characteristics.

6. **Visualization**  
   Produce figures and diagnostics to summarize experiment design, simulated observations, and retrieval performance.

## Repository Structure

```text
mw-sat-atmos-retrieval/
├── configs/
├── data/
│   ├── raw/
│   │   ├── era5/
│   │   ├── igra/
│   │   ├── atms/
│   │   └── chalmers_scattering/
│   ├── interim/
│   └── processed/
├── docs/
├── notebooks/
├── reports/
│   └── figures/
├── scripts/
├── src/
│   └── mwsat/
│       ├── io/
│       ├── profiles/
│       ├── forward/
│       ├── instrument/
│       ├── retrieval/
│       ├── evaluation/
│       ├── plotting/
│       └── utils/
├── tests/
├── .gitignore
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

## Current Demo Workflow

The current pipeline can be exercised with a small generated ERA5-like NetCDF file that serves as a placeholder input for development and testing.

```bash
python scripts/create_dummy_era5.py
python scripts/run_experiment_summary.py
```

This currently runs configuration loading, ERA5-like profile ingestion, mock forward simulation, baseline retrieval, and metric aggregation.

The package-level forward model and retrieval components are still placeholders intended to support pipeline development. They will later be replaced or extended with the validated ARTS/PyARTS setup and more realistic retrieval methods.

## PyARTS Validation Scripts

The repository includes standalone PyARTS smoke tests for the current clear-sky forward-model work. These scripts are intentionally kept outside the package until the workflow is ready to be promoted into reusable production code.

Run the controlled synthetic validation:

```bash
conda run -n mwsat python scripts/test_arts_minimal.py
```

This validates a minimal clear-sky O2-band ARTS setup using `O2-PWR98`, `iy_unit = "PlanckBT"`, a blackbody surface, and three channels at 50.3, 52.8, and 54.4 GHz. It checks baseline brightness temperatures plus uniform atmospheric, surface, lower-layer, and upper-layer temperature perturbations.

Run the ERA5 profile bridge smoke test:

```bash
conda run -n mwsat python scripts/test_arts_era5_profile.py
```

By default this uses `data/raw/era5/real_era5.nc` if present, otherwise `data/raw/era5/dummy_era5.nc`. You can also pass an explicit NetCDF path:

```bash
conda run -n mwsat python scripts/test_arts_era5_profile.py data/raw/era5/real_era5.nc
```

The ERA5 script loads one pressure-level temperature profile, prepares ARTS-compatible pressure, temperature, and altitude fields, runs the same clear-sky O2-band setup, and prints profile metadata plus channel brightness temperatures.

## ERA5 Sample Data

Small ERA5 sample data can be downloaded with the CDS API helper script:

```bash
python scripts/download_era5_sample.py
python scripts/run_forward_from_file.py data/raw/era5/real_era5.nc
```

For the next ARTS-oriented ERA5 smoke tests, download a compact pressure-level sample containing temperature, specific humidity, and geopotential:

```bash
conda run -n mwsat python scripts/download_era5_arts_sample.py
```

The output defaults to `data/raw/era5/arts_era5_sample.nc`. Use `--output` to choose a different path and `--overwrite` to replace an existing file.

CDS API credentials must be configured in `~/.cdsapirc` before download. Downloaded NetCDF files under `data/` are ignored by git and should not be committed.

## Current Project Status

The project is in its initial development phase.

Current work is focused on building the foundation for a research-grade simulation and retrieval framework. This includes establishing the package structure, environment definition, data organization, and the modular components needed to support clear-sky microwave temperature sounding studies.

The minimal ARTS forward model has now passed controlled standalone checks and has been connected to one ERA5 pressure-level profile. The next step is to extend the ERA5 smoke test to use humidity and geopotential from the richer ERA5 sample, then promote the validated setup into the package-level forward backend.

At this stage, the repository is intended as a scaffold for reproducible scientific development rather than a completed analysis system.

## Planned Roadmap

Planned development areas include:

- configuration-driven experiment workflows,
- ingestion and preprocessing of atmospheric input data,
- integration of ARTS/PyARTS into the forward simulation workflow,
- implementation of initial temperature retrieval methods,
- evaluation metrics for retrieval performance and instrument sensitivity,
- improved visualization and reporting utilities, and
- testing for reliability and reproducibility.

## Future Extensions

Future extensions may include the integration of Chalmers-related cloud and precipitation scattering datasets to support more advanced simulation studies beyond the initial clear-sky focus.

These additions are intended as possible next steps rather than current project capabilities.

This project is intended as a foundation for systematic investigation rather than a finished application.
