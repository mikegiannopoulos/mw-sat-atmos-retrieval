# mw-sat-atmos-retrieval

## Overview

`mw-sat-atmos-retrieval` is a research-oriented Python project focused on simulating microwave satellite observations of the atmosphere and investigating how instrument characteristics influence atmospheric temperature retrieval accuracy.

The project is centered on three connected components: forward simulation of microwave radiances, representation of instrument characteristics, and evaluation of how those characteristics influence retrieval performance. The overall aim is to support controlled, reproducible studies of temperature sounding in a modular scientific workflow designed for controlled numerical experimentation.

ARTS/PyARTS will be used as the main radiative transfer framework. The initial scope is clear-sky temperature sounding, with the codebase structured so that later extensions can be added without redesigning the project.

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

The present forward model and retrieval components are placeholders intended to support pipeline development. They will later be replaced or extended with ARTS/PyARTS-based forward simulation and more realistic retrieval methods.

## Current Project Status

The project is in its initial development phase.

Current work is focused on building the foundation for a research-grade simulation and retrieval framework. This includes establishing the package structure, environment definition, data organization, and the modular components needed to support clear-sky microwave temperature sounding studies.

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
