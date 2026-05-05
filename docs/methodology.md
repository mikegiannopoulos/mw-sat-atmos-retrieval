# Methodology

## Philosophy

The project follows a physics-first and interpretability-first methodology:

- begin with validated clear-sky forward physics,
- add minimal instrument assumptions,
- define simple scalar retrieval targets,
- evaluate controlled perturbations before scaling,
- prefer transparent models over flexible black-box models.

This keeps the scientific meaning of each experiment traceable.

## Forward Modeling

The forward model uses:

- ERA5 pressure-level profiles,
- temperature,
- specific humidity,
- geopotential-derived altitude,
- `O2-PWR98` and `H2O-PWR98`,
- PyARTS clear-sky brightness temperature simulation.

No scattering, cloud physics, or line-catalog complexity is introduced in the current phase.

## Instrument Model

The current instrument abstraction contains:

- three channels:
  - `50.3 GHz`
  - `52.8 GHz`
  - `54.4 GHz`
- channel-specific `NEΔT`
- additive Gaussian noise
- deterministic seeding for reproducibility

This is sufficient for structured noise-propagation and trade-off experiments.

## Retrieval Design

Retrieval targets are scalar layer-mean temperatures:

- lower-layer mean temperature
- upper-layer mean temperature

Retrieval models are limited to:

- mean baseline
- linear regression
- ridge regression

This keeps the mapping from channel signal to retrieval output easy to interpret.

## Evaluation Strategy

The repository uses several complementary evaluation styles:

- profile-level leave-one-out cross-validation,
- source-specific leave-one-out cross-validation,
- leave-one-regime-out testing,
- synthetic stress-test evaluation,
- sensitivity experiments,
- noise sweeps,
- channel ablation.

These are used as controlled diagnostics, not as operational performance certification.
