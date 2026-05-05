# Project Summary

## Motivation

This project investigates a focused microwave remote-sensing question: how do oxygen-band channel selection and instrument noise affect atmospheric temperature retrieval accuracy under a simple, interpretable clear-sky framework?

The work is motivated by a common challenge in atmospheric retrieval design. Channels may look useful from radiative-transfer theory alone, but their practical retrieval value depends on noise, target definition, atmospheric regime, and overlap with other channels. The project therefore builds a controlled workflow that links forward physics, instrument assumptions, and retrieval behavior directly.

## Methods

The repository combines:

- ERA5 pressure-level atmospheric profiles,
- clear-sky PyARTS forward simulation with oxygen and water-vapor absorption,
- a minimal three-channel instrument model (`50.3`, `52.8`, `54.4 GHz`),
- channel-specific `NEΔT` noise,
- simple scalar retrieval targets:
  - lower-layer mean temperature
  - upper-layer mean temperature,
- interpretable linear retrieval models:
  - mean baseline
  - linear regression
  - ridge regression.

The experiments were intentionally designed to stay physics-first and minimal. No nonlinear retrievals, no black-box models, and no expanded predictor set were introduced.

## Experiments

The completed workflow includes four main experiment classes:

1. **Sensitivity analysis**
   - temperature perturbations
   - lower- vs upper-layer sensitivity
   - humidity sensitivity

2. **Noise sweep retrieval experiment**
   - global and source-specific retrieval performance under increasing `NEΔT`

3. **Channel-specific noise sensitivity**
   - degrade one channel at a time to isolate target-dependent vulnerability

4. **Channel ablation**
   - test all 3-channel, 2-channel, and 1-channel combinations to identify minimal useful subsets

The dataset includes three ERA5-derived regimes plus labeled synthetic stress-test profiles used as controlled out-of-distribution cases.

## Main Results

The main conclusions are consistent across experiments:

- Global linear retrieval is not regime-invariant.
- Source-specific retrieval improves within-regime performance.
- Lower-layer temperature retrieval is mainly controlled by `52.8 GHz`.
- Upper-layer temperature retrieval is mainly controlled by `54.4 GHz`.
- Retrieval error increases smoothly with increasing `NEΔT`.
- Channel-specific degradation confirms that different retrieval targets depend on different channels.
- `52.8 + 54.4 GHz` is the strongest reduced channel pair.
- `50.3 GHz` contributes less unique information for the current scalar targets.

Taken together, these results show that the three-channel system is physically informative, but not uniformly so across targets or regimes.

## Why This Project Is Scientifically Useful

The project is useful because it goes beyond “does the retrieval work?” and asks a more informative question:

**Why does it work, when does it fail, and which parts of the instrument matter most?**

That makes the repository valuable as a research portfolio project:

- it demonstrates physics-aware model building,
- it links forward simulation to retrieval behavior,
- it treats uncertainty propagation explicitly,
- and it frames retrieval performance in terms of instrument trade-offs rather than raw benchmark scores.

## Limitations

The current system is intentionally constrained:

- clear-sky only
- synthetic noisy observations
- small dataset
- linear retrieval only
- scalar targets only
- not an operational retrieval system

These limitations are important. The repository should be read as a controlled research prototype, not as a finished atmospheric retrieval product.

## Future Extensions

The most natural next steps are:

- broader atmospheric sampling,
- richer retrieval targets,
- more realistic uncertainty sources,
- later cloud/scattering extensions,
- and more formal retrieval frameworks once the simple linear baseline is fully understood.

Even in its current form, the project already functions as a coherent research-grade framework for interpreting channel sensitivity, noise propagation, and early-stage instrument design choices.
