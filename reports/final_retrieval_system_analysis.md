# Final Retrieval System Analysis

## 1. Problem definition

This project studies a minimal microwave temperature-retrieval system built on a clear-sky PyARTS forward model, a small set of ERA5-derived atmospheric profiles, and a simple instrument abstraction with channel-specific `NEΔT`. The current observation vector consists of three oxygen-band channels:

- `50.3 GHz`
- `52.8 GHz`
- `54.4 GHz`

The current scalar retrieval targets are:

- `lower_layer_mean_temperature_k`
- `upper_layer_mean_temperature_k`

The system has been developed under explicit physics-first constraints:

- clear-sky only
- no scattering or cloud physics
- no black-box retrieval models
- minimal complexity and full interpretability
- validate controlled behavior before scaling

The goal has not been to build an operational retrieval system, but to understand how channel choice, instrument noise, and atmospheric regime affect the usable temperature information in a small, interpretable retrieval framework.

## 2. Key findings

### Regime dependence

The retrieval is strongly regime-dependent. A single global linear mapping can perform reasonably under in-distribution profile-level cross-validation, but it fails badly under leave-one-regime-out testing. This showed that the global linear retrieval is still heavily influenced by regime-specific covariance structure rather than by a regime-invariant mapping from `Tb_obs` to temperature target.

At the same time, source-specific retrieval improves substantially within regime. That indicates that the retrieval is not purely arbitrary; rather, the effective linear mapping differs across atmospheric regimes in a physically meaningful way.

### Vertical channel sensitivity

The sensitivity diagnostics and target-by-target retrieval experiments gave a vertically consistent picture:

- `52.8 GHz` is the most important channel for the lower-layer target
- `54.4 GHz` is the most important channel for the upper-layer target
- `50.3 GHz` plays a smaller and more secondary role under the current scalar-target setup

This is consistent with the earlier perturbation experiments, where the lower-layer response was strongest around `52.8 GHz`, while the upper-layer response was strongest around `54.4 GHz`.

### Noise propagation behavior

The instrument-noise sweep showed that retrieval error increases smoothly as channel `NEΔT` increases. This was true for both targets and for both global and source-specific retrieval setups. The lower-layer target degraded more slowly with increasing noise than the upper-layer target, both in absolute RMSE and in normalized RMSE.

This indicates that the current three-channel system carries a more robust lower-layer temperature signal than upper-layer signal under the present clear-sky assumptions.

### Channel-specific degradation effects

The one-channel-at-a-time noise experiment clarified which parts of the instrument matter most:

- degrading `52.8 GHz` hurts lower-layer retrieval most
- degrading `54.4 GHz` hurts upper-layer retrieval most
- degrading `50.3 GHz` has relatively little effect on upper-layer retrieval

These results match the sensitivity and dropout findings and provide a direct instrument-noise interpretation: not all channel noise is equally important for all retrieval targets.

### Minimal channel subset

The ablation experiment showed that the strongest reduced channel pair is:

- `52.8 + 54.4 GHz`

For the current scalar targets, this pair retains nearly all of the useful retrieval skill of the full three-channel system. The full set still performs best or near-best overall, but `50.3 GHz` contributes the least unique information under the current setup.

## 3. System understanding

### Which channels control which targets

The current system can be understood in a fairly direct way:

- `52.8 GHz` is the main lower-layer temperature channel
- `54.4 GHz` is the main upper-layer temperature channel
- `50.3 GHz` provides supplementary information, but less uniquely than the other two

This does not mean the channels measure completely separate layers. Their sensitivity overlaps. But their relative importance shifts in a physically consistent way depending on the retrieval target.

### Redundancy vs unique information

The experiments show that the three channels are partly redundant and partly complementary. The fact that `52.8 + 54.4 GHz` retains most skill means that a large fraction of the usable information for the current scalar targets is concentrated in those two channels. The smaller marginal value of `50.3 GHz` suggests that some of its information overlaps with the other channels, or that its unique contribution is weaker for these two targets.

At the same time, the three-channel system is not fully redundant. The full configuration still gives the most stable overall behavior, and the relative value of `50.3 GHz` can increase in specific warm/humid regimes or for other diagnostic purposes.

### Why upper-layer retrieval is harder

Upper-layer retrieval is harder for several linked reasons:

- the current observation vector has only three channels
- the upper-layer target is less directly constrained than the lower-layer target
- the system relies more strongly on the higher-frequency oxygen-band response, especially `54.4 GHz`
- the upper-layer retrieval is more sensitive to noise and to sample-specific covariance structure

This makes upper-layer performance more fragile and less cleanly separated in the smaller source-specific experiments.

### Limitations of the 3-channel system

The present three-channel system can support meaningful scalar temperature diagnostics, but it is limited. It does not provide a regime-invariant global linear retrieval, and it does not fully resolve vertical structure. It is best understood as a small, interpretable testbed for studying how physically motivated channel choices map into scalar retrieval performance.

## 4. Instrument-design implications

### Role of 52.8 GHz

`52.8 GHz` is the most important channel for lower-layer temperature retrieval. It dominates the lower-layer sensitivity diagnostics, its degradation causes the largest lower-layer performance penalty, and removing it is more damaging than removing `50.3 GHz`.

From an instrument-design perspective, this channel should be prioritized if lower-tropospheric temperature information is a primary objective.

### Role of 54.4 GHz

`54.4 GHz` is the most important channel for upper-layer temperature retrieval. It dominates the upper-layer sensitivity diagnostics, its degradation causes the largest upper-layer performance penalty, and any channel subset without it loses substantial upper-layer skill.

If upper-layer temperature sensitivity is required, this channel is the clearest priority in the current three-channel system.

### Limited role of 50.3 GHz under current targets

Under the current scalar targets, `50.3 GHz` has the weakest unique contribution. It is not useless, but the experiments do not strongly justify it as a core channel for these two targets alone. The ablation and channel-specific noise results both show that its removal or degradation is less damaging than comparable changes to `52.8 GHz` or `54.4 GHz`.

That said, it would be premature to dismiss `50.3 GHz` more generally. Its value may change for humidity-related diagnostics, broader retrieval targets, different regimes, or more realistic uncertainty structures.

### Implications for channel prioritization

A clear design lesson from the current framework is that channel prioritization should not rely only on broad sensitivity arguments or uniform noise assumptions. The experiments here show that:

- lower-layer and upper-layer targets do not weight channels the same way
- channel-specific noise degradation matters
- a channel can be scientifically interesting yet still contribute limited unique retrieval skill for a specific target set

For the present controlled problem, `52.8 + 54.4 GHz` is the minimal strong pair, and preserving the quality of those channels is more important than preserving uniform performance across all three equally.

## 5. Limitations

Several limitations remain important:

- clear-sky only: no clouds, precipitation, or scattering
- synthetic observations: no real measurement/calibration/representativeness error
- small dataset: only a limited number of ERA5-derived profiles across three regimes
- linear retrieval only: no nonlinear or more formal inverse methods
- scalar targets only: results apply to layer-mean temperatures, not full profile retrieval

In addition, the experiments here assess empirically usable information under the current retrieval design, not full radiative-transfer information content in a formal optimal-estimation or Jacobian framework.

## 6. Final conclusion

The current project has demonstrated a coherent and physically interpretable retrieval-analysis framework. Across sensitivity experiments, noise sweeps, channel-specific degradation tests, and channel ablation, the same broad picture emerges:

- lower-layer retrieval is controlled mainly by `52.8 GHz`
- upper-layer retrieval is controlled mainly by `54.4 GHz`
- retrieval behavior is regime-dependent
- added instrument noise degrades performance smoothly and predictably
- the strongest reduced channel subset for the present scalar targets is `52.8 + 54.4 GHz`

What remains unresolved is the extent to which these conclusions hold under broader atmospheric sampling, more realistic observational uncertainty, clouds/scattering, and richer retrieval targets. The current framework does not yet support operational claims, nor does it establish a regime-invariant global retrieval.

What it does provide is a research-grade instrument-analysis workflow: a controlled way to connect clear-sky radiative-transfer behavior, channel sensitivity, instrument noise, and retrieval performance in one consistent, interpretable pipeline. That makes it useful not only as a retrieval prototype, but also as a structured tool for reasoning about channel necessity, channel prioritization, and early-stage instrument trade-offs.
