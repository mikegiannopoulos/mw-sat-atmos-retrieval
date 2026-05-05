# Channel-Specific Noise Sensitivity Interpretation

## Experiment design

This experiment isolated the effect of degrading individual channels while keeping the underlying radiative-transfer signal fixed. The forward-simulated `Tb_true` values were unchanged, the channel set remained fixed at `50.3`, `52.8`, and `54.4 GHz`, and the scalar retrieval targets remained:

- `lower_layer_mean_temperature_k`
- `upper_layer_mean_temperature_k`

Only one channel `NEΔT` was scaled at a time. The other two channels were kept at their baseline `NEΔT` values. The multiplicative scaling factors applied to the degraded channel were:

- `1.0`
- `2.0`
- `3.0`
- `5.0`

For each scenario, deterministic Gaussian noise was applied using reproducible per-profile seeding. The retrieval models were unchanged from the existing pipeline:

- mean baseline
- linear regression
- ridge regression

No nonlinear models, no new predictors, and no changes to PyARTS physics were introduced.

## Main findings

The lower-layer retrieval was most sensitive to degradation of `52.8 GHz`. In the global ridge setup, increasing noise on `52.8 GHz` produced the largest increase in lower-layer RMSE, larger than the corresponding degradation applied to either `50.3 GHz` or `54.4 GHz`.

The upper-layer retrieval was most sensitive to degradation of `54.4 GHz`. In the global ridge setup, increasing noise on `54.4 GHz` caused by far the largest upper-layer RMSE increase. The effect of degrading `52.8 GHz` was smaller, and degrading `50.3 GHz` had almost no effect on upper-layer performance.

The `50.3 GHz` channel had weak influence on the upper-layer retrieval. Its degradation produced almost no change in global upper-layer RMSE across the full factor sweep, which is consistent with the earlier sensitivity and channel-dropout experiments.

The source-specific results were less clearly separated than the global results, especially for the upper-layer target. Those source-specific differences remain directionally useful, but they should be interpreted cautiously because the within-source sample size is limited and the source-specific upper-layer mapping is less decisively constrained than the pooled global comparison.

## Scientific interpretation

These results are consistent with the earlier oxygen-band sensitivity analysis. The three-channel set samples different parts of the oxygen absorption structure, so degrading one channel does not simply remove all information about one atmospheric layer. Instead, each channel contributes overlapping but differently weighted information about temperature structure.

The `52.8 GHz` channel affects lower-layer retrieval most because it sits in the part of the oxygen band that showed the strongest lower-atmospheric temperature sensitivity in the previous perturbation experiments. When only this channel becomes noisier, the lower-layer scalar retrieval loses its most informative lower/mid-tropospheric constraint, so retrieval error increases most strongly.

The `54.4 GHz` channel affects upper-layer retrieval most because it showed the strongest upper-atmospheric temperature sensitivity in the earlier layer perturbation diagnostics. As its noise increases, the retrieval loses the channel that most directly supports the upper-layer scalar target, so upper-layer RMSE grows most steeply.

The channel effects are overlapping rather than perfectly isolated because microwave temperature sounding in the oxygen band is inherently distributed across channels. Even when one channel is degraded, the remaining channels still carry partial information about the same target. This is why the degradation curves are smooth and why no single channel completely determines either retrieval target on its own.

## Instrument-design implication

For the current three-channel configuration, improving or stabilizing `52.8 GHz` provides the clearest benefit for lower-layer temperature retrieval. That channel is the most important single contributor to lower-layer retrieval robustness under the present clear-sky setup.

Improving or stabilizing `54.4 GHz` provides the clearest benefit for upper-layer temperature retrieval. Its degradation causes the strongest upper-layer retrieval penalty, so it is the most critical channel for preserving upper-layer sensitivity in this channel set.

This experiment also shows that uniform `NEΔT` assumptions can hide channel-specific weaknesses. A bulk noise sweep is useful, but it does not reveal which part of the instrument is driving retrieval degradation. The one-channel-at-a-time experiment therefore adds important structure to the instrument trade-off interpretation.

## Caveats

This remains a small-sample experiment based on a limited set of ERA5-derived profiles and a few synthetic stress-test profiles elsewhere in the pipeline. The quantitative sensitivity rankings should therefore be interpreted as controlled diagnostics, not as final climatological results.

The experiment is clear-sky only. No clouds, scattering, or precipitation effects are included, so channel roles under all-weather conditions are outside the scope of this result.

The observations are synthetic noisy realizations of forward-model output rather than real satellite measurements. This isolates the measurement-noise effect cleanly, but it does not include broader observational error sources.

The retrievals remain linear. That is appropriate for the current interpretability-first phase, but it also means that some regime-dependent or nonlinear channel interactions may be compressed into a simple linear mapping.

The source-specific retrieval should be interpreted as a diagnostic within-regime comparison, not as an operationally deployable regime-aware retrieval system.

## Final conclusion

The experiment confirms channel-specific noise sensitivity in a physically consistent way. Lower-layer retrieval is most vulnerable to degradation of `52.8 GHz`, while upper-layer retrieval is most vulnerable to degradation of `54.4 GHz`. The weak influence of `50.3 GHz` on upper-layer retrieval is also consistent with the earlier vertical-sensitivity results.

This strengthens the project’s instrument trade-off framework by linking retrieval degradation not only to overall instrument noise, but also to which specific channel is degraded. That makes the current pipeline more useful for research-grade design reasoning: it can now distinguish between bulk noise sensitivity and channel-specific performance bottlenecks while staying within a simple, interpretable, physics-first retrieval framework.
