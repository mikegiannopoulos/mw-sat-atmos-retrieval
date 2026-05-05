# Noise-Sweep Retrieval Interpretation

## Experiment design

This experiment tested how retrieval accuracy changes as instrument noise increases, while keeping the radiative-transfer problem itself fixed. The forward-simulated brightness temperatures (`Tb_true`) were not recomputed or perturbed physically. The channel set remained fixed at `50.3`, `52.8`, and `54.4 GHz`, and the retrieval targets remained fixed at:

- `lower_layer_mean_temperature_k`
- `upper_layer_mean_temperature_k`

The baseline channel-specific `NEΔT` values were scaled by multiplicative factors:

- `0.5`
- `1.0`
- `2.0`
- `3.0`
- `5.0`

For each factor, deterministic Gaussian noise was applied independently by channel and profile using reproducible seeding, so that:

`Tb_obs = Tb_true + noise`

with unchanged `Tb_true` and unchanged channel definitions. The retrieval models were intentionally kept minimal and interpretable:

- mean baseline
- linear regression
- ridge regression

No nonlinear models, no additional predictors, and no changes to PyARTS physics were introduced.

## Main findings

Retrieval error increased smoothly as `NEΔT` increased. This was true for both targets and for both retrieval setups. In the global ridge retrieval, lower-layer RMSE increased from about `1.27 K` at `0.5x` noise to `2.07 K` at `5x` noise. Over the same sweep, upper-layer RMSE increased from about `2.93 K` to `4.28 K`. The normalized RMSE curves showed the same qualitative behavior.

The lower-layer target was consistently less noise-sensitive than the upper-layer target. At all noise factors, both absolute and normalized errors were smaller for `lower_layer_mean_temperature_k` than for `upper_layer_mean_temperature_k`. This indicates that the current three-channel configuration carries more robust information for the lower-layer scalar target than for the upper-layer scalar target.

Source-specific retrieval remained beneficial across the full sweep. For both targets, source-specific ridge retrieval produced lower RMSE than the global ridge retrieval at every `NEΔT` factor. At baseline noise, lower-layer RMSE was about `0.77 K` for the source-specific setup versus about `1.41 K` for the global setup. For the upper-layer target, the corresponding values were about `1.64 K` versus `3.28 K`.

Skill relative to the mean baseline degraded as noise increased, but did not collapse abruptly. For the global ridge retrieval, lower-layer RMSE reduction versus the mean baseline decreased from about `91%` at `0.5x` noise to about `86%` at `5x` noise. For the upper-layer target, the corresponding skill reduction was larger, decreasing from about `56%` to about `36%`. The source-specific setup retained positive skill throughout, although that skill also weakened with increasing noise.

## Scientific interpretation

The lower-layer retrieval is more robust because the current channel set contains stronger and cleaner information about lower-atmospheric temperature structure than about upper-atmospheric structure. This is consistent with the earlier sensitivity experiments, which showed that `52.8 GHz` is especially informative for the lower-layer target and that the lower-layer temperature response is relatively strong compared with channel noise.

The upper-layer retrieval is more fragile because it relies more heavily on the higher-frequency oxygen-band channel behavior, especially `54.4 GHz`, and because the upper-layer scalar target is less directly constrained by the available three-channel observation vector. This makes the upper-layer regression more sensitive to added measurement noise, even when the underlying forward model is unchanged.

Source-specific retrieval remains beneficial because the mapping from `Tb_obs` to scalar temperature target is not regime-invariant in the current dataset. Training within a thermodynamically narrower source reduces regime-mixing and allows the linear model to fit a more coherent local relationship. In this sense, the source-specific setup is not merely a performance trick; it reflects the fact that different atmospheric regimes produce different effective retrieval mappings even under the same clear-sky physics and fixed instrument.

The noise sweep connects instrument noise directly to retrieval uncertainty in a controlled way. Because `Tb_true` was fixed and only `NEΔT` was varied, the resulting degradation can be attributed directly to measurement noise rather than to changing atmospheric states, changing channels, or changing forward-model assumptions. This makes the experiment useful for early-stage instrument trade-off analysis.

## Important caveats

This remains a small-sample experiment. The dataset includes only a limited number of ERA5-derived profiles across three regimes, so the curves should be interpreted as controlled diagnostics rather than stable climatological estimates.

The observations are synthetic noisy realizations generated from forward-model output, not real satellite measurements. The experiment therefore isolates retrieval sensitivity to prescribed instrument noise, but it does not capture broader observational error sources such as calibration biases, representativeness error, or cloud contamination.

The full pipeline remains clear-sky only. No clouds, precipitation, or scattering effects are included, so these results should not be extrapolated to all-weather retrieval conditions.

The profile-level leave-one-out cross-validation used here is not equivalent to leave-one-regime-out generalization. Good behavior under this setup does not imply that the model generalizes well across unseen regimes; earlier cross-regime experiments showed that global linear retrieval can fail badly in that setting.

The source-specific retrieval should therefore be interpreted as a controlled diagnostic of regime-aware behavior, not as an operational regime classifier or a general solution to out-of-distribution retrieval.

## Final conclusion

The success criterion was met. Retrieval performance degraded smoothly and in a physically interpretable way as instrument noise increased, and the experiment clearly linked `NEΔT` assumptions to retrieval uncertainty for both lower-layer and upper-layer scalar temperature targets.

The results strengthen the research-grade pipeline in two ways. First, they confirm that the current retrieval framework responds sensibly to controlled instrument degradation, which is essential for instrument trade-off work. Second, they reinforce the earlier physical interpretation of the channel set: the current three-channel configuration is better suited to the lower-layer target than to the upper-layer target, and regime-aware linear retrieval remains useful even under noisier instrument assumptions.

At the same time, the experiment does not remove the broader limitations already identified in the project. It should therefore be viewed as a validated diagnostic step that improves interpretability and supports future controlled trade-off studies, rather than as evidence of operational retrieval readiness.
