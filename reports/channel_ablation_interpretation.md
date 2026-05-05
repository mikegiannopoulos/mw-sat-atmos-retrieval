# Channel Ablation Interpretation

## Experiment design

This experiment tested which subset of the existing three-channel configuration is sufficient for the current scalar retrieval targets. The dataset was unchanged, the observation vectors used the existing `Tb_obs` values with the baseline instrument-noise setup, and the retrieval targets remained:

- `lower_layer_mean_temperature_k`
- `upper_layer_mean_temperature_k`

The same three simple models were used throughout:

- mean baseline
- linear regression
- ridge regression

All channel subsets were evaluated:

- 3-channel: `50.3 + 52.8 + 54.4 GHz`
- 2-channel: `50.3 + 52.8`, `50.3 + 54.4`, `52.8 + 54.4 GHz`
- 1-channel: `50.3`, `52.8`, `54.4 GHz`

Two evaluation setups were retained:

- global leave-one-profile-out cross-validation across all ERA5 profiles
- source-specific leave-one-profile-out cross-validation within each `source_name`

No changes were made to the forward model, the instrument definition, or the retrieval methodology beyond channel removal.

## Main findings

The `52.8 + 54.4 GHz` pair was the strongest reduced channel combination. In both the lower-layer and upper-layer global ridge results, this pair retained nearly all of the skill of the full three-channel setup.

Lower-layer retrieval depended strongly on `52.8 GHz`. Removing it produced a larger penalty than removing `50.3 GHz`, and a single-channel `52.8 GHz` retrieval was substantially stronger than a single-channel `54.4 GHz` retrieval for the lower-layer target.

Upper-layer retrieval depended strongly on `54.4 GHz`. Any global upper-layer configuration without `54.4 GHz` degraded substantially, while `54.4 GHz` alone still outperformed the single-channel `50.3` and `52.8 GHz` cases.

The `50.3 GHz` channel contributed the least unique information for these two scalar retrieval targets. Removing it from the full set produced only a small penalty, and two-channel combinations without `50.3 GHz` remained very strong.

The source-specific upper-layer results were less clean than the global results and should be interpreted cautiously. Several reduced configurations clustered closely, and some differences were small enough that they are likely sensitive to limited within-source sample size and covariance structure.

## Scientific interpretation

These results are consistent with the earlier oxygen-band sensitivity diagnostics. The three channels do not measure fully separate pieces of atmospheric information; rather, they sample overlapping but differently weighted temperature sensitivity across the oxygen band.

The `52.8 GHz` channel is central for lower-layer retrieval because it occupies the part of the oxygen band that previously showed the strongest lower-atmospheric temperature sensitivity relative to noise. When that channel is available, the retrieval retains its main lower-layer constraint. When it is absent, lower-layer skill drops noticeably.

The `54.4 GHz` channel is essential for upper-layer retrieval because it previously showed the strongest upper-atmospheric temperature sensitivity. Removing it weakens the observation vector’s access to upper-layer temperature structure more than removing the other channels does.

The `52.8 + 54.4 GHz` pair retains most skill because it combines the strongest lower/mid-level temperature-sensitive channel with the strongest upper-level temperature-sensitive channel. Under the current scalar-target setup, that pairing captures most of the practically useful information in the full three-channel configuration.

Cases where a reduced channel set slightly outperforms the full set should not be overinterpreted. In a small-sample linear retrieval experiment, small differences of that kind can arise from covariance structure, mild redundancy, and estimation noise rather than from a definitive physical argument that the removed channel is harmful. The broader pattern is more important than tiny numerical reversals.

## Instrument-design implication

Under the current clear-sky, three-channel, scalar-target setup, the minimal strong channel pair is `52.8 + 54.4 GHz`. That pair preserves most of the useful retrieval skill for both lower-layer and upper-layer mean temperature.

The `50.3 GHz` channel is not strongly justified by these two scalar retrieval targets alone. It may still have value for other targets, other regimes, robustness, or future retrieval formulations, but its unique contribution here is smaller than that of `52.8 GHz` and `54.4 GHz`.

Final channel-design conclusions should still be deferred until broader retrieval targets, broader atmospheric regimes, and more realistic uncertainty sources are included. This experiment identifies a strong reduced subset for the current controlled problem, not a universally optimal operational design.

## Caveats

This remains a small-sample study based on a limited set of ERA5-derived profiles and synthetic forward-model observations. The quantitative rankings should therefore be interpreted as controlled diagnostics rather than stable final design numbers.

The experiment is clear-sky only. It does not include clouds, precipitation, or scattering, so the channel roles here apply only within the current clear-sky framework.

The observations are synthetic and model-based rather than real satellite measurements. That is appropriate for controlled interpretation, but it omits many practical observational error sources.

The retrieval models are linear only. This is intentional for interpretability, but it also means the channel rankings reflect what is usable in a simple linear mapping, not necessarily the maximum possible performance under a more complex retrieval.

The targets are scalar only. These conclusions apply to lower-layer and upper-layer mean temperature, not to full profile retrieval or to broader state-vector estimation.

Finally, the ablation experiment tests empirically usable information under the current retrieval setup, not full radiative-transfer information content in a formal inverse-theory sense.

## Final conclusion

The experiment does identify a minimal strong channel subset for the current problem. The most robust reduced configuration is `52.8 + 54.4 GHz`, with `52.8 GHz` carrying the most important lower-layer temperature information and `54.4 GHz` carrying the most important upper-layer temperature information.

This closes the loop between the earlier sensitivity, noise, and retrieval experiments. The vertical sensitivity diagnostics indicated which channels should matter most; the channel-specific noise experiment showed which degraded channels hurt each target most; and the ablation experiment now shows which channels are actually necessary to retain most of the usable retrieval skill. Together, these results strengthen the project’s instrument trade-off framework while remaining within a simple, interpretable, physics-first pipeline.
