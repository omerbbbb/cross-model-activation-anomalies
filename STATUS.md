# Project status

## Completed

- Initial hidden-state/RMS exploration
- Identification of a dominant-dimension confound
- Per-layer/per-dimension robust anomaly score
- Single-model candidate search and matched controls
- Cross-model adaptive search and held-out transfer tests
- Fixed 10,000-string candidate-pool search on five model families
- Repetition-vs-pattern experiment on seven model families
- Token-count-matched controls
- BLOOM float16 NaN diagnostic
- Full BLOOM repetition retest in float32

## Current finding

Repeated structured strings can be high-percentile activation outliers in several model families, but the effect is not universal.

Falcon shows a particularly strong response to the repeated `%_B%W` pattern. Corrected BLOOM float32 results behave very differently.

## Active work

Investigate the BLOOM discrepancy:

- numerical precision
- tokenizer segmentation
- architecture / normalization behavior
- winning layer and hidden dimension
- sensitivity of the current max-activation anomaly metric

## Not claimed

- No universal LLM vulnerability
- No causal mechanism yet
- No claim that `%_B%W` is a universal trigger
- No claim that the anomaly metric is a standard interpretability measure
