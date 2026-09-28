# Project status

## Status

**Exploratory pilot, not a confirmed result.**

## Completed

- Initial hidden-state/RMS exploration
- Identification of a dominant-dimension confound
- Per-layer/per-dimension robust anomaly score
- Single-model candidate search and matched controls
- Cross-model adaptive search and held-out transfer tests
- Fixed 10,000-string candidate-pool search on five model families
- Repetition-vs-pattern pilot on seven model families
- Token-count-matched controls
- BLOOM float16 NaN diagnostic
- Full BLOOM repetition retest in float32
- Maintained scoring code now rejects NaN/Inf hidden activations

## Current observation

Repeated structured strings can rank highly under the current activation-anomaly metric in several model families, but the effect is not universal and six non-BLOOM repetition results still need float32 confirmation.

Corrected BLOOM float32 results behave very differently from several of the other model families.

## Active work

Investigate the BLOOM discrepancy and the possible role of:

- numerical precision
- tokenizer segmentation
- architecture / normalization behavior
- winning layer, token position, and hidden dimension
- sensitivity of the current global-max anomaly metric

## Planned validation

- float32 rerun of all remaining models
- direct tokenizer analysis
- layer/position/dimension logging
- confidence intervals / permutation tests
- multiple-comparison handling
- additional held-out architectures

## Not claimed

- No universal LLM vulnerability
- No confirmed causal role for tokenization
- No universal `%_B%W` trigger
- No established causal mechanism
- No claim that the anomaly metric is a standard interpretability benchmark
