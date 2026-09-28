# Research notes

This file summarizes the experimental reasoning behind the repository.

## 1. Initial question

Can a character string produce unusually large hidden-state activations across models with different weights and tokenizers?

## 2. First false lead

Large RMS values appeared in Qwen. Inspecting the responsible activation showed that one model-specific dimension could dominate the result. Raw global RMS was therefore not a useful cross-model anomaly measure.

## 3. Robust per-dimension score

The experiment moved to per-layer/per-dimension normal-text baselines using a median, 95th percentile and robust scale.

## 4. Single-model candidate

A searched Qwen string scored strongly against matched controls but failed to transfer to TinyLlama. This demonstrated that a model-specific search can overfit the activation landscape.

## 5. Adaptive multi-model search

A search over Qwen, TinyLlama and Pythia initially used a flawed normalized raw score. Empirical percentiles exposed the problem, and the objective was corrected.

A later candidate looked strong on the three search models and moderately strong on OPT, but failed on Falcon. Again, transfer was incomplete.

## 6. Fixed candidate-pool experiment

To reduce adaptive-search bias, 10,000 candidate strings were generated before model scoring and evaluated on Qwen, TinyLlama, Pythia, OPT and Falcon.

The highest-ranked strings were dominated by repeated short blocks.

## 7. Repetition-vs-pattern experiment

A new fixed experiment compared:

- `%_B%W` repeated,
- `abcde` repeated,
- the same characters shuffled,
- random repeated blocks,
- random ASCII,

across repetition counts 1, 2, 4, 8 and 12.

The experiment used 120 normal baseline texts, 160 experimental strings and 3,000 token-count-matched controls for each model.

## 8. BLOOM numerical issue

The first seven-model run used float16 and BLOOM produced NaNs. This made the original BLOOM percentiles unreliable.

BLOOM was then:

1. diagnosed in float32,
2. shown to have finite hidden states and a finite baseline,
3. rerun on the complete repetition experiment in float32.

The corrected result was qualitatively different: `%_B%W` was not high-percentile in BLOOM for 2–12 repetitions.

## 9. Current question

Why does the repeated pattern produce strong responses in some model families — especially Falcon — while corrected BLOOM behaves differently?

Possible explanations under investigation include tokenizer segmentation, architecture, numerical precision, normalization, and properties of the anomaly metric itself.

The discrepancy is currently treated as a research result to explain, not as noise to remove.
