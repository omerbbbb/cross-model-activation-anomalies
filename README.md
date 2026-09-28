# Cross-Model Activation Anomaly Search

**Exploratory pilot:** short repeated-pattern strings (for example `%_B%W` repeated) often receive unusually high hidden-state activation-anomaly scores relative to token-count-matched controls in several small open language models (0.1–1.1B parameters), but the effect is not consistent across all models and still needs stronger confirmation.

![Repeated pattern percentiles](assets/repetition_target_percentiles.png)

## Motivation

I started this project from an independent intuition: a human can often read a short repeated character pattern as one simple visual/structural object, while a language model receives a tokenizer-dependent sequence of discrete tokens. I wanted to test whether inputs that look simple as a whole to a person could nevertheless be internally unusual for a model because of this representation gap.

**Tokenization is currently a hypothesis, not an established explanation.** The experiments below measure hidden-state activation anomalies and match controls by token count, but they do not yet directly test how tokenizer segmentation causes the effect. Direct tokenizer analysis is listed under Future work.

> **Status: exploratory pilot, not a confirmed result.** No security vulnerability, universal trigger, or causal mechanism is claimed.

## Related work

I began this project independently and only afterwards learned that the question sits near several established lines of LLM research:

- **Sun et al. (2024), _Massive Activations in Large Language Models_** — reports rare hidden activations with magnitudes far larger than typical activations and studies their locations and functional role across LLMs.  
  Paper: https://arxiv.org/abs/2402.17762

- **Xiao et al. (2023), _Efficient Streaming Language Models with Attention Sinks_** — identifies "attention sinks": tokens, especially early tokens, that can receive unusually strong attention even when they are not semantically important. This is related background, but it is not the same quantity measured in this repository.  
  Paper: https://arxiv.org/abs/2309.17453

The measurement goal here is different from characterizing massive activations or attention sinks themselves. This project **searches for literal input strings** that produce high activation-anomaly scores across different model families, then uses locked candidates, held-out model families, and token-count-matched controls to test whether the effect survives beyond the models used to find it.

## Research question

Can the same input structure produce unusually large internal activations across independently trained language models — and, if so, what property of the input causes the effect?

The project started as a search for individual cross-model anomalous strings. The current focus is **mechanism**: repetition, tokenization, architecture, numerical precision, and the anomaly metric itself.

## Experimental pipeline

```text
candidate generation
        ↓
per-model activation baselines
        ↓
cross-model scoring
        ↓
locked candidates
        ↓
held-out model evaluation
        ↓
token-count-matched controls
        ↓
mechanism / confound analysis
```

A major theme of the project is that several apparently exciting results became weaker or changed interpretation once stronger controls were added.

## Experiment 1 — 10,000-string cross-model search

10,000 candidate strings were generated **before model scoring** and evaluated on five model families:

- Qwen2.5-0.5B
- TinyLlama-1.1B
- Pythia-410M
- OPT-350M
- Falcon-RW-1B

Candidates were ranked by their **worst empirical percentile** across the five models.

The top search candidate was:

```text
%_B%W%_B%W%_B%W%_B%W%_B%W%_B%W%_B%W%_B%W%_B%W%_B%W%_B%W%_B%W
```

| Search model | Percentile |
|---|---:|
| Qwen2.5-0.5B | 98.16 |
| TinyLlama-1.1B | 96.66 |
| Pythia-410M | 97.01 |
| OPT-350M | 95.33 |
| Falcon-RW-1B | 98.86 |

**Selection caveat:** these five percentiles are **in-sample with respect to candidate selection**: the same five models were used to rank and choose the candidate. They show that the search objective succeeded on its search models; they are not independent validation. The held-out model families are the fairer transfer test.

The candidate was locked before testing additional models. GPT-Neo was an unseen holdout in that stage. BLOOM was also intended as a holdout, but its original float16 result was later invalidated by a numerical issue described below.

### Important BLOOM correction

An initial BLOOM holdout run appeared extremely strong, but later diagnostics showed that **BLOOM produced NaNs in the float16 measurement pipeline**. That earlier BLOOM result is retained only as experiment history and should **not** be treated as reliable evidence.

This discovery motivated a stricter mechanism experiment and a BLOOM float32 retest.

## Experiment 2 — repetition vs. specific pattern

The next experiment tested whether the effect was caused by:

1. repetition in general,
2. the specific `%_B%W` pattern,
3. the same characters shuffled,
4. random repeated blocks,
5. or unusual/random ASCII.

The experiment used:

- 7 model families,
- 120 normal baseline texts per model,
- 160 experimental strings,
- a 3,000-string broad control pool,
- token-count-matched percentile comparisons.

Models:

- Qwen
- TinyLlama
- Pythia
- OPT
- Falcon
- BLOOM
- GPT-Neo

No adaptive search or optimization was used in this experiment.

## Current result

The specific `%_B%W` repetition pattern is **not a universal cross-model anomaly**.

Its token-matched median percentiles are:

| Repetitions | Qwen | TinyLlama | Pythia | OPT | Falcon | BLOOM* | GPT-Neo |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 61.7 | 62.7 | 45.8 | 89.4 | 100.0 | 53.0 | 51.1 |
| 2 | 40.6 | 92.3 | 83.6 | 82.5 | 96.8 | 15.6 | 100.0 |
| 4 | 92.2 | 89.7 | 100.0 | 75.3 | 100.0 | 10.1 | 96.3 |
| 8 | 82.1 | 98.3 | 95.6 | 42.9 | 97.4 | 15.1 | 88.3 |
| 12 | 69.8 | 89.8 | 94.1 | 87.0 | 97.8 | 11.8 | 87.0 |

\* BLOOM values are from the corrected **float32** retest.

> **Precision caveat:** the historical seven-model repetition run used float16 for Qwen, TinyLlama, Pythia, OPT, Falcon, BLOOM, and GPT-Neo, and at that time the code did not explicitly reject NaN/Inf activations. BLOOM was the model in which a float16 NaN artifact was later discovered, so BLOOM alone was rerun in float32. The other six models have **not yet been float32-confirmed**. Extreme pilot values such as `100.0` should therefore not be overinterpreted until that confirmation is done. The maintained scoring code now raises an error if non-finite hidden activations are detected.

The pattern is consistently extreme in Falcon in this pilot and often high in several other model families, while corrected BLOOM behaves very differently. Other repeated/control families also become high-percentile in some architectures.

The evidence therefore currently supports a cautious working interpretation:

> **Repeated structure may interact with model/tokenizer architecture in ways that affect the current activation-anomaly score, but the mechanism is not established and the effect is not universal.**

## Active investigation: why is BLOOM different?

This is the current research focus.

BLOOM initially behaved pathologically in float16, producing non-finite values. A diagnostic rerun in float32 showed finite hidden states and a completely finite baseline.

The full BLOOM repetition experiment was then repeated in float32 with the same:

- 120-text baseline,
- 160 experiment strings,
- 3,000-string control pool,
- random seeds,
- token-count matching procedure.

For `%_B%W`, the corrected BLOOM percentiles fell to roughly 10–16% for 2–12 repetitions.

I am currently investigating **why BLOOM differs from the other model families**, including whether the discrepancy is driven by:

- numerical precision,
- tokenizer behavior,
- architectural differences,
- layer/dimension behavior,
- or limitations of the current anomaly score.

This is ongoing work; the repository intentionally records the discrepancy instead of smoothing it away.

## Activation anomaly metric

For each model:

1. Run normal baseline texts and collect hidden states.
2. For every layer and hidden dimension, record the maximum absolute activation over non-padding token positions.
3. Estimate the baseline median and 95th percentile for each layer/dimension.
4. Use a robust scale:

```text
scale = max(q95 - median, 0.1 * median, 1e-3)
```

5. Candidate dimension score:

```text
(candidate_max - q95) / scale
```

6. The exploratory anomaly score is the maximum standardized score across non-embedding hidden states and hidden dimensions.

This is an **exploratory heuristic**, not a probability, z-score, or established interpretability benchmark.

## What changed during the research

The repo deliberately preserves failed and corrected approaches.

### Early RMS result

Raw RMS appeared to expose very large activations. Inspection showed that stable, model-specific dimensions could dominate the measurement.

### Per-dimension baseline

The metric was changed to compare each layer/dimension only against its own normal-text distribution.

### Flawed cross-model normalization

An early joint search normalized raw scores by a calibration quantile. This could make a model look strong even when its empirical percentile was ordinary.

It was replaced by actual empirical percentiles.

### Search-set overfitting

A string optimized on several models transferred poorly to Falcon, demonstrating why held-out model families matter.

### Fixed 10,000-string search

The search procedure was replaced by a pre-generated candidate pool scored identically across five model families.

### Repetition follow-up

Top candidates were dominated by repeated short patterns, so the project shifted from "find a magic string" to controlled tests of repetition and token structure.

### BLOOM numerical failure

A later control experiment exposed float16 NaNs in BLOOM. The model was rerun in float32, which changed the interpretation substantially.

## Repository layout

```text
.
├── README.md
├── STATUS.md
├── RESEARCH_NOTES.md
├── SECURITY.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   ├── research_log.ipynb
│   └── research_log_clean.ipynb
├── src/cross_model_anomalies/
│   ├── candidates.py
│   ├── data.py
│   ├── modeling.py
│   └── scoring.py
├── scripts/
│   └── run_cross_model_search.py
├── experiments/
│   ├── repetition_vs_pattern.py
│   ├── bloom_float32_diagnostic.py
│   └── bloom_float32_retest.py
├── results/
│   ├── top20_search.csv
│   ├── initial_holdout_results_legacy.csv
│   ├── initial_cross_model_results_legacy.csv
│   ├── repetition_family_summary_corrected.csv
│   ├── repetition_target_percentiles_corrected.csv
│   ├── repetition_cross_model_corrected.csv
│   └── bloom_float32_retest.csv
└── assets/
    └── repetition_target_percentiles.png
```

## Reproducing the completed cross-model search

A CUDA GPU is strongly recommended.

```bash
git clone https://github.com/omerbbbb/cross-model-activation-anomalies
cd cross-model-activation-anomalies

python -m venv .venv
source .venv/bin/activate

pip install -e .
python scripts/run_cross_model_search.py
```

For a smaller smoke test:

```bash
python scripts/run_cross_model_search.py \
  --n-baseline 40 \
  --n-candidates 500 \
  --n-holdout-controls 200 \
  --top-k 5
```

The later experimental scripts are preserved separately because they correspond to evolving research questions rather than one frozen benchmark.

## Notebook

`notebooks/research_log.ipynb` is the latest Colab research log with outputs preserved.

It includes the progression from initial RMS measurements through:

- baseline debugging,
- single-model search,
- failed transfer,
- corrected multi-model search,
- fixed candidate-pool search,
- repetition controls,
- BLOOM NaN diagnosis,
- and the float32 BLOOM retest.

`research_log_clean.ipynb` contains the same code with cell outputs removed for easier inspection.

## Limitations

- This is an exploratory pilot, not a confirmatory study.
- The anomaly score is a custom heuristic based on a global maximum.
- The candidate generator defines the population used for search percentiles.
- The five Experiment 1 search-model percentiles are selection-set results, not independent validation.
- Several experiments use relatively small baseline samples.
- Token-count matching reduces one confound but does not establish causality.
- The matched-control set for a given string can be only **tens of examples**; in the corrected BLOOM retest it ranged from **33 to 88** controls across the five `%_B%W` conditions.
- No confidence intervals are reported yet.
- No correction for multiple comparisons has been applied.
- Six non-BLOOM model families in the repetition table remain float16 pilot results without a float32 confirmation run.
- The original BLOOM float16 holdout result is invalid as confirmatory evidence.
- Models are relatively small open-weight LMs (approximately 0.1–1.1B parameters).
- Tokenizer segmentation is a proposed mechanism, not something directly established by the current experiments.
- The mechanism behind the architecture-dependent repetition effect is unresolved.
- Repeated runs, stronger statistical tests, and additional held-out architectures are still needed.

## Future work

The next planned checks are:

- **Float32 confirmation across all models** — rerun Qwen, TinyLlama, Pythia, OPT, Falcon, and GPT-Neo with explicit finiteness checks.
- **Direct tokenizer analysis** — record exactly how each model splits the top strings and test activation score against metrics such as tokens per character, token-boundary density, and repeated-token structure.
- **Localization of the anomaly** — log the layer, token position, and hidden dimension responsible for each winning score rather than retaining only the global maximum.
- **Uncertainty and significance** — add bootstrap confidence intervals and/or permutation tests.
- **Multiple-comparison handling** — predefine primary comparisons or apply an appropriate correction when testing many string families/conditions.
- **Additional held-out architectures** — replicate on further model families not involved in candidate selection.

## Not claimed

- No universal LLM vulnerability.
- No confirmed causal role for tokenization.
- No universal `%_B%W` trigger.
- No claim that the anomaly metric is a standard interpretability measure.
- No claim that the current pilot establishes a new general phenomenon.

## Security

The maintained code and public notebooks use `trust_remote_code=False`. No API keys or private credentials are required. See [`SECURITY.md`](SECURITY.md) for the public-release security notes and safe-running guidance.
