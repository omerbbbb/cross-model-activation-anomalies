# Completed seven-model repetition-vs-pattern experiment.
# Extracted from notebooks/research_log.ipynb.

# ================================================================
# REPETITION vs SPECIFIC TOKEN PATTERN
#
# Question:
# Is the cross-model anomaly caused by:
#
#   A) repetition in general?
#   B) the specific pattern "%_B%W"?
#   C) merely unusual/random characters?
#
# Models:
#   Qwen
#   TinyLlama
#   Pythia
#   OPT
#   Falcon
#   BLOOM
#   GPT-Neo
#
# No search / optimization in this experiment.
# Everything is generated BEFORE seeing model results.
# ================================================================


import gc
import random
import string
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ================================================================
# 0. SETTINGS
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected. In Colab choose "
        "Runtime -> Change runtime type -> T4 GPU."
    )

DEVICE = "cuda"
DTYPE = torch.float16

REPEAT_COUNTS = [1, 2, 4, 8, 12]

TARGET_BLOCK = "%_B%W"
PLAIN_BLOCK = "abcde"

# Number of different shuffled/random examples per condition.
N_REPLICATES = 10

# Normal-text baseline used to define the activation anomaly.
N_BASELINE = 120

# Broad weird-string pool used ONLY for token-count-matched controls.
N_CONTROL_POOL = 3000

MAX_LENGTH = 96
BATCH_SIZE = 16

random.seed(28092026)
np.random.seed(28092026)
torch.manual_seed(28092026)


MODELS = [
    {
        "name": "Qwen",
        "repo": "Qwen/Qwen2.5-0.5B",
        "trust_remote_code": False,
    },
    {
        "name": "TinyLlama",
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "trust_remote_code": False,
    },
    {
        "name": "Pythia",
        "repo": "EleutherAI/pythia-410m",
        "trust_remote_code": False,
    },
    {
        "name": "OPT",
        "repo": "facebook/opt-350m",
        "trust_remote_code": False,
    },
    {
        "name": "Falcon",
        "repo": "tiiuae/falcon-rw-1b",

        # Native Transformers implementation.
        # This is the version that worked in our previous run.
        "trust_remote_code": False,
    },
    {
        "name": "BLOOM",
        "repo": "bigscience/bloom-560m",
        "trust_remote_code": False,
    },
    {
        "name": "GPT-Neo",
        "repo": "EleutherAI/gpt-neo-125m",
        "trust_remote_code": False,
    },
]


SYMBOLS = "!@#$%^&*()[]{}<>?/\\|+=_-:;."
PRINTABLE = string.ascii_letters + string.digits + SYMBOLS


# ================================================================
# 1. LOAD NORMAL TEXTS FOR BASELINE
# ================================================================

print("Loading dataset...")

dataset = load_dataset(
    "fancyzhx/ag_news",
    split="train"
).shuffle(seed=8127)


def clean_normal_text(text):

    text = " ".join(
        text.replace("\n", " ").split()
    )

    # Keep lengths in roughly the same regime as our test strings.
    return text[:70]


baseline_texts = []

for row in dataset:

    text = clean_normal_text(
        row["text"]
    )

    if len(text) >= 20:

        baseline_texts.append(
            text
        )

    if len(baseline_texts) >= N_BASELINE:
        break


print(
    "Baseline texts:",
    len(baseline_texts)
)


# ================================================================
# 2. DEFINE TEST STRINGS BEFORE LOADING MODELS
# ================================================================

test_cases = []


def add_case(
    family,
    repetitions,
    replicate,
    text,
):

    test_cases.append(
        {
            "family": family,
            "repetitions": repetitions,
            "replicate": replicate,
            "text": text,
        }
    )


for repetitions in REPEAT_COUNTS:

    # ------------------------------------------------------------
    # 1. The pattern discovered in our previous experiment
    # ------------------------------------------------------------

    target = (
        TARGET_BLOCK
        * repetitions
    )

    add_case(
        "TARGET_%_B%W",
        repetitions,
        0,
        target,
    )


    # ------------------------------------------------------------
    # 2. Completely ordinary repeated block
    # ------------------------------------------------------------

    plain = (
        PLAIN_BLOCK
        * repetitions
    )

    add_case(
        "PLAIN_abcde",
        repetitions,
        0,
        plain,
    )


    # ------------------------------------------------------------
    # 3. SAME EXACT CHARACTERS as target,
    #    but shuffle them and destroy periodic structure
    # ------------------------------------------------------------

    for rep in range(
        N_REPLICATES
    ):

        rng = random.Random(
            100000
            + repetitions * 100
            + rep
        )

        chars = list(
            target
        )

        rng.shuffle(
            chars
        )

        shuffled = "".join(
            chars
        )

        # Make sure we did not accidentally reconstruct target.
        if shuffled == target:

            chars.reverse()

            shuffled = "".join(
                chars
            )

        add_case(
            "SHUFFLED_SAME_CHARS",
            repetitions,
            rep,
            shuffled,
        )


    # ------------------------------------------------------------
    # 4. Random block of SAME LENGTH as "%_B%W",
    #    repeated the same number of times.
    #
    # Tests whether repetition itself is enough.
    # ------------------------------------------------------------

    for rep in range(
        N_REPLICATES
    ):

        rng = random.Random(
            200000
            + repetitions * 100
            + rep
        )

        block = "".join(
            rng.choice(
                PRINTABLE
            )
            for _ in range(
                len(TARGET_BLOCK)
            )
        )

        repeated = (
            block
            * repetitions
        )

        add_case(
            "RANDOM_BLOCK_REPEAT",
            repetitions,
            rep,
            repeated,
        )


    # ------------------------------------------------------------
    # 5. Random characters — same CHARACTER LENGTH,
    #    but no repeated block.
    # ------------------------------------------------------------

    for rep in range(
        N_REPLICATES
    ):

        rng = random.Random(
            300000
            + repetitions * 100
            + rep
        )

        random_text = "".join(
            rng.choice(
                PRINTABLE
            )
            for _ in range(
                len(target)
            )
        )

        add_case(
            "RANDOM_ASCII",
            repetitions,
            rep,
            random_text,
        )


test_df = pd.DataFrame(
    test_cases
)


print(
    "\nTest strings:",
    len(test_df)
)


print(
    "\n===================================="
)
print(
    "EXAMPLES"
)
print(
    "===================================="
)


for repetitions in REPEAT_COUNTS:

    print(
        f"\n--- {repetitions} repetitions ---"
    )

    for family in [
        "TARGET_%_B%W",
        "PLAIN_abcde",
        "SHUFFLED_SAME_CHARS",
        "RANDOM_BLOCK_REPEAT",
        "RANDOM_ASCII",
    ]:

        example = test_df[
            (
                test_df["repetitions"]
                == repetitions
            )
            &
            (
                test_df["family"]
                == family
            )
        ].iloc[0]

        print(
            f"{family:22s}",
            repr(
                example["text"]
            )
        )


# ================================================================
# 3. CREATE BROAD CONTROL POOL
#
# IMPORTANT:
# This is generated before any model results.
#
# We later select controls with the SAME TOKEN COUNT
# as each experimental string.
# ================================================================

control_pool = []

seen_controls = set()


def add_control(text):

    text = text.strip()

    if (
        len(text) >= 3
        and text not in seen_controls
    ):

        seen_controls.add(
            text
        )

        control_pool.append(
            text
        )


control_index = 0


while len(
    control_pool
) < N_CONTROL_POOL:

    rng = random.Random(
        900000 + control_index
    )

    control_index += 1

    mode = rng.randrange(
        4
    )


    # ------------------------------------------------------------
    # random ASCII
    # ------------------------------------------------------------

    if mode == 0:

        length = rng.randint(
            4,
            65
        )

        text = "".join(
            rng.choice(
                PRINTABLE
            )
            for _ in range(
                length
            )
        )


    # ------------------------------------------------------------
    # random repeated block
    # ------------------------------------------------------------

    elif mode == 1:

        block_length = rng.randint(
            2,
            8
        )

        block = "".join(
            rng.choice(
                PRINTABLE
            )
            for _ in range(
                block_length
            )
        )

        repeats = rng.randint(
            1,
            15
        )

        text = (
            block
            * repeats
        )[:70]


    # ------------------------------------------------------------
    # letters / digits / punctuation mixture
    # ------------------------------------------------------------

    elif mode == 2:

        chunks = []

        for _ in range(
            rng.randint(
                2,
                12
            )
        ):

            chunk = "".join(
                rng.choice(
                    PRINTABLE
                )
                for _ in range(
                    rng.randint(
                        1,
                        7
                    )
                )
            )

            chunks.append(
                chunk
            )

        text = rng.choice(
            [
                "",
                " ",
                "_",
                "-",
                ".",
            ]
        ).join(
            chunks
        )[:70]


    # ------------------------------------------------------------
    # repeated normal-looking strings
    # ------------------------------------------------------------

    else:

        alphabet = (
            string.ascii_letters
        )

        block = "".join(
            rng.choice(
                alphabet
            )
            for _ in range(
                rng.randint(
                    3,
                    8
                )
            )
        )

        text = (
            block
            * rng.randint(
                1,
                15
            )
        )[:70]


    add_control(
        text
    )


print(
    "\nMatched-control pool:",
    len(control_pool)
)


# ================================================================
# 4. LOAD MODEL
# ================================================================

def load_model(config):

    print(
        "\n\n========================================"
    )

    print(
        "LOADING:",
        config["name"]
    )

    print(
        config["repo"]
    )

    print(
        "========================================"
    )


    tokenizer = AutoTokenizer.from_pretrained(
        config["repo"],
        trust_remote_code=
            config["trust_remote_code"],
    )


    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    model = AutoModelForCausalLM.from_pretrained(
        config["repo"],
        dtype=DTYPE,
        trust_remote_code=
            config["trust_remote_code"],
        low_cpu_mem_usage=True,
    )


    model = model.to(
        DEVICE
    )

    model.eval()


    print(
        "Loaded:",
        config["name"]
    )


    return tokenizer, model


# ================================================================
# 5. TOKEN COUNTS
# ================================================================

def get_token_counts(
    tokenizer,
    texts,
):

    counts = []


    for start in range(
        0,
        len(texts),
        256
    ):

        batch = texts[
            start:
            start + 256
        ]


        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )


        counts.extend(
            [
                len(ids)
                for ids in encoded[
                    "input_ids"
                ]
            ]
        )


    return np.array(
        counts,
        dtype=int
    )


# ================================================================
# 6. GET LAYER MAXIMA FOR A BATCH
# ================================================================

@torch.inference_mode()
def layer_maxima_batch(
    model,
    tokenizer,
    texts,
):

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    ).to(
        DEVICE
    )


    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )


    mask = inputs[
        "attention_mask"
    ].bool()


    result = []


    for h in outputs.hidden_states:

        # [batch, sequence, dimension]

        values = (
            h.float()
            .abs()
        )


        values = values.masked_fill(
            ~mask.unsqueeze(-1),
            -float("inf")
        )


        maxima = values.amax(
            dim=1
        )


        result.append(
            maxima.cpu()
        )


    return result


# ================================================================
# 7. BUILD NORMAL BASELINE
# ================================================================

def build_baseline(
    model,
    tokenizer,
    model_name,
):

    print(
        "\nBuilding activation baseline..."
    )


    layer_chunks = None


    for start in range(
        0,
        len(baseline_texts),
        BATCH_SIZE
    ):

        batch = baseline_texts[
            start:
            start + BATCH_SIZE
        ]


        maxima = layer_maxima_batch(
            model,
            tokenizer,
            batch,
        )


        if layer_chunks is None:

            layer_chunks = [
                []
                for _ in maxima
            ]


            print(
                "Hidden states:",
                len(maxima)
            )

            print(
                "Layer widths:",
                [
                    x.shape[1]
                    for x in maxima
                ]
            )


        for layer, tensor in enumerate(
            maxima
        ):

            layer_chunks[
                layer
            ].append(
                tensor
            )


        print(
            f"{model_name} baseline: "
            f"{min(start+BATCH_SIZE, len(baseline_texts))}"
            f"/{len(baseline_texts)}",
            end="\r"
        )


    print()


    stats = []


    for chunks in layer_chunks:

        values = torch.cat(
            chunks,
            dim=0
        )


        median = torch.median(
            values,
            dim=0
        ).values


        q95 = torch.quantile(
            values,
            0.95,
            dim=0
        )


        spread = (
            q95 - median
        )


        minimum_scale = torch.maximum(
            median * 0.10,
            torch.full_like(
                median,
                1e-3
            )
        )


        scale = torch.maximum(
            spread,
            minimum_scale
        )


        stats.append(
            {
                "q95": q95,
                "scale": scale,
            }
        )


    return stats


# ================================================================
# 8. SCORE MANY STRINGS
# ================================================================

def score_texts(
    model,
    tokenizer,
    baseline_stats,
    texts,
    label,
):

    scores = []


    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch = texts[
            start:
            start + BATCH_SIZE
        ]


        maxima = layer_maxima_batch(
            model,
            tokenizer,
            batch,
        )


        best = torch.full(
            (len(batch),),
            -float("inf")
        )


        # Layer 0 = embeddings.
        for layer in range(
            1,
            len(maxima)
        ):

            anomaly = (
                maxima[layer]
                - baseline_stats[
                    layer
                ]["q95"]
            ) / baseline_stats[
                layer
            ]["scale"]


            layer_best = (
                anomaly
                .max(dim=1)
                .values
            )


            best = torch.maximum(
                best,
                layer_best
            )


        scores.extend(
            best.numpy().tolist()
        )


        print(
            f"{label}: "
            f"{min(start+BATCH_SIZE, len(texts))}"
            f"/{len(texts)}",
            end="\r"
        )


    print()


    return np.array(
        scores,
        dtype=float
    )


# ================================================================
# 9. MATCHED-CONTROL PERCENTILE
#
# First try EXACT SAME token count.
#
# If fewer than 30 controls exist:
# allow ±1 token.
#
# If still fewer:
# allow ±2.
# ================================================================

def matched_percentile(
    candidate_score,
    candidate_tokens,
    control_scores,
    control_token_counts,
):

    tolerance_used = None
    matched = None


    for tolerance in [
        0,
        1,
        2,
        3,
    ]:

        mask = (
            np.abs(
                control_token_counts
                - candidate_tokens
            )
            <= tolerance
        )


        subset = control_scores[
            mask
        ]


        if len(subset) >= 30:

            matched = subset

            tolerance_used = tolerance

            break


    if matched is None:

        # Last-resort nearest 30 controls.

        distances = np.abs(
            control_token_counts
            - candidate_tokens
        )


        nearest = np.argsort(
            distances
        )[:30]


        matched = control_scores[
            nearest
        ]


        tolerance_used = int(
            distances[
                nearest
            ].max()
        )


    below_or_equal = np.sum(
        matched
        <= candidate_score
    )


    percentile = (
        (below_or_equal + 1)
        /
        (len(matched) + 1)
    )


    return (
        percentile,
        len(matched),
        tolerance_used,
    )


# ================================================================
# 10. RUN ONE MODEL
# ================================================================

all_rows = []


for config in MODELS:

    tokenizer, model = load_model(
        config
    )


    baseline_stats = build_baseline(
        model,
        tokenizer,
        config["name"],
    )


    # ------------------------------------------------------------
    # Score broad weird-string control pool
    # ------------------------------------------------------------

    print(
        "\nTokenizing matched controls..."
    )


    control_token_counts = get_token_counts(
        tokenizer,
        control_pool,
    )


    print(
        "Scoring matched controls..."
    )


    control_scores = score_texts(
        model,
        tokenizer,
        baseline_stats,
        control_pool,
        config["name"]
        + " controls",
    )


    # ------------------------------------------------------------
    # Score experiment strings
    # ------------------------------------------------------------

    experimental_strings = (
        test_df["text"]
        .tolist()
    )


    experimental_token_counts = (
        get_token_counts(
            tokenizer,
            experimental_strings,
        )
    )


    print(
        "\nScoring experimental strings..."
    )


    experimental_scores = score_texts(
        model,
        tokenizer,
        baseline_stats,
        experimental_strings,
        config["name"]
        + " experiment",
    )


    # ------------------------------------------------------------
    # Convert each result to token-matched percentile
    # ------------------------------------------------------------

    for i, row in test_df.iterrows():

        pct, matched_n, tolerance = (
            matched_percentile(
                experimental_scores[i],
                experimental_token_counts[i],
                control_scores,
                control_token_counts,
            )
        )


        all_rows.append(
            {
                "model":
                    config["name"],

                "family":
                    row["family"],

                "repetitions":
                    int(
                        row[
                            "repetitions"
                        ]
                    ),

                "replicate":
                    int(
                        row[
                            "replicate"
                        ]
                    ),

                "text":
                    row["text"],

                "token_count":
                    int(
                        experimental_token_counts[
                            i
                        ]
                    ),

                "raw_anomaly":
                    float(
                        experimental_scores[
                            i
                        ]
                    ),

                "matched_percentile":
                    pct * 100,

                "matched_controls":
                    matched_n,

                "token_tolerance":
                    tolerance,
            }
        )


    # ------------------------------------------------------------
    # Unload model
    # ------------------------------------------------------------

    model.to(
        "cpu"
    )


    del model
    del tokenizer
    del baseline_stats

    gc.collect()

    torch.cuda.empty_cache()


# ================================================================
# 11. DETAILED DATAFRAME
# ================================================================

results = pd.DataFrame(
    all_rows
)


# ================================================================
# 12. AGGREGATE REPLICATES
# ================================================================

summary = (
    results
    .groupby(
        [
            "model",
            "family",
            "repetitions",
        ]
    )
    .agg(
        median_percentile=(
            "matched_percentile",
            "median"
        ),

        min_percentile=(
            "matched_percentile",
            "min"
        ),

        max_percentile=(
            "matched_percentile",
            "max"
        ),

        median_raw=(
            "raw_anomaly",
            "median"
        ),

        median_tokens=(
            "token_count",
            "median"
        ),

        n=(
            "matched_percentile",
            "size"
        ),
    )
    .reset_index()
)


# ================================================================
# 13. CROSS-MODEL SUMMARY
#
# For each family/repetition level:
#
# - average model percentile
# - weakest model percentile
#
# For replicated families, we first use the median within each model.
# ================================================================

cross_summary = (
    summary
    .groupby(
        [
            "family",
            "repetitions",
        ]
    )
    .agg(
        mean_across_models=(
            "median_percentile",
            "mean"
        ),

        worst_model=(
            "median_percentile",
            "min"
        ),

        best_model=(
            "median_percentile",
            "max"
        ),
    )
    .reset_index()
)


# ================================================================
# 14. PRINT PER-MODEL RESULTS
# ================================================================

print(
    "\n\n################################################"
)

print(
    "PER-MODEL MEDIAN TOKEN-MATCHED PERCENTILES"
)

print(
    "################################################"
)


for model_name in [
    x["name"]
    for x in MODELS
]:

    print(
        f"\n\n===== {model_name} ====="
    )


    table = summary[
        summary["model"]
        == model_name
    ].pivot(
        index="repetitions",
        columns="family",
        values="median_percentile",
    )


    print(
        table.to_string(
            float_format=
                lambda x: f"{x:6.1f}"
        )
    )


# ================================================================
# 15. PRINT CROSS-MODEL RESULTS
# ================================================================

print(
    "\n\n################################################"
)

print(
    "CROSS-MODEL: WORST MODEL PERCENTILE"
)

print(
    "################################################"
)


worst_table = cross_summary.pivot(
    index="repetitions",
    columns="family",
    values="worst_model",
)


print(
    worst_table.to_string(
        float_format=
            lambda x: f"{x:6.1f}"
    )
)


print(
    "\n\n################################################"
)

print(
    "CROSS-MODEL: MEAN PERCENTILE"
)

print(
    "################################################"
)


mean_table = cross_summary.pivot(
    index="repetitions",
    columns="family",
    values="mean_across_models",
)


print(
    mean_table.to_string(
        float_format=
            lambda x: f"{x:6.1f}"
    )
)


# ================================================================
# 16. DIRECT TARGET COMPARISON
# ================================================================

print(
    "\n\n################################################"
)

print(
    "TARGET %_B%W — ALL SEVEN MODELS"
)

print(
    "################################################"
)


target_table = summary[
    summary["family"]
    == "TARGET_%_B%W"
][
    [
        "model",
        "repetitions",
        "median_tokens",
        "median_raw",
        "median_percentile",
    ]
]


print(
    target_table.to_string(
        index=False,
        float_format=
            lambda x: f"{x:7.2f}"
    )
)


# ================================================================
# 17. SAVE EVERYTHING
# ================================================================

results.to_csv(
    "/content/repetition_detailed.csv",
    index=False
)


summary.to_csv(
    "/content/repetition_model_summary.csv",
    index=False
)


cross_summary.to_csv(
    "/content/repetition_cross_model.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "/content/repetition_detailed.csv"
)

print(
    "/content/repetition_model_summary.csv"
)

print(
    "/content/repetition_cross_model.csv"
)


print(
    "\n========================================"
)

print(
    "EXPERIMENT COMPLETE"
)

print(
    "========================================"
)
