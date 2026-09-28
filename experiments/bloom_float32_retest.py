# Corrected BLOOM-only repetition experiment in float32.
# Same baseline size, candidate set, controls, seeds and token matching.

# ================================================================
# BLOOM-ONLY FIXED REPETITION EXPERIMENT
#
# Re-runs EXACTLY the previous repetition experiment for BLOOM only,
# but in float32 to avoid the float16 NaN problem.
#
# Same:
#   - 120 normal baseline texts
#   - 160 experiment strings
#   - 3000 matched controls
#   - seeds
#   - token-count matching
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
# 0. SETTINGS — IDENTICAL TO PREVIOUS EXPERIMENT
# ================================================================

if not torch.cuda.is_available():
    raise RuntimeError("GPU not detected.")

DEVICE = "cuda"

MODEL_NAME = "bigscience/bloom-560m"

REPEAT_COUNTS = [1, 2, 4, 8, 12]

TARGET_BLOCK = "%_B%W"
PLAIN_BLOCK = "abcde"

N_REPLICATES = 10

N_BASELINE = 120
N_CONTROL_POOL = 3000

MAX_LENGTH = 96
BATCH_SIZE = 16

random.seed(28092026)
np.random.seed(28092026)
torch.manual_seed(28092026)

SYMBOLS = "!@#$%^&*()[]{}<>?/\\|+=_-:;."
PRINTABLE = string.ascii_letters + string.digits + SYMBOLS


# ================================================================
# 1. LOAD SAME NORMAL BASELINE
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

    return text[:70]


baseline_texts = []

for row in dataset:

    text = clean_normal_text(
        row["text"]
    )

    if len(text) >= 20:
        baseline_texts.append(text)

    if len(baseline_texts) >= N_BASELINE:
        break


print("Baseline texts:", len(baseline_texts))


# ================================================================
# 2. RECREATE EXACT SAME TEST STRINGS
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

    # TARGET
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


    # PLAIN
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


    # SAME CHARS, SHUFFLED
    for rep in range(N_REPLICATES):

        rng = random.Random(
            100000
            + repetitions * 100
            + rep
        )

        chars = list(target)

        rng.shuffle(chars)

        shuffled = "".join(chars)

        if shuffled == target:

            chars.reverse()

            shuffled = "".join(chars)


        add_case(
            "SHUFFLED_SAME_CHARS",
            repetitions,
            rep,
            shuffled,
        )


    # RANDOM BLOCK REPEATED
    for rep in range(N_REPLICATES):

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


    # RANDOM ASCII
    for rep in range(N_REPLICATES):

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


test_df = pd.DataFrame(test_cases)

print(
    "Test strings:",
    len(test_df)
)


# ================================================================
# 3. RECREATE EXACT SAME 3000 CONTROL STRINGS
# ================================================================

control_pool = []
seen_controls = set()


def add_control(text):

    text = text.strip()

    if (
        len(text) >= 3
        and text not in seen_controls
    ):

        seen_controls.add(text)

        control_pool.append(text)


control_index = 0


while len(control_pool) < N_CONTROL_POOL:

    rng = random.Random(
        900000 + control_index
    )

    control_index += 1

    mode = rng.randrange(4)


    # random ASCII
    if mode == 0:

        length = rng.randint(
            4,
            65
        )

        text = "".join(
            rng.choice(
                PRINTABLE
            )
            for _ in range(length)
        )


    # repeated random block
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


    # chunks
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

            chunks.append(chunk)


        text = rng.choice(
            [
                "",
                " ",
                "_",
                "-",
                ".",
            ]
        ).join(chunks)[:70]


    # repeated normal-ish letters
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


    add_control(text)


print(
    "Matched-control pool:",
    len(control_pool)
)


# ================================================================
# 4. LOAD BLOOM IN FLOAT32
# ================================================================

print(
    "\n========================================"
)

print(
    "LOADING BLOOM IN FLOAT32"
)

print(
    "========================================"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)


model = model.to(DEVICE)

model.eval()


print("Loaded BLOOM.")


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
# 6. ACTIVATION MAXIMA
# ================================================================

@torch.inference_mode()
def layer_maxima_batch(
    texts
):

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    ).to(DEVICE)


    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )


    mask = inputs[
        "attention_mask"
    ].bool()


    result = []


    for layer, h in enumerate(
        outputs.hidden_states
    ):

        h = h.float()


        if not torch.isfinite(
            h
        ).all():

            raise RuntimeError(
                f"NaN/Inf in hidden state "
                f"layer {layer}"
            )


        values = h.abs()


        values = values.masked_fill(
            ~mask.unsqueeze(-1),
            -float("inf")
        )


        maxima = values.amax(
            dim=1
        )


        if not torch.isfinite(
            maxima
        ).all():

            raise RuntimeError(
                f"NaN/Inf in maxima "
                f"layer {layer}"
            )


        result.append(
            maxima.cpu()
        )


    return result


# ================================================================
# 7. BUILD BLOOM BASELINE
# ================================================================

print(
    "\nBuilding float32 activation baseline..."
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
        batch
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
        f"Baseline: "
        f"{min(start+BATCH_SIZE, len(baseline_texts))}"
        f"/{len(baseline_texts)}",
        end="\r"
    )


print()


baseline_stats = []


for layer, chunks in enumerate(
    layer_chunks
):

    values = torch.cat(
        chunks,
        dim=0
    )


    if not torch.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"Baseline NaN/Inf "
            f"layer {layer}"
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
        q95
        - median
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


    baseline_stats.append(
        {
            "q95": q95,
            "scale": scale,
        }
    )


print(
    "Baseline finite: YES"
)


# ================================================================
# 8. SCORE STRINGS
# ================================================================

def score_texts(
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
            batch
        )


        best = torch.full(
            (len(batch),),
            -float("inf")
        )


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


            if not torch.isfinite(
                anomaly
            ).all():

                raise RuntimeError(
                    f"NaN/Inf anomaly "
                    f"layer {layer}"
                )


            layer_best = (
                anomaly
                .max(dim=1)
                .values
            )


            best = torch.maximum(
                best,
                layer_best
            )


        if not torch.isfinite(
            best
        ).all():

            raise RuntimeError(
                "Final scores contain NaN/Inf"
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
# 9. SCORE CONTROLS
# ================================================================

print(
    "\nTokenizing 3000 controls..."
)


control_token_counts = get_token_counts(
    tokenizer,
    control_pool
)


print(
    "Scoring controls..."
)


control_scores = score_texts(
    control_pool,
    "BLOOM controls"
)


print(
    "\nControl raw-score distribution:"
)


print(
    "median =",
    round(
        float(
            np.median(
                control_scores
            )
        ),
        4
    )
)


print(
    "95%    =",
    round(
        float(
            np.percentile(
                control_scores,
                95
            )
        ),
        4
    )
)


print(
    "99%    =",
    round(
        float(
            np.percentile(
                control_scores,
                99
            )
        ),
        4
    )
)


print(
    "max    =",
    round(
        float(
            np.max(
                control_scores
            )
        ),
        4
    )
)


# ================================================================
# 10. SCORE EXPERIMENT
# ================================================================

experimental_strings = (
    test_df["text"]
    .tolist()
)


experimental_token_counts = (
    get_token_counts(
        tokenizer,
        experimental_strings
    )
)


print(
    "\nScoring experiment..."
)


experimental_scores = score_texts(
    experimental_strings,
    "BLOOM experiment"
)


# ================================================================
# 11. TOKEN-MATCHED PERCENTILE
# ================================================================

def matched_percentile(
    candidate_score,
    candidate_tokens,
):

    matched = None
    tolerance_used = None


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


        subset = (
            control_scores[
                mask
            ]
        )


        if len(subset) >= 30:

            matched = subset

            tolerance_used = tolerance

            break


    if matched is None:

        distances = np.abs(
            control_token_counts
            - candidate_tokens
        )


        nearest = np.argsort(
            distances
        )[:30]


        matched = (
            control_scores[
                nearest
            ]
        )


        tolerance_used = int(
            distances[
                nearest
            ].max()
        )


    below = np.sum(
        matched
        <= candidate_score
    )


    percentile = (
        (below + 1)
        /
        (len(matched) + 1)
    )


    return (
        percentile * 100,
        len(matched),
        tolerance_used,
    )


# ================================================================
# 12. BUILD RESULTS
# ================================================================

rows = []


for i, row in test_df.iterrows():

    pct, n_controls, tolerance = (
        matched_percentile(
            experimental_scores[i],
            experimental_token_counts[i],
        )
    )


    rows.append(
        {
            "family":
                row["family"],

            "repetitions":
                int(
                    row["repetitions"]
                ),

            "replicate":
                int(
                    row["replicate"]
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
                pct,

            "matched_controls":
                n_controls,

            "token_tolerance":
                tolerance,
        }
    )


results = pd.DataFrame(
    rows
)


# ================================================================
# 13. SUMMARY
# ================================================================

summary = (
    results
    .groupby(
        [
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
# 14. FINAL TABLE
# ================================================================

print(
    "\n\n################################################"
)

print(
    "FIXED BLOOM — TOKEN-MATCHED PERCENTILES"
)

print(
    "################################################"
)


table = summary.pivot(
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
# 15. TARGET DETAILS
# ================================================================

print(
    "\n\n################################################"
)

print(
    "TARGET %_B%W — FIXED BLOOM"
)

print(
    "################################################"
)


target = summary[
    summary["family"]
    == "TARGET_%_B%W"
][
    [
        "repetitions",
        "median_tokens",
        "median_raw",
        "median_percentile",
    ]
]


print(
    target.to_string(
        index=False,
        float_format=
            lambda x: f"{x:8.3f}"
    )
)


# ================================================================
# 16. RAW INDIVIDUAL TARGET SCORES
# ================================================================

print(
    "\n\n################################################"
)

print(
    "TARGET INDIVIDUAL VALUES"
)

print(
    "################################################"
)


target_raw = results[
    results["family"]
    == "TARGET_%_B%W"
][
    [
        "repetitions",
        "token_count",
        "raw_anomaly",
        "matched_percentile",
        "matched_controls",
        "token_tolerance",
    ]
]


print(
    target_raw.to_string(
        index=False,
        float_format=
            lambda x: f"{x:8.3f}"
    )
)


# ================================================================
# 17. SAVE
# ================================================================

results.to_csv(
    "/content/bloom_fixed_detailed.csv",
    index=False
)


summary.to_csv(
    "/content/bloom_fixed_summary.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "/content/bloom_fixed_detailed.csv"
)

print(
    "/content/bloom_fixed_summary.csv"
)


print(
    "\n========================================"
)

print(
    "BLOOM RETEST COMPLETE"
)

print(
    "========================================"
)
