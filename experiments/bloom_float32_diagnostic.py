# BLOOM numerical diagnostic: float16 NaNs vs float32 finite activations.
# Extracted from notebooks/research_log.ipynb.

# ================================================================
# BLOOM NaN DIAGNOSTIC + RETEST
# Runs BLOOM only, in float32
# ================================================================

import gc
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


DEVICE = "cuda"

MODEL_NAME = "bigscience/bloom-560m"

TARGETS = [
    "%_B%W",
    "%_B%W" * 2,
    "%_B%W" * 4,
    "%_B%W" * 8,
    "%_B%W" * 12,
]

N_BASELINE = 120
MAX_LENGTH = 96


# ================================================================
# 1. LOAD BLOOM IN FLOAT32
# ================================================================

print("Loading BLOOM in float32...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)

model = model.to(DEVICE)
model.eval()

print("Loaded.")


# ================================================================
# 2. CHECK RAW HIDDEN STATES FIRST
# ================================================================

@torch.inference_mode()
def inspect_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(DEVICE)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    print("\nTEXT:")
    print(repr(text))

    print(
        "tokens:",
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )
    )

    print(
        "token count:",
        inputs["input_ids"].shape[1]
    )


    for layer, h in enumerate(
        outputs.hidden_states
    ):

        h = h.float()

        finite = torch.isfinite(
            h
        )

        if not finite.all():

            n_nan = torch.isnan(
                h
            ).sum().item()

            n_inf = torch.isinf(
                h
            ).sum().item()

            print(
                f"!!! layer {layer}: "
                f"NaN={n_nan}, Inf={n_inf}"
            )

            return False


    print(
        "All hidden states finite."
    )

    return True


print("\n================================")
print("RAW NUMERICAL CHECK")
print("================================")


for text in TARGETS:

    inspect_text(
        text
    )


# ================================================================
# 3. LOAD NORMAL BASELINE
# ================================================================

dataset = load_dataset(
    "fancyzhx/ag_news",
    split="train"
).shuffle(seed=8127)


baseline_texts = []

for row in dataset:

    text = " ".join(
        row["text"]
        .replace("\n", " ")
        .split()
    )[:70]

    if len(text) >= 20:

        baseline_texts.append(
            text
        )

    if len(
        baseline_texts
    ) >= N_BASELINE:

        break


# ================================================================
# 4. GET MAX ACTIVATIONS
# ================================================================

@torch.inference_mode()
def get_layer_maxima(
    text
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    ).to(DEVICE)


    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )


    result = []


    for h in outputs.hidden_states:

        h = h[0].float()


        if not torch.isfinite(
            h
        ).all():

            raise RuntimeError(
                "Non-finite hidden state found!"
            )


        maxima = (
            h.abs()
            .amax(dim=0)
            .cpu()
        )


        result.append(
            maxima
        )


    return result


# ================================================================
# 5. BUILD BASELINE
# ================================================================

print("\n================================")
print("BUILDING FLOAT32 BLOOM BASELINE")
print("================================")


layer_samples = None


for i, text in enumerate(
    baseline_texts
):

    maxima = get_layer_maxima(
        text
    )


    if layer_samples is None:

        layer_samples = [
            []
            for _ in maxima
        ]


    for layer, vector in enumerate(
        maxima
    ):

        layer_samples[
            layer
        ].append(
            vector
        )


    print(
        f"{i+1}/{N_BASELINE}",
        end="\r"
    )


print()


baseline_stats = []


for layer, vectors in enumerate(
    layer_samples
):

    values = torch.stack(
        vectors
    )


    if not torch.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"Baseline contains NaN/Inf "
            f"at layer {layer}"
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


    scale = torch.maximum(
        q95 - median,
        torch.maximum(
            median * 0.10,
            torch.full_like(
                median,
                1e-3
            )
        )
    )


    baseline_stats.append(
        {
            "q95": q95,
            "scale": scale,
        }
    )


print(
    "Baseline is completely finite."
)


# ================================================================
# 6. SCORE TARGET
# ================================================================

def score_text(
    text
):

    maxima = get_layer_maxima(
        text
    )


    best_score = -float("inf")
    best_layer = None
    best_dim = None


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
                f"Anomaly became NaN/Inf "
                f"at layer {layer}"
            )


        dim = torch.argmax(
            anomaly
        ).item()


        score = anomaly[
            dim
        ].item()


        if score > best_score:

            best_score = score
            best_layer = layer
            best_dim = dim


    return (
        best_score,
        best_layer,
        best_dim,
    )


# ================================================================
# 7. FINAL BLOOM RESULTS
# ================================================================

print("\n\n================================")
print("BLOOM FLOAT32 RESULTS")
print("================================")


for repetitions, text in zip(
    [1, 2, 4, 8, 12],
    TARGETS
):

    raw, layer, dim = score_text(
        text
    )


    print(
        f"\n{repetitions} repetitions"
    )

    print(
        "token count:",
        len(
            tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]
        )
    )

    print(
        "raw anomaly:",
        raw
    )

    print(
        "winning layer:",
        layer
    )

    print(
        "winning dimension:",
        dim
    )


print(
    "\n================================"
)
print(
    "DONE"
)
print(
    "================================"
)
