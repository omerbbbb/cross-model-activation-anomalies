from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cross_model_anomalies.candidates import generate_unique_strings
from cross_model_anomalies.data import load_ag_news_texts
from cross_model_anomalies.modeling import load_causal_lm
from cross_model_anomalies.scoring import (
    build_baseline,
    heldout_percentile,
    pool_percentiles,
    score_texts,
)


SEARCH_MODELS = [
    ("Qwen", "Qwen/Qwen2.5-0.5B"),
    ("TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ("Pythia", "EleutherAI/pythia-410m"),
    ("OPT", "facebook/opt-350m"),
    ("Falcon", "tiiuae/falcon-rw-1b"),
]

HOLDOUT_MODELS = [
    ("BLOOM", "bigscience/bloom-560m"),
    ("GPT-Neo", "EleutherAI/gpt-neo-125m"),
]


def unload(model):
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model hidden-state activation anomaly search."
    )
    parser.add_argument("--n-baseline", type=int, default=200)
    parser.add_argument("--n-candidates", type=int, default=10_000)
    parser.add_argument("--n-holdout-controls", type=int, default=2_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the default experiment.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(20260928)
    torch.manual_seed(20260928)

    normal_texts = load_ag_news_texts(n=2000)
    baseline_texts = normal_texts[: args.n_baseline]
    seed_texts = normal_texts[args.n_baseline:1000]

    candidates, next_index, seen = generate_unique_strings(
        args.n_candidates,
        seed_texts,
    )
    holdout_controls, _, _ = generate_unique_strings(
        args.n_holdout_controls,
        seed_texts,
        start_index=next_index,
        seen=seen,
    )

    search_raw = {}

    for model_name, repo in SEARCH_MODELS:
        print(f"\n=== {model_name} ===")
        tokenizer, model = load_causal_lm(repo)

        baseline = build_baseline(
            model,
            tokenizer,
            baseline_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        search_raw[model_name] = score_texts(
            model,
            tokenizer,
            baseline,
            candidates,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        unload(model)
        del tokenizer, baseline

    percentiles = {
        name: pool_percentiles(search_raw[name])
        for name, _ in SEARCH_MODELS
    }

    matrix = np.vstack(
        [percentiles[name] for name, _ in SEARCH_MODELS]
    ).T

    joint = matrix.min(axis=1)
    mean = matrix.mean(axis=1)

    ranking = sorted(
        range(len(candidates)),
        key=lambda i: (joint[i], mean[i]),
        reverse=True,
    )

    top_indices = ranking[: args.top_k]
    locked = [candidates[i] for i in top_indices]

    search_rows = []

    for rank, i in enumerate(top_indices, 1):
        row = {
            "rank": rank,
            "text": candidates[i],
            "worst_search_percentile": joint[i] * 100,
            "mean_search_percentile": mean[i] * 100,
        }
        for name, _ in SEARCH_MODELS:
            row[name] = percentiles[name][i] * 100
        search_rows.append(row)

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(
        args.output_dir / "locked_top20.csv",
        index=False,
    )

    holdout_rows = []

    for model_name, repo in HOLDOUT_MODELS:
        print(f"\n=== HOLDOUT: {model_name} ===")
        tokenizer, model = load_causal_lm(repo)

        baseline = build_baseline(
            model,
            tokenizer,
            baseline_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        control_scores = score_texts(
            model,
            tokenizer,
            baseline,
            holdout_controls,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        locked_scores = score_texts(
            model,
            tokenizer,
            baseline,
            locked,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        for rank, (text, raw) in enumerate(
            zip(locked, locked_scores),
            1,
        ):
            holdout_rows.append(
                {
                    "rank": rank,
                    "text": text,
                    "holdout_model": model_name,
                    "raw_score": raw,
                    "percentile": (
                        heldout_percentile(control_scores, raw) * 100
                    ),
                }
            )

        unload(model)
        del tokenizer, baseline

    holdout_long = pd.DataFrame(holdout_rows)
    holdout_long.to_csv(
        args.output_dir / "holdout_long.csv",
        index=False,
    )

    pivot = holdout_long.pivot(
        index=["rank", "text"],
        columns="holdout_model",
        values="percentile",
    ).reset_index()

    pivot["worst_holdout_percentile"] = pivot[
        [name for name, _ in HOLDOUT_MODELS]
    ].min(axis=1)

    final_df = search_df.merge(
        pivot,
        on=["rank", "text"],
        how="left",
    ).sort_values("rank")

    final_df.to_csv(
        args.output_dir / "final_cross_model_results.csv",
        index=False,
    )

    print("\nLocked candidates:")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
