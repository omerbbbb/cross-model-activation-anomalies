from __future__ import annotations

import numpy as np
import torch

from .modeling import layer_maxima_batch


def build_baseline(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    device: str = "cuda",
    max_length: int = 64,
):
    """
    Build per-layer, per-dimension robust activation baselines.

    For every dimension:
      center reference = 95th percentile over normal texts
      scale = max(q95 - median, 10% of median, 1e-3)

    The embedding state (layer 0) is retained in the returned object,
    but ignored by score_texts by default.
    """
    layer_chunks = None

    for start in range(0, len(texts), batch_size):
        maxima = layer_maxima_batch(
            model,
            tokenizer,
            texts[start:start + batch_size],
            device=device,
            max_length=max_length,
        )

        if layer_chunks is None:
            layer_chunks = [[] for _ in maxima]

        for layer, values in enumerate(maxima):
            layer_chunks[layer].append(values)

    stats = []

    for chunks in layer_chunks:
        values = torch.cat(chunks, dim=0)
        median = torch.median(values, dim=0).values
        q95 = torch.quantile(values, 0.95, dim=0)

        spread = q95 - median
        floor = torch.maximum(
            median * 0.10,
            torch.full_like(median, 1e-3),
        )
        scale = torch.maximum(spread, floor)

        stats.append(
            {
                "median": median,
                "q95": q95,
                "scale": scale,
            }
        )

    return stats


def score_texts(
    model,
    tokenizer,
    baseline_stats,
    texts: list[str],
    batch_size: int = 32,
    device: str = "cuda",
    max_length: int = 64,
    ignore_embedding_state: bool = True,
) -> np.ndarray:
    """
    Score each text by its largest standardized activation anomaly
    anywhere in the retained hidden states.

    This is an exploratory metric, not a standard deviation or a
    calibrated probability.
    """
    all_scores = []
    first_layer = 1 if ignore_embedding_state else 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        maxima = layer_maxima_batch(
            model,
            tokenizer,
            batch,
            device=device,
            max_length=max_length,
        )

        best = torch.full((len(batch),), -float("inf"))

        for layer in range(first_layer, len(maxima)):
            anomaly = (
                maxima[layer] - baseline_stats[layer]["q95"]
            ) / baseline_stats[layer]["scale"]

            layer_best = anomaly.max(dim=1).values
            best = torch.maximum(best, layer_best)

        all_scores.extend(best.numpy().tolist())

    return np.asarray(all_scores, dtype=float)


def pool_percentiles(scores: np.ndarray) -> np.ndarray:
    """Empirical percentile of every score within the same score pool."""
    sorted_scores = np.sort(scores)
    return np.searchsorted(
        sorted_scores,
        scores,
        side="right",
    ) / len(sorted_scores)


def heldout_percentile(control_scores: np.ndarray, score: float) -> float:
    """Smoothed empirical percentile against an independent control pool."""
    sorted_controls = np.sort(control_scores)
    below = np.searchsorted(sorted_controls, score, side="right")
    return (below + 1) / (len(sorted_controls) + 1)
