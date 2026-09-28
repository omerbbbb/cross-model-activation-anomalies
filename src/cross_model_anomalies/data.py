from __future__ import annotations

from datasets import load_dataset


def load_ag_news_texts(
    n: int = 2000,
    seed: int = 12345,
    max_chars: int = 60,
    min_chars: int = 20,
) -> list[str]:
    dataset = load_dataset(
        "fancyzhx/ag_news",
        split="train",
    ).shuffle(seed=seed)

    texts = []

    for row in dataset:
        text = " ".join(
            row["text"].replace("\n", " ").split()
        )[:max_chars]

        if len(text) >= min_chars:
            texts.append(text)

        if len(texts) >= n:
            break

    return texts
