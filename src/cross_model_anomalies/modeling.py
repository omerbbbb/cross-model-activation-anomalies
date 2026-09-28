from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(
    repo: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load a causal LM and tokenizer using native Transformers code."""
    tokenizer = AutoTokenizer.from_pretrained(
        repo,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        repo,
        dtype=dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def layer_maxima_batch(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cuda",
    max_length: int = 64,
) -> list[torch.Tensor]:
    """
    Return, for every hidden state, the maximum absolute activation
    over non-padding token positions for each hidden dimension.

    Each list entry has shape [batch, hidden_width].
    Keeping layers as a list supports models such as OPT whose
    final hidden-state width can differ from intermediate layers.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
    )

    mask = inputs["attention_mask"].bool()
    result = []

    for hidden in outputs.hidden_states:
        values = hidden.float().abs()
        values = values.masked_fill(
            ~mask.unsqueeze(-1),
            -float("inf"),
        )
        result.append(values.amax(dim=1).cpu())

    return result
