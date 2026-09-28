from __future__ import annotations

import random
import string

SYMBOLS = "!@#$%^&*()[]{}<>?/\\|+=_-:;."
ALNUM = string.ascii_letters + string.digits
ALL_CHARS = ALNUM + SYMBOLS + " "


def mutate_natural(text: str, rng: random.Random) -> str:
    chars = list(text)
    operations = rng.randint(2, 10)

    for _ in range(operations):
        if not chars:
            break

        op = rng.choice(
            ["replace", "insert", "delete", "case", "duplicate", "space"]
        )
        pos = rng.randrange(len(chars))

        if op == "replace":
            chars[pos] = rng.choice(ALL_CHARS)
        elif op == "insert":
            chars.insert(pos, rng.choice(SYMBOLS + string.digits))
        elif op == "delete" and len(chars) > 5:
            chars.pop(pos)
        elif op == "case" and chars[pos].isalpha():
            chars[pos] = chars[pos].swapcase()
        elif op == "duplicate":
            chars.insert(pos, chars[pos])
        elif op == "space":
            chars.insert(pos, " ")

    return "".join(chars)[:60]


def fragment_words(text: str, rng: random.Random) -> str:
    result = []
    for word in text.split():
        if len(word) < 3:
            result.append(word)
            continue

        separator = rng.choice(SYMBOLS + string.digits)
        pieces = []

        for char in word:
            pieces.append(char)
            if rng.random() < 0.25:
                pieces.append(separator)

        result.append("".join(pieces))

    return " ".join(result)[:60]


def mixed_alphanumeric(rng: random.Random) -> str:
    chunks = []
    for _ in range(rng.randint(3, 12)):
        length = rng.randint(1, 8)
        chunks.append(
            "".join(
                rng.choice(ALNUM + SYMBOLS)
                for _ in range(length)
            )
        )

    separator = rng.choice(["", " ", "_", "-", ".", "/"])
    return separator.join(chunks)[:60]


def repetition_string(rng: random.Random) -> str:
    fragment = "".join(
        rng.choice(ALNUM + SYMBOLS)
        for _ in range(rng.randint(1, 8))
    )
    return (fragment * rng.randint(3, 15))[:60]


def boundary_string(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 2:
        return text

    separator = rng.choice(["", "$", "%", "7", "][", "::", "__", "#", "}{"])
    return separator.join(words)[:60]


def spaced_characters(text: str, rng: random.Random) -> str:
    text = text.replace(" ", "")
    separator = rng.choice([" ", "  ", ".", "_", "$", "7", "%"])
    return separator.join(text)[:60]


def random_ascii(rng: random.Random) -> str:
    length = rng.randint(10, 60)
    return "".join(rng.choice(ALL_CHARS) for _ in range(length))


def generate_string(
    index: int,
    seed_texts: list[str],
    seed_offset: int = 900000,
) -> str:
    rng = random.Random(seed_offset + index)
    mode = rng.randrange(7)
    base = rng.choice(seed_texts)

    if mode == 0:
        return mutate_natural(base, rng)
    if mode == 1:
        return fragment_words(base, rng)
    if mode == 2:
        return mixed_alphanumeric(rng)
    if mode == 3:
        return repetition_string(rng)
    if mode == 4:
        return boundary_string(base, rng)
    if mode == 5:
        return spaced_characters(base, rng)
    return random_ascii(rng)


def generate_unique_strings(
    count: int,
    seed_texts: list[str],
    start_index: int = 0,
    seen: set[str] | None = None,
):
    if seen is None:
        seen = set()

    result = []
    index = start_index

    while len(result) < count:
        text = generate_string(index, seed_texts).strip()
        index += 1

        if len(text) < 5 or text in seen:
            continue

        seen.add(text)
        result.append(text)

    return result, index, seen
