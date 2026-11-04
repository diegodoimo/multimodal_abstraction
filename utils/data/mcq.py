"""N-option multiple-choice prompt with a given correct-option position and a seeded distractor order."""

from __future__ import annotations

import random
from typing import Sequence


def build_mcq_prompt(
    *,
    question: str,
    correct: str,
    pool: Sequence[str],
    n_choices: int,
    correct_pos: int,
    shuffle_seed: str,
) -> str:
    """Return "question\\nA. option\\nB. option..." with ``correct`` at position ``correct_pos``.

    Arguments:
        question: The question stem rendered on the first line.
        correct: The correct option.
        pool: Candidate distractors (``correct`` excluded); the first ``n_choices - 1`` of a seeded
            shuffle are used.
        n_choices: Number of options.
        correct_pos: Position of the correct option, ``0 <= correct_pos < n_choices``.
        shuffle_seed: Seed of the distractor shuffle.
    """
    if n_choices < 2:
        raise ValueError(f"n_choices must be >= 2, got {n_choices}")
    if not (0 <= correct_pos < n_choices):
        raise ValueError(f"correct_pos {correct_pos} out of range for n_choices={n_choices}")
    if len(pool) < n_choices - 1:
        raise ValueError(f"pool has {len(pool)} distractors, need at least {n_choices - 1}")

    shuffled = list(pool)
    random.Random(shuffle_seed).shuffle(shuffled)
    choices = shuffled[: n_choices - 1]
    choices.insert(correct_pos, correct)

    lines = [question]
    for i, choice in enumerate(choices):
        lines.append(f"{chr(ord('A') + i)}. {choice}")
    return "\n".join(lines)
