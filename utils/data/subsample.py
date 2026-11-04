"""Seeded class-stratified subsampling.

Dataset-agnostic helper used by ``ClassificationDataset`` to produce a balanced
random subsample. Works on any sequence of hashable labels, returning indices
into the original sequence. Sorted ascending so the downstream ``dataset.select``
preserves original ordering.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Hashable, Sequence


def stratified_indices(
    labels: Sequence[Hashable],
    target_size: int,
    seed: int = 42,
) -> list[int]:
    """Return a seeded stratified random subsample of indices.

    Arguments:
        labels: Sequence of class labels, one per sample in the source dataset.
        target_size: Number of indices to return. Must be positive and not exceed
            ``len(labels)``.
        seed: RNG seed used both for the within-class pick and the remainder
            distribution.

    Returns:
        Sorted list of ``target_size`` indices.

    Raises:
        ValueError: If ``target_size`` is not in ``[1, len(labels)]``, if any
            class has fewer samples than its stratified quota, or if ``labels``
            is empty.
    """
    n = len(labels)
    if n == 0:
        raise ValueError("labels is empty")
    if not (1 <= target_size <= n):
        raise ValueError(
            f"target_size {target_size} out of range [1, {n}]"
        )

    class_to_indices: dict[Hashable, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    sorted_labels = sorted(class_to_indices.keys())
    n_classes = len(sorted_labels)
    base_quota = target_size // n_classes
    remainder = target_size % n_classes

    # Remainder classes: first `remainder` in sorted-label order. Deterministic
    # without consuming rng state — so changing target_size does not shuffle the
    # within-class picks for classes that kept the same quota.
    extra_classes: set[Hashable] = set(sorted_labels[:remainder]) if remainder else set()

    rng = random.Random(seed)

    picked: list[int] = []
    for label in sorted_labels:
        pool = class_to_indices[label]
        quota = base_quota + (1 if label in extra_classes else 0)
        if quota == 0:
            continue
        if len(pool) < quota:
            raise ValueError(
                f"class {label!r} has {len(pool)} samples, quota {quota} "
                f"(target_size={target_size}, n_classes={n_classes})"
            )
        picked.extend(rng.sample(pool, quota))

    picked.sort()
    return picked
