from __future__ import annotations

import random


def make_train_val_test_split(
    complex_ids: list[str],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0,1)")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0,1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    ids = sorted(set(complex_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(train_fraction * n))
    n_val = int(round(val_fraction * n))
    n_train = min(max(n_train, 1), max(n - 2, 1))
    n_val = min(max(n_val, 1), max(n - n_train - 1, 0))

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    if not test_ids and val_ids:
        test_ids = [val_ids.pop()]
    if not test_ids and train_ids:
        test_ids = [train_ids.pop()]

    split: dict[str, str] = {}
    for cid in train_ids:
        split[cid] = "train"
    for cid in val_ids:
        split[cid] = "val"
    for cid in test_ids:
        split[cid] = "test"
    return split

