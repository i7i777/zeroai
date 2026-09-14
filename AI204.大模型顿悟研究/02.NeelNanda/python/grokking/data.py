"""Modular addition dataset, exactly as used in the paper.

The input to the model is of the form "a b =", where a and b are encoded as
P-dimensional one-hot vectors and '=' is a special token above which the output
c = (a + b) mod P is read.  The full dataset has all P*P ordered pairs; a random
`frac_train` fraction is used for training and the rest for testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ModularAdditionData:
    p: int
    frac_train: float
    seed: int

    # full grid of all ordered pairs (a, b) -> c = (a+b) mod p
    a: np.ndarray                      # (P*P,) int64
    b: np.ndarray                      # (P*P,) int64
    c: np.ndarray                      # (P*P,) int64
    train_mask: np.ndarray             # (P*P,) bool
    test_mask: np.ndarray              # (P*P,) bool

    @property
    def n_pairs(self) -> int:
        return self.a.size

    # ---- convenience accessors -------------------------------------------
    def idx_train(self) -> np.ndarray:
        return np.nonzero(self.train_mask)[0]

    def idx_test(self) -> np.ndarray:
        return np.nonzero(self.test_mask)[0]

    def train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.train_mask
        return self.a[m], self.b[m], self.c[m]

    def test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.test_mask
        return self.a[m], self.b[m], self.c[m]

    def label_grid(self) -> np.ndarray:
        """Ground-truth c for every (a, b), shaped [P, P]."""
        return self.c.reshape(self.p, self.p)


def make_modular_addition_data(p: int, frac_train: float, seed: int) -> ModularAdditionData:
    aa, bb = np.meshgrid(np.arange(p, dtype=np.int64), np.arange(p, dtype=np.int64), indexing="ij")
    a = aa.ravel()
    b = bb.ravel()
    c = (a + b) % p

    rng = np.random.default_rng(seed)
    perm = rng.permutation(a.size)
    n_train = int(round(a.size * frac_train))
    train_mask = np.zeros(a.size, dtype=bool)
    train_mask[perm[:n_train]] = True

    return ModularAdditionData(
        p=p, frac_train=frac_train, seed=seed,
        a=a, b=b, c=c,
        train_mask=train_mask,
        test_mask=~train_mask,
    )
