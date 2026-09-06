"""The three splitting rules compared in Versuemer et al. (2025), Sec. II D.

ARAUS is clustered: each participant rates ~42 stimuli. How those clusters are
handled decides whether the reported score is honest.

    gkf   Group-K-Fold. Groups kept whole, no target stratification.
          Used in the inner loop of the no-leakage configuration.

    sgkf  Stratified-Group-K-Fold. Groups kept whole AND the target
          distribution approximated in every fold. This is the honest rule.

    cskf  "Custom Stratified-K-Fold". Stratifies the target but spreads each
          participant's ratings across folds, deliberately inducing group
          leakage. Reproduced here because the paper uses it to quantify the
          illusory performance gain - not because it should be used.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

N_STRATA = 10


def target_strata(y, n_strata: int = N_STRATA) -> np.ndarray:
    """Discretise the target into equal-width classes for stratification."""
    y = np.asarray(y, dtype=float)
    edges = np.linspace(y.min(), y.max(), n_strata + 1)[1:-1]
    return np.digitize(y, edges)


def make_splitter(rule: str, n_splits: int = 5, seed: int = 0):
    """Return an object with .split(X, y, groups) for the given rule."""
    rule = rule.lower()
    if rule == "gkf":
        return _Wrapped(GroupKFold(n_splits=n_splits), use_strata=False, use_groups=True)
    if rule == "sgkf":
        return _Wrapped(
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            use_strata=True, use_groups=True,
        )
    if rule == "cskf":
        return _Wrapped(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
            use_strata=True, use_groups=False,
        )
    raise ValueError(f"unknown splitting rule: {rule!r}")


class _Wrapped:
    def __init__(self, splitter, use_strata: bool, use_groups: bool):
        self.splitter = splitter
        self.use_strata = use_strata
        self.use_groups = use_groups

    def split(self, X, y, groups=None):
        strata = target_strata(y) if self.use_strata else None
        g = groups if self.use_groups else None
        return self.splitter.split(X, strata, g)

    @property
    def n_splits(self) -> int:
        return self.splitter.get_n_splits()


def leakage_report(groups, train_idx, test_idx) -> int:
    """Number of groups appearing on both sides of a split (0 = no leakage)."""
    groups = np.asarray(groups)
    return len(np.intersect1d(np.unique(groups[train_idx]), np.unique(groups[test_idx])))
