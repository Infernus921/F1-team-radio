# Race-grouped splits. Every split in the project comes from here

# Imports
from __future__ import annotations

import numpy as np
import pandas as pd


# Functions
def race_year(race_id: pd.Series) -> pd.Series:
    return race_id.str.slice(0, 4).astype(int)


def chronological_split(df: pd.DataFrame, train_years, val_years, test_years) -> pd.Series:
    year = race_year(df["race_id"])
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out[year.isin(train_years)] = "train"
    out[year.isin(val_years)] = "val"
    out[year.isin(test_years)] = "test"
    if out.isna().any():
        missing = sorted(year[out.isna()].unique())
        raise ValueError(f"seasons not assigned to any split: {missing}")
    return out


def grouped_kfold(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    sizes = df.groupby("race_id").size()
    races = sizes.index.to_numpy()
    rng.shuffle(races)
    races = sorted(races, key=lambda r: -sizes[r])
    load = np.zeros(n_folds, dtype=int)
    fold_of = {}
    for r in races:
        k = int(np.argmin(load))
        fold_of[r] = k
        load[k] += sizes[r]
    return df["race_id"].map(fold_of).astype(int)


def top_drivers(df: pd.DataFrame, n: int) -> list[str]:
    return df["driver_id"].value_counts().head(n).index.tolist()


def lodo_folds(df: pd.DataFrame, race_fold: pd.Series, drivers: list[str]):
    for d in drivers:
        is_d = df["driver_id"] == d
        for k in sorted(race_fold.unique()):
            in_k = race_fold == k
            test_idx = df.index[is_d & in_k]
            train_idx = df.index[~is_d & ~in_k]
            if len(test_idx):
                yield d, k, train_idx, test_idx


def assert_no_race_leakage(race_id: pd.Series, fold: pd.Series) -> None:
    n = pd.DataFrame({"race_id": race_id, "fold": fold}).groupby("race_id")["fold"].nunique()
    bad = n[n > 1]
    if len(bad):
        raise AssertionError(f"race_id in >1 fold: {bad.index.tolist()[:10]}")
