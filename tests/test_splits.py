# Leakage assertions (BUILD_BRIEF §2). These must fail if any race_id spans two folds

# Imports
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1radio.config import ROOT, load_config
from f1radio.splits import (
    assert_no_race_leakage,
    chronological_split,
    grouped_kfold,
    lodo_folds,
    top_drivers,
)

# Settings
META = ROOT / "data" / "interim" / "metadata.parquet"


# Functions
def _synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    races = [f"{y}_GP{i}" for y in range(2018, 2026) for i in range(8)]
    return pd.DataFrame({
        "race_id": rng.choice(races, n),
        "driver_id": rng.choice([f"D{i}" for i in range(12)], n),
    })


@pytest.fixture(params=["synthetic", "real"])
def df(request):
    if request.param == "real":
        if not META.exists():
            pytest.skip("metadata.parquet not built yet")
        return pd.read_parquet(META, columns=["race_id", "driver_id"])
    return _synthetic()


def test_assertion_catches_leak():
    race = pd.Series(["a", "a", "b"])
    with pytest.raises(AssertionError):
        assert_no_race_leakage(race, pd.Series([0, 1, 0]))


def test_chronological_no_leak(df):
    cfg = load_config()["splits"]
    split = chronological_split(df, cfg["train_years"], cfg["val_years"], cfg["test_years"])
    assert_no_race_leakage(df["race_id"], split)


def test_grouped_kfold_no_leak_and_deterministic(df):
    f1 = grouped_kfold(df, 5, seed=1337)
    f2 = grouped_kfold(df, 5, seed=1337)
    assert (f1 == f2).all()
    assert_no_race_leakage(df["race_id"], f1)
    assert f1.nunique() == 5


def test_lodo_is_driver_and_race_disjoint(df):
    fold = grouped_kfold(df, 5, seed=1337)
    for d, k, tr, te in lodo_folds(df, fold, top_drivers(df, 8)):
        assert set(df.loc[tr, "race_id"]).isdisjoint(df.loc[te, "race_id"])
        assert d not in set(df.loc[tr, "driver_id"])
        assert set(df.loc[te, "driver_id"]) == {d}
