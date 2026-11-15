import numpy as np
import pandas as pd

from src import features as F


def toy_frame(days=700, stores=2, items=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=days)
    rows = [(d, s, i, float(rng.poisson(40 + 10 * (d.dayofweek >= 5)))) for d in dates for s in range(1, stores + 1) for i in range(1, items + 1)]
    return pd.DataFrame(rows, columns=["date", "store", "item", "sales"])


def test_features_ignore_data_inside_horizon():
    ok, diff = F.leakage_check(toy_frame(), "2016-09-15")
    assert ok, diff


def test_row_matches_manual_calculation():
    df = toy_frame()
    frame = F.build(df)
    checks = F.verify_row(frame, df, "2016-09-15", 2, 3)
    assert all(v[2] for v in checks.values())


def test_splits_are_chronological():
    df = toy_frame(days=900)
    frame = F.build(df)
    tr, va, te = F.split_masks(frame, "2016-04-30", "2016-06-30")
    assert frame.loc[tr, "date"].max() < frame.loc[va, "date"].min()
    assert frame.loc[va, "date"].max() < frame.loc[te, "date"].min()
