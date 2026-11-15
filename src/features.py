import numpy as np
import pandas as pd

from . import config as C
from .data import holidays

H = C.HORIZON
LAGS = [28, 35, 42, 49, 56, 91, 364]
ROLLS = [7, 14, 28, 56, 91]
CORE_MIN_HISTORY = 91 + H
CATEGORICAL = ["store", "item"]
CALENDAR = [
    "dow", "dom", "month", "quarter", "doy", "woy", "is_weekend",
    "is_holiday", "days_to_holiday", "is_month_start", "is_month_end",
]
LAG_FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in ROLLS]
    + ["roll_std_28", "dow_mean_5", "yoy_week_mean", "trend_ratio"]
)
FEATURES = CATEGORICAL + CALENDAR + LAG_FEATURES


def extend_calendar(df, days=H):
    last = df["date"].max()
    future = pd.date_range(last + pd.Timedelta(days=1), periods=days)
    keys = df[["store", "item"]].drop_duplicates()
    fut = keys.merge(pd.DataFrame({"date": future}), how="cross")
    fut["sales"] = np.nan
    return pd.concat([df[["date", "store", "item", "sales"]], fut], ignore_index=True)


def calendar_frame(index):
    hol = holidays(index.min() - pd.Timedelta(days=30), index.max() + pd.Timedelta(days=30))
    hol_vals = hol.values
    pos = np.searchsorted(hol_vals, index.values)
    prev_ = hol_vals[np.clip(pos - 1, 0, len(hol_vals) - 1)]
    next_ = hol_vals[np.clip(pos, 0, len(hol_vals) - 1)]
    d_prev = (index.values - prev_).astype("timedelta64[D]").astype(int)
    d_next = (next_ - index.values).astype("timedelta64[D]").astype(int)
    nearest = np.where(d_next <= d_prev, d_next, -d_prev)
    return pd.DataFrame({
        "dow": index.dayofweek,
        "dom": index.day,
        "month": index.month,
        "quarter": index.quarter,
        "doy": index.dayofyear,
        "woy": index.isocalendar().week.astype(int).values,
        "is_weekend": (index.dayofweek >= 5).astype(int),
        "is_holiday": index.isin(hol).astype(int),
        "days_to_holiday": np.clip(nearest, -7, 7),
        "is_month_start": index.is_month_start.astype(int),
        "is_month_end": index.is_month_end.astype(int),
    }, index=index)


def build(df):
    w = df.pivot(index="date", columns=["store", "item"], values="sales").sort_index()
    idx, cols = w.index, w.columns
    s = w.shift(H)
    feats = {f"lag_{k}": w.shift(k) for k in LAGS}
    for win in ROLLS:
        feats[f"roll_mean_{win}"] = s.rolling(win, min_periods=win).mean()
    feats["roll_std_28"] = s.rolling(28, min_periods=28).std()
    feats["dow_mean_5"] = sum(w.shift(k) for k in (28, 35, 42, 49, 56)) / 5
    feats["yoy_week_mean"] = w.shift(361).rolling(7, min_periods=7).mean()
    prior = w.shift(H + 364).rolling(91, min_periods=91).mean()
    feats["trend_ratio"] = feats["roll_mean_91"] / prior

    n, m = len(idx), len(cols)
    out = {
        "date": np.repeat(idx.values, m),
        "store": np.tile(cols.get_level_values(0).values.astype("int16"), n),
        "item": np.tile(cols.get_level_values(1).values.astype("int16"), n),
    }
    cal = calendar_frame(idx)
    for c in CALENDAR:
        out[c] = np.repeat(cal[c].to_numpy(dtype="int16"), m)
    for k, v in feats.items():
        out[k] = v.to_numpy(dtype="float32").ravel()
    out["sales"] = w.to_numpy(dtype="float32").ravel()
    return pd.DataFrame(out)


def usable(frame):
    first_ok = frame["date"].min() + pd.Timedelta(days=CORE_MIN_HISTORY)
    core = [f"lag_{k}" for k in LAGS if k <= 91] + [f"roll_mean_{w}" for w in ROLLS] + ["roll_std_28", "dow_mean_5"]
    return (frame["date"] >= first_ok) & frame[core].notna().all(axis=1)


def split_masks(frame, train_end=C.TRAIN_END, val_end=C.VAL_END):
    ok = usable(frame) & frame["sales"].notna()
    train = ok & (frame["date"] <= pd.Timestamp(train_end))
    val = ok & (frame["date"] > pd.Timestamp(train_end)) & (frame["date"] <= pd.Timestamp(val_end))
    test = ok & (frame["date"] > pd.Timestamp(val_end))
    return train, val, test


def leakage_check(df, probe_date, tolerance=0.0):
    probe = pd.Timestamp(probe_date)
    base = build(df)
    altered = df.copy()
    mask = altered["date"] > probe - pd.Timedelta(days=H)
    altered.loc[mask, "sales"] = altered.loc[mask, "sales"] + 12345.0
    other = build(altered)
    a = base[base["date"] == probe][FEATURES].reset_index(drop=True)
    b = other[other["date"] == probe][FEATURES].reset_index(drop=True)
    diff = (a.fillna(-1) - b.fillna(-1)).abs().to_numpy().max()
    return bool(diff <= tolerance), float(diff)


def verify_row(frame, df, date, store, item):
    d = pd.Timestamp(date)
    series = df[(df["store"] == store) & (df["item"] == item)].set_index("date")["sales"]
    row = frame[(frame["date"] == d) & (frame["store"] == store) & (frame["item"] == item)].iloc[0]
    checks = {
        "lag_28": series.loc[d - pd.Timedelta(days=28)],
        "roll_mean_28": series.loc[d - pd.Timedelta(days=55): d - pd.Timedelta(days=28)].mean(),
        "dow_mean_5": np.mean([series.loc[d - pd.Timedelta(days=k)] for k in (28, 35, 42, 49, 56)]),
    }
    return {k: (float(row[k]), float(v), bool(np.isclose(row[k], v, rtol=1e-4))) for k, v in checks.items()}
