import numpy as np
import pandas as pd


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = np.abs(y) + np.abs(p)
    out = np.where(denom == 0, 0.0, 2 * np.abs(y - p) / np.where(denom == 0, 1, denom))
    return float(out.mean() * 100)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = y - p
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "sMAPE": smape(y, p),
        "WAPE": float(np.abs(err).sum() / y.sum() * 100),
        "Bias": float(-err.mean()),
        "R2": 1 - ss_res / ss_tot,
    }


def interval_report(y, lo, hi):
    y = np.asarray(y)
    return {
        "coverage": float(((y >= lo) & (y <= hi)).mean()),
        "avg_width": float(np.mean(np.asarray(hi) - np.asarray(lo))),
    }


def by_group(df, pred_col, group):
    target = "actual" if "actual" in df else "sales"
    g = df.assign(abs_err=(df[target] - df[pred_col]).abs(), _y=df[target]).groupby(group)
    return g.agg(MAE=("abs_err", "mean"), demand=("_y", "mean")).assign(
        WAPE=lambda t: t["MAE"] / t["demand"] * 100
    )
