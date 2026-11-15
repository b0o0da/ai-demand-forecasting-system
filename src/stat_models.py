import json
import time

import numpy as np
import pandas as pd

from . import config as C
from . import models as M
from .evaluate import metrics

SAMPLE_SERIES = 60


def run():
    df = pd.read_parquet(C.PROCESSED / "daily_sales.parquet")
    wide = M.wide_sales(df)
    dates = list(wide.index)
    rng = np.random.default_rng(C.SEED)
    pos = sorted(rng.choice(wide.shape[1], SAMPLE_SERIES, replace=False).tolist())
    preds = pd.read_parquet(C.PROCESSED / "predictions.parquet")
    val = preds[preds["split"] == "validation"]
    test = preds[preds["split"] == "test"]
    blocks = M.blocks(dates, val["date"].min(), val["date"].max()) + M.blocks(dates, test["date"].min(), test["date"].max())
    t0 = time.time()
    out = M.ets_forecasts(wide, blocks, pos)
    secs = time.time() - t0
    row = {d: i for i, d in enumerate(wide.index)}
    col = {c: j for j, c in enumerate(wide.columns)}
    preds["ets"] = [out[row[d], col[(s, i)]] for d, s, i in zip(preds["date"], preds["store"], preds["item"])]
    preds["ets"] = preds["ets"].astype("float32")
    preds.to_parquet(C.PROCESSED / "predictions.parquet", index=False)

    sample = preds[preds["ets"].notna()]
    res = {"sample_series": SAMPLE_SERIES, "ets_fit_seconds_total": secs, "like_for_like": {}}
    for split in ("validation", "test"):
        part = sample[sample["split"] == split]
        res["like_for_like"][split] = {
            name: metrics(part["actual"], part[c])
            for name, c in {
                "Seasonal naive (last same weekday)": "naive_origin",
                "Exponential smoothing (Holt-Winters)": "ets",
                "Random Forest": "rf",
                "LightGBM": "lgbm",
            }.items()
        }
    (C.REPORTS / "ets_comparison.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: round(v["MAE"], 3) for k, v in res["like_for_like"]["test"].items()}), secs)


if __name__ == "__main__":
    run()
