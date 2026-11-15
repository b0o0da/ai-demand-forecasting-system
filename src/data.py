import json
import urllib.request

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config as C


def download(force=False):
    C.RAW.parent.mkdir(parents=True, exist_ok=True)
    if C.RAW.exists() and not force:
        return C.RAW
    error = None
    for url in C.DATA_URLS:
        try:
            urllib.request.urlretrieve(url, C.RAW)
            pd.read_csv(C.RAW, nrows=5)
            return C.RAW
        except Exception as exc:
            error = exc
    raise RuntimeError(f"Could not download dataset: {error}")


def load_raw():
    return pd.read_csv(C.RAW, parse_dates=["date"])


def holidays(start, end):
    return USFederalHolidayCalendar().holidays(start, end)


def profile(df):
    full = pd.date_range(df["date"].min(), df["date"].max())
    per_series = df.groupby(["store", "item"])["date"].agg(["min", "max", "count"])
    s = df["sales"]
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "exact_duplicates": int(df.duplicated().sum()),
        "repeated_keys": int(df.duplicated(["date", "store", "item"]).sum()),
        "stores": int(df["store"].nunique()),
        "items": int(df["item"].nunique()),
        "series": int(len(per_series)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "calendar_days": int(len(full)),
        "missing_calendar_dates": int(len(full.difference(pd.DatetimeIndex(df["date"].unique())))),
        "series_with_incomplete_history": int((per_series["count"] != len(full)).sum()),
        "negative_sales": int((s < 0).sum()),
        "zero_share": float((s == 0).mean()),
        "skew": float(s.skew()),
        "summary": {k: float(v) for k, v in s.describe().items()},
    }


def flag_outliers(df, threshold=4.0):
    d = df[["date", "store", "item", "sales"]].copy()
    d["year"] = d["date"].dt.year
    d["dow"] = d["date"].dt.dayofweek
    base = d.groupby(["store", "item", "year", "dow"])["sales"].transform("median")
    r = np.log1p(d["sales"]) - np.log1p(base)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    d["robust_z"] = 0.6745 * (r - med) / mad
    d["is_outlier"] = d["robust_z"].abs() > threshold
    return d


def clean(df):
    log = {}
    n0 = len(df)
    df = df.drop_duplicates()
    log["exact_duplicates_removed"] = int(n0 - len(df))
    log["repeated_keys_found"] = int(df.duplicated(["date", "store", "item"]).sum())

    bad = df["sales"] < 0
    log["negative_sales_set_missing"] = int(bad.sum())
    df = df.assign(sales=df["sales"].where(~bad))

    full = pd.date_range(df["date"].min(), df["date"].max())
    keys = df[["store", "item"]].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"date": full}), how="cross")
    df = grid.merge(df, on=["store", "item", "date"], how="left")
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    log["records_added_by_calendar_reindex"] = int(df["sales"].isna().sum())

    df["was_missing"] = df["sales"].isna()
    if df["was_missing"].any():
        df["sales"] = df.groupby(["store", "item"])["sales"].transform(
            lambda s: s.interpolate(limit=3, limit_direction="both")
        )
    log["missing_after_imputation"] = int(df["sales"].isna().sum())

    flags = flag_outliers(df)
    hol = holidays(full.min(), full.max())
    flagged = flags[flags["is_outlier"]].copy()
    log["outliers_flagged"] = int(len(flagged))
    log["outlier_share"] = float(len(flagged) / len(flags))
    log["outliers_on_holidays_share"] = float(flagged["date"].isin(hol).mean()) if len(flagged) else 0.0
    log["outliers_removed"] = 0
    flagged.sort_values("robust_z", key=np.abs, ascending=False).head(500).to_csv(
        C.REPORTS / "flagged_outliers.csv", index=False
    )

    df["store"] = df["store"].astype("int8")
    df["item"] = df["item"].astype("int8")
    df["sales"] = df["sales"].astype("float32")
    return df.drop(columns="was_missing"), log


def prepare(force_download=False):
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    download(force_download)
    raw = load_raw()
    report = {"raw_profile": profile(raw)}
    clean_df, log = clean(raw)
    report["cleaning_log"] = log
    clean_df.to_parquet(C.PROCESSED / "daily_sales.parquet", index=False)
    (C.REPORTS / "data_quality.json").write_text(json.dumps(report, indent=2))
    return clean_df, report


if __name__ == "__main__":
    _, rep = prepare()
    print(json.dumps(rep, indent=2))
