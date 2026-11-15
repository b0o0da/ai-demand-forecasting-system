import json

import numpy as np
import pandas as pd

from . import config as C

Z95 = 1.645
Z80 = 1.2816


def compute():
    hist = pd.read_parquet(C.PROCESSED / "daily_sales.parquet")
    fut = pd.read_parquet(C.PROCESSED / "future_forecast.parquet")
    f0, f1 = fut["date"].min(), fut["date"].max()
    ly0, ly1 = f0 - pd.Timedelta(days=364), f1 - pd.Timedelta(days=364)
    last0 = f0 - pd.Timedelta(days=C.HORIZON)

    ly = hist[(hist["date"] >= ly0) & (hist["date"] <= ly1)].groupby(["store", "item"])["sales"].sum().rename("same_period_last_year")
    recent = hist[hist["date"] >= last0].groupby(["store", "item"])["sales"].sum().rename("last_28_days")
    agg = fut.groupby(["store", "item"]).agg(forecast=("p50", "sum"), low=("p10", "sum"), high=("p90", "sum"))
    si = agg.join(ly).join(recent).reset_index()
    si["yoy_change_pct"] = (si["forecast"] / si["same_period_last_year"] - 1) * 100
    si["vs_last_28_days_pct"] = (si["forecast"] / si["last_28_days"] - 1) * 100

    item = si.groupby("item")[["forecast", "same_period_last_year", "last_28_days"]].sum()
    item["yoy_change_pct"] = (item["forecast"] / item["same_period_last_year"] - 1) * 100
    daily_item = fut.groupby(["item", "date"])[["p10", "p50", "p90"]].sum().reset_index()
    width = daily_item.groupby("item").apply(lambda g: ((g["p90"] - g["p10"]) / g["p50"]).mean(), include_groups=False)
    item["relative_interval_width"] = width
    item = item.reset_index()
    item.to_csv(C.REPORTS / "item_insights.csv", index=False)

    store = si.groupby("store")[["forecast", "same_period_last_year"]].sum()
    store["yoy_change_pct"] = (store["forecast"] / store["same_period_last_year"] - 1) * 100
    store.reset_index().to_csv(C.REPORTS / "store_insights.csv", index=False)

    lt = fut[fut["date"] < f0 + pd.Timedelta(days=C.LEAD_TIME_DAYS)]
    pol = lt.groupby(["store", "item"]).agg(lead_time_demand=("p50", "sum")).reset_index()
    sigma = fut.assign(s=(fut["p90"] - fut["p10"]) / (2 * Z80)).groupby(["store", "item"])["s"].mean().rename("daily_sigma").reset_index()
    pol = pol.merge(sigma, on=["store", "item"])
    pol["safety_stock"] = Z95 * pol["daily_sigma"] * np.sqrt(C.LEAD_TIME_DAYS)
    pol["reorder_point"] = pol["lead_time_demand"] + pol["safety_stock"]
    pol.round(1).to_csv(C.REPORTS / "inventory_policy_example.csv", index=False)

    d = hist.assign(woy=hist["date"].dt.isocalendar().week.astype(int))
    wk = d.groupby("woy")["sales"].mean()
    wk = wk / wk.mean() * 100
    wk.rename("demand_index").to_csv(C.REPORTS / "weekly_seasonal_index.csv")

    top_growth = item.sort_values("yoy_change_pct", ascending=False).head(5)
    bottom = item.sort_values("yoy_change_pct").head(5)
    volatile = item.sort_values("relative_interval_width", ascending=False).head(5)
    slow = item.sort_values("forecast").head(5)
    summary = {
        "forecast_window": [str(f0.date()), str(f1.date())],
        "total_forecast_units": float(fut["p50"].sum()),
        "same_period_last_year_units": float(ly.sum()),
        "yoy_change_pct": float(fut["p50"].sum() / ly.sum() * 100 - 100),
        "vs_last_28_days_pct": float(fut["p50"].sum() / recent.sum() * 100 - 100),
        "items_with_negative_yoy": int((item["yoy_change_pct"] < 0).sum()),
        "yoy_range_items_pct": [float(item["yoy_change_pct"].min()), float(item["yoy_change_pct"].max())],
        "top_growth_items": {int(r.item): round(float(r.yoy_change_pct), 2) for r in top_growth.itertuples()},
        "lowest_growth_items": {int(r.item): round(float(r.yoy_change_pct), 2) for r in bottom.itertuples()},
        "most_volatile_items": {int(r.item): round(float(r.relative_interval_width), 3) for r in volatile.itertuples()},
        "slowest_items_forecast_units": {int(r.item): round(float(r.forecast), 0) for r in slow.itertuples()},
        "peak_weeks": {int(k): round(float(v), 1) for k, v in wk.sort_values(ascending=False).head(5).items()},
        "trough_weeks": {int(k): round(float(v), 1) for k, v in wk.sort_values().head(5).items()},
        "daily_peak_forecast": {str(r.date.date()): round(float(r.p50), 0) for r in fut.groupby("date")["p50"].sum().sort_values(ascending=False).head(3).reset_index().itertuples()},
        "lead_time_days": C.LEAD_TIME_DAYS,
        "safety_stock_share_of_lead_time_demand_pct": float(pol["safety_stock"].sum() / pol["lead_time_demand"].sum() * 100),
    }
    (C.REPORTS / "insights_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(compute(), indent=1))
