import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller, kpss

from . import config as C
from .evaluate import by_group, metrics

BLUE, ORANGE, GREY, DARK = "#2F5D8C", "#D9822B", "#9AA5B1", "#1B2430"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def save(name):
    plt.tight_layout()
    plt.savefig(C.IMAGES / name)
    plt.close()


def eda():
    df = pd.read_parquet(C.PROCESSED / "daily_sales.parquet")
    total = df.groupby("date")["sales"].sum()
    out = {}

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(total.index, total, color=GREY, lw=0.6, label="Daily units sold")
    for w, c in ((30, BLUE), (90, ORANGE)):
        ax.plot(total.index, total.rolling(w).mean(), color=c, lw=1.8, label=f"{w}-day rolling mean")
    ax.set(title="Total daily units sold across all stores and items", xlabel="Date", ylabel="Units sold per day")
    ax.legend()
    save("01_total_demand_trend.png")
    yearly = total.groupby(total.index.year).mean()
    out["yearly_avg_daily_units"] = {int(k): float(v) for k, v in yearly.items()}
    out["yearly_growth_pct"] = {int(k): float(v) for k, v in (yearly.pct_change() * 100).dropna().items()}

    d = df.assign(month=df["date"].dt.month, dow=df["date"].dt.dayofweek)
    heat = d.groupby(["month", "dow"])["sales"].mean().unstack()
    heat = heat / heat.values.mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    im = axes[0].imshow(heat.values, cmap="Blues", aspect="auto")
    axes[0].set(xticks=range(7), xticklabels=DAYS, yticks=range(12), yticklabels=MONTHS,
                title="Demand index by month and weekday (100 = overall mean)")
    axes[0].grid(False)
    plt.colorbar(im, ax=axes[0])
    dow = d.groupby("dow")["sales"].mean()
    axes[1].bar(DAYS, dow.values, color=[ORANGE if v == dow.max() else BLUE for v in dow.values])
    axes[1].set(title="Average units per store-item by weekday", ylabel="Units sold per day")
    save("02_seasonality_heatmap.png")
    out["weekday_index"] = {DAYS[i]: float(v / dow.mean() * 100) for i, v in dow.items()}
    mon = d.groupby("month")["sales"].mean()
    out["month_index"] = {MONTHS[i - 1]: float(v / mon.mean() * 100) for i, v in mon.items()}

    sub = d.groupby([d["date"].dt.year.rename("year"), "month"])["sales"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    for y in sub.columns:
        ax.plot(range(1, 13), sub[y], marker="o", ms=3, label=str(y))
    ax.set(xticks=range(1, 13), xticklabels=MONTHS, title="Monthly seasonal subseries by year",
           xlabel="Month", ylabel="Average units per store-item per day")
    ax.legend(ncol=5)
    save("03_seasonal_subseries.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["sales"], bins=60, color=BLUE)
    axes[0].set(title="Distribution of daily units per store-item", xlabel="Units sold per day", ylabel="Number of records")
    axes[1].hist(np.log1p(df["sales"]), bins=60, color=ORANGE)
    axes[1].set(title="Same distribution on log scale", xlabel="log(1 + units sold)", ylabel="Number of records")
    save("04_demand_distribution.png")
    q = df["sales"].quantile([0.5, 0.9, 0.99]).to_dict()
    out["distribution"] = {"p50": q[0.5], "p90": q[0.9], "p99": q[0.99], "max": float(df["sales"].max()),
                           "cv": float(df["sales"].std() / df["sales"].mean())}

    store = df.groupby("store")["sales"].mean().sort_values()
    item = df.groupby("item")["sales"].mean().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh([f"Store {s}" for s in store.index], store.values, color=BLUE)
    axes[0].set(title="Average daily units per item, by store", xlabel="Units sold per day")
    sel = pd.concat([item.head(8), item.tail(8)])
    axes[1].barh([f"Item {i}" for i in sel.index], sel.values, color=[GREY] * 8 + [ORANGE] * 8)
    axes[1].set(title="Lowest 8 and highest 8 items", xlabel="Units sold per day (avg across stores)")
    save("05_store_item_volume.png")
    out["store_avg"] = {int(k): float(v) for k, v in store.items()}
    out["item_avg_range"] = [float(item.min()), float(item.max())]

    yr = df.assign(year=df["date"].dt.year).groupby(["item", "year"])["sales"].sum().unstack()
    growth = (yr[2017] / yr[2016] - 1) * 100
    fig, ax = plt.subplots(figsize=(11, 3.8))
    order = growth.sort_values()
    ax.bar(order.index.astype(str), order.values, color=[ORANGE if v < growth.median() else BLUE for v in order.values])
    ax.set(title="Item-level demand growth, 2017 vs 2016", xlabel="Item", ylabel="Change in units sold (%)")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    save("06_item_growth.png")
    out["item_growth_2017_pct"] = {"min": float(growth.min()), "median": float(growth.median()), "max": float(growth.max())}

    store_shape = d.groupby(["store", "month"])["sales"].mean().unstack(0)
    store_shape = store_shape / store_shape.mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    for s in store_shape.columns:
        ax.plot(range(1, 13), store_shape[s], lw=1.2, label=f"Store {s}")
    ax.set(xticks=range(1, 13), xticklabels=MONTHS, title="Seasonal shape by store (each store scaled to its own mean)",
           xlabel="Month", ylabel="Relative demand")
    ax.legend(ncol=5, fontsize=7)
    save("07_store_seasonal_shape.png")
    out["store_shape_spread_peak_month"] = float(store_shape.loc[7].max() - store_shape.loc[7].min())

    mstl = MSTL(np.log(total), periods=[7, 365]).fit()
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    parts = [("Observed (log units)", np.log(total)), ("Trend", mstl.trend),
             ("Weekly seasonality", mstl.seasonal["seasonal_7"]),
             ("Annual seasonality", mstl.seasonal["seasonal_365"])]
    for ax, (name, ser) in zip(axes, parts):
        ax.plot(total.index, ser, color=BLUE, lw=0.9)
        ax.set_ylabel(name, fontsize=8)
    axes[-1].set_xlabel("Date")
    axes[0].set_title("Multiplicative decomposition (MSTL on log scale): trend, weekly and annual seasonality")
    save("08_decomposition.png")
    resid = mstl.resid
    out["decomposition"] = {
        "method": "MSTL on log(total daily units), periods 7 and 365 (log-additive = multiplicative)",
        "weekly_amplitude_pct": float((np.exp(mstl.seasonal["seasonal_7"].max()) - np.exp(mstl.seasonal["seasonal_7"].min())) * 100),
        "annual_amplitude_pct": float((np.exp(mstl.seasonal["seasonal_365"].max()) - np.exp(mstl.seasonal["seasonal_365"].min())) * 100),
        "residual_std_log": float(resid.std()),
        "residual_autocorr_lag1": float(resid.autocorr(1)),
        "residual_autocorr_lag7": float(resid.autocorr(7)),
    }
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(total.index, resid, color=GREY, lw=0.6)
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set(title="Decomposition residual (log scale)", xlabel="Date", ylabel="Residual")
    save("09_decomposition_residual.png")

    adf = adfuller(total.values, autolag="AIC")
    adf_d = adfuller(total.diff().dropna().values, autolag="AIC")
    adf_s = adfuller(total.diff(7).dropna().values, autolag="AIC")
    kp = kpss(total.values, regression="c", nlags="auto")
    out["stationarity"] = {
        "adf_level_pvalue": float(adf[1]), "adf_diff1_pvalue": float(adf_d[1]),
        "adf_seasonal_diff7_pvalue": float(adf_s[1]), "kpss_level_pvalue": float(kp[1]),
    }
    rm = total.rolling(90)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(total.index, rm.mean(), color=BLUE, label="90-day rolling mean")
    ax2 = ax.twinx()
    ax2.plot(total.index, rm.std(), color=ORANGE, label="90-day rolling std")
    ax.set(title="Rolling mean and rolling standard deviation (stability of level and variance)", xlabel="Date", ylabel="Mean units per day")
    ax2.set_ylabel("Std of units per day")
    ax2.grid(False)
    save("10_rolling_stats.png")

    (C.REPORTS / "eda_findings.json").write_text(json.dumps(out, indent=2))
    return out


def results():
    preds = pd.read_parquet(C.PROCESSED / "predictions.parquet")
    fut = pd.read_parquet(C.PROCESSED / "future_forecast.parquet")
    hist = pd.read_parquet(C.PROCESSED / "daily_sales.parquet")
    metr = json.loads((C.REPORTS / "metrics.json").read_text())
    out = {}

    labels = ["Seasonal naive (last same weekday)", "Moving average (28 days)", "Random Forest", "LightGBM"]
    vals = [metr["validation"][k]["MAE"] for k in labels]
    tests = [metr["test"][k]["MAE"] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, vals, 0.4, color=BLUE, label="Validation")
    ax.bar(x + 0.2, tests, 0.4, color=ORANGE, label="Test")
    for i, (a, b) in enumerate(zip(vals, tests)):
        ax.text(i - 0.2, a + 0.1, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.1, f"{b:.2f}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=[l.replace(" (", "\n(") for l in labels], ylabel="Mean absolute error (units per store-item per day)",
           title="Forecast error by model against the baselines (lower is better)")
    ax.legend()
    save("11_model_comparison.png")

    test = preds[preds["split"] == "test"].copy()
    test["resid"] = test["actual"] - test["lgbm"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(test["resid"], bins=80, color=BLUE)
    axes[0].axvline(0, color=ORANGE)
    axes[0].set(title="Residual distribution, LightGBM on test", xlabel="Actual minus forecast (units)", ylabel="Number of records")
    daily = test.groupby("date").agg(actual=("actual", "sum"), pred=("lgbm", "sum"))
    axes[1].plot(daily.index, daily["actual"], color=BLUE, label="Actual")
    axes[1].plot(daily.index, daily["pred"], color=ORANGE, label="Forecast")
    axes[1].set(title="Actual vs forecast, all stores and items (test period)", xlabel="Date", ylabel="Units sold per day")
    axes[1].legend()
    save("12_actual_vs_forecast_total.png")
    dm = daily["actual"] - daily["pred"]
    out["total_level"] = {"MAE": float(dm.abs().mean()), "WAPE_pct": float(dm.abs().sum() / daily["actual"].sum() * 100)}

    s, i = 1, 1
    one = preds[(preds["store"] == s) & (preds["item"] == i) & (preds["split"] == "test")].sort_values("date")
    one_f = fut[(fut["store"] == s) & (fut["item"] == i)].sort_values("date")
    h = hist[(hist["store"] == s) & (hist["item"] == i) & (hist["date"] > "2017-09-30")]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(h["date"], h["sales"], color=BLUE, lw=1, label="Actual demand")
    ax.plot(one["date"], one["lgbm"], color=ORANGE, lw=1, alpha=0.8, label="Forecast on test period")
    ax.fill_between(one["date"], one["p10"], one["p90"], color=ORANGE, alpha=0.15, label="80% prediction interval")
    ax.plot(one_f["date"], one_f["p50"], color=ORANGE, lw=1.8, label="28-day forecast")
    ax.fill_between(one_f["date"], one_f["p10"], one_f["p90"], color=ORANGE, alpha=0.3)
    ax.axvline(pd.Timestamp("2018-01-01"), color=DARK, ls="--", lw=1)
    ax.text(pd.Timestamp("2018-01-01"), ax.get_ylim()[1] * 0.96, " Forecast origin", fontsize=9)
    ax.set(title="Store 1, Item 1: history, test-period forecast and 28-day outlook", xlabel="Date", ylabel="Units sold per day")
    ax.legend(loc="lower left", fontsize=8)
    save("13_forecast_example.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    err = test.assign(abs_err=test["resid"].abs())
    err.groupby("date")["abs_err"].mean().plot(ax=axes[0, 0], color=BLUE, lw=0.9)
    axes[0, 0].set(title="Mean absolute error by date (test)", xlabel="Date", ylabel="Units per store-item")
    g = by_group(test.assign(dow=test["date"].dt.dayofweek), "lgbm", "dow")
    axes[0, 1].bar(DAYS, g["MAE"], color=BLUE)
    axes[0, 1].set(title="Mean absolute error by weekday", ylabel="Units per store-item")
    g = by_group(test, "lgbm", "store")
    axes[1, 0].bar(g.index.astype(str), g["WAPE"], color=BLUE)
    axes[1, 0].set(title="Weighted error by store", xlabel="Store", ylabel="WAPE (%)")
    g = by_group(test.assign(month=test["date"].dt.month), "lgbm", "month")
    axes[1, 1].bar([MONTHS[m - 1] for m in g.index], g["WAPE"], color=BLUE)
    axes[1, 1].set(title="Weighted error by month", ylabel="WAPE (%)")
    save("14_error_analysis.png")

    daily_err = err.groupby("date")["abs_err"].mean()
    worst_dates = daily_err.sort_values(ascending=False).head(10)
    by_item = by_group(test, "lgbm", "item").sort_values("WAPE", ascending=False)
    by_store = by_group(test, "lgbm", "store").sort_values("WAPE", ascending=False)
    by_dow = by_group(test.assign(dow=test["date"].dt.dayofweek), "lgbm", "dow")
    by_month = by_group(test.assign(month=test["date"].dt.month), "lgbm", "month")
    test["bias_dow"] = test["resid"]
    out["errors"] = {
        "worst_dates": {str(k.date()): float(v) for k, v in worst_dates.items()},
        "worst_items_wape": {int(k): float(v) for k, v in by_item["WAPE"].head(5).items()},
        "best_items_wape": {int(k): float(v) for k, v in by_item["WAPE"].tail(3).items()},
        "store_wape": {int(k): float(v) for k, v in by_store["WAPE"].items()},
        "dow_wape": {DAYS[k]: float(v) for k, v in by_dow["WAPE"].items()},
        "month_wape": {MONTHS[k - 1]: float(v) for k, v in by_month["WAPE"].items()},
        "bias_by_month": {MONTHS[k - 1]: float(v) for k, v in test.groupby(test["date"].dt.month)["resid"].mean().items()},
        "bias_overall": float(test["resid"].mean()),
        "share_abs_error_top1pct_rows": float(err["abs_err"].nlargest(int(len(err) * 0.01)).sum() / err["abs_err"].sum()),
        "interval_coverage_by_month": {MONTHS[k - 1]: float(v) for k, v in
                                       test.assign(c=(test["actual"] >= test["p10"]) & (test["actual"] <= test["p90"]))
                                       .groupby(test["date"].dt.month)["c"].mean().items()},
    }
    by_item.to_csv(C.REPORTS / "error_by_item.csv")
    by_store.to_csv(C.REPORTS / "error_by_store.csv")
    (C.REPORTS / "error_findings.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import sys
    C.IMAGES.mkdir(exist_ok=True)
    if "eda" in sys.argv or len(sys.argv) == 1:
        print(json.dumps(eda(), indent=1))
    if "results" in sys.argv or len(sys.argv) == 1:
        print(json.dumps(results(), indent=1))
