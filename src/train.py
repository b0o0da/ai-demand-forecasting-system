import json
import os
import platform
import time

import lightgbm
import numpy as np
import pandas as pd
import sklearn
import statsmodels

from . import config as C
from . import features as F
from . import models as M
from .evaluate import interval_report, metrics

PRED_COLS = {
    "Seasonal naive (last same weekday)": "naive_origin",
    "Seasonal naive (same weekday, 28 days earlier)": "naive_lag28",
    "Moving average (28 days)": "ma_origin",
    "Exponential smoothing (Holt-Winters)": "ets",
    "Random Forest": "rf",
    "LightGBM": "lgbm",
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run():
    t_start = time.time()
    df = pd.read_parquet(C.PROCESSED / "daily_sales.parquet")
    frame = F.build(F.extend_calendar(df))
    train_m, val_m, test_m = F.split_masks(frame)
    train, val, test = frame[train_m], frame[val_m], frame[test_m]
    trainval = frame[train_m | val_m]
    allhist = frame[train_m | val_m | test_m]
    future = frame[frame["date"] > df["date"].max()].copy()
    log(f"rows train={len(train)} val={len(val)} test={len(test)} future={len(future)}")

    wide = M.wide_sales(df)
    dates = list(wide.index)
    val_blocks = M.blocks(dates, val["date"].min(), val["date"].max())
    test_blocks = M.blocks(dates, test["date"].min(), test["date"].max())
    seas, mov = M.origin_baselines(wide, val_blocks + test_blocks)
    n, m = wide.shape
    row_pos = {d: i for i, d in enumerate(wide.index)}
    series_pos = {c: j for j, c in enumerate(wide.columns)}

    def lookup(arr, part):
        i = part["date"].map(row_pos).to_numpy()
        j = pd.Series(list(zip(part["store"], part["item"]))).map(series_pos).to_numpy()
        return arr[i, j]

    for part in (val, test):
        part["naive_origin"] = lookup(seas, part)
        part["ma_origin"] = lookup(mov, part)
        part["naive_lag28"] = part["lag_28"]

    log("LightGBM tuning on validation")
    grid = [
        ("raw", "regression", 63),
        ("raw", "regression_l1", 63),
        ("ratio", "regression_l1", 63),
        ("ratio", "regression_l1", 127),
    ]
    tuning = []
    best = None
    for mode, obj, leaves in grid:
        model, secs = M.fit_lgbm(train, M.lgb_params(obj, leaves), mode, valid=val)
        pred = M.predict_lgbm(model, val, mode, model.best_iteration)
        mae = float(np.abs(val["sales"] - pred).mean())
        tuning.append({"target": mode, "objective": obj, "num_leaves": leaves,
                       "best_iteration": int(model.best_iteration), "val_MAE": mae, "seconds": secs})
        log(f"  {mode}/{obj}/{leaves}: val MAE {mae:.3f} iters {model.best_iteration} ({secs:.0f}s)")
        if best is None or mae < best["val_MAE"]:
            best = {**tuning[-1], "model": model, "pred": pred}
    mode, obj, leaves, iters = best["target"], best["objective"], best["num_leaves"], best["best_iteration"]
    val["lgbm"] = best["pred"]
    timing = {"LightGBM": {"train_seconds": best["seconds"]}}

    log("Quantile models (train only)")
    q = {}
    for name, alpha in (("p10", 0.1), ("p90", 0.9)):
        qm, secs = M.fit_lgbm(train, M.lgb_params("quantile", leaves, alpha=alpha), mode, rounds=iters)
        q[name] = qm
        val[name] = M.predict_lgbm(qm, val, mode)
    val["p10"], val["p90"] = np.minimum(val["p10"], val["lgbm"]), np.maximum(val["p90"], val["lgbm"])

    log("Random Forest on validation")
    rf_params = dict(n_estimators=60, max_depth=14, min_samples_leaf=30, max_features=0.5, max_samples=0.25)
    rf, secs = M.fit_rf(train, rf_params)
    val["rf"] = M.predict_rf(rf, val)
    timing["Random Forest"] = {"train_seconds": secs}
    del rf

    log("Refit on train+validation for test")
    rounds_final = int(iters * 1.15)
    final_tv, secs = M.fit_lgbm(trainval, M.lgb_params(obj, leaves), mode, rounds=rounds_final)
    test["lgbm"] = M.predict_lgbm(final_tv, test, mode)
    qtv = {}
    for name, alpha in (("p10", 0.1), ("p90", 0.9)):
        qtv[name], _ = M.fit_lgbm(trainval, M.lgb_params("quantile", leaves, alpha=alpha), mode, rounds=rounds_final)
        test[name] = M.predict_lgbm(qtv[name], test, mode)
    test["p10"], test["p90"] = np.minimum(test["p10"], test["lgbm"]), np.maximum(test["p90"], test["lgbm"])
    rf, _ = M.fit_rf(trainval, rf_params)
    test["rf"] = M.predict_rf(rf, test)
    del rf

    log("Final models on all history, future forecast")
    rounds_all = int(iters * 1.2)
    final_all, _ = M.fit_lgbm(allhist, M.lgb_params(obj, leaves), mode, rounds=rounds_all)
    future["p50"] = M.predict_lgbm(final_all, future, mode)
    for name, alpha in (("p10", 0.1), ("p90", 0.9)):
        qa, _ = M.fit_lgbm(allhist, M.lgb_params("quantile", leaves, alpha=alpha), mode, rounds=rounds_all)
        future[name] = M.predict_lgbm(qa, future, mode)
        qa.save_model(str(C.MODELS / f"lgbm_{name}.txt"))
    future["p10"], future["p90"] = np.minimum(future["p10"], future["p50"]), np.maximum(future["p90"], future["p50"])
    final_all.save_model(str(C.MODELS / "lgbm_point.txt"))

    imp = pd.DataFrame({
        "feature": F.FEATURES,
        "gain": final_all.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp.to_csv(C.REPORTS / "feature_importance.csv", index=False)

    keep = ["date", "store", "item", "sales", "naive_origin", "naive_lag28", "ma_origin", "rf", "lgbm", "p10", "p90"]
    val["split"], test["split"] = "validation", "test"
    preds = pd.concat([val[keep + ["split"]], test[keep + ["split"]]], ignore_index=True)
    preds = preds.rename(columns={"sales": "actual"})
    preds["ets"] = np.nan
    for c in preds.columns:
        if preds[c].dtype == "float64":
            preds[c] = preds[c].astype("float32")
    preds.to_parquet(C.PROCESSED / "predictions.parquet", index=False)
    future[["date", "store", "item", "p10", "p50", "p90"]].astype({"p10": "float32", "p50": "float32", "p90": "float32"}).to_parquet(
        C.PROCESSED / "future_forecast.parquet", index=False
    )

    result = {"validation": {}, "test": {}}
    for split, part in (("validation", val), ("test", test)):
        for label, col in PRED_COLS.items():
            if col in part and part[col].notna().any():
                result[split][label] = metrics(part["sales"], part[col])
        result[split]["_interval_80"] = interval_report(part["sales"], part["p10"], part["p90"])
    result["tuning"] = tuning
    result["chosen"] = {"target": mode, "objective": obj, "num_leaves": leaves,
                        "best_iteration_on_validation": iters, "rounds_train_plus_val": rounds_final,
                        "rounds_all_history": rounds_all}
    result["timing"] = timing
    result["splits"] = {
        "train": [str(train["date"].min().date()), C.TRAIN_END],
        "validation": [str(val["date"].min().date()), C.VAL_END],
        "test": [str(test["date"].min().date()), str(test["date"].max().date())],
        "rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
    }
    result["environment"] = {
        "python": platform.python_version(), "lightgbm": lightgbm.__version__,
        "scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
        "pandas": pd.__version__, "numpy": np.__version__, "cpu_count": os.cpu_count(),
        "seed": C.SEED, "horizon_days": C.HORIZON, "rf_params": rf_params,
        "total_pipeline_seconds": time.time() - t_start,
    }
    (C.REPORTS / "metrics.json").write_text(json.dumps(result, indent=2))
    log("done " + json.dumps({k: round(v["MAE"], 3) for k, v in result["test"].items() if not k.startswith("_")}))


if __name__ == "__main__":
    run()
