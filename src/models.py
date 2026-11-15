import math
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config as C
from .features import FEATURES, CATEGORICAL


def blocks(dates, start, end, horizon=C.HORIZON):
    d = [x for x in dates if pd.Timestamp(start) <= x <= pd.Timestamp(end)]
    return [d[i:i + horizon] for i in range(0, len(d), horizon)]


def wide_sales(df):
    return df.pivot(index="date", columns=["store", "item"], values="sales").sort_index()


def origin_baselines(wide, block_list):
    idx = {d: i for i, d in enumerate(wide.index)}
    arr = wide.to_numpy(dtype="float64")
    seasonal = np.full(arr.shape, np.nan)
    moving = np.full(arr.shape, np.nan)
    for blk in block_list:
        o = idx[blk[0]] - 1
        avg = arr[o - 27:o + 1].mean(axis=0)
        for j, d in enumerate(blk):
            h = j + 1
            k = 7 * math.ceil(h / 7)
            seasonal[idx[d]] = arr[o + h - k]
            moving[idx[d]] = avg
    return seasonal, moving


def ets_forecasts(wide, block_list, series_positions, window=730):
    idx = {d: i for i, d in enumerate(wide.index)}
    arr = wide.to_numpy(dtype="float64")
    out = np.full(arr.shape, np.nan)
    warnings.filterwarnings("ignore")
    for blk in block_list:
        o = idx[blk[0]] - 1
        for p in series_positions:
            y = arr[max(0, o - window + 1):o + 1, p]
            try:
                fit = ExponentialSmoothing(
                    y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7
                ).fit()
                out[idx[blk[0]]:idx[blk[0]] + len(blk), p] = np.clip(fit.forecast(len(blk)), 0, None)
            except Exception:
                out[idx[blk[0]]:idx[blk[0]] + len(blk), p] = arr[o - 6:o + 1, p].mean()
    return out


def base_level(frame):
    return (frame["roll_mean_91"].to_numpy(dtype="float64") + 1.0)


def target_and_weight(frame, target_mode):
    y = frame["sales"].to_numpy(dtype="float64")
    if target_mode == "ratio":
        b = base_level(frame)
        return y / b, b
    return y, None


def to_dataset(frame, target_mode):
    y, w = target_and_weight(frame, target_mode)
    return lgb.Dataset(
        frame[FEATURES], label=y, weight=w, categorical_feature=CATEGORICAL, free_raw_data=False
    )


def lgb_params(objective, num_leaves, alpha=None, lr=0.06):
    p = {
        "objective": objective,
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "seed": C.SEED,
        "verbosity": -1,
        "num_threads": 0,
    }
    if alpha is not None:
        p["alpha"] = alpha
    return p


def fit_lgbm(train, params, target_mode, valid=None, rounds=2000, patience=80):
    t0 = time.time()
    dtrain = to_dataset(train, target_mode)
    callbacks = []
    valid_sets = None
    if valid is not None:
        valid_sets = [to_dataset(valid, target_mode)]
        callbacks = [lgb.early_stopping(patience, verbose=False)]
    model = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)
    return model, time.time() - t0


def predict_lgbm(model, frame, target_mode, num_iteration=None):
    p = model.predict(frame[FEATURES], num_iteration=num_iteration)
    if target_mode == "ratio":
        p = p * base_level(frame)
    return np.clip(p, 0, None)


def fit_rf(train, params):
    t0 = time.time()
    X = train[FEATURES].astype("float32")
    model = RandomForestRegressor(**params, random_state=C.SEED, n_jobs=-1)
    model.fit(X, train["sales"].to_numpy())
    return model, time.time() - t0


def predict_rf(model, frame):
    return np.clip(model.predict(frame[FEATURES].astype("float32")), 0, None)
