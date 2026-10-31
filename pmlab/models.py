"""Time-ordered splits, LightGBM fits and scoring for notebook 02."""

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import BOOK_FEATURES, CEX_FEATURES, ORACLE_FEATURES
from .stats import brier, cluster_bootstrap, log_loss, sigmoid

SEED = 0
ROUNDS = 400
VALIDATION = pd.Timedelta(hours=2)
FOLDS = ((12, 16), (16, 20), (20, 24))  # train until the first hour, test until the second

# name: (columns, start from the market price and learn only a correction)
FEATURE_SETS = {
    "binance": (CEX_FEATURES, False),
    "binance+oracle": (CEX_FEATURES + ORACLE_FEATURES, False),
    "binance+oracle+book": (CEX_FEATURES + ORACLE_FEATURES + BOOK_FEATURES, False),
    "price+correction": (CEX_FEATURES + ORACLE_FEATURES + BOOK_FEATURES, True),
}


def split(frame, train_end, test_end):
    """Masks for train, validation and test, by the hours of one UTC day.

    Each market sits entirely in one part: a market counts only if it has ended before
    the next part begins, so no outcome is shared across the boundary.
    """
    day = frame["ts"].min().normalize()
    cut = day + pd.Timedelta(hours=train_end)
    stop = day + pd.Timedelta(hours=test_end)
    start, end = frame["market_start_ts"], frame["market_end_ts"]
    return (end <= cut - VALIDATION,
            (start >= cut - VALIDATION) & (end <= cut),
            (start >= cut) & (end <= stop))


def fit(train, valid, columns, anchored, params):
    def dataset(d):
        return lgb.Dataset(d[columns].astype("float32"), label=d["y"].to_numpy(),
                           init_score=d["pm_logit"].to_numpy() if anchored else None)

    settings = dict(objective="binary", metric="binary_logloss", lambda_l2=20.0,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                    seed=SEED, deterministic=True, force_row_wise=True, **params)
    return lgb.train(settings, dataset(train), num_boost_round=ROUNDS,
                     valid_sets=[dataset(valid)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def predict(model, frame, columns, anchored):
    raw = model.predict(frame[columns].astype("float32"), raw_score=True)
    return sigmoid(raw + frame["pm_logit"].to_numpy() if anchored else raw)


def score(frame, p, baseline="mid"):
    """Brier, log loss and AUC, with the Brier gap to the market per opening time."""
    y = frame["y"].to_numpy()
    gap = (p - y) ** 2 - (frame[baseline].to_numpy() - y) ** 2
    mean, low, high = cluster_bootstrap(gap, frame["market_start_ts"].astype("int64").to_numpy())
    return {"markets": frame["market"].nunique(), "brier": brier(p, y), "log_loss": log_loss(p, y),
            "auc": roc_auc_score(y, p), "gap_to_mid": mean, "gap_low": low, "gap_high": high}
