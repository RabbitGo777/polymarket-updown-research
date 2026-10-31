"""A deliberately small feature set for predicting whether Up wins.

Every feature uses only what was known at the end of the bar it is stamped with.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import load_exchange, load_oracle, load_polymarket

PM_COLUMNS = ["bid_price", "ask_price", "bid_size", "ask_size", "mid_close"]
CEX_COLUMNS = ["mid_close", "quote_volume", "taker_buy_quote_volume"]

CEX_FEATURES = ["tau_sec", "ret_start_bps", "ret_5s_bps", "sigma_60_bps", "taker_imb_60"]
ORACLE_FEATURES = ["oracle_ret_bps"]
BOOK_FEATURES = ["pm_logit", "pm_spread_c", "pm_imb_top"]


def exchange_features(cex):
    logp = np.log(cex["mid_close"])
    step = logp.diff()
    quote = cex["quote_volume"].rolling(60, min_periods=1).sum()
    buy = cex["taker_buy_quote_volume"].rolling(60, min_periods=1).sum()
    return pd.DataFrame({
        "logp": logp,
        "ret_5s_bps": (logp - logp.shift(5)) * 1e4,
        "sigma_60_bps": step.rolling(60, min_periods=20).std() * 1e4,
        "taker_imb_60": ((2 * buy - quote) / quote.where(quote > 0)).fillna(0.0),
    })


def oracle_features(frame, oracle):
    """Oracle 60s TWAP versus its value at the market start, as known at each second.

    Up wins if the TWAP at the end is at or above the TWAP at the start. Rows before
    the starting value has reached us stay NaN.
    """
    oracle = oracle.dropna(subset=["twap_60s"])
    seen = pd.merge_asof(frame[["ts"]].assign(known=frame["ts"] + pd.Timedelta(seconds=1)),
                         oracle[["arrival", "twap_60s"]], left_on="known", right_on="arrival")
    strike = pd.merge_asof(frame[["market_start_ts"]].sort_values("market_start_ts"),
                           oracle[["ts", "twap_60s", "arrival"]], left_on="market_start_ts",
                           right_on="ts").reindex(frame.index)
    known = strike["arrival"] <= seen["known"]
    ret = (seen["twap_60s"].to_numpy() / strike["twap_60s"].to_numpy() - 1.0) * 1e4
    return np.where(known, ret, np.nan)


def build_features(coin, tf="5m", step=5):
    """One row per `step` seconds per market, on the Up token."""
    pm = load_polymarket(coin, tf, PM_COLUMNS)
    pm = pm[(pm["outcome"] == "Up") & pm["winning_outcome"].notna()]
    pm = pm.sort_values("ts", ignore_index=True)
    cex = exchange_features(load_exchange(coin, columns=CEX_COLUMNS))

    f = pm[["ts", "market_start_ts", "market_end_ts", "tau_sec", "coin"]].copy()
    f["market"] = pm["condition_id"].astype(str)
    f["y"] = (pm["winning_outcome"].astype(str) == "Up").astype(np.int8)
    f["mid"] = pm["mid_close"]
    f["pm_logit"] = np.log(pm["mid_close"].clip(0.005, 0.995)
                           / (1 - pm["mid_close"].clip(0.005, 0.995)))
    f["pm_spread_c"] = (pm["ask_price"] - pm["bid_price"]) * 100.0
    size = pm["bid_size"] + pm["ask_size"]
    f["pm_imb_top"] = ((pm["bid_size"] - pm["ask_size"]) / size.where(size > 0)).fillna(0.0)

    at = cex.reindex(pm["ts"])
    f["ret_5s_bps"] = at["ret_5s_bps"].to_numpy()
    f["sigma_60_bps"] = at["sigma_60_bps"].to_numpy()
    f["taker_imb_60"] = at["taker_imb_60"].to_numpy()
    f["ret_start_bps"] = (at["logp"].to_numpy()
                          - cex["logp"].reindex(pm["market_start_ts"]).to_numpy()) * 1e4
    f["oracle_ret_bps"] = oracle_features(f, load_oracle(coin))

    sd = f["sigma_60_bps"] * np.sqrt(f["tau_sec"])
    f["digital"] = norm.cdf(f["ret_start_bps"] / sd.where(sd > 0))
    elapsed = (f["ts"] - f["market_start_ts"]).dt.total_seconds().astype(int)
    return f[elapsed % step == 0].reset_index(drop=True)
