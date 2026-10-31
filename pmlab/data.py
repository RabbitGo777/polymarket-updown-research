"""Download and load the free one-day sample published on Hugging Face."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

DAY = "20261020"
COINS = ("btc", "eth", "sol", "xrp", "doge", "bnb")
HF_USER = "RabbitGo777"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PMLAB_DATA", ROOT / "data"))


def agg_repo(coin):
    return f"{HF_USER}/polymarket.{coin}_{DAY}.v4.1s.parquet"


def raw_repo(coin):
    return f"{HF_USER}/polymarket.{coin}_{DAY}.v4.parquet"


def download_agg(coins=COINS):
    """Fetch the 1-second aggregates (about 1.1 GB for six coins). No account needed."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    for coin in coins:
        snapshot_download(agg_repo(coin), repo_type="dataset",
                          local_dir=DATA_DIR / "agg", allow_patterns="*.parquet")


def download_raw(coin, files):
    """Fetch named raw tick files for one coin into data/raw."""
    from huggingface_hub import hf_hub_download

    return [Path(hf_hub_download(raw_repo(coin), name, repo_type="dataset",
                                 local_dir=DATA_DIR / "raw")) for name in files]


def _agg(name):
    return DATA_DIR / "agg" / f"{name}_{DAY}.v4.1s.parquet"


def in_window(pm):
    """Keep the seconds a market is live.

    Polymarket quotes a market about 30 seconds before it opens and keeps
    publishing after it ends, so these rows must be dropped by hand.
    """
    return pm[(pm["ts"] >= pm["market_start_ts"]) & (pm["ts"] < pm["market_end_ts"])]


def load_polymarket(coin, tf="5m", columns=None):
    """One row per token per second, live window only, with seconds to expiry."""
    keys = ["ts", "condition_id", "outcome", "market_start_ts", "market_end_ts",
            "winning_outcome"]
    cols = None if columns is None else list(dict.fromkeys(keys + list(columns)))
    pm = in_window(pd.read_parquet(_agg(f"polymarket.{coin}.{tf}"), columns=cols)).copy()
    pm["outcome"] = pm["outcome"].astype(str)
    pm["coin"] = coin
    pm["tau_sec"] = (pm["market_end_ts"] - pm["ts"]).dt.total_seconds()
    return pm.reset_index(drop=True)


def load_exchange(coin, venue="binance", instrument_class="perp", columns=("mid_close",)):
    """One instrument on a complete 1-second grid; prices carried forward."""
    cols = ["ts", "instrument_class", *columns]
    d = pd.read_parquet(_agg(f"{venue}.{coin}"), columns=list(dict.fromkeys(cols)))
    d = d[d["instrument_class"] == instrument_class].drop(columns="instrument_class")
    d = d.drop_duplicates("ts").set_index("ts").sort_index().asfreq("s")
    flows = [c for c in d.columns if c.endswith("volume") or c == "trades"]
    d[flows] = d[flows].fillna(0.0)
    return d.ffill()


def load_oracle(coin):
    """Chainlink rounds with `arrival`, the moment a value was actually known.

    `ts` is the oracle's round time, and the value reaches the collector later,
    so point-in-time joins must key on `arrival`.
    """
    o = pd.read_parquet(_agg(f"chainlink.{coin}"),
                        columns=["ts", "price", "twap_60s", "event_lag_ms", "n_topics"])
    o = o.dropna(subset=["price"]).sort_values("ts").reset_index(drop=True)
    o["arrival"] = o["ts"] + pd.to_timedelta(o["event_lag_ms"].clip(lower=0), unit="ms")
    o["arrival"] = o["arrival"].cummax()
    return o


def market_outcomes(pm):
    """One row per market with its result, for markets that settled inside the day."""
    m = (pm.groupby("condition_id", observed=True)
         .agg(start=("market_start_ts", "first"), end=("market_end_ts", "first"),
              winner=("winning_outcome", "first"))
         .reset_index().dropna(subset=["winner"]))
    m["up_won"] = (m["winner"].astype(str) == "Up").astype(np.int8)
    return m.sort_values("start").reset_index(drop=True)
