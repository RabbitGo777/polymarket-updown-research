import numpy as np
import pandas as pd

from pmlab.data import in_window
from pmlab.features import exchange_features, oracle_features
from pmlab.models import split
from pmlab.stats import brier, cluster_bootstrap, log_loss, taker_fee


def exchange(n=300, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-10-20", periods=n, freq="s", name="ts")
    return pd.DataFrame({"mid_close": 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n))),
                         "quote_volume": rng.uniform(1, 9, n),
                         "taker_buy_quote_volume": rng.uniform(0, 1, n)}, index=index)


def test_in_window_drops_rows_outside_the_market():
    start = pd.Timestamp("2026-10-20 00:05")
    ts = start + pd.to_timedelta([-30, 0, 299, 300], unit="s")
    pm = pd.DataFrame({"ts": ts, "market_start_ts": start,
                       "market_end_ts": start + pd.Timedelta(minutes=5)})
    assert list(in_window(pm)["ts"]) == [ts[1], ts[2]]


def test_exchange_features_ignore_the_future():
    data = exchange()
    changed = data.copy()
    changed.iloc[200:, 0] *= 1.5
    changed.iloc[200:, 1] *= 9
    pd.testing.assert_frame_equal(exchange_features(data).iloc[:200],
                                  exchange_features(changed).iloc[:200])


def test_oracle_return_waits_for_the_opening_value_to_arrive():
    start = pd.Timestamp("2026-10-20 00:05")
    frame = pd.DataFrame({"ts": start + pd.to_timedelta([0, 1, 2, 3], unit="s"),
                          "market_start_ts": start})
    oracle = pd.DataFrame({"ts": start + pd.to_timedelta([0, 1, 2], unit="s"),
                           "twap_60s": [100.0, 101.0, 102.0]})
    oracle["arrival"] = oracle["ts"] + pd.Timedelta(seconds=2)
    ret = oracle_features(frame, oracle)
    assert np.isnan(ret[0])
    assert ret[1] == 0
    assert np.isclose(ret[2], 100.0)
    assert np.isclose(ret[3], 200.0)


def test_oracle_rounds_without_a_twap_are_skipped():
    start = pd.Timestamp("2026-10-20 00:05")
    frame = pd.DataFrame({"ts": [start + pd.Timedelta(seconds=9)], "market_start_ts": start})
    oracle = pd.DataFrame({"ts": start + pd.to_timedelta([0, 1, 2], unit="s"),
                           "twap_60s": [100.0, 101.0, np.nan]})
    oracle["arrival"] = oracle["ts"]
    assert np.isclose(oracle_features(frame, oracle)[0], 100.0)


def test_split_keeps_each_market_in_one_part():
    starts = pd.date_range("2026-10-20", periods=288, freq="5min")
    frame = pd.DataFrame({"ts": starts, "market_start_ts": starts,
                          "market_end_ts": starts + pd.Timedelta(minutes=5)})
    train, valid, test = split(frame, 12, 16)
    assert not (train & valid).any() and not (valid & test).any() and not (train & test).any()
    assert frame.loc[train, "market_end_ts"].max() <= frame.loc[valid, "market_start_ts"].min()
    assert frame.loc[valid, "market_end_ts"].max() <= frame.loc[test, "market_start_ts"].min()
    assert test.sum() == 48


def test_scores():
    assert brier([1.0, 0.0], [1, 0]) == 0
    assert np.isclose(log_loss([0.5, 0.5], [1, 0]), np.log(2))
    assert np.isclose(taker_fee(0.5), 0.0175)


def test_bootstrap_widens_when_rows_are_clustered():
    rng = np.random.default_rng(0)
    values = np.repeat(rng.normal(size=20), 50)
    clusters = np.repeat(np.arange(20), 50)
    _, low, high = cluster_bootstrap(values, clusters)
    _, low_rows, high_rows = cluster_bootstrap(values, np.arange(len(values)))
    assert high - low > 3 * (high_rows - low_rows)
