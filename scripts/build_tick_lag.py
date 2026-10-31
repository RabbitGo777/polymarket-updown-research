"""Measure how fast Polymarket follows Binance, from raw tick data.

The aggregates are 1-second bars, too coarse for a reaction that takes a few hundred
milliseconds. This script reads the raw messages of one day and writes small CSVs to
results/, which notebook 01 plots.

    python scripts/build_tick_lag.py                  # downloads about 7 GB
    python scripts/build_tick_lag.py --coins btc      # about 2 GB
    python scripts/build_tick_lag.py --raw-dir DIR    # raw files already on disk
"""

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmlab.data import COINS, DATA_DIR, DAY, ROOT, download_raw

BIN_MS = 50
WINDOW = 4  # the move is measured over 4 bins, 200 ms
MIN_GAP = 60  # bins between events, 3 s
BEFORE, AFTER = 40, 120  # the event window, -2 s to +6 s
TAIL_QUANTILE = 0.9995  # an event is a move in the top 0.05% of 200 ms moves
MIN_PRICE, MAX_PRICE = 0.2, 0.8  # far from 0.5 the Up price barely responds
XCORR_BINS = np.arange(-20, 41)
BOOT = 500


def read_ticks(con, coin, raw_dir, clock="receive"):
    """Binance perp and Polymarket Up mids, binned on one of two clocks.

    "receive" is the collector's clock, so it includes each feed's delivery delay.
    "venue" is the timestamp each exchange put on the message, so it removes that
    delay but rests on the two exchanges' clocks agreeing.
    """
    cex_ts, pm_ts = (("ts_local", "ts_local") if clock == "receive"
                     else ("E_ * 1000", '"timestamp" * 1000'))
    binance = raw_dir / f"binance.{coin}_{DAY}.v4.parquet"
    poly = raw_dir / f"polymarket.{coin}.5m_{DAY}.v4.parquet"
    if not (binance.exists() and poly.exists()):
        download_raw(coin, [binance.name, poly.name])
    us = BIN_MS * 1000
    cex = con.execute(f"""
        select ({cex_ts} // {us})::bigint as bin, arg_max((b + a) / 2, seq_local) as mid
        from '{binance}'
        where kind_local like 'binance.futures%' and e = 'bookTicker'
        group by 1 order by 1""").df()
    pm = con.execute(f"""
        with m as (select distinct y_token, start_ts, end_ts from '{poly}'
                   where source_local = 'collector' and y_token is not null)
        select (p.{pm_ts} // {us})::bigint as bin, m.start_ts, m.end_ts,
               arg_max((best_bid + best_ask) / 2, seq_local) as mid
        from '{poly}' p join m on p.asset_id = m.y_token
        where p.event_type in ('price_change', 'best_bid_ask')
          and p.best_bid > 0 and p.best_ask < 1
          and p.ts_local >= m.start_ts * 1000000 and p.ts_local < m.end_ts * 1000000
        group by 1, 2, 3""").df()
    lag = con.execute(f"""
        select (select median(ts_local - E_ * 1000) / 1000 from '{binance}'
                where kind_local like 'binance.futures%' and e = 'bookTicker') as binance,
               (select median(ts_local - "timestamp" * 1000) / 1000 from '{poly}'
                where event_type = 'price_change') as polymarket""").fetchone()
    return cex, pm, lag


def to_grid(cex, pm):
    """Last known price per bin on one grid, and the market live in each bin."""
    first, last = int(pm["bin"].min()), int(pm["bin"].max()) + 1
    grid = np.arange(first, last)
    binance = pd.Series(cex["mid"].to_numpy(), index=cex["bin"]).reindex(grid).ffill()
    price = np.full(len(grid), np.nan)
    market = np.full(len(grid), -1, dtype=np.int64)
    per_second = 1000 // BIN_MS
    for (start, end), d in pm.groupby(["start_ts", "end_ts"]):
        lo = max(start * per_second - first, 0)
        hi = min(end * per_second - first, len(grid))
        price[lo:hi] = (pd.Series(d["mid"].to_numpy(), index=d["bin"].to_numpy() - first)
                        .sort_index().reindex(np.arange(lo, hi)).ffill().to_numpy())
        market[lo:hi] = start
    return np.log(binance.to_numpy()) * 1e4, price, market


def find_events(log_price, price, market):
    moves = log_price[WINDOW:] - log_price[:-WINDOW]
    threshold = float(np.nanquantile(np.abs(moves), TAIL_QUANTILE))
    events, last = [], -MIN_GAP
    for i in np.flatnonzero(np.abs(moves) >= threshold) + WINDOW:
        if i - last < MIN_GAP:
            continue
        last = i
        lo, hi = i - WINDOW - BEFORE, i + AFTER
        if lo < 0 or hi >= len(price) or market[lo] < 0 or market[lo] != market[hi]:
            continue
        if MIN_PRICE <= price[i - WINDOW] <= MAX_PRICE and np.isfinite(price[lo:hi + 1]).all():
            events.append(i)
    return np.array(events), threshold


def crossing(curve, offsets, share):
    """First time at or after the event that a curve reaches `share` of its final level."""
    final = curve[-20:].mean()
    return offsets[(offsets >= 0) & (curve >= share * final)][0] * BIN_MS


def event_study(log_price, price, events, seed=0):
    offsets = np.arange(-BEFORE, AFTER + 1)
    base = events - WINDOW
    sign = np.sign(log_price[events] - log_price[base])
    pm = np.stack([sign * (price[events + h] - price[base]) for h in offsets], axis=1)
    cex = np.stack([sign * (log_price[events + h] - log_price[base]) for h in offsets], axis=1)
    pm_curve, cex_curve = pm.mean(0), cex.mean(0)
    rng = np.random.default_rng(seed)
    half = [crossing(pm[rng.integers(0, len(pm), len(pm))].mean(0), offsets, 0.5)
            for _ in range(BOOT)]
    summary = {
        "events": len(events),
        "pm_move_cents": pm_curve[-20:].mean() * 100,
        "binance_move_bps": cex_curve[-20:].mean(),
        "pm_t50_ms": crossing(pm_curve, offsets, 0.5),
        "pm_t50_lo": np.percentile(half, 2.5),
        "pm_t50_hi": np.percentile(half, 97.5),
        "pm_t90_ms": crossing(pm_curve, offsets, 0.9),
    }
    curve = pd.DataFrame({"ms": offsets * BIN_MS,
                          "polymarket_pct": 100 * pm_curve / pm_curve[-20:].mean(),
                          "binance_pct": 100 * cex_curve / cex_curve[-20:].mean()})
    return summary, curve


def cross_correlation(log_price, price, market):
    """Correlation of 100 ms changes; a positive lag means Binance moved first."""
    n = 2
    a = log_price[n:] - log_price[:-n]
    b = price[n:] - price[:-n]
    b[market[n:] != market[:-n]] = np.nan
    rows = []
    for lag in XCORR_BINS:
        x = a[max(0, -lag):len(a) - max(0, lag)]
        y = b[max(0, lag):len(b) - max(0, -lag)]
        ok = np.isfinite(x) & np.isfinite(y)
        rows.append((lag * BIN_MS, np.corrcoef(x[ok], y[ok])[0, 1]))
    return pd.DataFrame(rows, columns=["lag_ms", "correlation"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coins", nargs="+", default=list(COINS), choices=COINS)
    ap.add_argument("--raw-dir", type=Path, default=DATA_DIR / "raw")
    args = ap.parse_args()

    con = duckdb.connect()
    summaries, curves, xcorrs = [], [], []
    for coin in args.coins:
        for clock in ("receive", "venue"):
            cex, pm, (binance_lag, pm_lag) = read_ticks(con, coin, args.raw_dir, clock)
            log_price, price, market = to_grid(cex, pm)
            events, threshold = find_events(log_price, price, market)
            summary, curve = event_study(log_price, price, events)
            xcorr = cross_correlation(log_price, price, market)
            summaries.append({"coin": coin, "clock": clock, "threshold_bps": threshold, **summary,
                              "xcorr_peak_ms": xcorr.loc[xcorr["correlation"].idxmax(), "lag_ms"],
                              "xcorr_peak": xcorr["correlation"].max(),
                              "binance_feed_ms": binance_lag, "polymarket_feed_ms": pm_lag})
            curves.append(curve.assign(coin=coin, clock=clock))
            xcorrs.append(xcorr.assign(coin=coin, clock=clock))
            print(coin, clock, summary, flush=True)

    out = ROOT / "results"
    pd.DataFrame(summaries).round(3).to_csv(out / "tick_lag_summary.csv", index=False)
    pd.concat(curves).round(2).to_csv(out / "tick_lag_response.csv", index=False)
    pd.concat(xcorrs).round(4).to_csv(out / "tick_lag_xcorr.csv", index=False)


if __name__ == "__main__":
    main()
