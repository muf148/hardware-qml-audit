"""A public fill-probability task, built from Binance USD-M futures data, as an analog of the RFQ fill task in
Ciceri et al. (HSBC/IBM, arXiv:2509.17715).

Event      : at a random time t, a passive limit order is posted on a random side of a random symbol, at a fixed
             relative distance delta from the last trade price before t (buy below, sell above).
Label      : 1 if the order would be touched within H seconds: a trade prints at or through its price in (t, t + H].
             This is the standard "touch" fill rule of fill-probability studies (queue position is ignored).
Features   : market state known at t only. Trade-flow statistics over several look-back windows for the
             symbol itself, for BTC and ETH (market), and averaged across all symbols (sector analog);
             order-book depth snapshots; open-interest and long/short metrics; event descriptors.
             216 features, the same count as the HSBC paper.

Everything uses information strictly before t, except the label.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "LINKUSDT"]
BASE = "https://data.binance.vision/data/futures/um/daily"
KINDS = ("aggTrades", "bookDepth", "metrics")
OWN_WINDOWS = (1, 5, 15, 60, 300, 900, 3600)          # seconds
MKT_WINDOWS = (5, 15, 60, 300, 900, 3600)
CROSS_WINDOWS = (5, 15, 60, 300, 900, 3600)
DEPTH_LEVELS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)       # percent from mid, as in the bookDepth files


# ----------------------------------------------------------------------------------------------- download

def download(raw_dir, days, symbols=SYMBOLS, kinds=KINDS):
    """Fetch daily zip files from data.binance.vision (public). Skips files already present."""
    import urllib.request
    raw = Path(raw_dir); raw.mkdir(parents=True, exist_ok=True)
    missing = []
    for s in symbols:
        for d in days:
            for k in kinds:
                f = raw / f"{s}-{k}-{d}.zip"
                if f.exists() and f.stat().st_size > 0:
                    continue
                try:
                    urllib.request.urlretrieve(f"{BASE}/{k}/{s}/{f.name}", f)
                except Exception:
                    missing.append(f.name)
    return missing


def _read_zip(path, **kw):
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as fh:
            return pd.read_csv(io.BytesIO(fh.read()), **kw)


# ----------------------------------------------------------------------------------------------- loading

class Trades:
    """Trade tape of one symbol with cumulative sums for O(log n) window statistics."""

    def __init__(self, df):
        df = df.sort_values("transact_time", kind="stable")
        self.t = df.transact_time.values.astype(np.int64)                 # ms
        self.p = df.price.values.astype(np.float64)
        q = df.quantity.values.astype(np.float64)
        sell_aggr = df.is_buyer_maker.values.astype(bool)                  # buyer is maker -> seller aggressed
        sgn = np.where(sell_aggr, -1.0, 1.0)
        lr = np.r_[0.0, np.diff(np.log(self.p))]
        self.c_vol = np.r_[0.0, np.cumsum(q)]
        self.c_svol = np.r_[0.0, np.cumsum(sgn * q)]
        self.c_not = np.r_[0.0, np.cumsum(q * self.p)]
        self.c_r2 = np.r_[0.0, np.cumsum(lr ** 2)]
        self.c_n = np.arange(len(self.t) + 1, dtype=np.float64)
        self.sell_aggr = sell_aggr
        self.q = q

    def idx(self, t_ms):
        """Index of the first trade at or after t (so trades before t are [:idx])."""
        return np.searchsorted(self.t, t_ms, side="left")

    def last_price(self, t_ms):
        i = self.idx(t_ms) - 1
        return self.p[np.clip(i, 0, None)]

    def window(self, t_ms, w_s):
        """Statistics of trades in [t - w, t). Returns dict of arrays."""
        a, b = self.idx(t_ms - int(w_s * 1000)), self.idx(t_ms)
        n = self.c_n[b] - self.c_n[a]
        vol = self.c_vol[b] - self.c_vol[a]
        svol = self.c_svol[b] - self.c_svol[a]
        notional = self.c_not[b] - self.c_not[a]
        r2 = self.c_r2[b] - self.c_r2[a]
        p_end = self.p[np.clip(b - 1, 0, None)]
        p_start = self.p[np.clip(a - 1, 0, None)]
        vwap = np.where(vol > 0, notional / np.where(vol > 0, vol, 1), p_end)
        return {"ret": np.log(p_end / p_start), "rv": np.sqrt(r2), "imb": np.where(vol > 0, svol / np.where(vol > 0, vol, 1), 0.0),
                "lcount": np.log1p(n), "lnotional": np.log1p(notional), "lsize": np.log1p(np.where(n > 0, notional / np.where(n > 0, n, 1), 0.0)),
                "vwapdev": np.log(p_end / vwap)}

    def range_(self, t_ms, w_s):
        out = np.zeros(len(t_ms))
        a, b = self.idx(t_ms - int(w_s * 1000)), self.idx(t_ms)
        for k in range(len(t_ms)):
            if b[k] - a[k] > 1:
                seg = self.p[a[k]:b[k]]
                out[k] = np.log(seg.max() / seg.min())
        return out

    def touched(self, t_ms, h_s, limit, side):
        """1 if a trade prints at or through `limit` in (t, t + h]. side +1 = buy order (needs price <= limit)."""
        a, b = np.searchsorted(self.t, t_ms, side="right"), np.searchsorted(self.t, t_ms + int(h_s * 1000), side="right")
        out = np.zeros(len(t_ms), dtype=int)
        for k in range(len(t_ms)):
            if b[k] > a[k]:
                seg = self.p[a[k]:b[k]]
                out[k] = int(seg.min() <= limit[k]) if side[k] > 0 else int(seg.max() >= limit[k])
        return out


def load_trades(raw_dir, symbol, days):
    raw = Path(raw_dir)
    cols = ["price", "quantity", "transact_time", "is_buyer_maker"]
    df = pd.concat([_read_zip(raw / f"{symbol}-aggTrades-{d}.zip", usecols=cols) for d in days], ignore_index=True)
    df["is_buyer_maker"] = df.is_buyer_maker.astype(str).str.lower().eq("true")
    return Trades(df)


def load_depth(raw_dir, symbol, days):
    raw = Path(raw_dir)
    df = pd.concat([_read_zip(raw / f"{symbol}-bookDepth-{d}.zip") for d in days], ignore_index=True)
    df["t"] = pd.to_datetime(df.timestamp).astype("int64") // 10 ** 6
    wide = df.pivot_table(index="t", columns="percentage", values="notional", aggfunc="last").sort_index()
    wide = wide[[c for c in wide.columns if round(float(c)) in DEPTH_LEVELS and abs(float(c)) >= 1]]
    wide.columns = [int(round(float(c))) for c in wide.columns]
    return wide[list(DEPTH_LEVELS)]


def load_metrics(raw_dir, symbol, days):
    raw = Path(raw_dir)
    df = pd.concat([_read_zip(raw / f"{symbol}-metrics-{d}.zip") for d in days], ignore_index=True)
    df["t"] = pd.to_datetime(df.create_time).astype("int64") // 10 ** 6
    return df.sort_values("t").set_index("t")


# ----------------------------------------------------------------------------------------------- events

def sample_events(study_days, n_per_day, symbols=SYMBOLS, seed=2026):
    """Random times (uniform within each UTC day), random symbol, random side."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in study_days:
        t0 = int(pd.Timestamp(d, tz="UTC").value // 10 ** 6)
        ts = np.sort(rng.integers(t0, t0 + 86_400_000 - 3_600_000, n_per_day))   # leave the last hour for labels
        for t in ts:
            rows.append({"t": int(t), "day": d, "symbol": symbols[rng.integers(len(symbols))], "side": int(rng.choice([-1, 1]))})
    ev = pd.DataFrame(rows)
    ev["event_id"] = np.arange(len(ev))
    return ev


def label_events(ev, trades, delta_bps, horizon_s):
    """Touch-fill label for each event."""
    y = np.zeros(len(ev), dtype=int)
    for s, g in ev.groupby("symbol"):
        T = trades[s]
        ref = T.last_price(g.t.values)
        limit = ref * (1 - g.side.values * delta_bps * 1e-4)
        y[g.index.values] = T.touched(g.t.values, horizon_s, limit, g.side.values)
    return y


def calibrate_delta(ev, trades, horizon_s, target=0.37, grid=None):
    """The distance (bps) whose pooled fill rate is closest to `target` (HSBC's 37%)."""
    grid = np.round(np.arange(0.5, 30.01, 0.25), 2) if grid is None else grid
    rates = np.array([label_events(ev, trades, d, horizon_s).mean() for d in grid])
    k = int(np.argmin(np.abs(rates - target)))
    return float(grid[k]), float(rates[k])


# ----------------------------------------------------------------------------------------------- features

STAT_KEYS = ("ret", "rv", "imb", "lcount", "lnotional", "lsize", "vwapdev")


def features(ev, trades, depth, metrics, symbols=SYMBOLS):
    """216 features known at event time t. Returns a DataFrame aligned with ev."""
    t = ev.t.values
    cols = {}
    # own symbol: 7 statistics + range over 7 windows = 56
    own = {w: {k: np.zeros(len(ev)) for k in (*STAT_KEYS, "range")} for w in OWN_WINDOWS}
    for s, g in ev.groupby("symbol"):
        T = trades[s]; ii = g.index.values
        for w in OWN_WINDOWS:
            st = T.window(g.t.values, w)
            for k in STAT_KEYS:
                own[w][k][ii] = st[k]
            own[w]["range"][ii] = T.range_(g.t.values, w)
    for w in OWN_WINDOWS:
        for k in (*STAT_KEYS, "range"):
            cols[f"own_{k}_{w}s"] = own[w][k]
    # market: BTC and ETH, 8 statistics x 6 windows = 96
    for m in ("BTCUSDT", "ETHUSDT"):
        for w in MKT_WINDOWS:
            st = trades[m].window(t, w)
            for k in STAT_KEYS:
                cols[f"{m[:3].lower()}_{k}_{w}s"] = st[k]
            cols[f"{m[:3].lower()}_range_{w}s"] = trades[m].range_(t, w)
    # cross-section: mean over all symbols of ret, rv, imb over 6 windows = 18
    for w in CROSS_WINDOWS:
        acc = {k: np.zeros(len(ev)) for k in ("ret", "rv", "imb")}
        for s in symbols:
            st = trades[s].window(t, w)
            for k in acc:
                acc[k] += st[k] / len(symbols)
        for k, v in acc.items():
            cols[f"xs_{k}_{w}s"] = v
    # order-book depth of the own symbol: latest snapshot (10) and its change vs 5 min earlier (10) = 20
    dep_now = np.zeros((len(ev), 10)); dep_old = np.zeros((len(ev), 10))
    for s, g in ev.groupby("symbol"):
        D = depth[s]; ti = D.index.values
        k_now = np.clip(np.searchsorted(ti, g.t.values, side="right") - 1, 0, None)
        k_old = np.clip(np.searchsorted(ti, g.t.values - 300_000, side="right") - 1, 0, None)
        dep_now[g.index.values] = np.log1p(D.values[k_now]); dep_old[g.index.values] = np.log1p(D.values[k_old])
    for j, lv in enumerate(DEPTH_LEVELS):
        cols[f"depth_{lv:+d}pct"] = dep_now[:, j]
        cols[f"depth_chg5m_{lv:+d}pct"] = dep_now[:, j] - dep_old[:, j]
    # open interest and positioning of the own symbol = 5
    mk = {c: np.zeros(len(ev)) for c in ("oi_chg15m", "oi_chg60m", "toptrader_ls", "ls_ratio", "taker_ls_vol")}
    for s, g in ev.groupby("symbol"):
        M = metrics[s]; ti = M.index.values
        def at(dt):
            return np.clip(np.searchsorted(ti, g.t.values - dt, side="right") - 1, 0, None)
        k0, k15, k60 = at(0), at(900_000), at(3_600_000)
        oi = M.sum_open_interest.values.astype(float)
        ii = g.index.values
        mk["oi_chg15m"][ii] = np.log(oi[k0] / oi[k15]); mk["oi_chg60m"][ii] = np.log(oi[k0] / oi[k60])
        mk["toptrader_ls"][ii] = np.log(M.sum_toptrader_long_short_ratio.values.astype(float)[k0])
        mk["ls_ratio"][ii] = np.log(M.count_long_short_ratio.values.astype(float)[k0])
        mk["taker_ls_vol"][ii] = np.log(M.sum_taker_long_short_vol_ratio.values.astype(float)[k0] + 1e-9)
    cols.update(mk)
    # event descriptors = 21: side, side x own imbalance (3 windows), seconds since last trade, last trade size,
    # symbol one-hot (8), time of day (2), day of week (2), side x own return (3)
    cols["side"] = ev.side.values.astype(float)
    for w in (5, 60, 300):
        cols[f"side_x_imb_{w}s"] = ev.side.values * own[w]["imb"]
        cols[f"side_x_ret_{w}s"] = ev.side.values * own[w]["ret"]
    since = np.zeros(len(ev)); lsize = np.zeros(len(ev))
    for s, g in ev.groupby("symbol"):
        T = trades[s]; i = np.clip(T.idx(g.t.values) - 1, 0, None)
        since[g.index.values] = np.log1p((g.t.values - T.t[i]) / 1000.0)
        lsize[g.index.values] = np.log1p(T.q[i] * T.p[i])
    cols["log_secs_since_trade"] = since; cols["log_last_trade_notional"] = lsize
    for s in symbols:
        cols[f"is_{s[:-4].lower()}"] = (ev.symbol.values == s).astype(float)
    tod = (t % 86_400_000) / 86_400_000 * 2 * np.pi
    cols["tod_sin"], cols["tod_cos"] = np.sin(tod), np.cos(tod)
    dow = pd.to_datetime(t, unit="ms").dayofweek.values / 7 * 2 * np.pi
    cols["dow_sin"], cols["dow_cos"] = np.sin(dow), np.cos(dow)
    F = pd.DataFrame(cols, index=ev.index)
    return F


def hsbc_scale(F, mu=None, sd=None):
    """HSBC preprocessing: standardize, then 2*pi*tanh(z/3) -> angles in (-2pi, 2pi)."""
    X = np.asarray(F, float)
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) if sd is None else sd
    Z = (X - mu) / np.where(sd > 0, sd, 1.0)
    return 2 * np.pi * np.tanh(Z / 3), mu, sd
