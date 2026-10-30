"""Design of the experiment: which event runs in which job on which day, and an emulator of hardware features with
job- and day-level fingerprints, used to write down predictions before the hardware runs.

Arms (all share the same jobs, so they see identical hardware conditions; all use the same 1,000 events):
    R   data circuit (HSBC "longer" PQFM on a 129-qubit chain); each event goes to a (session, job) slot at random
    C   data circuit; market day d runs on session d, split into that session's jobs in market-time order
        (the natural pipeline)
    N   null circuit (the same gates, every data angle zero), placed exactly like C. Its rows cannot depend on the
        event, so they measure hardware fingerprints, and any submission order can be replayed in silico.
Anchors: one fixed event, repeated in every job, never scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _chunks(idx, k):
    return np.array_split(np.asarray(idx), k)


def make_schedule(ev, hw, n_sessions=8, jobs=3, anchor=None, seed=11):
    """Rows: arm, event (pool index), session (0-based), job, null, anchor."""
    rng = np.random.default_rng(seed)
    days = sorted(ev.day.unique())
    assert len(days) == n_sessions, "one market day per session"
    rows = []
    def add(arm, e, s, j, null=False, anchor_=False):
        rows.append({"arm": arm, "event": int(e), "session": int(s), "job": int(j), "null": null, "anchor": anchor_})
    for s, d in enumerate(days):
        ed = hw[ev.day.values[hw] == d]
        ed = ed[np.argsort(ev.t.values[ed], kind="stable")]
        for j, part in enumerate(_chunks(ed, jobs)):
            for e in part:
                add("C", e, s, j)
                add("N", e, s, j, null=True)
    slots = [(s, j) for s in range(n_sessions) for j in range(jobs)]
    for k, e in enumerate(rng.permutation(hw)):
        s, j = slots[k % len(slots)]
        add("R", e, s, j)
    if anchor is not None:
        for s in range(n_sessions):
            for j in range(jobs):
                add("anchor", anchor, s, j, anchor_=True)
    return pd.DataFrame(rows)


def reassign(slots, ev, pool, order, rng):
    """Give the null-arm slots of each session to that market day's events under a submission order.

    slots: DataFrame with session (0-based schedule index), job, and a row id column 'row'.
    Returns a list of (row, event) pairs. Orders: random, chronological, by symbol, by outcome."""
    days = sorted(ev.day.unique())
    out = []
    for s, g in slots.sort_values(["session", "job", "row"]).groupby("session"):
        e = pool[ev.day.values[pool] == days[s]]
        if order == "random":
            e = rng.permutation(e)
        elif order == "chronological":
            e = e[np.argsort(ev.t.values[e], kind="stable")]
        elif order == "by symbol":
            e = e[np.lexsort((ev.t.values[e], ev.symbol.values[e]))]
        elif order == "by outcome":
            e = e[np.lexsort((ev.t.values[e], ev.y.values[e]))]
        else:
            raise ValueError(order)
        k = min(len(g), len(e))
        out += list(zip(g.row.values[:k], e[:k]))
    return out


def emulate(E, sched_rows, retention, shots, sigma_job=0.0, sigma_day=0.0, s_gate=0.005, seed=0):
    """Hardware-like features for scheduled rows.

    E: (n_rows, 3, n) noiseless features of each row (null rows already constant).
    Fingerprint offsets are drawn per (session, job) and per session, per feature, in units of the shot-noise sd of a
    feature at this shot count (1/sqrt(shots)). retention: (n,) attenuation per qubit.
    """
    rng = np.random.default_rng(seed)
    n = E.shape[2]
    sd_shot = 1 / np.sqrt(shots)
    jobs = sched_rows[["session", "job"]].drop_duplicates().itertuples(index=False)
    o_job = {(s, j): rng.normal(size=(3, n)) * sigma_job * sd_shot for s, j in jobs}
    o_day = {s: rng.normal(size=(3, n)) * sigma_day * sd_shot for s in sched_rows.session.unique()}
    off = np.stack([o_job[(s, j)] + o_day[s] for s, j in zip(sched_rows.session, sched_rows.job)])
    m = np.clip(E * retention[None, None, :] + off + rng.normal(size=E.shape) * s_gate, -1, 1)
    return m + rng.normal(size=m.shape) * np.sqrt((1 - m ** 2) / shots)


PREREG_CODE = ("pqfm.py", "cryptofill.py", "backtest.py", "design.py", "pauliprop.py", "lightcone.py", "featuremap.py")


def prereg_digest(root):
    """SHA-256 over every file in outputs/prereg (except SHA256.txt) and the analysis code. Used by H3 and H4."""
    import hashlib
    from pathlib import Path
    root = Path(root); pre = root / "outputs" / "prereg"
    h = hashlib.sha256()
    for f in sorted(p for p in pre.glob("*") if p.name != "SHA256.txt"):
        h.update(f.name.encode()); h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    for m in PREREG_CODE:
        h.update((root / "hsbcrep" / m).read_bytes().replace(b"\r\n", b"\n"))   # same digest on Windows and Linux
    return h.hexdigest()
