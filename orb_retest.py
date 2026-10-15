"""ORB Retest backtest: implementation of the frozen specification, Review Revision 05.

Spec: "ORB Retest - Frozen Test Spec", Review Revision 05 (strategy rules FROZEN).
Source: flux_orb_algo.pine, SHA256 fec9db1eea0d427112fddf7dbe557b3aa97f193833f1d3f6784905b283601424.
Rule numbers in comments ("SR n" = Session rules item n, "XR n" = Exit rules item n,
"EO n" = Execution order step n, "DC n" = Doubled-cost item n, "DATA" = conventions table)
refer to that spec and to RULE_MAP.md.

Read-only: instrument-candle GET requests on the OANDA PRACTICE host only. No account, order,
position, trade or pricing calls.
Run:     python -u orb_retest.py --out DIR [--run-tests] [--expect-sha H] [--expect-test-sha H]
                                 [--require-volume] [--export-after]
Export:  python -u orb_retest.py --out DIR --export-only
Exit codes: 0 COMPLETE (each session's verdict in the summary); 1 INCOMPLETE (data/download failure
or unresolved trades; files still written); 2 internal error; 3 refused at startup (hash mismatch,
failed or skipped tests, missing persistent storage, or no token) before any OANDA request.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from array import array
from datetime import date, datetime, timedelta, timezone

import requests

SPEC_VERSION = "ORB Retest Review Revision 05 (strategy rules FROZEN)"
SOURCE_SHA256 = "fec9db1eea0d427112fddf7dbe557b3aa97f193833f1d3f6784905b283601424"
HOST = "https://api-fxpractice.oanda.com"

# ---------------------------------------------------------------- frozen settings (Settings table)
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD",
         "USD_CHF", "EUR_JPY", "GBP_JPY", "EUR_GBP", "AUD_JPY", "EUR_AUD"]
SESSIONS = [  # (name, zone, hour, minute) - session start S in local time (Settings: session start)
    ("London", "LON", 8, 0),
    ("NY_forex", "NY", 8, 0),
    ("US_stocks", "NY", 9, 30),
]
STEP = 300                    # M5, seconds
RANGE_CANDLES = 6             # 30-minute opening range
EMA_ALPHA = 2.0 / 10.0        # 9-period EMA
GATE_PCT = 0.20               # minimumProfitPercent
SPACING_PCT = 0.075           # minimumProfitIncrementPercent
SHARES = (0.90, 0.05, 0.05)   # exit quantities of the original size
FIRST_DATE = date(2010, 1, 1)             # DATA: session dates
LAST_DATE = date(2026, 10, 9)
DL_START = datetime(2009, 12, 28, tzinfo=timezone.utc)   # DATA: download window
DL_END = datetime(2026, 10, 10, tzinfo=timezone.utc)     # exclusive
RUNS = {"normal": {"slip": 0.1, "widen": False},          # Fill prices; DC 1-2
        "double": {"slip": 0.2, "widen": True}}

# ---------------------------------------------------------------- time zones (DST, no tz database)
def _nth_sunday(y, m, n):
    d = date(y, m, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_sunday(y, m):
    d = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - 6) % 7)


DST_DATES = (date(2009, 12, 1), date(2026, 12, 31))
DST_TIMES = {"LON": {(8, 0)}, "NY": {(8, 0), (9, 30), (16, 45), (17, 0)}}


def utc_offset_hours(zone, d, hour, minute):
    """UTC offset of a local wall-clock time. Rules valid 2007-2030 (US) and 1996-2030 (EU).
    US: DST from 2nd Sunday March 02:00 to 1st Sunday November 02:00 local.
    UK: BST from last Sunday March 01:00 UTC to last Sunday October 01:00 UTC."""
    if not (DST_DATES[0] <= d <= DST_DATES[1]) or (hour, minute) not in DST_TIMES.get(zone, ()):
        raise ValueError(f"fixed DST rules are accepted only for the frozen dates and clock times: {zone} {d} {hour:02d}:{minute:02d}")
    local_min = hour * 60 + minute
    if zone == "NY":
        start, end = _nth_sunday(d.year, 3, 2), _nth_sunday(d.year, 11, 1)
        dst = (start < d < end) or (d == start and local_min >= 180) or (d == end and local_min < 60)
        return -4 if dst else -5
    if zone == "LON":
        start, end = _last_sunday(d.year, 3), _last_sunday(d.year, 10)
        dst = (start < d < end) or (d == start and local_min >= 120) or (d == end and local_min < 60)
        return 1 if dst else 0
    raise ValueError(zone)


def local_to_epoch(zone, d, hour, minute):
    off = utc_offset_hours(zone, d, hour, minute)
    dt = datetime(d.year, d.month, d.day, hour, minute, tzinfo=timezone.utc) - timedelta(hours=off)
    return int(dt.timestamp())


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pip(pair):
    return 0.01 if "JPY" in pair else 0.0001


# ---------------------------------------------------------------- candles (DATA rows)
FIELDS = ("mo", "mh", "ml", "mc", "bo", "bh", "bl", "bc", "ao", "ah", "al", "ac")
TIME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.(\d{1,9}))?Z$")


class DataError(Exception):
    """A data error from the DATA 'Data errors' row: the run stops, no verdict."""


class DownloadError(Exception):
    """A failed or incomplete download: never treated as a quiet period."""


def parse_time(s):
    """Full RFC3339 parse; fractional seconds must be zero (M5 candles open on the minute)."""
    m = TIME_RE.match(s) if isinstance(s, str) else None
    if not m:
        raise DataError(f"bad timestamp {s!r}")
    if m.group(3) and int(m.group(3)) != 0:
        raise DataError(f"non-zero fractional seconds in {s!r}")
    return int(datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


class Candles:
    """Column store: t (epoch seconds, candle open) and twelve price columns."""

    def __init__(self):
        self.t = array("q")
        self.cols = {f: array("d") for f in FIELDS}

    def __len__(self):
        return len(self.t)

    def append(self, t, vals):
        self.t.append(t)
        for f in FIELDS:
            self.cols[f].append(vals[f])

    def row(self, i):
        return {f: self.cols[f][i] for f in FIELDS}


def validate_candle(c, prev_t):
    """DATA 'Data errors' row. Returns (t, values)."""
    if not isinstance(c, dict):
        raise DataError("candle is not an object")
    for k in ("time", "complete", "mid", "bid", "ask"):
        if k not in c:
            raise DataError(f"missing field {k!r}")
    t = parse_time(c["time"])
    if t % STEP:
        raise DataError(f"{c['time']}: not 5-minute aligned")
    if c["complete"] is not True:
        raise DataError(f"{c['time']}: incomplete candle")
    if prev_t is not None and t <= prev_t:
        raise DataError(f"{c['time']}: duplicate or out of order")
    vals = {}
    for side, p in (("mid", "m"), ("bid", "b"), ("ask", "a")):
        q = c[side]
        if not isinstance(q, dict):
            raise DataError(f"{c['time']}: {side} not an object")
        for f in "ohlc":
            if f not in q:
                raise DataError(f"{c['time']}: missing {side}.{f}")
            try:
                v = float(q[f])
            except (TypeError, ValueError):
                raise DataError(f"{c['time']}: {side}.{f} not a number") from None
            if not math.isfinite(v) or v <= 0:
                raise DataError(f"{c['time']}: {side}.{f} not finite and positive")
            vals[p + f] = v
        o, h, l, cl = (vals[p + f] for f in "ohlc")
        if not (l <= min(o, cl) <= max(o, cl) <= h):
            raise DataError(f"{c['time']}: invalid {side} OHLC")
    for f in "ohlc":
        if vals["b" + f] > vals["a" + f]:
            raise DataError(f"{c['time']}: bid above ask in field {f}")
    return t, vals


_session = None
_sleep = time.sleep


def _get(url, params, token):
    """One planned request: at most 3 HTTP attempts, no redirects, explicit timeouts."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.trust_env = False
    last = "no attempt"
    for attempt in range(1, 4):
        if attempt > 1:
            _sleep((2.0, 5.0)[attempt - 2])
        try:
            r = _session.get(url, params=params, timeout=(10, 30), allow_redirects=False,
                             headers={"Authorization": f"Bearer {token}",
                                      "Accept-Datetime-Format": "RFC3339"})
        except requests.RequestException as e:
            last = f"network error ({type(e).__name__})"
            continue
        if r.status_code in (401, 403):
            raise DownloadError(f"authentication failure (HTTP {r.status_code}); stopping")
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                last = "unreadable JSON"
                continue
        last = f"HTTP {r.status_code}"
        if r.status_code not in (429, 500, 502, 503, 504):
            raise DownloadError(f"non-retryable {last}")
    raise DownloadError(f"request failed after 3 attempts (last: {last})")


DOWNLOADER = {"host": HOST, "endpoint": "/v3/instruments/{pair}/candles", "granularity": "M5",
              "price": "BAM", "smooth": "false", "includeFirst": "true", "count": 5000,
              "timeout_s": [10, 30], "max_attempts": 3, "retry_delays_s": [2.0, 5.0],
              "allow_redirects": False, "pagination": "next from = last returned candle open + 5 minutes"}


def download(pair, token, start=DL_START, end=DL_END, page=5000, log=print):
    """DATA 'Pagination' and 'Candles' rows (Review fix 1). Retrieval completes only when a candle
    at or beyond the exclusive end has been seen or the next page would start at or after it.
    An empty page before completion is a failed download; short pages simply continue paging.
    Raises DataError / DownloadError; never treats a failed or truncated download as a gap."""
    url = f"{HOST}/v3/instruments/{pair}/candles"
    out, cur, prev_t, requests_made = Candles(), int(start.timestamp()), None, 0
    end_ts = int(end.timestamp())
    while True:
        params = {"granularity": "M5", "price": "BAM", "smooth": "false", "includeFirst": "true",
                  "from": iso(cur), "count": str(page)}
        data = _get(url, params, token)
        requests_made += 1
        if not isinstance(data, dict) or data.get("instrument") != pair or data.get("granularity") != "M5":
            raise DataError(f"{pair}: response instrument/granularity mismatch")
        candles = data.get("candles")
        if not isinstance(candles, list):
            raise DataError(f"{pair}: response has no candle list")
        if not candles:
            raise DownloadError(f"{pair}: empty page from {iso(cur)} before reaching {iso(end_ts)}")
        reached, last_t = False, None
        for c in candles:
            t, vals = validate_candle(c, prev_t)
            if t < cur:
                raise DataError(f"{pair}: candle {iso(t)} before requested from {iso(cur)}")
            prev_t = last_t = t
            if t >= end_ts:            # exclusive end reached: excluded from the simulation
                reached = True
                break
            out.append(t, vals)
        if reached:
            break
        nxt = last_t + STEP            # one full candle after the last one returned
        if nxt <= cur:
            raise DownloadError(f"{pair}: pagination failed to advance at {iso(cur)}")
        if nxt >= end_ts:
            break
        cur = nxt
        _sleep(0.1)
    log(f"{pair}: downloaded {len(out)} candles in {requests_made} requests "
        f"({iso(out.t[0]) if len(out) else '-'} to {iso(out.t[-1]) if len(out) else '-'})")
    return out, requests_made


def ema_series(c):
    """DATA 'EMA': alpha 2/10, seeded with the first (h+l)/2, continuous, never reset."""
    mh, ml = c.cols["mh"], c.cols["ml"]
    out = array("d")
    e = None
    for i in range(len(c)):
        v = (mh[i] + ml[i]) / 2.0
        e = v if e is None else EMA_ALPHA * v + (1 - EMA_ALPHA) * e
        out.append(e)
    return out


def widen(vals):
    """DC 1: double each field's spread, then an OHLC envelope per series. Mid unchanged."""
    w = dict(vals)
    for p, sign in (("b", -1), ("a", 1)):
        raw = {}
        for f in "ohlc":
            h = (vals["a" + f] - vals["b" + f]) / 2.0
            raw[f] = vals[p + f] + sign * h
        w[p + "o"], w[p + "c"] = raw["o"], raw["c"]
        w[p + "h"], w[p + "l"] = max(raw.values()), min(raw.values())
    return w


# ---------------------------------------------------------------- one session date
def session_times(zone, hour, minute, d):
    S = local_to_epoch(zone, d, hour, minute)
    deadline = local_to_epoch("NY", d, 16, 45)        # 16:45 New York, same session date D
    cutoff = local_to_epoch("NY", d, 17, 0)           # strictly before 17:00
    return S, deadline, cutoff


def find_signal(c, idx, S, deadline):
    """Session rules SR 1-10 on midpoint candles. Returns a dict describing the day's outcome
    before any fills: 'skip' (incomplete range), 'none', or a signal with direction and E."""
    t, mo, mh, ml, mc = c.t, c.cols["mo"], c.cols["mh"], c.cols["ml"], c.cols["mc"]
    i0 = idx.get(S)
    rng = [idx.get(S + k * STEP) for k in range(RANGE_CANDLES)]
    if any(i is None for i in rng):                              # SR 1
        return {"kind": "skip"}
    RH = max(mh[i] for i in rng)
    RL = min(ml[i] for i in rng)
    last_signal_open = deadline - 2 * STEP                       # SR 10: candle opening 16:35
    state, bdir, bstart, retests = "waiting", 0, None, 0
    i = rng[-1] + 1
    while i < len(t) and t[i] <= last_signal_open:
        if t[i] < S + RANGE_CANDLES * STEP:
            i += 1
            continue
        close, low, high = mc[i], ml[i], mh[i]
        if state == "waiting":                                   # SR 2-3 (breakout block first)
            if close > RH:
                state, bdir, bstart, retests = "breakout", 1, i, 0
            elif close < RL:
                state, bdir, bstart, retests = "breakout", -1, i, 0
        if state == "breakout":
            if (bdir == 1 and close < RH) or (bdir == -1 and close > RL):   # SR 4-5
                state = "waiting"                                # no new breakout this candle
            if i > bstart and ((bdir == 1 and close > RH and low < RH) or
                               (bdir == -1 and close < RL and high > RL)):  # SR 7
                retests += 1
            if retests >= 1 and state == "breakout":             # SR 8
                return {"kind": "signal", "dir": bdir, "sig_i": i, "E": close, "RH": RH, "RL": RL,
                        "stop": (RH + RL) / 2.0}
        i += 1
    return {"kind": "none", "RH": RH, "RL": RL}


def simulate_trade(c, ema, sig, deadline, cutoff, pair, run):
    """Execution order EO 1-4, Exit rules XR 1-6, fills, liquidation. One cost run, own state.
    Returns a record with status 'closed', 'unresolved', 'cancelled' or 'invalid'."""
    t = c.t
    d = sig["dir"]
    s = RUNS[run]["slip"] * pip(pair)
    q = (lambda i: widen(c.row(i))) if RUNS[run]["widen"] else c.row
    base = {"pair": pair, "dir": d, "signal_time": t[sig["sig_i"]], "E": sig["E"], "stop0": sig["stop"]}
    sig_close = t[sig["sig_i"]] + STEP
    # DATA 'Missing entry candle': first candle opening >= signal close, strictly before 16:45, same date
    j = sig["sig_i"] + 1
    while j < len(t) and t[j] < sig_close:
        j += 1
    if j >= len(t) or t[j] >= deadline:
        return dict(base, status="cancelled",
                    reason="no candle opening at or after the signal close and before 16:45 New York")
    v = q(j)
    entry_t = t[j]
    entry_fill = v["ao"] + s if d == 1 else v["bo"] - s
    stop = sig["stop"]
    R = d * (entry_fill - stop)
    if R <= 0:
        return dict(base, status="invalid", entry_time=entry_t, entry_fill=entry_fill,
                    reason="initial R <= 0 (entry fill not beyond the starting stop)")
    trade = dict(base, status="closed", entry_time=entry_t, entry_fill=entry_fill, R=R, exits=[],
                 entry_cost_r=d * (entry_fill - c.cols["mo"][j]) / R)   # spread + slippage paid at entry, in R
    remaining, stage, pending, P = 1.0, 0, None, None

    def add_exit(fill, when, reason, share):
        trade["exits"].append({"time": when, "fill": fill, "share": share, "reason": reason})

    def close_rest(fill, when, reason):
        nonlocal remaining
        add_exit(fill, when, reason, remaining)
        remaining = 0.0

    def exit_side_open(vv):
        return vv["bo"] if d == 1 else vv["ao"]

    def through(price):
        return price <= stop if d == 1 else price >= stop

    def at_quote(o):
        return o - s if d == 1 else o + s

    first = True
    k = j
    while True:
        if k >= len(t) or t[k] >= cutoff or (not first and t[k] >= deadline):
            # 16:45 clock event (EO 2 / DATA 'Missing 16:45 candle'): pending partial cancelled,
            # no EMA exits; liquidate on the first candle in [16:45, 17:00) of date D, gap stop first.
            pending = None
            if k < len(t) and deadline <= t[k] < cutoff:
                o = exit_side_open(q(k))
                close_rest(at_quote(o), t[k], "gap_stop" if through(o) else "end_of_day")
            else:
                trade["status"] = "unresolved"
            break
        v = q(k)
        o = exit_side_open(v)
        if not first:
            if through(o):                                                  # EO 1 gap stop
                close_rest(at_quote(o), t[k], "gap_stop")
                break
            if pending is not None:                                         # EO 3 pending partial
                share = SHARES[pending - 1]
                add_exit(at_quote(o), t[k], f"partial_{pending}", share)
                remaining = round(remaining - share, 10)
                stage, pending = pending, None
                if stage == 1:                                              # adaptive stop
                    stop = entry_fill
                    if through(o):
                        close_rest(at_quote(o), t[k], "gap_stop")
                        break
                if stage == 3 or remaining <= 1e-12:
                    break
        else:
            first = False
            if through(o):                                                  # EO 4
                close_rest(at_quote(o), t[k], "gap_stop")
                break
        if (d == 1 and v["bl"] <= stop) or (d == -1 and v["ah"] >= stop):   # during the candle, inclusive
            close_rest(stop - s if d == 1 else stop + s, t[k], "stop")
            break
        if t[k] + STEP < deadline:                                          # at the close
            e, cl = ema[k], c.cols["mc"][k]
            gate = d * (e - sig["E"]) > 0 and abs(e - sig["E"]) / sig["E"] * 100.0 >= GATE_PCT   # XR 1
            trig = (cl < e) if d == 1 else (cl > e)                                               # XR 2
            if gate and trig:
                if stage == 0:                                                                    # XR 3
                    pending, P = 1, e
                elif stage in (1, 2):                                                             # XR 4-5
                    if d * (e - P) > 0 and abs(e - P) / P * 100.0 >= SPACING_PCT:
                        pending, P = stage + 1, e
        k += 1
    trade["remaining"] = remaining
    trade["exit_slippage_r"] = sum(x["share"] for x in trade["exits"]) * s / R
    if trade["status"] == "closed":
        trade["exit_time"] = trade["exits"][-1]["time"]
        trade["r"] = trade_r(trade)
    return trade


def trade_r(tr):
    """Trade result: sum over exits of d x (exit fill - entry fill) x share, divided by initial R."""
    return sum(tr["dir"] * (x["fill"] - tr["entry_fill"]) * x["share"] for x in tr["exits"]) / tr["R"]


def run_pair(c, pair, sessions=SESSIONS, first=FIRST_DATE, last=LAST_DATE):
    """All session dates for one pair, both cost runs. Returns per session: diagnostics, per-date
    event records (skipped, cancelled, invalid) and per run the closed and unresolved trades."""
    idx = {c.t[i]: i for i in range(len(c))}
    ema = ema_series(c)
    out = {}
    for name, zone, hh, mm in sessions:
        diag = {"skipped": 0, "no_signal": 0, "signals": 0}
        events = []
        res = {r: {"trades": [], "unresolved": []} for r in RUNS}
        d = first
        while d <= last:
            if d.weekday() < 5:
                S, deadline, cutoff = session_times(zone, hh, mm, d)
                sig = find_signal(c, idx, S, deadline)
                if sig["kind"] == "skip":
                    diag["skipped"] += 1
                    missing = [iso(S + k * STEP) for k in range(RANGE_CANDLES) if (S + k * STEP) not in idx]
                    events.append({"session": name, "pair": pair, "date": d.isoformat(), "run": "both",
                                   "kind": "skipped", "detail": "range candles missing: " + " ".join(missing)})
                elif sig["kind"] == "none":
                    diag["no_signal"] += 1
                else:
                    diag["signals"] += 1
                    for r in RUNS:
                        tr = simulate_trade(c, ema, sig, deadline, cutoff, pair, r)
                        tr["session"], tr["date"], tr["run"] = name, d.isoformat(), r
                        if tr["status"] in ("cancelled", "invalid"):
                            detail = tr["reason"] + f"; signal {iso(tr['signal_time'])}"
                            if tr["status"] == "invalid":
                                detail += f"; entry fill {tr['entry_fill']!r}; stop {tr['stop0']!r}"
                            events.append({"session": name, "pair": pair, "date": d.isoformat(), "run": r,
                                           "kind": tr["status"], "detail": detail})
                        else:
                            tr["risk_pips"] = tr["R"] / pip(pair)
                            res[r]["trades" if tr["status"] == "closed" else "unresolved"].append(tr)
            d += timedelta(days=1)
        out[name] = {"diag": diag, "events": events, "runs": res}
    return out


# ---------------------------------------------------------------- statistics and verdict
def stats(trades, pairs=PAIRS):
    """Completed (closed) trades only."""
    by_entry = sorted(trades, key=lambda x: (x["entry_time"], x["pair"]))
    r = [x["r"] for x in by_entry]
    n = len(r)
    gains = sum(v for v in r if v > 0)
    losses = -sum(v for v in r if v < 0)
    pf = (gains / losses) if losses > 0 else ("infinite" if gains > 0 else "undefined")
    h = n // 2
    first, second = r[:h], r[h:]
    eq, peak, dd = 0.0, 0.0, 0.0
    for x in sorted(trades, key=lambda x: (x["exit_time"], x["entry_time"], x["pair"])):
        eq += x["r"]
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    per_pair = {}
    for p in pairs:
        pr = [x["r"] for x in trades if x["pair"] == p]
        per_pair[p] = {"n": len(pr), "avg_r": (sum(pr) / len(pr)) if pr else None}
    return {
        "n": n, "win_rate": (sum(1 for v in r if v > 0) / n) if n else None,
        "avg_r": (sum(r) / n) if n else None, "pf": pf, "total_r": sum(r), "max_dd_r": dd,
        "first_half_avg": (sum(first) / len(first)) if first else None,
        "second_half_avg": (sum(second) / len(second)) if second else None,
        "split_time": iso(by_entry[h]["entry_time"]) if n else None,
        "pairs_positive": sum(1 for v in per_pair.values() if v["avg_r"] is not None and v["avg_r"] > 0),
        "per_pair": per_pair,
    }


def run_status(unresolved):
    return "INCOMPLETE" if unresolved else "COMPLETE"


def verdict(st, unresolved):
    """Pass bar (normal costs, per session). Only COMPLETE runs get PASS or FAIL."""
    if unresolved:
        return "NO VERDICT (INCOMPLETE)"
    pf_ok = st["pf"] == "infinite" or (isinstance(st["pf"], float) and st["pf"] >= 1.2)
    ok = (st["n"] >= 300 and pf_ok and st["first_half_avg"] is not None and st["first_half_avg"] > 0
          and st["second_half_avg"] is not None and st["second_half_avg"] > 0 and st["pairs_positive"] >= 7)
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------- diagnostics (not filters)
def gap_records(c):
    """Every observed gap between consecutive candles: start (first absent slot), end (next candle
    open), absent M5 slots. No gap is discarded and no cause is inferred from its length."""
    out = []
    for i in range(1, len(c)):
        g = c.t[i] - c.t[i - 1]
        if g > STEP:
            out.append((c.t[i - 1] + STEP, c.t[i], g // STEP - 1))
    return out


def gap_summary(gaps):
    by_year = {}
    for a, b, n in gaps:
        y = datetime.fromtimestamp(a, tz=timezone.utc).year
        e = by_year.setdefault(y, {"gaps": 0, "absent_slots": 0})
        e["gaps"] += 1
        e["absent_slots"] += n
    return by_year


def spread_diagnostics(c, pair):
    """Yearly closing-spread distribution in pips, from every downloaded candle."""
    by_year = {}
    ac, bc = c.cols["ac"], c.cols["bc"]
    for i in range(len(c)):
        y = datetime.fromtimestamp(c.t[i], tz=timezone.utc).year
        by_year.setdefault(y, []).append((ac[i] - bc[i]) / pip(pair))
    out = {}
    for y, v in sorted(by_year.items()):
        v.sort()
        out[y] = {"n": len(v), "median": statistics.median(v), "p95": v[int(0.95 * (len(v) - 1))], "max": v[-1]}
    return out


def fingerprint(c):
    h = hashlib.sha256()
    h.update(c.t.tobytes())
    for f in FIELDS:
        h.update(c.cols[f].tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- exports (round-trip safe)
TRADE_FIELDS = ["session", "run", "status", "pair", "date", "dir", "signal_time", "entry_time", "entry_fill",
                "stop0", "R", "risk_pips", "E", "entry_cost_r", "exit_slippage_r", "remaining",
                "exit_time", "r", "exits"]
TIME_FIELDS = ("signal_time", "entry_time", "exit_time")


def write_trades(path, trades):
    """Closed and unresolved trades. Floats as repr (exact round trip). Unresolved trades keep
    every filled partial and the remaining quantity; their exit_time and r stay empty."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRADE_FIELDS)
        for x in sorted(trades, key=lambda x: (x["entry_time"], x["pair"])):
            row = []
            for f in TRADE_FIELDS:
                v = x.get(f)
                if v is None:
                    row.append("")
                elif f in TIME_FIELDS:
                    row.append(iso(v))
                elif f == "exits":
                    row.append(";".join(f"{iso(e['time'])}|{e['reason']}|{e['share']!r}|{e['fill']!r}" for e in v))
                elif isinstance(v, float):
                    row.append(repr(v))
                else:
                    row.append(v)
            w.writerow(row)


def read_trades(path):
    """Inverse of write_trades (used by tests and reviewers)."""
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            x = dict(row)
            for f in ("entry_fill", "stop0", "R", "risk_pips", "E", "entry_cost_r", "exit_slippage_r",
                      "remaining", "r"):
                x[f] = float(x[f]) if x[f] != "" else None
            x["dir"] = int(x["dir"])
            for f in TIME_FIELDS:
                x[f] = parse_time(x[f]) if x[f] else None
            x["exits"] = [{"time": parse_time(a), "reason": b, "share": float(c_), "fill": float(d_)}
                          for a, b, c_, d_ in (e.split("|") for e in x["exits"].split(";") if e)]
            out.append(x)
    return out


def write_events(path, events):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "pair", "date", "run", "kind", "detail"])
        for e in events:
            w.writerow([e["session"], e["pair"], e["date"], e["run"], e["kind"], e["detail"]])


def write_gaps(path, gaps):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["first_absent_slot_utc", "next_candle_utc", "absent_m5_slots"])
        for a, b, n in gaps:
            w.writerow([iso(a), iso(b), n])


def write_manifest(out_dir):
    names = sorted(n for n in os.listdir(out_dir) if n != "MANIFEST.sha256")
    with open(os.path.join(out_dir, "MANIFEST.sha256"), "w") as fh:
        for n in names:
            with open(os.path.join(out_dir, n), "rb") as f:
                fh.write(f"{hashlib.sha256(f.read()).hexdigest()}  {n}\n")
    return names


# ---------------------------------------------------------------- summary
def build_summary(all_res, identity, failure=None, pairs=PAIRS, sessions=SESSIONS):
    """Every session/cost run gets COMPLETE or INCOMPLETE; only COMPLETE normal runs get PASS/FAIL.
    The doubled-cost run is diagnostic. Incompleteness is stated at the top."""
    incomplete = []
    sess = {}
    for name, *_ in sessions:
        sess[name] = {}
        done = [p for p in pairs if p in all_res]
        diag = {k: sum(all_res[p][name]["diag"][k] for p in done) for k in ("skipped", "no_signal", "signals")}
        events = [e for p in done for e in all_res[p][name]["events"]]
        for r in RUNS:
            trades = [x for p in done for x in all_res[p][name]["runs"][r]["trades"]]
            unresolved = [x for p in done for x in all_res[p][name]["runs"][r]["unresolved"]]
            st = stats(trades, pairs)
            status = "INCOMPLETE" if (failure or unresolved) else "COMPLETE"
            if status == "INCOMPLETE":
                incomplete.append(f"{name}/{r}")
            reasons = {}
            for x in trades:
                for e in x["exits"]:
                    reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
            st.update({
                "status": status,
                "unresolved": [{"pair": x["pair"], "date": x["date"], "entry_time": iso(x["entry_time"]),
                                "remaining": x["remaining"], "filled_exits": len(x["exits"])} for x in unresolved],
                "cancelled": sum(1 for e in events if e["kind"] == "cancelled" and e["run"] == r),
                "invalid": sum(1 for e in events if e["kind"] == "invalid" and e["run"] == r),
                "exit_reasons": reasons, "diagnostics": diag,
                "median_risk_pips": statistics.median([x["risk_pips"] for x in trades]) if trades else None,
                "median_entry_cost_r": statistics.median([x["entry_cost_r"] for x in trades]) if trades else None,
                "median_exit_slippage_r": statistics.median([x["exit_slippage_r"] for x in trades]) if trades else None,
            })
            if r == "normal":
                st["verdict"] = "NO VERDICT (INCOMPLETE)" if status == "INCOMPLETE" else verdict(st, [])
            else:
                st["verdict"] = "diagnostic only (doubled costs)"
            sess[name][r] = st
    overall = "INCOMPLETE" if (failure or incomplete) else "COMPLETE"
    return {"status": overall, "incomplete_runs": incomplete, "failure": failure,
            "pairs_completed": [p for p in pairs if p in all_res], "spec": SPEC_VERSION,
            "identity": identity, "sessions": sess}


# ---------------------------------------------------------------- startup refusals
def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run_offline_tests():
    """Run only test_orb_retest.py. Returns (ok, ran, failures, errors, skipped)."""
    import unittest
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    suite = unittest.defaultTestLoader.loadTestsFromName("test_orb_retest")
    res = unittest.TextTestRunner(stream=sys.stdout, verbosity=1).run(suite)
    return (res.wasSuccessful() and not res.skipped, res.testsRun, len(res.failures), len(res.errors),
            len(res.skipped))


def preflight(args, here):
    """Refuse to start (exit 3) on hash mismatch, failed or skipped tests, or missing persistent storage."""
    problems = []
    code_sha = sha256_file(os.path.join(here, "orb_retest.py"))
    test_path = os.path.join(here, "test_orb_retest.py")
    test_sha = sha256_file(test_path) if os.path.exists(test_path) else None
    if args.expect_sha and code_sha != args.expect_sha:
        problems.append(f"orb_retest.py SHA256 {code_sha} != reviewed {args.expect_sha}")
    if args.expect_test_sha and test_sha != args.expect_test_sha:
        problems.append(f"test_orb_retest.py SHA256 {test_sha} != reviewed {args.expect_test_sha}")
    if args.require_volume:
        mount = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "")
        out = os.path.abspath(args.out)
        if not mount or not os.path.isdir(mount):
            problems.append("persistent volume not mounted (RAILWAY_VOLUME_MOUNT_PATH missing)")
        elif not out.startswith(os.path.abspath(mount) + os.sep):
            problems.append(f"output directory {out} is not on the persistent volume {mount}")
        elif os.path.exists(out) and os.listdir(out):
            problems.append(f"output directory {out} already exists and is not empty")
    return problems, code_sha, test_sha


# ---------------------------------------------------------------- export (logs, rate-limited)
EXPORT_CHUNK = 3000


def export_dir(out_dir, emit=print, pause=0.05):
    """Print every file in out_dir as gzip+base64 chunks, then its SHA256, at <= 20 lines/second."""
    import base64
    import gzip
    for n in sorted(os.listdir(out_dir)):
        with open(os.path.join(out_dir, n), "rb") as fh:
            raw = fh.read()
        b = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
        parts = [b[i:i + EXPORT_CHUNK] for i in range(0, len(b), EXPORT_CHUNK)] or [""]
        for i, part in enumerate(parts, 1):
            emit(f"EXPORT {n} {i}/{len(parts)} {part}")
            _sleep(pause)
        emit(f"EXPORT-SHA {n} {hashlib.sha256(raw).hexdigest()}")
        _sleep(pause)


class ExportError(Exception):
    """The exported files cannot be fully and exactly reconstructed."""


def rebuild_from_lines(lines):
    """Reassemble exported files from log lines. Rejects missing chunks, conflicting duplicate
    chunks (identical repeats are accepted), inconsistent chunk counts, missing or wrong per-file
    SHA256 lines. Returns {name: bytes}. This alone does NOT prove the export is complete; use
    verify_export for that."""
    import base64
    import gzip
    chunks, totals, shas = {}, {}, {}
    for ln in lines:
        if ln.startswith("EXPORT-SHA "):
            _, n, h = ln.split(" ")
            if n in shas and shas[n] != h:
                raise ExportError(f"{n}: conflicting SHA256 lines")
            shas[n] = h
        elif ln.startswith("EXPORT "):
            _, n, pos, *rest = ln.split(" ")
            i, total = map(int, pos.split("/"))
            if totals.setdefault(n, total) != total:
                raise ExportError(f"{n}: inconsistent chunk count")
            part = rest[0] if rest else ""
            slot = chunks.setdefault(n, [None] * total)
            if slot[i - 1] is not None and slot[i - 1] != part:
                raise ExportError(f"{n}: conflicting duplicate chunk {i}/{total}")
            slot[i - 1] = part
    files = {}
    for n, parts in chunks.items():
        missing = [i + 1 for i, p in enumerate(parts) if p is None]
        if missing:
            raise ExportError(f"{n}: missing export chunks {missing[:10]}")
        if n not in shas:
            raise ExportError(f"{n}: no EXPORT-SHA line")
        raw = gzip.decompress(base64.b64decode("".join(parts)))
        if hashlib.sha256(raw).hexdigest() != shas[n]:
            raise ExportError(f"{n}: SHA256 mismatch after rebuild")
        files[n] = raw
    for n in shas:
        if n not in files:
            raise ExportError(f"{n}: SHA256 line present but no chunks")
    return files


def verify_export(lines):
    """The only function that declares an export verified (DEPLOY.md section 8). Requires a
    reconstructed MANIFEST.sha256 whose own exported checksum matches; every file it lists
    reconstructed with the listed checksum; and no unexpected files besides the manifest.
    Returns {name: bytes} on success; raises ExportError otherwise."""
    files = rebuild_from_lines(lines)
    if "MANIFEST.sha256" not in files:
        raise ExportError("MANIFEST.sha256 missing from the export")
    listed = {}
    for ln in files["MANIFEST.sha256"].decode().splitlines():
        if not ln.strip():
            continue
        h, n = ln.split("  ", 1)
        listed[n] = h
    missing = sorted(n for n in listed if n not in files)
    if missing:
        raise ExportError(f"files listed in the manifest but not exported: {missing[:10]}")
    bad = sorted(n for n, h in listed.items() if hashlib.sha256(files[n]).hexdigest() != h)
    if bad:
        raise ExportError(f"checksum differs from the manifest: {bad[:10]}")
    extra = sorted(n for n in files if n != "MANIFEST.sha256" and n not in listed)
    if extra:
        raise ExportError(f"unexpected files not in the manifest: {extra[:10]}")
    return files


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-sha", default="")
    ap.add_argument("--expect-test-sha", default="")
    ap.add_argument("--require-volume", action="store_true")
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--export-after", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    if args.export_only:
        export_dir(args.out)
        return 0
    problems, code_sha, test_sha = preflight(args, here)
    if args.run_tests:
        ok, ran, fails, errs, skips = run_offline_tests()
        print(f"OFFLINE TESTS: ran {ran}, failures {fails}, errors {errs}, skipped {skips}")
        if not ok:
            problems.append("offline tests failed or were skipped")
    if problems:
        for p in problems:
            print(f"REFUSED: {p}")
        print("No OANDA request was made. No verdict.")
        return 3
    token = next((os.getenv(n).strip() for n in ("OANDA_API_KEY", "OANDA_API_TOKEN", "OANDA_TOKEN",
                                                  "OANDA_ACCESS_TOKEN") if os.getenv(n, "").strip()), None)
    if not token:
        print("REFUSED: no OANDA token in the environment. No verdict.")
        return 3
    os.makedirs(args.out, exist_ok=True)
    started = datetime.now(timezone.utc)
    identity = {"commit": os.getenv("RAILWAY_GIT_COMMIT_SHA"), "deployment": os.getenv("RAILWAY_DEPLOYMENT_ID"),
                "code_sha256": code_sha, "test_sha256": test_sha, "source_sha256": SOURCE_SHA256,
                "start_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ")}
    print(f"ORB RETEST BACKTEST | {SPEC_VERSION} | {json.dumps(identity)}")
    record = {"spec": SPEC_VERSION, "identity": identity, "status": "RUNNING",
              "settings": {"pairs": PAIRS, "sessions": SESSIONS, "range_candles": RANGE_CANDLES, "ema_alpha": EMA_ALPHA,
                           "gate_pct": GATE_PCT, "spacing_pct": SPACING_PCT, "shares": SHARES,
                           "session_dates": [FIRST_DATE.isoformat(), LAST_DATE.isoformat()], "runs": RUNS},
              "downloader": dict(DOWNLOADER, window=[iso(int(DL_START.timestamp())), iso(int(DL_END.timestamp()))]),
              "data": {}}
    all_res, failure = {}, None
    try:
        for pair in PAIRS:
            c, nreq = download(pair, token)
            gaps = gap_records(c)
            write_gaps(os.path.join(args.out, f"gaps_{pair}.csv"), gaps)
            record["data"][pair] = {"candles": len(c), "requests": nreq,
                                    "first": iso(c.t[0]) if len(c) else None, "last": iso(c.t[-1]) if len(c) else None,
                                    "sha256": fingerprint(c), "gaps_by_year": gap_summary(gaps),
                                    "close_spread_pips_by_year": spread_diagnostics(c, pair)}
            if not len(c) or c.t[0] >= int(datetime(2010, 1, 1, tzinfo=timezone.utc).timestamp()):
                raise DataError(f"{pair}: no candles before 2010-01-01; coverage incomplete")
            all_res[pair] = run_pair(c, pair)
            del c
            print(f"{pair}: simulated")
    except (DataError, DownloadError) as e:
        failure = {"type": type(e).__name__, "error": str(e), "pair": pair}
        print(f"STOP: {type(e).__name__}: {e}. Run INCOMPLETE; no verdict. Dates and pairs are not changed.")
    summary = build_summary(all_res, identity, failure)
    trades_all = {(n, r): [] for n, *_ in SESSIONS for r in RUNS}
    events = []
    for p in all_res:
        for n, *_ in SESSIONS:
            events += all_res[p][n]["events"]
            for r in RUNS:
                trades_all[(n, r)] += all_res[p][n]["runs"][r]["trades"] + all_res[p][n]["runs"][r]["unresolved"]
    for (n, r), tr in trades_all.items():
        write_trades(os.path.join(args.out, f"trades_{n}_{r}.csv"), tr)
    write_events(os.path.join(args.out, "events.csv"), events)
    record["status"] = summary["status"]
    record["failure"] = failure
    record["identity"]["end_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(os.path.join(args.out, "data_record.json"), "w") as fh:
        json.dump(record, fh, indent=1, default=str)
    names = write_manifest(args.out)
    print(f"STATUS: {summary['status']}" + (f" | incomplete: {summary['incomplete_runs']}" if summary["incomplete_runs"] else "")
          + (f" | failure: {failure}" if failure else ""))
    for name in summary["sessions"]:
        for r in RUNS:
            st = summary["sessions"][name][r]
            print(f"RESULT {name} {r} [{st['status']}]: n {st['n']} | win {st['win_rate']} | avgR {st['avg_r']} | "
                  f"PF {st['pf']} | total {st['total_r']} | maxDD {st['max_dd_r']} | halves {st['first_half_avg']} / "
                  f"{st['second_half_avg']} (split {st['split_time']}) | pairs+ {st['pairs_positive']}/12 | "
                  f"cancelled {st['cancelled']} invalid {st['invalid']} unresolved {len(st['unresolved'])} | "
                  f"VERDICT {st['verdict']}")
    print(f"FILES: {len(names)} written to {args.out} with MANIFEST.sha256")
    if args.export_after:
        export_dir(args.out)
    print(f"end {record['identity']['end_utc']}")
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # never leak credentials; never restart
        print(f"INTERNAL ERROR: {type(e).__name__}: {e}")
        sys.exit(2)
