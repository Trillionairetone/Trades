"""Opening range breakout (ORB) backtest. BACKTEST ONLY: instrument candle GETs only, no account or order endpoints.

Strategy rules (fixed before testing, unchanged):
  range   = high/low (mid) of the first 15 minutes after the session open (the first three M5 candles)
  signal  = first M5 candle (mid) to CLOSE above the range high (buy) or below the range low (sell),
            searched for 3 hours (by the clock) after the range ends
  entry   = next M5 open (buy at ask, sell at bid); a breakout only counts if that entry candle closes
            no later than the session exit time
  stop    = opposite side of the range (buy -> range low, sell -> range high)
  target  = 2R
  exit    = stop, target, or the close of the M5 candle ending at 16:00 New York (12:30 for the 09:30 open)
  limits  = one trade per pair per session per New York date
Sessions: London 08:00 London time, New York forex 08:00 NY, New York stock open 09:30 NY.

Accounting conventions:
  * Exit engine checks each candle's OPEN first, including the entry candle (entry and exit use opposite
    quote sides, so the exit-side open can already be past the stop). Opened past the stop -> fill at that
    open; opened past the target -> fill at the target. Then intrabar: both touched -> stop (ambiguous).
  * Returns come from bid/ask fills; spread is never deducted again. Slippage 0.1 pips/side and swap
    0.5 pips per 17:00 New York rollover (calendar dates, DST aware) are deducted once.
  * cost_r is ATTRIBUTION vs mid: half the entry-candle open spread + half the exit-side spread. Exit
    spread: the candle's close spread for close-price exits, its open spread for open (gap) fills, and its
    open spread as a proxy for intrabar stop/target fills (OANDA candles carry no intrabar spread).
  * Stress test (2x spread around mid, 0.2 pips/side slippage) reprices the EXACT baseline trades: same
    trade IDs, entry candle, direction, stop price and target price (the same bracket orders). Only the
    fills change. Its R denominator is the BASELINE risk (same position size), so every extra pip of cost
    shows up as lost R. No trade can be added or dropped by the stress run.
  * ORB_END fixes the endpoint; candles whose CLOSE is after it are dropped before simulation. A trade cut
    off by the endpoint is marked 'end', a scheduled session exit 'session', and a session exit forced
    early by missing candles 'session_gap'.
  * Data are streamed in 14-day chunks of whole New York dates, one pair at a time. Every session (London
    open through the 16:00 NY exit) lies inside one New York date, so no session spans two chunks.
  * If a pair fails to download, or fewer than 90% of weekdays have a valid opening range for a session,
    the run is INCOMPLETE and no PASS is printed.

Run: python orb_test.py
Env: ORB_END (ISO timestamp, default now), ORB_DAYS (default 3650), ORB_PAIRS (default EUR_USD,GBP_USD,USD_JPY)
"""
import gc
import logging
import os
import time
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from bot import Oanda, pip, parse_candles

log = logging.getLogger("orb")
NY = ZoneInfo("America/New_York")
LDN = ZoneInfo("Europe/London")
M5 = pd.Timedelta(minutes=5)

RR = 2.0
SEARCH = pd.Timedelta(hours=3)
SLIP = 0.1
SWAP = 0.5
CHUNK_DAYS = 14
MIN_COVERAGE = 0.90

# name, timezone, open time, exit time (New York)
SESSIONS = [
    ("London 08:00", LDN, dtime(8, 0), dtime(16, 0)),
    ("NY fx 08:00", NY, dtime(8, 0), dtime(16, 0)),
    ("NY stock 09:30", NY, dtime(9, 30), dtime(12, 30)),
]
QUOTES = ("bo", "bh", "bl", "bc", "ao", "ah", "al", "ac")


# ----------------------------------------------------------------------------- time helpers
def ny_midnight(day):
    return pd.Timestamp(datetime.combine(day, dtime(0, 0)), tz=NY).tz_convert("UTC")


def nights(entry, exit_):
    """17:00 New York rollovers in (entry, exit], counted on New York calendar dates (DST safe)."""
    def bucket(ts):
        local = pd.Timestamp(ts).tz_convert(NY)
        return local.date().toordinal() - (1 if local.time() < dtime(17, 0) else 0)
    return max(0, bucket(exit_) - bucket(entry))


# ----------------------------------------------------------------------------- data
def fetch_window(api, pair, a, b):
    """M5 candles with open >= a and CLOSE <= b, paging in 5000s. Candle GETs only."""
    parts, cur = [], a
    while cur < b:
        data = api.get(f"/v3/instruments/{pair}/candles",
                       {"granularity": "M5", "price": "BAM", "count": 5000,
                        "from": cur.strftime("%Y-%m-%dT%H:%M:%SZ")})
        raw = data.get("candles", [])
        if not raw:
            break
        df = parse_candles(raw)
        last_time = pd.Timestamp(raw[-1]["time"])
        got = len(raw)
        del raw, data
        if not df.empty:
            parts.append(df)
        if got < 5000 or last_time + M5 >= b:
            break
        cur = last_time + pd.Timedelta(seconds=1)
        time.sleep(0.1)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
    return d[(d["time"] >= a) & (d["time"] + M5 <= b)].reset_index(drop=True)


def widen_arrays(q, mid, factor=2.0):
    """Quote arrays with the bid/ask spread widened around mid."""
    out = {}
    for f in "ohlc":
        out["b" + f] = mid[f] - factor * (mid[f] - q["b" + f])
        out["a" + f] = mid[f] + factor * (q["a" + f] - mid[f])
    return out


# ----------------------------------------------------------------------------- exit engine
def exit_trade(direction, j, sl, tp, last, q):
    """Walk candles j..last. Open-first on EVERY candle, including the entry candle.
    Returns (price, k, kind, ambiguous, spread_side) with spread_side 'o' or 'c'."""
    bo, bh, bl, bc, ao, ah, al, ac = (q[c] for c in QUOTES)
    for k in range(j, last + 1):
        if direction == 1:
            if bo[k] <= sl:
                return bo[k], k, "stop_gap", False, "o"
            if bo[k] >= tp:
                return tp, k, "target_gap", False, "o"
            hit_sl, hit_tp = bl[k] <= sl, bh[k] >= tp
        else:
            if ao[k] >= sl:
                return ao[k], k, "stop_gap", False, "o"
            if ao[k] <= tp:
                return tp, k, "target_gap", False, "o"
            hit_sl, hit_tp = ah[k] >= sl, al[k] <= tp
        if hit_sl:
            return sl, k, "stop", bool(hit_tp), "o"     # open spread = proxy for an intrabar fill
        if hit_tp:
            return tp, k, "target", False, "o"
    return (bc[last] if direction == 1 else ac[last]), last, "close", False, "c"


def fill(pair, direction, j, entry, sl, tp, risk_den, last, last_kind, t, q, slip):
    """Simulate one trade's exit and costs. last_kind labels a close-price exit."""
    p = pip(pair)
    px, k, kind, amb, side = exit_trade(direction, j, sl, tp, last, q)
    if kind == "close":
        kind = last_kind
    exit_time = t[k] + M5 if side == "c" else t[k]          # close-price exits record the candle close
    n = nights(t[j], exit_time)
    half_in = (q["ao"][j] - q["bo"][j]) / 2
    half_out = (q["a" + side][k] - q["b" + side][k]) / 2
    extra = (2 * slip + n * SWAP) * p
    return {
        "exit": exit_time, "kind": kind, "amb": amb, "approx_time": side == "o" and kind not in ("stop_gap", "target_gap"),
        "entry_px": entry, "exit_px": px,
        "r": float((direction * (px - entry) - extra) / risk_den),
        "cost_r": float((half_in + half_out + extra) / risk_den),
        "spread_r": float((half_in + half_out) / risk_den),
        "nights": n, "hold_min": (exit_time - t[j]).total_seconds() / 60,
    }


# ----------------------------------------------------------------------------- one chunk
def simulate_chunk(d, pair, dates, end_ts, coverage, slip=SLIP):
    """All sessions for the given New York dates. Returns (base_trades, stress_trades, invalid_risk)."""
    base, stress, invalid = [], [], 0
    if d.empty:
        for day in dates:
            if day.weekday() < 5:
                for s in SESSIONS:
                    coverage[s[0]][1] += 1
        return base, stress, invalid
    t = pd.DatetimeIndex(d["time"])
    pos = {ts: i for i, ts in enumerate(t)}
    mid = {f: d[f].to_numpy() for f in "ohlc"}
    q = {c: d[c].to_numpy() for c in QUOTES}
    qs = widen_arrays(q, mid)
    n = len(t)
    for day in dates:
        if day.weekday() >= 5:
            continue
        for name, tz, t_open, t_exit in SESSIONS:
            coverage[name][1] += 1
            o_utc = pd.Timestamp(datetime.combine(day, t_open), tz=tz).tz_convert("UTC")
            i0 = pos.get(o_utc)
            if i0 is None or i0 + 2 >= n or t[i0 + 2] != o_utc + 2 * M5:
                continue
            hi, lo = mid["h"][i0:i0 + 3].max(), mid["l"][i0:i0 + 3].min()
            if not hi > lo:
                continue
            coverage[name][0] += 1
            exit_end = pd.Timestamp(datetime.combine(day, t_exit), tz=NY).tz_convert("UTC")
            stop_at = min(exit_end, end_ts)
            search_end = o_utc + 3 * M5 + SEARCH
            sig = None
            for s in range(i0 + 3, n - 1):
                if t[s] >= search_end or t[s + 1] + M5 > stop_at:
                    break
                c = mid["c"][s]
                if c > hi or c < lo:
                    sig = (s, 1 if c > hi else -1)
                    break
            if sig is None:
                continue
            s, direction = sig
            j = s + 1
            last = j
            while last + 1 < n and t[last + 1] + M5 <= stop_at:
                last += 1
            if exit_end > end_ts:
                last_kind = "end"
            elif t[last] + M5 == exit_end:
                last_kind = "session"
            else:
                last_kind = "session_gap"
            sl = lo if direction == 1 else hi
            entry = q["ao"][j] if direction == 1 else q["bo"][j]
            risk = direction * (entry - sl)
            if risk <= 0:            # entry already beyond the stop: baseline rejects it (and so does stress)
                invalid += 1
                continue
            tp = entry + direction * RR * risk
            meta = {"id": f"{pair}|{name}|{day}", "pair": pair, "session": name, "entry": t[j],
                    "dir": direction, "stop": sl, "target": tp, "risk_pips": risk / pip(pair),
                    "spread_frac": (q["ao"][j] - q["bo"][j]) / risk}
            base.append({**meta, **fill(pair, direction, j, entry, sl, tp, risk, last, last_kind, t, q, slip)})
            entry_s = qs["ao"][j] if direction == 1 else qs["bo"][j]
            stress.append({**meta, **fill(pair, direction, j, entry_s, sl, tp, risk, last, last_kind, t, qs, 2 * slip)})
    return base, stress, invalid


# ----------------------------------------------------------------------------- run
def run(api, pairs, start, end, chunk_days=CHUNK_DAYS, fetch=fetch_window):
    """Stream each pair in chunks of whole New York dates. Returns a results dict."""
    end_ts = pd.Timestamp(end)
    first = (pd.Timestamp(start).tz_convert(NY) + pd.Timedelta(days=1)).date()   # first full NY date
    last_day = end_ts.tz_convert(NY).date()
    out = {"base": [], "stress": [], "invalid_risk": 0, "incomplete": [], "max_rows": 0, "coverage": {}}
    for pair in pairs:
        coverage = {s[0]: [0, 0] for s in SESSIONS}
        try:
            day = first
            while day <= last_day:
                dates = [day + timedelta(days=i) for i in range(chunk_days) if day + timedelta(days=i) <= last_day]
                a = ny_midnight(dates[0])
                b = min(ny_midnight(dates[-1] + timedelta(days=1)), end_ts)
                d = fetch(api, pair, a, b)
                if not d.empty:
                    d = d[d["time"] + M5 <= end_ts].reset_index(drop=True)   # close must be <= endpoint
                out["max_rows"] = max(out["max_rows"], len(d))
                b_tr, s_tr, inv = simulate_chunk(d, pair, dates, end_ts, coverage)
                out["base"] += b_tr
                out["stress"] += s_tr
                out["invalid_risk"] += inv
                del d
                day = dates[-1] + timedelta(days=1)
            gc.collect()
        except Exception as e:
            out["incomplete"].append(f"{pair}: download/processing failed ({type(e).__name__}: {e})")
            continue
        out["coverage"][pair] = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in coverage.items()}
        for sess, frac in out["coverage"][pair].items():
            if frac < MIN_COVERAGE:
                out["incomplete"].append(f"{pair} {sess}: only {frac:.0%} of weekdays had a valid opening range")
    return out


# ----------------------------------------------------------------------------- reporting
def stats(trades):
    if not trades:
        return {"n": 0, "win": 0.0, "avg": 0.0, "pf": 0.0, "dd": 0.0, "total": 0.0, "hold": 0.0, "cost": 0.0}
    ts = sorted(trades, key=lambda x: x["exit"])
    r = np.array([x["r"] for x in ts])
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    curve = np.r_[0.0, np.cumsum(r)]
    return {"n": len(r), "win": 100 * (r > 0).mean(), "avg": float(r.mean()),
            "pf": float(min(gp / gl, 99.0)) if gl > 0 else 99.0,
            "dd": float((np.maximum.accumulate(curve) - curve).max()), "total": float(r.sum()),
            "hold": float(np.mean([x["hold_min"] for x in ts])), "cost": float(np.mean([x["cost_r"] for x in ts]))}


HEADER = f"{'':<16}{'n':>6}{'win%':>7}{'avgR':>8}{'PF':>6}{'maxDD':>8}{'totalR':>9}{'hold':>7}{'costR':>7}"


def line(label, s):
    return (f"{label:<16}{s['n']:>6}{s['win']:>7.1f}{s['avg']:>8.3f}{s['pf']:>6.2f}{s['dd']:>8.1f}"
            f"{s['total']:>9.1f}{s['hold']:>7.0f}{s['cost']:>7.3f}")


def verdict(base, boundary, incomplete):
    ev = stats([x for x in base if x["entry"] < boundary])
    ho = stats([x for x in base if x["entry"] >= boundary])
    pooled = stats(base)
    if incomplete:
        return "INCOMPLETE"
    ok = ev["avg"] > 0 and ho["avg"] > 0 and pooled["pf"] >= 1.2 and pooled["n"] >= 300
    return "PASS" if ok else "FAIL"


def report(res, pairs, boundary):
    results = {}
    for name, *_ in SESSIONS:
        base = [x for x in res["base"] if x["session"] == name]
        stress = [x for x in res["stress"] if x["session"] == name]
        log.info("")
        log.info(f"===== ORB | {name} | evaluation before {boundary:%Y-%m-%d %H:%M} UTC, holdout after =====")
        log.info(HEADER + "   (hold in minutes; costR = half spreads + slippage + swap, in R)")
        for label, subset in (("evaluation", [x for x in base if x["entry"] < boundary]),
                              ("holdout", [x for x in base if x["entry"] >= boundary]), ("pooled", base)):
            log.info(line(label, stats(subset)))
        for label, subset in (("2x evaluation", [x for x in stress if x["entry"] < boundary]),
                              ("2x holdout", [x for x in stress if x["entry"] >= boundary]), ("2x pooled", stress)):
            log.info(line(label, stats(subset)))
        for pair in pairs:
            sp = stats([x for x in base if x["pair"] == pair])
            log.info(f"  {pair:<9} n {sp['n']:>5}  avgR {sp['avg']:>7.3f}  totalR {sp['total']:>8.1f}")
        kinds = pd.Series([x["kind"] for x in base]).value_counts().to_dict() if base else {}
        log.info(f"  exits {kinds} | ambiguous candles {sum(x['amb'] for x in base)} | "
                 f"avg entry spread {np.mean([x['spread_frac'] for x in base]) * 100 if base else 0:.1f}% of risk")
        results[name] = verdict(base, boundary, res["incomplete"])
        log.info(f"  RESULT {name}: {results[name]} (needs avgR > 0 in both periods, pooled PF >= 1.2, 300+ trades)")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    end = pd.Timestamp(os.getenv("ORB_END")) if os.getenv("ORB_END") else pd.Timestamp.now(tz="UTC").floor("min")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end > pd.Timestamp.now(tz="UTC"):
        raise ValueError("ORB_END cannot be in the future")
    days = int(os.getenv("ORB_DAYS", "3650"))
    pairs = [p.strip().upper() for p in os.getenv("ORB_PAIRS", "EUR_USD,GBP_USD,USD_JPY").split(",") if p.strip()]
    start = end - pd.Timedelta(days=days)
    boundary = start + (end - start) * 2 / 3
    log.info(f"ORB BACKTEST | M5 | window {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC "
             f"(ORB_END={end.isoformat()}) | split {boundary:%Y-%m-%d %H:%M} UTC | pairs {pairs}")
    log.info(f"Target {RR:g}R | slippage {SLIP:g} pips/side | swap {SWAP:g} pips per 17:00 NY rollover | "
             f"stress = same trades, 2x spread, {2 * SLIP:g} pips/side, baseline risk as R denominator")
    log.info("Earlier tests covered these years; the holdout is not untouched history. 3 sessions are tested, "
             "so one passing by luck is more likely than for a single test.")
    res = run(Oanda(), pairs, start, end)
    assert len(res["base"]) == len(res["stress"]), "stress run must reprice exactly the baseline trades"
    log.info(f"Data: largest chunk {res['max_rows']} candles | entries rejected (entry beyond stop): {res['invalid_risk']}")
    for pair, cov in res["coverage"].items():
        log.info(f"  coverage {pair}: " + ", ".join(f"{k} {v:.1%}" for k, v in cov.items()))
    results = report(res, pairs, boundary)
    log.info("")
    if res["incomplete"]:
        log.info("RUN INCOMPLETE - no PASS can be declared:")
        for msg in res["incomplete"]:
            log.info(f"  {msg}")
    log.info("SUMMARY | " + " | ".join(f"{k}: {v}" for k, v in results.items()))
    log.info("ORB backtest finished. Nothing was traded.")


if __name__ == "__main__":
    main()
