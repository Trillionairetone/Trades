"""ORB Retest data-coverage probe (read-only).

Implements ORB_Review_Notes.txt section D (Review Revision 05) with ChatGPT's probe fixes:
  - 48 planned requests: 12 pairs x (R1 start-of-history + R2-R4 sample Tuesdays)
  - instrument-candle GET requests only, on the OANDA PRACTICE host
  - at most 3 HTTP attempts per planned request (at most 144 in total), explicit timeouts,
    bounded retry delays, immediate stop on authentication failure, never restarts itself
  - stop conditions S1-S5; on a stop: reason, final summary, unperformed requests marked
    NOT RUN, unsuccessful exit code
  - credentials are read from the environment and never printed

It does not place, modify or read orders, trades, positions or accounts, and it does not
import any other module from the repository. Full historical coverage is NOT proven by
this probe; only the full validated download in the backtest can establish it.

Exit codes: 0 = PROBE GO, 1 = PROBE STOP, 2 = internal error (also reported as STOP).
"""
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

SPEC_VERSION = "ORB Retest Review Revision 05 (strategy rules FROZEN, data probe pending)"
HOST = "https://api-fxpractice.oanda.com"          # practice host, fixed
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD",
         "USD_CHF", "EUR_JPY", "GBP_JPY", "EUR_GBP", "AUD_JPY", "EUR_AUD"]
COMMON = {"granularity": "M5", "price": "BAM", "smooth": "false", "includeFirst": "true"}
PLAN = [  # (request id, from, count)
    ("R1", "2009-12-28T00:00:00Z", 5),
    ("R2", "2010-06-15T06:00:00Z", 180),
    ("R3", "2015-06-16T06:00:00Z", 180),
    ("R4", "2020-06-16T06:00:00Z", 180),
]
PRE_2010 = datetime(2010, 1, 1, tzinfo=timezone.utc)
MAX_ATTEMPTS = 3
RETRY_DELAYS = (2.0, 5.0)        # seconds before attempt 2 and attempt 3 (bounded)
TIMEOUT_S = 20                   # socket timeout: applies to connecting and to each read
WINDOW = ((6, 0), (21, 0))       # [06:00, 21:00) UTC on the sample date
# June sample dates: London (BST) and New York (EDT) are both on summer time, so in UTC:
SESSIONS_UTC = {"London 08:00": (7, 0), "NY forex 08:00": (12, 0), "US stocks 09:30": (13, 30)}
LIQ_WINDOW_UTC = ((20, 45), (21, 0))   # New York 16:45-17:00 EDT = [20:45, 21:00) UTC

_urlopen = urllib.request.urlopen      # replaced by mocks in the unit tests
_sleep = time.sleep


class Stop(Exception):
    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def get_token():
    for name in ("OANDA_API_KEY", "OANDA_API_TOKEN", "OANDA_TOKEN", "OANDA_ACCESS_TOKEN"):
        v = os.getenv(name)
        if v and v.strip():
            return v.strip()
    return None


def redact(text, token):
    text = str(text)
    return text.replace(token, "[REDACTED]") if token else text


def pip(pair):
    return 0.01 if "JPY" in pair else 0.0001


def build_url(pair, frm, count):
    q = dict(COMMON, **{"from": frm, "count": str(count)})
    return f"{HOST}/v3/instruments/{pair}/candles?" + urllib.parse.urlencode(q)


def fetch(url, token, log, counter):
    """At most MAX_ATTEMPTS HTTP attempts; counter[0] counts every attempt made.
    Returns the parsed JSON. Raises Stop (S1)."""
    last = "no attempt made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            _sleep(RETRY_DELAYS[attempt - 2])
        counter[0] += 1
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                                   "Accept-Datetime-Format": "RFC3339"})
        try:
            with _urlopen(req, timeout=TIMEOUT_S) as resp:
                body = resp.read()
            return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise Stop("S1", f"authentication failure (HTTP {e.code}) on attempt {attempt}; stopping immediately") from None
            last = f"HTTP {e.code}"
            if e.code not in (429, 500, 502, 503, 504):
                raise Stop("S1", f"non-retryable {last} on attempt {attempt}") from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            last = f"network error ({type(e).__name__})"
        except (ValueError, UnicodeDecodeError):
            last = "unreadable response body"
        log.append(redact(f"    attempt {attempt} failed: {last}", token))
    raise Stop("S1", f"request failed after {MAX_ATTEMPTS} attempts (last: {last})")


def parse_time(s):
    # RFC3339 with nanoseconds, e.g. 2010-06-15T06:00:00.000000000Z
    if not isinstance(s, str) or not s.endswith("Z") or "T" not in s:
        raise ValueError("bad time format")
    base = s[:-1].split(".")[0]
    return datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def validate(candles):
    """Spec data checks. Returns (list of (time, candle)), raises ValueError with the reason."""
    out, prev = [], None
    for i, c in enumerate(candles):
        if not isinstance(c, dict):
            raise ValueError(f"candle {i}: not an object")
        for k in ("time", "complete", "bid", "ask", "mid"):
            if k not in c:
                raise ValueError(f"candle {i}: missing field '{k}'")
        t = parse_time(c["time"])
        if t.second or t.minute % 5:
            raise ValueError(f"candle {i} ({c['time']}): not aligned to 5 minutes")
        if c["complete"] is not True:
            raise ValueError(f"candle {i} ({c['time']}): incomplete")
        if prev is not None and t <= prev:
            raise ValueError(f"candle {i} ({c['time']}): duplicate or out of order")
        vals = {}
        for side in ("bid", "ask", "mid"):
            q = c[side]
            if not isinstance(q, dict):
                raise ValueError(f"candle {i}: '{side}' not an object")
            for f in "ohlc":
                if f not in q:
                    raise ValueError(f"candle {i}: missing {side}.{f}")
                try:
                    v = float(q[f])
                except (TypeError, ValueError):
                    raise ValueError(f"candle {i}: {side}.{f} not a number") from None
                if not math.isfinite(v) or v <= 0:
                    raise ValueError(f"candle {i}: {side}.{f} not finite and positive")
                vals[(side, f)] = v
            o, h, l, cl = (vals[(side, f)] for f in "ohlc")
            if not (l <= min(o, cl) <= max(o, cl) <= h):
                raise ValueError(f"candle {i} ({c['time']}): invalid {side} OHLC")
        for f in "ohlc":
            if vals[("bid", f)] > vals[("ask", f)]:
                raise ValueError(f"candle {i} ({c['time']}): bid above ask in field {f}")
        out.append((t, vals))
        prev = t
    return out


def at(day, hm):
    return day.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)


def sample_report(pair, day, rows):
    """Coverage numbers for one sample date, from candles opening in [06:00, 21:00) UTC."""
    w0, w1 = at(day, WINDOW[0]), at(day, WINDOW[1])
    inside = [(t, v) for t, v in rows if w0 <= t < w1]
    outside = [t for t, _ in rows if not (w0 <= t < w1)]
    times = {t for t, _ in inside}
    missing = {}
    for name, hm in SESSIONS_UTC.items():
        s = at(day, hm)
        need = [s + timedelta(minutes=5 * k) for k in range(6)]
        missing[name] = [x.strftime("%H:%M") for x in need if x not in times]
    l0, l1 = at(day, LIQ_WINDOW_UTC[0]), at(day, LIQ_WINDOW_UTC[1])
    liq = sorted(t.strftime("%H:%M") for t in times if l0 <= t < l1)
    spreads = [(v[("ask", "c")] - v[("bid", "c")]) / pip(pair) for _, v in inside]
    return {
        "in_window": len(inside), "outside": [t.strftime("%Y-%m-%dT%H:%M") for t in outside],
        "missing_range": missing, "liquidation_candles": liq,
        "spread_median": round(statistics.median(spreads), 2) if spreads else None,
        "spread_max": round(max(spreads), 2) if spreads else None,
    }


def run(out=print):
    started = datetime.now(timezone.utc)
    token = get_token()
    with open(__file__, "rb") as fh:
        code_hash = hashlib.sha256(fh.read()).hexdigest()
    out("ORB RETEST DATA PROBE (read-only, practice host, instrument candles only)")
    out(f"Spec: {SPEC_VERSION}")
    out(f"Probe file SHA256: {code_hash}")
    out(f"Commit: {os.getenv('RAILWAY_GIT_COMMIT_SHA', 'unknown')} | "
        f"Deployment: {os.getenv('RAILWAY_DEPLOYMENT_ID', 'unknown')} | Start (UTC): {started:%Y-%m-%dT%H:%M:%SZ}")
    out(f"Plan: {len(PAIRS)} pairs x {len(PLAN)} requests = {len(PAIRS) * len(PLAN)} planned; "
        f"max {MAX_ATTEMPTS} attempts each ({len(PAIRS) * len(PLAN) * MAX_ATTEMPTS} max); "
        f"timeout {TIMEOUT_S}s; common params {json.dumps(COMMON)}")

    results, details, triggered = [], [], []
    total_attempts = 0
    stop = None
    if not token:
        stop = Stop("S1", "no OANDA token found in the environment")
        triggered.append(stop)

    deficient_days = {}
    for pair in PAIRS:
        for rid, frm, count in PLAN:
            row = {"pair": pair, "req": rid, "from": frm, "count": count,
                   "attempts": 0, "outcome": "NOT RUN", "candles": "-", "first": "-",
                   "last": "-", "validation": "-"}
            results.append(row)
            if stop:
                continue
            log = []
            try:
                counter = [0]
                try:
                    data = fetch(build_url(pair, frm, count), token, log, counter)
                finally:
                    row["attempts"] = counter[0]
                row["outcome"] = "HTTP 200"
                candles = data.get("candles", []) if isinstance(data, dict) else None
                if not isinstance(candles, list):
                    raise Stop("S3", f"{pair} {rid}: response has no candle list")
                row["candles"] = len(candles)
                if candles:
                    row["first"] = candles[0].get("time", "?") if isinstance(candles[0], dict) else "?"
                    row["last"] = candles[-1].get("time", "?") if isinstance(candles[-1], dict) else "?"
                try:
                    rows = validate(candles)
                except ValueError as e:
                    row["validation"] = "FAIL"
                    raise Stop("S3", f"{pair} {rid}: {e}") from None
                row["validation"] = "OK"
                if rid == "R1":
                    if not rows or rows[0][0] >= PRE_2010:
                        raise Stop("S2", f"{pair} R1: no candle before 2010-01-01T00:00:00Z "
                                         f"(first returned: {row['first']})")
                else:
                    day = datetime.strptime(frm[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    rep = sample_report(pair, day, rows)
                    details.append((pair, frm[:10], rep))
                    if any(rep["missing_range"].values()):
                        deficient_days[pair] = deficient_days.get(pair, 0) + 1
                    if not rep["liquidation_candles"]:
                        raise Stop("S5", f"{pair} {frm[:10]}: no candle opens in [20:45, 21:00) UTC; "
                                         f"liquidation-coverage risk (not a strategy FAIL)")
                    if deficient_days.get(pair, 0) >= 2:
                        raise Stop("S4", f"{pair}: range candles missing on {deficient_days[pair]} of 3 sample days")
            except Stop as s:
                stop = s
                if row["outcome"] == "NOT RUN":
                    row["outcome"] = "FAILED"
                triggered.append(s)
            finally:
                total_attempts += row["attempts"]
                for line in log:
                    out(line)

    out("")
    out("REQUESTS")
    out("pair     req from                  count att outcome   candles first                          last                           validation")
    for r in results:
        out(f"{r['pair']:<8} {r['req']:<3} {r['from']:<21} {r['count']:>5} {r['attempts']:>3} {r['outcome']:<9} "
            f"{str(r['candles']):>7} {str(r['first']):<30} {str(r['last']):<30} {r['validation']}")
    out("")
    out("SAMPLE DATES (candles opening in [06:00, 21:00) UTC)")
    for pair, day, rep in details:
        miss = "; ".join(f"{k}: {', '.join(v) if v else 'none missing'}" for k, v in rep["missing_range"].items())
        out(f"{pair} {day}: in-window {rep['in_window']}/180 | missing range candles -> {miss} | "
            f"liquidation-window candles: {', '.join(rep['liquidation_candles']) or 'NONE'} | "
            f"close spread pips median {rep['spread_median']} max {rep['spread_max']} | "
            f"outside window: {', '.join(rep['outside']) or 'none'}")
    out("")
    planned = len(results)
    completed = sum(1 for r in results if r["outcome"] == "HTTP 200" and r["validation"] == "OK")
    attempted = sum(1 for r in results if r["outcome"] != "NOT RUN")
    out(f"TOTALS planned {planned} | attempted {attempted} | completed {completed} | "
        f"not run {planned - attempted} | HTTP attempts {total_attempts}")
    if triggered:
        out("STOP CONDITIONS TRIGGERED")
        for s in triggered:
            out(redact(f"  {s.code}: {s.detail}", token))
    ended = datetime.now(timezone.utc)
    verdict = "PROBE STOP" if stop else "PROBE GO"
    out(f"End (UTC): {ended:%Y-%m-%dT%H:%M:%SZ}")
    out(f"RESULT: {verdict}. Full historical coverage remains UNVERIFIED; only the full validated "
        f"download can establish it. A stop never shortens dates, drops pairs or changes rules.")
    return 0 if not stop else 1


if __name__ == "__main__":
    try:
        code = run()
    except Exception as e:  # report without credentials, never restart
        print(redact(f"INTERNAL ERROR: {type(e).__name__}: {e}", get_token()))
        print("RESULT: PROBE STOP (internal error). Full historical coverage remains UNVERIFIED.")
        code = 2
    sys.exit(code)
