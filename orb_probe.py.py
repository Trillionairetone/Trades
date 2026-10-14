"""Frozen R05 data feasibility probe. No strategy, account or trading imports.

Run only after test_orb_probe.py passes. Emits JSON lines to stdout; exits 1
on PROBE STOP. Uses existing OANDA token aliases, never account credentials.
"""
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import statistics
import time

import requests

BASE = "https://api-fxpractice.oanda.com"
PAIRS = ("EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD",
         "USD_CHF", "EUR_JPY", "GBP_JPY", "EUR_GBP", "AUD_JPY", "EUR_AUD")
WINDOWS = (("R1", "2009-12-28T00:00:00Z", 5),
           ("R2", "2010-06-15T06:00:00Z", 180),
           ("R3", "2015-06-16T06:00:00Z", 180),
           ("R4", "2020-06-16T06:00:00Z", 180))
ALIASES = ("OANDA_API_KEY", "OANDA_API_TOKEN", "OANDA_TOKEN", "OANDA_ACCESS_TOKEN")


def emit(value):
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def params_for(window):
    return dict(granularity="M5", price="BAM", smooth="false",
                includeFirst="true", **{"from": window[1], "count": window[2]})


def timestamp(value):
    # Check fractional digits before parsing: datetime otherwise truncates nanos.
    if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0+)?Z", value):
        raise ValueError("timestamp must be an aligned UTC candle open")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.minute % 5:
        raise ValueError("timestamp not five-minute aligned")
    return result


def iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def validate(data, pair, window):
    """Return validated candles and a safe, specific error without raw payloads."""
    if not isinstance(data, dict) or data.get("instrument") != pair or data.get("granularity") != "M5":
        return [], "response instrument/granularity mismatch or missing"
    raw = data.get("candles")
    if not isinstance(raw, list) or len(raw) > window[2]:
        return [], "candles missing/not a list or exceeds requested count"
    parsed, previous = [], None
    start = timestamp(window[1])
    for index, candle in enumerate(raw):
        try:
            t = timestamp(candle["time"])
            if t < start or (previous is not None and t <= previous):
                raise ValueError("timestamp before request, duplicate or out of order")
            if candle.get("complete") is not True:
                raise ValueError("candle is not complete")
            series = {}
            for side in ("bid", "ask", "mid"):
                values = {}
                for field in "ohlc":
                    v = candle[side][field]
                    if isinstance(v, bool):
                        raise ValueError("boolean price")
                    values[field] = float(v)
                if not all(math.isfinite(v) and v > 0 for v in values.values()):
                    raise ValueError("non-finite or non-positive price")
                if not values["l"] <= min(values["o"], values["c"]) <= max(values["o"], values["c"]) <= values["h"]:
                    raise ValueError("invalid OHLC envelope")
                series[side] = values
            if any(series["bid"][f] > series["ask"][f] for f in "ohlc"):
                raise ValueError("bid exceeds ask")
            parsed.append((t, series))
            previous = t
        except (KeyError, TypeError, ValueError, OverflowError):
            return [], f"invalid candle at response index {index} (timestamp/completeness/required BAM fields/prices/OHLC)"
    return parsed, None


def analyze(data, pair, window):
    candles, error = validate(data, pair, window)
    result = {"validation": "ERROR" if error else "OK", "error": error,
              "range_missing": {}, "bad_range_day": False}
    if error:
        return result, ["S3"]
    result.update(returned=len(candles), first=iso(candles[0][0]) if candles else None,
                  last=iso(candles[-1][0]) if candles else None)
    if window[0] == "R1":
        before = bool(candles and candles[0][0] < timestamp("2010-01-01T00:00:00Z"))
        result["pre_2010_returned"] = before
        return result, [] if before else ["S2"]
    start = timestamp(window[1])
    end = start.replace(hour=21)
    inside = [(t, s) for t, s in candles if start <= t < end]
    times = {t for t, _ in inside}
    missing = {}
    for name, hour, minute in (("London", 7, 0), ("NY_forex", 12, 0), ("US_stocks", 13, 30)):
        opening = start.replace(hour=hour, minute=minute)
        missing[name] = [iso(opening + timedelta(minutes=5*i)) for i in range(6)
                         if opening + timedelta(minutes=5*i) not in times]
    liquid = sorted(iso(t) for t in times if start.replace(hour=20, minute=45) <= t < end)
    pip = .01 if pair.endswith("JPY") else .0001
    spreads = [(s["ask"]["c"] - s["bid"]["c"]) / pip for _, s in inside]
    result.update(in_window=len(inside), expected_slots=180,
                  outside_window=[iso(t) for t, _ in candles if not start <= t < end],
                  range_missing=missing, bad_range_day=any(missing.values()),
                  liquidation_candles=liquid,
                  median_close_spread_pips=statistics.median(spreads) if spreads else None,
                  max_close_spread_pips=max(spreads) if spreads else None)
    return result, [] if liquid else ["S5"]


def fetch(session, token, pair, window, sleep=time.sleep):
    """Exactly one GET per attempt; no redirects or adapter-level retries."""
    last = None
    for attempt in range(1, 4):
        try:
            with session.get(BASE + f"/v3/instruments/{pair}/candles",
                             params=params_for(window),
                             headers={"Authorization": "Bearer " + token,
                                      "Accept-Datetime-Format": "RFC3339"},
                             timeout=(10, 30), allow_redirects=False) as response:
                last = response.status_code
                if last in (401, 403):
                    return None, attempt, last, "S1"
                if last == 200:
                    try:
                        return response.json(), attempt, last, None
                    except ValueError:
                        return None, attempt, last, "S3"
        except requests.RequestException:
            last = "TRANSPORT_ERROR"
        if attempt < 3:
            sleep(attempt)
    return None, 3, last, "S1"


def run(session, token, output=emit, sleep=time.sleep):
    started = utcnow()
    output({"type": "START", "utc": started, "spec": "R05",
            "commit": os.getenv("RAILWAY_GIT_COMMIT_SHA"),
            "deployment": os.getenv("RAILWAY_DEPLOYMENT_ID"),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "planned": 48, "attempt_limit": 144, "endpoint_base": BASE})
    attempts = attempted = completed = 0
    stops = []
    bad_days = {p: 0 for p in PAIRS}
    if not token:
        stops = [{"condition": "S1", "reason": "No existing token available; no requests made"}]
    for pair in PAIRS:
        for window in WINDOWS:
            row = {"type": "REQUEST", "pair": pair, "request": window[0],
                   "params": params_for(window)}
            if stops:
                output(dict(row, status="NOT RUN", attempts=0))
                continue
            attempted += 1
            data, count, http, failure = fetch(session, token, pair, window, sleep)
            attempts += count
            row.update(attempts=count, http=http)
            if failure:
                conditions = [failure]
                row["error"] = "Malformed JSON" if failure == "S3" else "HTTP/authentication/transport failure"
            else:
                completed += 1
                details, conditions = analyze(data, pair, window)
                row.update(details)
                if details["bad_range_day"]:
                    bad_days[pair] += 1
                if bad_days[pair] >= 2:
                    conditions.append("S4")
            row.update(status="STOP" if conditions else "CHECKED", stop_conditions=conditions)
            output(row)
            stops.extend({"condition": s, "pair": pair, "request": window[0],
                          "date": window[1][:10]} for s in conditions)
            if not stops:
                sleep(.2)
    code = 1 if stops else 0
    output({"type": "SUMMARY", "verdict": "PROBE STOP" if stops else "PROBE GO",
            "started_utc": started, "ended_utc": utcnow(), "exit_status": code,
            "planned": 48, "attempted": attempted, "completed_http_json": completed,
            "not_run": 48-attempted, "http_attempts": attempts, "stops": stops,
            "full_historical_coverage_verified": False,
            "strategy_verdict": "NOT TESTED"})
    return code


def main():
    token = next((os.getenv(k, "").strip() for k in ALIASES if os.getenv(k, "").strip()), "")
    with requests.Session() as session:
        session.trust_env = False
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
        try:
            return run(session, token)
        except Exception:
            # Never print exception messages, request headers or response bodies.
            emit({"type": "SUMMARY", "verdict": "PROBE STOP", "exit_status": 1,
                  "reason": "Unexpected local error; inspect code with mocked tests",
                  "full_historical_coverage_verified": False})
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
