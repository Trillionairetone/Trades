#!/usr/bin/env python3
"""
pair_profile.py - READ-ONLY pair selection study (no signals, no orders).
Implements "Pair Profile Spec - Revision 03".

Ranks pairs by "movement per unit of cost" using OANDA M5 bid/ask candles,
STUDY PERIOD ONLY: [2010-01-01, 2020-01-01). No request ever asks for 2020+ data.

For each pair and window (full UTC day, London 08-16 local, New York 08-17 local):
  range  = median of daily high-low (mid prices), in pips
  spread = median close spread (ask - bid) of all M5 candles in valid windows, in pips
  score  = range / spread   (higher = more movement per pip of cost)

Integrity: bounded 14-day requests ending at 2019-12-31T23:59:59Z, duplicate and
out-of-order candles dropped and counted, per-pair completeness checks, and the
output directory must be on the Railway volume (--require-volume) before any request.
A ranking is only printed if every pair is COMPLETE; otherwise STATUS INCOMPLETE, exit 2.

Env: OANDA_TOKEN, OANDA_ENV ("practice" default, or "live"). Only GET /v3/instruments/*/candles.
"""
import argparse, csv, hashlib, json, math, os, sys, time, tempfile
import urllib.request, urllib.error, urllib.parse
from array import array
from datetime import datetime, timedelta, timezone, date
from functools import lru_cache
from statistics import median

UTC = timezone.utc
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD",
         "USD_CHF", "EUR_JPY", "GBP_JPY", "EUR_GBP", "AUD_JPY", "EUR_AUD"]
START = datetime(2010, 1, 1, tzinfo=UTC)
END = datetime(2020, 1, 1, tzinfo=UTC)          # exclusive
LAST_REQUEST_TO = END - timedelta(seconds=1)    # 2019-12-31T23:59:59Z
CHUNK_DAYS = 14                                  # max 4032 M5 candles < OANDA's 5000 cap
MAX_PER_REQUEST = 5000
FIRST_CANDLE_BY = datetime(2010, 1, 5, tzinfo=UTC)
LAST_CANDLE_FROM = datetime(2019, 12, 31, 12, tzinfo=UTC)
MAX_GAP = timedelta(hours=96)                    # longer than any weekend + holiday close
MIN_VALID_DAYS = 2000                            # per window; ~2,600 weekdays in the period
PRACTICE_HOST = "api-fxpractice.oanda.com"       # Practice only; there is no live option
# window: (zone, start minute local, end minute local, minimum distinct M5 candles for a valid day)
WINDOWS = {
    "day":     ("UTC", 0, 24 * 60, 240),
    "london":  ("EU", 8 * 60, 16 * 60, 80),
    "newyork": ("US", 8 * 60, 17 * 60, 90),
}


def log(msg):
    print(msg, flush=True)


def fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def pip_size(inst):
    return 0.01 if "JPY" in inst else 0.0001


def _last_sunday(y, m):
    d = (date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)) - timedelta(days=1)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d


def _nth_sunday(y, m, n):
    d = date(y, m, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)
    return d + timedelta(days=7 * (n - 1))


@lru_cache(maxsize=None)
def _dst_bounds(zone, y):
    if zone == "EU":   # last Sun Mar 01:00 UTC -> last Sun Oct 01:00 UTC
        s, e = _last_sunday(y, 3), _last_sunday(y, 10)
        return (datetime(s.year, s.month, s.day, 1, tzinfo=UTC),
                datetime(e.year, e.month, e.day, 1, tzinfo=UTC))
    if zone == "US":   # 2nd Sun Mar 07:00 UTC -> 1st Sun Nov 06:00 UTC
        s, e = _nth_sunday(y, 3, 2), _nth_sunday(y, 11, 1)
        return (datetime(s.year, s.month, s.day, 7, tzinfo=UTC),
                datetime(e.year, e.month, e.day, 6, tzinfo=UTC))
    return None


def utc_offset_hours(dt, zone):
    if zone == "UTC":
        return 0
    s, e = _dst_bounds(zone, dt.year)
    summer = s <= dt < e
    if zone == "EU":
        return 1 if summer else 0
    return -4 if summer else -5


def local_minutes(dt, zone):
    loc = dt + timedelta(hours=utc_offset_hours(dt, zone))
    return loc.hour * 60 + loc.minute


def make_fetch(host, token):
    def fetch(path, params):
        url = f"https://{host}{path}?{urllib.parse.urlencode(params)}"
        for attempt in range(1, 7):
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {token}", "Accept-Datetime-Format": "RFC3339"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (401, 429, 500, 502, 503, 504) and attempt < 6:
                    wait = min(60, 5 * 2 ** (attempt - 1))
                    log(f"RETRY {path} HTTP {e.code} attempt {attempt} of 6, waiting {wait}s")
                    time.sleep(wait)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 6:
                    log(f"RETRY {path} {e} attempt {attempt} of 6, waiting 10s")
                    time.sleep(10)
                    continue
                raise
    return fetch


def parse_time(t):
    return datetime.fromisoformat(t[:19]).replace(tzinfo=UTC)


def new_stats():
    return {"requests": 0, "candles": 0, "empty_chunks": [], "truncated_chunks": [],
            "duplicates": 0, "out_of_order": 0, "outside_chunk": 0, "incomplete_candles": 0,
            "first": None, "last": None, "max_gap_h": 0.0, "max_gap_at": None}


def oanda_candles(fetch, inst, stats):
    """Yield (dt, bid, ask) for complete M5 candles in [START, END), strictly increasing in time.
    Requests are bounded 14-day chunks; the last one ends at 2019-12-31T23:59:59Z."""
    cur, last_dt = START, None
    while cur < END:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), END)
        to = nxt - timedelta(seconds=1)
        data = fetch(f"/v3/instruments/{inst}/candles",
                     {"price": "BA", "granularity": "M5", "from": fmt(cur), "to": fmt(to)})
        stats["requests"] += 1
        cs = data.get("candles", [])
        if not cs:
            stats["empty_chunks"].append(fmt(cur))
        if len(cs) >= MAX_PER_REQUEST:
            stats["truncated_chunks"].append(fmt(cur))
        for c in cs:
            dt = parse_time(c["time"])
            if dt < cur or dt >= nxt:
                stats["outside_chunk"] += 1
                continue
            if last_dt is not None and dt <= last_dt:
                stats["duplicates" if dt == last_dt else "out_of_order"] += 1
                continue
            if not c.get("complete", True):
                stats["incomplete_candles"] += 1
                continue
            if last_dt is not None and dt - last_dt > timedelta(hours=stats["max_gap_h"]):
                stats["max_gap_h"] = (dt - last_dt).total_seconds() / 3600
                stats["max_gap_at"] = fmt(last_dt)
            if stats["first"] is None:
                stats["first"] = fmt(dt)
            stats["last"] = fmt(dt)
            last_dt = dt
            stats["candles"] += 1
            yield dt, c["bid"], c["ask"]
        cur = nxt


def completeness_problems(stats):
    p = []
    if stats["empty_chunks"]:
        p.append(f"{len(stats['empty_chunks'])} empty chunk(s) from {stats['empty_chunks'][0]}")
    if stats["truncated_chunks"]:
        p.append(f"{len(stats['truncated_chunks'])} truncated chunk(s)")
    if stats["out_of_order"]:
        p.append(f"{stats['out_of_order']} out-of-order candle(s)")
    if stats["outside_chunk"]:
        p.append(f"{stats['outside_chunk']} candle(s) outside requested range")
    if stats["first"] is None:
        p.append("no candles")
    else:
        if parse_time(stats["first"]) > FIRST_CANDLE_BY:
            p.append(f"history starts late ({stats['first']})")
        if parse_time(stats["last"]) < LAST_CANDLE_FROM:
            p.append(f"history ends early ({stats['last']})")
    if stats["max_gap_h"] > MAX_GAP.total_seconds() / 3600:
        p.append(f"gap of {stats['max_gap_h']:.1f}h after {stats['max_gap_at']}")
    return p


def data_problems(row):
    """Spec Rev 03 valid-data requirement: every window needs >= MIN_VALID_DAYS valid days
    and finite range, finite spread > 0, and finite score (math.isfinite rejects NaN and +/-inf)."""
    p = []
    for w in WINDOWS:
        d, r, sp, sc = row[f"{w}_days"], row[f"{w}_range_pips"], row[f"{w}_spread_pips"], row[f"{w}_score"]
        if d < MIN_VALID_DAYS:
            p.append(f"{w}: only {d} valid days (need {MIN_VALID_DAYS})")
        if not (math.isfinite(r) and math.isfinite(sp) and math.isfinite(sc)) or sp <= 0:
            p.append(f"{w}: unusable metrics (range {r}, spread {sp}, score {sc})")
    return p


def practice_host():
    """Returns (host, error). Refuses anything but Practice."""
    env = os.environ.get("OANDA_ENV", "practice").strip().lower()
    if env != "practice":
        return None, f"OANDA_ENV={env!r} but this study is Practice-only"
    return PRACTICE_HOST, None


def profile(inst, candles):
    pip = pip_size(inst)
    buckets = {}  # (date, window) -> [high, low, count, spreads]
    seen, dups = set(), 0
    for dt, b, a in candles:
        if dt in seen:             # second line of defence: one candle per timestamp
            dups += 1
            continue
        seen.add(dt)
        if dt.weekday() >= 5:      # skip Saturday/Sunday UTC partial sessions
            continue
        hi = (float(b["h"]) + float(a["h"])) / 2
        lo = (float(b["l"]) + float(a["l"])) / 2
        spr = (float(a["c"]) - float(b["c"])) / pip
        for w, (zone, s, e, _) in WINDOWS.items():
            if s <= local_minutes(dt, zone) < e:
                k = (dt.date(), w)
                bk = buckets.get(k)
                if bk is None:
                    buckets[k] = [hi, lo, 1, array("d", [spr])]
                else:
                    bk[0] = max(bk[0], hi); bk[1] = min(bk[1], lo); bk[2] += 1; bk[3].append(spr)
    row = {"pair": inst, "dup_dropped_in_profile": dups}
    for w, (_, _, _, minc) in WINDOWS.items():
        valid = [v for (d, ww), v in buckets.items() if ww == w and v[2] >= minc]
        ranges = [(v[0] - v[1]) / pip for v in valid]
        spreads = array("d")
        for v in valid:
            spreads.extend(v[3])
        r = median(ranges) if ranges else float("nan")
        s = median(spreads) if spreads else float("nan")
        row[f"{w}_days"] = len(valid)
        row[f"{w}_range_pips"] = round(r, 2)
        row[f"{w}_spread_pips"] = round(s, 3)
        row[f"{w}_score"] = round(r / s, 2) if s and s > 0 else float("nan")
    return row


DECISION_WINDOWS = ("london", "newyork")   # Spec Rev 02: user trades both sessions
TIE_BAND = 0.10                             # runner-up within 10% of the top -> both go forward


def decide(rows):
    """Decision score = mean of London and New York scores. Returns (ranked rows, selected pairs)."""
    for r in rows:
        vals = [r[f"{w}_score"] for w in DECISION_WINDOWS]
        r["decision_score"] = round(sum(vals) / len(vals), 2) if all(math.isfinite(v) for v in vals) else float("nan")
    ranked = sorted([r for r in rows if math.isfinite(r["decision_score"])],
                    key=lambda r: -r["decision_score"])
    if not ranked:
        return ranked, []
    top = ranked[0]["decision_score"]
    picked = [ranked[0]["pair"]]
    if len(ranked) > 1 and ranked[1]["decision_score"] >= top * (1 - TIE_BAND):
        picked.append(ranked[1]["pair"])
    return ranked, picked


def verify_volume(out):
    """Refuse unless `out` is inside RAILWAY_VOLUME_MOUNT_PATH, that path is a separate mount,
    and a marker file can be written and read back there. Returns an error string or None."""
    mnt = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    if not mnt:
        return "RAILWAY_VOLUME_MOUNT_PATH not set"
    m, o = os.path.realpath(mnt), os.path.realpath(out)
    if not (o == m or o.startswith(m.rstrip("/") + "/")):
        return f"output {o} is not inside volume {m}"
    if not os.path.isdir(m):
        return f"volume path {m} does not exist"
    if not (os.path.ismount(m) or os.stat(m).st_dev != os.stat("/").st_dev):
        return f"{m} is not a separate mount (not persistent storage)"
    try:
        os.makedirs(o, exist_ok=True)
        marker = os.path.join(o, ".volume_check")
        with open(marker, "w") as f:
            f.write("ok")
        with open(marker) as f:
            ok = f.read() == "ok"
        os.remove(marker)
        if not ok:
            return "marker read-back mismatch"
    except OSError as e:
        return f"cannot write to {o}: {e}"
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fake_fetch(step_hours=6, empty_from=None, dup_every=0, calls=None):
    """Fake OANDA: weekday candles every step_hours inside each requested range."""
    def fetch(path, p):
        frm, to = parse_time(p["from"]), parse_time(p["to"])
        if calls is not None:
            calls.append((frm, to))
        if empty_from and frm >= empty_from:
            return {"candles": []}
        out, t = [], frm.replace(minute=0, second=0)
        while t <= to:
            if t >= frm and t.weekday() < 5:
                c = {"time": t.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"), "complete": True,
                     "bid": {"h": "1.1", "l": "1.0", "c": "1.05"}, "ask": {"h": "1.1", "l": "1.0", "c": "1.05"}}
                out.append(c)
                if dup_every and len(out) % dup_every == 0:
                    out.append(dict(c))
            t += timedelta(hours=step_hours)
        return {"candles": out}
    return fetch


def run_tests():
    import unittest

    def bar(t0, i, hi=1.1020, lo=1.1010):
        b = {"h": f"{hi - 0.00005:.5f}", "l": f"{lo - 0.00005:.5f}", "c": "1.10150"}
        a = {"h": f"{hi + 0.00005:.5f}", "l": f"{lo + 0.00005:.5f}", "c": "1.10160"}
        return (t0 + timedelta(minutes=5 * i), b, a)

    class T(unittest.TestCase):
        def test_pip(self):
            self.assertEqual(pip_size("GBP_JPY"), 0.01)
            self.assertEqual(pip_size("EUR_USD"), 0.0001)

        def test_eu_dst(self):  # 2019: BST 31 Mar 01:00 UTC -> 27 Oct 01:00 UTC
            self.assertEqual(utc_offset_hours(datetime(2019, 3, 31, 0, 59, tzinfo=UTC), "EU"), 0)
            self.assertEqual(utc_offset_hours(datetime(2019, 3, 31, 1, 0, tzinfo=UTC), "EU"), 1)
            self.assertEqual(utc_offset_hours(datetime(2019, 10, 27, 1, 0, tzinfo=UTC), "EU"), 0)

        def test_us_dst(self):  # 2019: EDT 10 Mar 07:00 UTC -> 3 Nov 06:00 UTC
            self.assertEqual(utc_offset_hours(datetime(2019, 3, 10, 6, 59, tzinfo=UTC), "US"), -5)
            self.assertEqual(utc_offset_hours(datetime(2019, 3, 10, 7, 0, tzinfo=UTC), "US"), -4)
            self.assertEqual(utc_offset_hours(datetime(2019, 11, 3, 6, 0, tzinfo=UTC), "US"), -5)

        def test_sessions(self):  # London/NY 08:00 local in summer and winter
            self.assertEqual(local_minutes(datetime(2019, 7, 1, 7, 0, tzinfo=UTC), "EU"), 480)
            self.assertEqual(local_minutes(datetime(2019, 1, 7, 8, 0, tzinfo=UTC), "EU"), 480)
            self.assertEqual(local_minutes(datetime(2019, 7, 1, 12, 0, tzinfo=UTC), "US"), 480)
            self.assertEqual(local_minutes(datetime(2019, 1, 7, 13, 0, tzinfo=UTC), "US"), 480)

        def test_profile_synthetic(self):  # 50-pip range, 1-pip spread -> score 50
            t0 = datetime(2019, 1, 7, tzinfo=UTC)
            cs = [bar(t0, i, hi=1.1050 if i == 100 else 1.1020, lo=1.1000 if i == 200 else 1.1010)
                  for i in range(288)]
            row = profile("EUR_USD", cs)
            self.assertEqual(row["day_days"], 1)
            self.assertAlmostEqual(row["day_range_pips"], 50.0, places=1)
            self.assertAlmostEqual(row["day_spread_pips"], 1.0, places=2)
            self.assertAlmostEqual(row["day_score"], 50.0, places=0)

        def test_weekend_and_thin_days_excluded(self):
            sat, mon = datetime(2019, 1, 5, tzinfo=UTC), datetime(2019, 1, 7, tzinfo=UTC)
            self.assertEqual(profile("EUR_USD", [bar(sat, i) for i in range(288)])["day_days"], 0)
            self.assertEqual(profile("EUR_USD", [bar(mon, i) for i in range(50)])["day_days"], 0)

        def test_duplicate_candles_do_not_make_a_valid_day(self):  # reviewer's case
            mon = datetime(2019, 1, 7, tzinfo=UTC)
            row = profile("EUR_USD", [bar(mon, 0)] * 240)
            self.assertEqual(row["day_days"], 0)
            self.assertEqual(row["dup_dropped_in_profile"], 239)

        def test_downloader_drops_duplicates(self):
            st = new_stats()
            n = sum(1 for _ in oanda_candles(_fake_fetch(dup_every=3), "EUR_USD", st))
            self.assertGreater(st["duplicates"], 0)
            self.assertEqual(n, st["candles"])
            self.assertEqual(completeness_problems(st), [])

        def test_requests_never_reach_2020(self):
            calls, st = [], new_stats()
            for _ in oanda_candles(_fake_fetch(calls=calls), "EUR_USD", st):
                pass
            self.assertEqual(calls[0][0], START)
            self.assertEqual(max(to for _, to in calls), LAST_REQUEST_TO)
            self.assertTrue(all(frm < END and to < END for frm, to in calls))
            self.assertTrue(all(to - frm < timedelta(days=CHUNK_DAYS) for frm, to in calls))
            self.assertTrue(all(calls[i][1] < calls[i + 1][0] for i in range(len(calls) - 1)))

        def test_full_history_is_complete(self):
            st = new_stats()
            for _ in oanda_candles(_fake_fetch(), "EUR_USD", st):
                pass
            self.assertEqual(completeness_problems(st), [])

        def test_empty_chunk_marks_incomplete(self):
            st = new_stats()
            for _ in oanda_candles(_fake_fetch(empty_from=datetime(2015, 6, 1, tzinfo=UTC)), "EUR_USD", st):
                pass
            probs = completeness_problems(st)
            self.assertTrue(any("empty chunk" in p for p in probs))
            self.assertTrue(any("ends early" in p for p in probs))

        def test_decision_rule(self):
            def r(p, lo, ny):
                return {"pair": p, "london_score": lo, "newyork_score": ny}
            ranked, picked = decide([r("A", 100, 60), r("B", 70, 80), r("C", 50, 50)])
            self.assertEqual([x["pair"] for x in ranked], ["A", "B", "C"])  # 80, 75, 50
            self.assertEqual(picked, ["A", "B"])                            # 75 >= 72 (within 10%)
            _, picked = decide([r("A", 100, 100), r("B", 80, 80)])
            self.assertEqual(picked, ["A"])                                 # 80 < 90

        def test_zero_valid_days_is_not_usable(self):  # reviewer's case
            sat = datetime(2019, 1, 5, tzinfo=UTC)
            row = profile("EUR_USD", [bar(sat, i) for i in range(288)])
            probs = data_problems(row)
            self.assertEqual(len([x for x in probs if "valid days" in x]), len(WINDOWS))
            self.assertTrue(any("unusable" in x for x in probs))

        def test_enough_valid_days_passes(self):
            row = {}
            for w in WINDOWS:
                row.update({f"{w}_days": MIN_VALID_DAYS, f"{w}_range_pips": 80.0,
                            f"{w}_spread_pips": 1.2, f"{w}_score": 66.7})
            self.assertEqual(data_problems(row), [])

        def test_non_finite_metrics_rejected(self):  # reviewer's case: inf passed the old x == x check
            def good():
                row = {}
                for w in WINDOWS:
                    row.update({f"{w}_days": MIN_VALID_DAYS, f"{w}_range_pips": 80.0,
                                f"{w}_spread_pips": 1.0, f"{w}_score": 80.0})
                return row
            bad_values = [float("inf"), float("-inf"), float("nan")]
            n = 0
            for w in WINDOWS:
                for metric in ("range_pips", "spread_pips", "score"):
                    for v in bad_values:
                        row = good(); row[f"{w}_{metric}"] = v
                        probs = data_problems(row)
                        self.assertTrue(any(x.startswith(f"{w}: unusable") for x in probs), (w, metric, v))
                        n += 1
                for sp in (0.0, -1.0):
                    row = good(); row[f"{w}_spread_pips"] = sp
                    self.assertTrue(any(x.startswith(f"{w}: unusable") for x in data_problems(row)), (w, sp))
                    n += 1
            self.assertEqual(n, len(WINDOWS) * 11)
            row = good(); row["day_range_pips"] = float("inf"); row["day_score"] = float("inf")
            self.assertNotEqual(data_problems(row), [])  # exact row ChatGPT reproduced
            self.assertEqual(data_problems(good()), [])   # valid finite metrics still pass

        def test_decision_ignores_non_finite(self):
            rows = [{"pair": "A", "london_score": float("inf"), "newyork_score": 50.0},
                    {"pair": "B", "london_score": 60.0, "newyork_score": 60.0}]
            ranked, picked = decide(rows)
            self.assertEqual([r["pair"] for r in ranked], ["B"])
            self.assertEqual(picked, ["B"])

        def test_practice_only(self):
            old = os.environ.get("OANDA_ENV")
            try:
                os.environ.pop("OANDA_ENV", None)
                self.assertEqual(practice_host(), (PRACTICE_HOST, None))
                os.environ["OANDA_ENV"] = "Practice"
                self.assertEqual(practice_host()[0], PRACTICE_HOST)
                for bad in ("live", "trade", ""):
                    os.environ["OANDA_ENV"] = bad
                    host, err = practice_host()
                    self.assertIsNone(host)
                    self.assertIn("Practice-only", err)
            finally:
                os.environ.pop("OANDA_ENV", None)
                if old is not None:
                    os.environ["OANDA_ENV"] = old

        def test_volume_check_refuses(self):
            old = os.environ.pop("RAILWAY_VOLUME_MOUNT_PATH", None)
            try:
                self.assertIn("not set", verify_volume("/tmp/x"))
                with tempfile.TemporaryDirectory() as d:
                    os.environ["RAILWAY_VOLUME_MOUNT_PATH"] = os.path.join(d, "vol")
                    os.makedirs(os.environ["RAILWAY_VOLUME_MOUNT_PATH"])
                    self.assertIn("not inside", verify_volume(os.path.join(d, "elsewhere")))
            finally:
                os.environ.pop("RAILWAY_VOLUME_MOUNT_PATH", None)
                if old is not None:
                    os.environ["RAILWAY_VOLUME_MOUNT_PATH"] = old

    res = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(T))
    log(f"OFFLINE TESTS: ran {res.testsRun}, failures {len(res.failures)}, errors {len(res.errors)}")
    return res.wasSuccessful()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tests", action="store_true", help="run offline tests first; abort if any fail")
    ap.add_argument("--tests-only", action="store_true")
    ap.add_argument("--require-volume", action="store_true", help="refuse unless --out is on the Railway volume")
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--out", default=os.path.join(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "."),
                                                  "pair_profile", os.environ.get("RAILWAY_DEPLOYMENT_ID", "local")))
    args = ap.parse_args()

    if args.run_tests or args.tests_only:
        if not run_tests():
            log("ABORT: offline tests failed. No OANDA request was made.")
            sys.exit(1)
        if args.tests_only:
            return

    if args.require_volume:
        err = verify_volume(args.out)
        if err:
            log(f"REFUSED: {err}. No OANDA request was made.")
            sys.exit(1)
    host, err = practice_host()
    if err:
        log(f"REFUSED: {err}. No OANDA request was made.")
        sys.exit(1)
    token = os.environ.get("OANDA_TOKEN")
    if not token:
        log("ABORT: OANDA_TOKEN not set. No OANDA request was made."); sys.exit(1)
    code_sha = sha256_file(os.path.abspath(__file__))
    log("PAIR PROFILE | Spec Revision 03 | read-only | " + json.dumps({
        "code_sha256": code_sha, "period": [fmt(START), fmt(END) + " (exclusive)"],
        "last_request_to": fmt(LAST_REQUEST_TO), "out": args.out,
        "deployment": os.environ.get("RAILWAY_DEPLOYMENT_ID"), "commit": os.environ.get("RAILWAY_GIT_COMMIT_SHA"),
        "start_utc": fmt(datetime.now(UTC))}))

    os.makedirs(args.out, exist_ok=True)
    fetch = make_fetch(host, token)
    rows, incomplete = [], []
    for inst in [p.strip() for p in args.pairs.split(",") if p.strip()]:
        st = new_stats()
        row = profile(inst, oanda_candles(fetch, inst, st))
        probs = completeness_problems(st) + data_problems(row)
        row.update({"status": "INCOMPLETE" if probs else "COMPLETE", "problems": "; ".join(probs),
                    "candles": st["candles"], "requests": st["requests"], "first": st["first"], "last": st["last"],
                    "duplicates_dropped": st["duplicates"], "max_gap_h": round(st["max_gap_h"], 1)})
        rows.append(row)
        if probs:
            incomplete.append(inst)
        log(f"{inst}: {row['status']} | {st['candles']} candles in {st['requests']} requests "
            f"({st['first']} to {st['last']}) | dups dropped {st['duplicates']} | max gap {row['max_gap_h']}h"
            + (f" | PROBLEMS: {row['problems']}" if probs else ""))
        log(f"{inst}: " + " | ".join(f"{w} range {row[w+'_range_pips']} spread {row[w+'_spread_pips']} "
                                     f"score {row[w+'_score']} ({row[w+'_days']} days)" for w in WINDOWS))

    ranked_decision, picked = decide(rows)
    if not incomplete and not picked:
        incomplete.append("NO_SELECTION")
    csv_path = os.path.join(args.out, "pair_profile.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(args.out, "MANIFEST.sha256"), "w") as f:
        f.write(f"{code_sha}  pair_profile.py\n{sha256_file(csv_path)}  pair_profile.csv\n")

    if incomplete:
        log(f"STATUS: INCOMPLETE | {incomplete} | NO RANKING, NO DECISION (spec: requires every pair COMPLETE with valid data)")
    else:
        log("STATUS: COMPLETE")
        for w in WINDOWS:
            ranked = sorted(rows, key=lambda r: -(r[f"{w}_score"] if math.isfinite(r[f"{w}_score"]) else -1))
            log(f"RANKING {w} (range pips / spread pips = score):")
            for i, r in enumerate(ranked, 1):
                log(f"  {i:2d}. {r['pair']:8s} {r[f'{w}_range_pips']:8.1f} / {r[f'{w}_spread_pips']:6.2f} = "
                    f"{r[f'{w}_score']:7.1f}  ({r[f'{w}_days']} days)")
        log("RANKING decision (mean of london and newyork scores):")
        for i, r in enumerate(ranked_decision, 1):
            log(f"  {i:2d}. {r['pair']:8s} {r['decision_score']:7.1f}")
        log(f"DECISION: {' + '.join(picked)}" + (" (runner-up within 10%: both go forward)" if len(picked) > 1 else ""))
    log(f"FILES: written to {args.out} with MANIFEST.sha256")
    log("end " + fmt(datetime.now(UTC)))
    sys.exit(2 if incomplete else 0)


if __name__ == "__main__":
    main()
