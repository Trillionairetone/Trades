"""Unit tests for orb_probe.py. All network access is mocked; nothing here contacts OANDA.

Run only this file:  python -m unittest -v test_orb_probe.py
"""
import ast
import io
import json
import os
import socket
import unittest
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from unittest import mock

import orb_probe as P

TOKEN = "secret-token-DO-NOT-PRINT-123"


def candle(t, mid=1.1000, half_spread=0.00005, complete=True):
    def q(off):
        return {"o": f"{mid + off:.5f}", "h": f"{mid + 0.0005 + off:.5f}",
                "l": f"{mid - 0.0005 + off:.5f}", "c": f"{mid + 0.0002 + off:.5f}"}
    return {"time": t.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"), "complete": complete,
            "volume": 10, "mid": q(0.0), "bid": q(-half_spread), "ask": q(half_spread)}


def series(start, n):
    return [candle(start + timedelta(minutes=5 * k)) for k in range(n)]


def default_candles(frm, count):
    start = datetime.strptime(frm, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return series(start, count)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FakeOanda:
    """Serves synthetic candles. overrides[(pair, from)] = list of actions, used in order:
    a list of candles, an Exception instance, or an int HTTP status."""

    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.calls = []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.calls.append((url, timeout, dict(req.header_items())))
        parts = urllib.parse.urlparse(url)
        assert parts.scheme == "https" and parts.netloc == "api-fxpractice.oanda.com", url
        segs = parts.path.split("/")
        assert segs[1:3] == ["v3", "instruments"] and segs[4] == "candles" and len(segs) == 5, url
        pair = segs[3]
        q = dict(urllib.parse.parse_qsl(parts.query))
        key = (pair, q["from"])
        actions = self.overrides.get(key)
        action = actions.pop(0) if actions else None
        if isinstance(action, Exception):
            raise action
        if isinstance(action, int):
            raise urllib.error.HTTPError(url, action, "err", {}, io.BytesIO(b'{"errorMessage":"x"}'))
        candles = action if isinstance(action, list) else default_candles(q["from"], int(q["count"]))
        return FakeResponse(json.dumps({"instrument": pair, "granularity": "M5", "candles": candles}).encode())


def http_error(code):
    return code


class ProbeTestBase(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"OANDA_TOKEN": TOKEN}, clear=False)
        self.env.start()
        for name in ("OANDA_API_KEY", "OANDA_API_TOKEN", "OANDA_ACCESS_TOKEN"):
            os.environ.pop(name, None)
        self.sleeps = []
        self.p_sleep = mock.patch.object(P, "_sleep", side_effect=lambda s: self.sleeps.append(s))
        self.p_sleep.start()
        # Any accidental real network call fails loudly.
        self.p_real = mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("real network call"))
        self.p_real.start()

    def tearDown(self):
        mock.patch.stopall()
        self.env.stop()

    def run_probe(self, fake):
        with mock.patch.object(P, "_urlopen", fake):
            lines = []
            code = P.run(out=lines.append)
        text = "\n".join(lines)
        self.assertNotIn(TOKEN, text)
        return code, text, lines


class TestPlanAndRequests(ProbeTestBase):
    def test_full_run_go_with_48_requests(self):
        fake = FakeOanda()
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 0)
        self.assertEqual(len(fake.calls), 48)
        self.assertIn("TOTALS planned 48 | attempted 48 | completed 48 | not run 0 | HTTP attempts 48", text)
        self.assertIn("RESULT: PROBE GO", text)
        self.assertIn("Full historical coverage remains UNVERIFIED", text)

    def test_exact_request_parameters_and_timeout(self):
        fake = FakeOanda()
        self.run_probe(fake)
        url, timeout, headers = fake.calls[0]
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        self.assertEqual(q, {"granularity": "M5", "price": "BAM", "smooth": "false", "includeFirst": "true",
                             "from": "2009-12-28T00:00:00Z", "count": "5"})
        self.assertEqual(timeout, P.TIMEOUT_S)
        froms = sorted({dict(urllib.parse.parse_qsl(urllib.parse.urlparse(u).query))["from"] for u, _, _ in fake.calls})
        self.assertEqual(froms, ["2009-12-28T00:00:00Z", "2010-06-15T06:00:00Z",
                                 "2015-06-16T06:00:00Z", "2020-06-16T06:00:00Z"])
        counts = {dict(urllib.parse.parse_qsl(urllib.parse.urlparse(u).query))["count"] for u, _, _ in fake.calls}
        self.assertEqual(counts, {"5", "180"})

    def test_only_stdlib_imports_no_repo_modules(self):
        with open(P.__file__) as fh:
            tree = ast.parse(fh.read())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods.add(n.module.split(".")[0])
        self.assertTrue(mods <= {"hashlib", "json", "math", "os", "socket", "statistics", "sys",
                                 "time", "urllib", "datetime"}, mods)

    def test_no_account_order_or_position_paths_in_source(self):
        with open(P.__file__) as fh:
            src = fh.read()
        for bad in ("/accounts", "/orders", "/trades", "/positions", "/pricing", "api-fxtrade"):
            self.assertNotIn(bad, src)


class TestAttemptsS1(ProbeTestBase):
    def test_retry_then_success_counts_attempts(self):
        fake = FakeOanda({("EUR_USD", "2009-12-28T00:00:00Z"): [503]})
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 0)
        self.assertIn("HTTP attempts 49", text)
        self.assertEqual(self.sleeps, [P.RETRY_DELAYS[0]])

    def test_three_attempt_limit_then_stop_and_not_run(self):
        fake = FakeOanda({("EUR_USD", "2010-06-15T06:00:00Z"): [503, socket.timeout("t"), 502]})
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.calls), 1 + 3)          # R1 once, R2 three times, nothing after
        self.assertIn("S1: request failed after 3 attempts", text)
        self.assertIn("TOTALS planned 48 | attempted 2 | completed 1 | not run 46 | HTTP attempts 4", text)
        self.assertIn("RESULT: PROBE STOP", text)
        self.assertTrue(all(s <= max(P.RETRY_DELAYS) for s in self.sleeps))
        self.assertEqual(text.count(" NOT RUN "), 46)

    def test_auth_failure_stops_immediately(self):
        fake = FakeOanda({("EUR_USD", "2009-12-28T00:00:00Z"): [401]})
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("authentication failure (HTTP 401)", text)
        self.assertIn("HTTP attempts 1", text)
        self.assertEqual(self.sleeps, [])

    def test_non_retryable_client_error_stops(self):
        fake = FakeOanda({("EUR_USD", "2009-12-28T00:00:00Z"): [400]})
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 1)
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("non-retryable HTTP 400", text)

    def test_missing_token_stops_without_requests(self):
        os.environ.pop("OANDA_TOKEN")
        fake = FakeOanda()
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 1)
        self.assertEqual(fake.calls, [])
        self.assertIn("no OANDA token", text)
        self.assertIn("not run 48", text)


class TestS2History(ProbeTestBase):
    def test_no_pre_2010_candle_stops(self):
        late = series(datetime(2010, 1, 4, 0, 0, tzinfo=timezone.utc), 5)
        fake = FakeOanda({("GBP_USD", "2009-12-28T00:00:00Z"): [late]})
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 1)
        self.assertIn("S2: GBP_USD R1: no candle before 2010-01-01", text)

    def test_empty_r1_stops(self):
        fake = FakeOanda({("EUR_USD", "2009-12-28T00:00:00Z"): [[]]})
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 1)
        self.assertIn("S2:", text)


class TestS3Validation(ProbeTestBase):
    def bad(self, mutate, expect):
        cs = default_candles("2015-06-16T06:00:00Z", 180)
        mutate(cs)
        fake = FakeOanda({("USD_JPY", "2015-06-16T06:00:00Z"): [cs]})
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 1, text)
        self.assertIn("S3: USD_JPY R3", text)
        self.assertIn(expect, text)

    def test_bid_above_ask(self):
        def m(cs):
            cs[10]["bid"]["c"] = "1.20000"
            cs[10]["bid"]["h"] = "1.20000"
        self.bad(m, "bid above ask")

    def test_missing_field(self):
        self.bad(lambda cs: cs[3].pop("ask"), "missing field 'ask'")

    def test_nan_price(self):
        def m(cs):
            cs[5]["mid"]["o"] = "NaN"
        self.bad(m, "not finite")

    def test_misaligned_time(self):
        def m(cs):
            cs[7]["time"] = "2015-06-16T06:36:00.000000000Z"
        self.bad(m, "not aligned")

    def test_invalid_ohlc(self):
        def m(cs):
            cs[8]["mid"]["l"] = "1.20000"
        self.bad(m, "invalid mid OHLC")

    def test_incomplete_candle(self):
        def m(cs):
            cs[9]["complete"] = False
        self.bad(m, "incomplete")

    def test_duplicate_time(self):
        def m(cs):
            cs[12]["time"] = cs[11]["time"]
        self.bad(m, "duplicate or out of order")


class TestWindowsS4S5(ProbeTestBase):
    def test_window_boundaries(self):
        day = datetime(2020, 6, 16, tzinfo=timezone.utc)
        times = [day.replace(hour=5, minute=55), day.replace(hour=6), day.replace(hour=20, minute=55),
                 day.replace(hour=21)]
        rows = P.validate([candle(t) for t in times])
        rep = P.sample_report("EUR_USD", day, rows)
        self.assertEqual(rep["in_window"], 2)                       # 06:00 and 20:55 only
        self.assertEqual(rep["outside"], ["2020-06-16T05:55", "2020-06-16T21:00"])
        self.assertEqual(rep["liquidation_candles"], ["20:55"])

    def test_session_and_liquidation_times_match_notes(self):
        self.assertEqual(P.SESSIONS_UTC, {"London 08:00": (7, 0), "NY forex 08:00": (12, 0),
                                          "US stocks 09:30": (13, 30)})
        self.assertEqual(P.LIQ_WINDOW_UTC, ((20, 45), (21, 0)))
        self.assertEqual(P.WINDOW, ((6, 0), (21, 0)))

    def test_spread_in_pips(self):
        day = datetime(2020, 6, 16, tzinfo=timezone.utc)
        rows = P.validate([candle(day.replace(hour=7), mid=150.0, half_spread=0.005)])
        rep = P.sample_report("USD_JPY", day, rows)
        self.assertAlmostEqual(rep["spread_median"], 1.0)

    def drop(self, frm, hm_list):
        cs = default_candles(frm, 180)
        drop = {f"{frm[:10]}T{h:02d}:{m:02d}" for h, m in hm_list}
        return [c for c in cs if c["time"][:16] not in drop]

    def test_s4_one_deficient_day_continues(self):
        fake = FakeOanda({("AUD_USD", "2010-06-15T06:00:00Z"): [self.drop("2010-06-15T06:00:00Z", [(7, 10)])]})
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 0, text)
        self.assertIn("AUD_USD 2010-06-15: in-window 179/180", text)
        self.assertIn("London 08:00: 07:10", text)

    def test_s4_two_deficient_days_stop(self):
        fake = FakeOanda({
            ("AUD_USD", "2010-06-15T06:00:00Z"): [self.drop("2010-06-15T06:00:00Z", [(7, 10)])],
            ("AUD_USD", "2015-06-16T06:00:00Z"): [self.drop("2015-06-16T06:00:00Z", [(13, 30)])],
        })
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 1)
        self.assertIn("S4: AUD_USD: range candles missing on 2 of 3 sample days", text)

    def test_s5_no_liquidation_candle_on_one_date_stops(self):
        frm = "2020-06-16T06:00:00Z"
        fake = FakeOanda({("EUR_GBP", frm): [self.drop(frm, [(20, 45), (20, 50), (20, 55)])]})
        code, text, _ = self.run_probe(fake)
        self.assertEqual(code, 1)
        self.assertIn("S5: EUR_GBP 2020-06-16: no candle opens in [20:45, 21:00) UTC", text)
        self.assertIn("not a strategy FAIL", text)


class TestCredentialSafety(ProbeTestBase):
    def test_token_never_printed_on_errors(self):
        err = urllib.error.URLError(f"connection refused while sending Bearer {TOKEN}")
        fake = FakeOanda({("EUR_USD", "2009-12-28T00:00:00Z"): [err, err, err]})
        code, text, _ = self.run_probe(fake)    # run_probe asserts the token is absent
        self.assertEqual(code, 1)

    def test_authorization_header_sent_but_not_logged(self):
        fake = FakeOanda()
        _, text, _ = self.run_probe(fake)
        hdrs = {k.lower(): v for k, v in fake.calls[0][2].items()}
        self.assertEqual(hdrs["authorization"], f"Bearer {TOKEN}")
        self.assertNotIn("Bearer", text)

    def test_redact(self):
        self.assertEqual(P.redact(f"x {TOKEN} y", TOKEN), "x [REDACTED] y")


if __name__ == "__main__":
    unittest.main()
