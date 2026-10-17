"""Offline tests for the h4_live.py execution fixes. No network: the HTTP session is mocked and
real sockets are blocked. Run only this file:  python -m unittest -v test_h4_live.py"""
import ast
import hashlib
import os
import socket
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import pandas as pd
import requests

os.environ.setdefault("OANDA_TOKEN", "test-token")
os.environ.setdefault("OANDA_ACCOUNT_ID", "test-account")
os.environ["OANDA_ENV"] = "practice"

import h4_live as H

_NET = mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network access in tests"))


def setUpModule():
    _NET.start()


def tearDownModule():
    _NET.stop()


class Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}
        self.text = str(self._body)

    def json(self):
        return self._body


class FakeSession:
    """script: list of Resp or Exception, consumed in order by get(); post() records calls."""

    def __init__(self, script=None, post_resp=None):
        self.script = list(script or [])
        self.gets, self.posts = [], []
        self.post_resp = post_resp or Resp(201, {"orderFillTransaction": {"price": "1.10000"}})
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.gets.append((url, params))
        act = self.script.pop(0)
        if isinstance(act, Exception):
            raise act
        return act

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        return self.post_resp


def make_api(session):
    api = H.Oanda()
    api.s = session
    return api


class TestSignalCodeUnchanged(unittest.TestCase):
    def test_signal_functions_fingerprint(self):
        # SHA256 of the AST of calc_atr, echo_direction, _exit_bar, scan_last_bar in the live
        # version 5a5fffb7... The execution fixes must not change them.
        tree = ast.parse(open(H.__file__).read())
        names = ["calc_atr", "echo_direction", "_exit_bar", "scan_last_bar"]
        src = "".join(ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
        self.assertEqual(hashlib.sha256(src.encode()).hexdigest(),
                         "9f774d2420bf589ea2c74e0314de11cac01ce3357df99034721d0290f27e81c9")

    def test_strategy_settings_unchanged(self):
        self.assertEqual((H.ACC_LEN, H.ACC_MULT, H.MANIP_MULT, H.SL_ATR, H.RR, H.ECHO_LEN, H.ECHO_FCAST,
                          H.GRAN, H.HISTORY, H.TAG), (40, 5.0, 0.6, 5.0, 0.86, 50, 50, "H4", 1000, "h4po3e"))
        self.assertEqual(len(H.PAIRS), 12)


class TestRetries(unittest.TestCase):
    def setUp(self):
        self.sleeps = []
        self.p = mock.patch.object(H.time, "sleep", side_effect=lambda s: self.sleeps.append(s))
        self.p.start()
        H.Oanda.retries = 0

    def tearDown(self):
        self.p.stop()

    def test_401_then_success_retries_after_one_minute(self):
        fs = FakeSession([Resp(401), Resp(200, {"trades": []})])
        self.assertEqual(make_api(fs).get("/v3/x"), {"trades": []})
        self.assertEqual(self.sleeps, [60])
        self.assertEqual(H.Oanda.retries, 1)

    def test_four_attempts_then_error(self):
        fs = FakeSession([Resp(401), Resp(503), requests.ConnectionError("x"), Resp(401)])
        with self.assertRaises(RuntimeError) as cm:
            make_api(fs).get("/v3/x")
        self.assertIn("after 4 attempts", str(cm.exception))
        self.assertEqual(self.sleeps, [60, 120, 240])
        self.assertEqual(len(fs.gets), 4)

    def test_other_client_errors_not_retried(self):
        fs = FakeSession([Resp(404)])
        with self.assertRaises(RuntimeError):
            make_api(fs).get("/v3/x")
        self.assertEqual((len(fs.gets), self.sleeps), (1, []))

    def test_orders_never_retried(self):
        fs = FakeSession(post_resp=Resp(503, {"errorMessage": "busy"}))
        status, _ = make_api(fs).post("/v3/accounts/a/orders", {"order": {}})
        self.assertEqual((status, len(fs.posts), self.sleeps), (503, 1, []))


def candles_ending(close_time):
    """300 flat H4 candles; the last one closes at close_time."""
    last_open = close_time - timedelta(hours=4)
    times = [last_open - timedelta(hours=4 * k) for k in range(299, -1, -1)]
    return pd.DataFrame({"time": pd.to_datetime(times, utc=True)})


class TestStaleGuardAndDuplicates(unittest.TestCase):
    NOW = datetime(2026, 10, 15, 13, 0, 20, tzinfo=timezone.utc)

    def run_check(self, close_time, open_pairs=()):
        placed = []
        with mock.patch.object(H, "utcnow", return_value=self.NOW), \
             mock.patch.object(H, "PAIRS", ["EUR_USD"]), \
             mock.patch.object(H, "open_tagged_trades", return_value=[{"instrument": p} for p in open_pairs]), \
             mock.patch.object(H, "fetch_recent", return_value=candles_ending(close_time)), \
             mock.patch.object(H, "scan_last_bar", return_value=(1, 0.002)), \
             mock.patch.object(H, "place", side_effect=lambda *a: placed.append(a)), \
             self.assertLogs(H.log, level="INFO") as logs:
            H.log.info("test marker")             # assertLogs needs at least one record
            H.check(object(), {})
        return placed, "\n".join(logs.output)

    def test_fresh_signal_is_placed(self):
        placed, _ = self.run_check(self.NOW - timedelta(minutes=1))
        self.assertEqual(len(placed), 1)

    def test_signal_exactly_at_limit_is_placed(self):
        placed, _ = self.run_check(self.NOW - timedelta(minutes=15))
        self.assertEqual(len(placed), 1)

    def test_stale_signal_is_skipped_and_logged(self):
        placed, out = self.run_check(self.NOW - timedelta(minutes=16))
        self.assertEqual(placed, [])
        self.assertIn("SKIPPED: stale", out)

    def test_signal_three_hours_late_is_skipped(self):
        placed, out = self.run_check(self.NOW - timedelta(hours=3))
        self.assertEqual(placed, [])
        self.assertIn("SKIPPED: stale", out)

    def test_open_position_still_skips_first(self):
        placed, out = self.run_check(self.NOW - timedelta(minutes=1), open_pairs=["EUR_USD"])
        self.assertEqual(placed, [])
        self.assertIn("position already open", out)

    def test_duplicate_signal_not_ordered(self):
        bar = pd.Timestamp("2026-10-15 09:00", tz="UTC")
        cid = f"h4po3e-EUR_USD-{bar:%Y%m%d%H}"
        fs = FakeSession([Resp(200, {"trades": [{"clientExtensions": {"id": cid}}]})])
        api = make_api(fs)
        with mock.patch.object(H, "quote_and_conv", return_value=(1.1, 1.1001, 1.0)), \
             mock.patch.object(H, "account_nav", return_value=10000.0):
            H.place(api, "EUR_USD", 1, 0.002, bar)
        self.assertEqual(fs.posts, [])
        self.assertIn("/trades", fs.gets[0][0])
        self.assertEqual(fs.gets[0][1], {"instrument": "EUR_USD", "state": "ALL", "count": 50})

    def test_new_signal_is_ordered_once_with_stop_and_target(self):
        bar = pd.Timestamp("2026-10-15 09:00", tz="UTC")
        fs = FakeSession([Resp(200, {"trades": [{"clientExtensions": {"id": "h4po3e-EUR_USD-2026101505"}}]})])
        api = make_api(fs)
        with mock.patch.object(H, "quote_and_conv", return_value=(1.1, 1.1001, 1.0)), \
             mock.patch.object(H, "account_nav", return_value=10000.0), \
             mock.patch.object(H, "utcnow", return_value=datetime(2026, 10, 15, 13, 1, tzinfo=timezone.utc)):
            H.place(api, "EUR_USD", 1, 0.002, bar)
        self.assertEqual(len(fs.posts), 1)
        order = fs.posts[0][1]["order"]
        self.assertIn("stopLossOnFill", order)
        self.assertIn("takeProfitOnFill", order)
        self.assertEqual(order["clientExtensions"]["id"], "h4po3e-EUR_USD-2026101509")


class TestSubmissionTimeGuard(unittest.TestCase):
    """Regression for the review finding: the guard passed at 14 minutes, a slow history lookup
    pushed submission to 16+ minutes, and an order was still sent."""
    BAR = pd.Timestamp("2026-10-15 09:00", tz="UTC")          # closes 13:00 UTC
    CLOSE = datetime(2026, 10, 15, 13, 0, tzinfo=timezone.utc)

    def setUp(self):
        self.clock = [self.CLOSE + timedelta(minutes=14)]
        self.p1 = mock.patch.object(H, "utcnow", side_effect=lambda: self.clock[0])
        self.p2 = mock.patch.object(H.time, "sleep", side_effect=lambda s: self.clock.__setitem__(0, self.clock[0] + timedelta(seconds=s)))
        self.p1.start(); self.p2.start()

    def tearDown(self):
        self.p1.stop(); self.p2.stop()

    def responses(self, history_failures=0):
        pricing = Resp(200, {"prices": [{"tradeable": True, "bids": [{"price": "1.10000"}], "asks": [{"price": "1.10010"}]}],
                             "homeConversions": [{"currency": "USD", "accountLoss": "1.0"}]})
        return ([Resp(401)] * history_failures + [Resp(200, {"trades": []}),        # history (duplicate guard)
                 Resp(200, {"account": {"NAV": "10000"}}),                           # account value
                 pricing])                                                           # fresh quote, last

    def test_slow_history_lookup_blocks_late_order(self):
        fs = FakeSession(self.responses(history_failures=2))            # 60 s + 120 s of retries
        with self.assertLogs(H.log, level="INFO") as logs:
            H.place(make_api(fs), "EUR_USD", 1, 0.002, self.BAR)
        self.assertEqual(fs.posts, [])                                  # zero orders at 17 minutes
        self.assertIn("SKIPPED at submission", "\n".join(logs.output))

    def test_fast_path_orders_once(self):
        fs = FakeSession(self.responses())
        with self.assertLogs(H.log, level="INFO"):
            H.place(make_api(fs), "EUR_USD", 1, 0.002, self.BAR)
        self.assertEqual(len(fs.posts), 1)

    def test_request_order_history_nav_then_fresh_quote(self):
        fs = FakeSession(self.responses())
        with self.assertLogs(H.log, level="INFO"):
            H.place(make_api(fs), "EUR_USD", 1, 0.002, self.BAR)
        paths = [u.split("/v3/accounts/test-account")[1] for u, _ in fs.gets]
        self.assertEqual(paths, ["/trades", "/summary", "/pricing"])

    def test_stop_and_target_come_from_the_fresh_quote(self):
        fs = FakeSession(self.responses())
        with self.assertLogs(H.log, level="INFO"):
            H.place(make_api(fs), "EUR_USD", 1, 0.002, self.BAR)
        order = fs.posts[0][1]["order"]
        risk = H.SL_ATR * 0.002
        self.assertEqual(order["stopLossOnFill"]["price"], H.px_round("EUR_USD", 1.10010 - risk))
        self.assertEqual(order["takeProfitOnFill"]["price"], H.px_round("EUR_USD", 1.10010 + H.RR * risk))

    def test_retry_policy_covers_all_5xx_but_not_other_4xx(self):
        self.assertTrue(all(H.retryable(c) for c in (401, 429, 500, 507, 599)))
        self.assertFalse(any(H.retryable(c) for c in (400, 403, 404, 600)))


class TestPracticeOnly(unittest.TestCase):
    def test_refuses_non_practice(self):
        with mock.patch.dict(os.environ, {"OANDA_ENV": "live"}):
            with self.assertRaises(RuntimeError):
                H.Oanda()

    def test_practice_host_hard_coded(self):
        self.assertEqual(H.Oanda().base, "https://api-fxpractice.oanda.com")


if __name__ == "__main__":
    unittest.main()
