"""Run with: python -m unittest -v test_orb.py   (synthetic candles, no network, no account)."""
import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

import orb_test as O

P = 0.0001       # EUR_USD pip
SP = 0.0001      # default 1-pip spread


def frame(start_utc, mids, highs=None, lows=None, opens=None, spread=SP):
    n = len(mids)
    t = pd.date_range(start_utc, periods=n, freq="5min", tz="UTC")
    c = np.array(mids, float)
    o = np.array(opens if opens is not None else np.r_[c[0], c[:-1]], float)
    h = np.array(highs if highs is not None else np.maximum(o, c), float)
    lo = np.array(lows if lows is not None else np.minimum(o, c), float)
    d = pd.DataFrame({"time": t, "o": o, "h": h, "l": lo, "c": c, "volume": 100.0})
    sp = np.broadcast_to(np.asarray(spread, float), (n,))
    for f in "ohlc":
        d["b" + f] = d[f] - sp / 2
        d["a" + f] = d[f] + sp / 2
    return d


def quotes(d):
    return {c: d[c].to_numpy() for c in O.QUOTES}


def day_frame(path, start="2026-01-05 13:00", spread=SP, range_hi=1.1010, range_lo=1.1000):
    """NY fx 08:00 session on Mon 2026-01-05 (13:00 UTC in winter): 3 range candles, then path."""
    mids = [1.1005, 1.1010, 1.1000] + list(path)
    highs = [1.1006, range_hi, 1.1005] + [m + 0.0001 for m in path]
    lows = [1.1004, 1.1006, range_lo] + [m - 0.0001 for m in path]
    return frame(start, mids, highs=highs, lows=lows, spread=spread)


DAY = date(2026, 1, 5)
END = pd.Timestamp("2026-01-07", tz="UTC")


def chunk(d, end=END, dates=(DAY,)):
    cov = {s[0]: [0, 0] for s in O.SESSIONS}
    return O.simulate_chunk(d, "EUR_USD", list(dates), end, cov)


def ny_fx(trades):
    return [x for x in trades if x["session"] == "NY fx 08:00"]


class ExitEngine(unittest.TestCase):
    def test_target_gap_beats_intrabar_stop_long(self):
        d = frame("2026-01-05 13:00", [1.1000, 1.0980], opens=[1.1000, 1.1030],
                  highs=[1.1001, 1.1030], lows=[1.0999, 1.0980])
        px, k, kind, amb, side = O.exit_trade(1, 0, 1.0990, 1.1020, 1, quotes(d))
        self.assertEqual((kind, px, k), ("target_gap", 1.1020, 1))

    def test_target_gap_short(self):
        d = frame("2026-01-05 13:00", [1.1000, 1.1010], opens=[1.1000, 1.0970],
                  highs=[1.1001, 1.1030], lows=[1.0999, 1.0970])
        px, k, kind, *_ = O.exit_trade(-1, 0, 1.1010, 1.0980, 1, quotes(d))
        self.assertEqual((kind, px), ("target_gap", 1.0980))

    def test_stop_gap_fills_at_open(self):
        d = frame("2026-01-05 13:00", [1.1000, 1.0970], opens=[1.1000, 1.0980],
                  highs=[1.1001, 1.0981], lows=[1.0999, 1.0960])
        px, k, kind, *_ = O.exit_trade(1, 0, 1.0990, 1.1020, 1, quotes(d))
        self.assertEqual(kind, "stop_gap")
        self.assertAlmostEqual(px, 1.0980 - SP / 2)

    def test_both_touched_is_stop_and_ambiguous(self):
        d = frame("2026-01-05 13:00", [1.1000, 1.1000], opens=[1.1000, 1.1000],
                  highs=[1.1001, 1.1030], lows=[1.0999, 1.0980])
        px, k, kind, amb, _ = O.exit_trade(1, 0, 1.0990, 1.1020, 1, quotes(d))
        self.assertEqual((kind, amb), ("stop", True))

    # Fix 1: the entry candle gets open-first checks too
    def test_entry_candle_long_stop_fills_at_bid_open(self):
        # ask 100.2 (entry), bid open 99.8, stop 100.0 -> fill at 99.8, not 100.0
        d = frame("2026-01-05 13:00", [100.0], opens=[100.0], highs=[100.1], lows=[99.9], spread=0.4)
        px, k, kind, *_ = O.exit_trade(1, 0, 100.0, 100.6, 0, quotes(d))
        self.assertEqual(kind, "stop_gap")
        self.assertAlmostEqual(px, 99.8)

    def test_entry_candle_short_stop_fills_at_ask_open(self):
        # bid 99.8 (entry), ask open 100.2, stop 100.0 -> fill at 100.2
        d = frame("2026-01-05 13:00", [100.0], opens=[100.0], highs=[100.1], lows=[99.9], spread=0.4)
        px, k, kind, *_ = O.exit_trade(-1, 0, 100.0, 99.4, 0, quotes(d))
        self.assertEqual(kind, "stop_gap")
        self.assertAlmostEqual(px, 100.2)


class Signals(unittest.TestCase):
    def test_breakout_buy_next_open_stop_at_range_low(self):
        base, stress, _ = chunk(day_frame([1.1008, 1.1015] + [1.1016] * 5 + [1.1040] * 3))
        t = ny_fx(base)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["dir"], 1)
        self.assertEqual(t[0]["entry"], pd.Timestamp("2026-01-05 13:25", tz="UTC"))
        self.assertEqual(t[0]["stop"], 1.1000)
        self.assertAlmostEqual(t[0]["target"], t[0]["entry_px"] + 2 * (t[0]["entry_px"] - 1.1000))

    def test_session_exit_records_close_and_one_trade_per_day(self):
        base, *_ = chunk(day_frame([1.1012] + [1.1013] * 120 + [1.0990] * 5))
        t = ny_fx(base)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["kind"], "session")
        self.assertEqual(t[0]["exit"], pd.Timestamp("2026-01-05 21:00", tz="UTC"))

    def test_no_trade_without_breakout_within_three_hours(self):
        base, *_ = chunk(day_frame([1.1005] * 36 + [1.1020] * 5))
        self.assertEqual(ny_fx(base), [])

    def test_search_window_is_by_clock_with_missing_candles(self):
        d = pd.concat([day_frame([1.1005, 1.1005]), frame("2026-01-05 17:20", [1.1020] * 5)], ignore_index=True)
        base, *_ = chunk(d)
        self.assertEqual(ny_fx(base), [])

    def test_no_entry_at_or_after_exit_time(self):
        t0 = "2026-01-05 14:30"   # NY stock open; exit 12:30 NY = 17:30 UTC
        mids = [1.1005, 1.1010, 1.1000] + [1.1005] * 32 + [1.1020, 1.1020]
        highs = [1.1006, 1.1010, 1.1005] + [1.1006] * 32 + [1.1021, 1.1021]
        lows = [1.1004, 1.1006, 1.1000] + [1.1004] * 32 + [1.1019, 1.1019]
        base, *_ = chunk(frame(t0, mids, highs=highs, lows=lows))
        self.assertEqual([x for x in base if x["session"] == "NY stock 09:30"], [])

    def test_london_session_uses_london_clock(self):
        d = day_frame([1.1012] + [1.1013] * 20, start="2026-01-05 08:00")
        base, *_ = chunk(d)
        lon = [x for x in base if x["session"] == "London 08:00"]
        self.assertEqual(lon[0]["entry"], pd.Timestamp("2026-01-05 08:20", tz="UTC"))


class Stress(unittest.TestCase):
    # Fix 2: stress reprices exactly the baseline trades
    def test_rejected_baseline_entry_is_not_added_by_stress(self):
        # breakout buy, next open gaps below the range low: base ask 1.09995 < stop 1.1000 -> rejected;
        # the 2x-spread ask is 1.10005 > stop, which the old code accepted as a stress-only trade.
        path = [1.1012, 1.09985, 1.0999]
        d = day_frame(path, spread=0.0001)
        d.loc[4, ["o", "h", "l", "c"]] = [1.09985, 1.0999, 1.0998, 1.09985]
        for f in "ohlc":
            d.loc[4, "b" + f] = d.loc[4, f] - 0.0001      # 2-pip spread on the entry candle
            d.loc[4, "a" + f] = d.loc[4, f] + 0.0001
        base, stress, invalid = chunk(d)
        self.assertEqual((ny_fx(base), ny_fx(stress)), ([], []))
        self.assertGreaterEqual(invalid, 1)

    def test_stress_keeps_ids_entries_directions_and_brackets(self):
        base, stress, _ = chunk(day_frame([1.1008, 1.1015] + [1.1016] * 5 + [1.1040] * 3))
        self.assertEqual(len(base), len(stress))
        for b, s in zip(base, stress):
            for key in ("id", "entry", "dir", "stop", "target", "risk_pips"):
                self.assertEqual(b[key], s[key])
            self.assertGreater(s["entry_px"], b["entry_px"])      # worse long fill
            self.assertLess(s["r"], b["r"])                        # same position size -> lower R


class Costs(unittest.TestCase):
    # Fix 4: half spreads, close spread for close exits, no double deduction
    def test_session_exit_costs(self):
        d = day_frame([1.1012] + [1.1013] * 120, spread=0.0002)
        base, *_ = chunk(d)
        t = ny_fx(base)[0]
        risk = t["risk_pips"] * P
        self.assertAlmostEqual(t["spread_r"], (0.0001 + 0.0001) / risk)          # half + half of 2 pips
        self.assertAlmostEqual(t["cost_r"], (0.0002 + 2 * 0.1 * P) / risk)
        expected_r = ((1.1013 - 0.0001) - t["entry_px"] - 2 * 0.1 * P) / risk     # bid close - ask entry - slip
        self.assertAlmostEqual(t["r"], expected_r)


class Rollover(unittest.TestCase):
    # Fix 5: New York calendar dates
    def test_autumn_clock_change(self):
        self.assertEqual(O.nights(pd.Timestamp("2026-10-30T22:00Z"), pd.Timestamp("2026-11-02T23:00Z")), 3)

    def test_spring_clock_change(self):
        self.assertEqual(O.nights(pd.Timestamp("2027-03-12T22:30Z"), pd.Timestamp("2027-03-15T21:30Z")), 3)

    def test_summer_and_winter_rollover_hour(self):
        self.assertEqual(O.nights(pd.Timestamp("2026-07-06T20:30Z"), pd.Timestamp("2026-07-06T21:30Z")), 1)
        self.assertEqual(O.nights(pd.Timestamp("2026-01-05T21:15Z"), pd.Timestamp("2026-01-05T21:45Z")), 0)
        self.assertEqual(O.nights(pd.Timestamp("2026-01-05T21:15Z"), pd.Timestamp("2026-01-05T22:15Z")), 1)


class EndpointAndRuns(unittest.TestCase):
    # Fix 6: endpoint, exit labels, incomplete runs
    def test_trade_cut_by_endpoint_is_marked_end(self):
        d = day_frame([1.1012] + [1.1013] * 120)
        end = pd.Timestamp("2026-01-05 14:02", tz="UTC")
        d = d[d["time"] + O.M5 <= end].reset_index(drop=True)
        base, *_ = chunk(d, end=end)
        t = ny_fx(base)[0]
        self.assertEqual(t["kind"], "end")
        self.assertLessEqual(t["exit"], end)

    def test_run_drops_candles_closing_after_endpoint(self):
        big = day_frame([1.1012] + [1.1013] * 120)
        end = pd.Timestamp("2026-01-05 14:02", tz="UTC")
        seen = []

        def fetch(api, pair, a, b):
            part = big[(big["time"] >= a) & (big["time"] < b)].reset_index(drop=True)
            seen.append(part["time"].max())
            return part
        res = O.run(None, ["EUR_USD"], pd.Timestamp("2026-01-04", tz="UTC"), end, fetch=fetch)
        self.assertTrue(all(x["exit"] <= end for x in res["base"]))
        self.assertTrue(res["base"] and res["base"][0]["kind"] == "end")

    def test_failed_pair_makes_run_incomplete_and_blocks_pass(self):
        def fetch(api, pair, a, b):
            raise RuntimeError("network down")
        res = O.run(None, ["EUR_USD"], pd.Timestamp("2026-01-04", tz="UTC"), END, fetch=fetch)
        self.assertTrue(res["incomplete"])
        self.assertEqual(O.verdict(res["base"], pd.Timestamp("2026-01-06", tz="UTC"), res["incomplete"]), "INCOMPLETE")

    def test_thin_data_makes_run_incomplete(self):
        def fetch(api, pair, a, b):
            return pd.DataFrame()
        res = O.run(None, ["EUR_USD"], pd.Timestamp("2026-01-04", tz="UTC"), END, fetch=fetch)
        self.assertTrue(any("valid opening range" in m for m in res["incomplete"]))


class Streaming(unittest.TestCase):
    # Fix 3: bounded chunks give identical trades to one big window, no duplicates or gaps
    @classmethod
    def setUpClass(cls):
        t = pd.date_range("2025-12-01", "2026-02-10", freq="5min", tz="UTC")
        t = t[t.dayofweek < 5]
        rng = np.random.default_rng(3)
        c = 1.10 + rng.normal(0, 0.0004, len(t)).cumsum()
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c) + np.abs(rng.normal(0, 0.0002, len(t)))
        lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.0002, len(t)))
        d = pd.DataFrame({"time": t, "o": o, "h": h, "l": lo, "c": c, "volume": 100.0})
        for f in "ohlc":
            d["b" + f] = d[f] - 0.00005
            d["a" + f] = d[f] + 0.00005
        cls.data = d

    def fetch(self, api, pair, a, b):
        d = self.data
        return d[(d["time"] >= a) & (d["time"] + O.M5 <= b)].reset_index(drop=True)

    def test_chunked_equals_single_window(self):
        start, end = pd.Timestamp("2025-12-01", tz="UTC"), pd.Timestamp("2026-02-09 12:00", tz="UTC")
        small = O.run(None, ["EUR_USD"], start, end, chunk_days=14, fetch=self.fetch)
        whole = O.run(None, ["EUR_USD"], start, end, chunk_days=1000, fetch=self.fetch)
        key = lambda r: (r["id"], r["entry"], r["exit"], round(r["r"], 12))
        self.assertEqual([key(x) for x in small["base"]], [key(x) for x in whole["base"]])
        self.assertEqual([key(x) for x in small["stress"]], [key(x) for x in whole["stress"]])
        ids = [x["id"] for x in small["base"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(ids), 50)
        self.assertLessEqual(small["max_rows"], 14 * 288)
        self.assertEqual(len(small["base"]), len(small["stress"]))


class Safety(unittest.TestCase):
    def test_candle_endpoint_only(self):
        with open(O.__file__) as fh:
            src = fh.read()
        self.assertNotIn("/v3/accounts", src)
        self.assertNotIn("/orders", src)
        self.assertNotIn("api.post", src)
        self.assertNotIn("api.put", src)


if __name__ == "__main__":
    unittest.main()
