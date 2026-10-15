"""Offline tests for orb_retest.py (Revision 05). No network: every HTTP call is mocked.
Run only this file:  python -m unittest -v test_orb_retest.py
Test names carry the spec rule they check (SR = Session rules, XR = Exit rules, EO = Execution
order, DC = Doubled-cost run, DATA = conventions table, PB = pass bar)."""
import json
import socket
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import orb_retest as O

_NET = mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network access in tests"))


def setUpModule():
    _NET.start()


def tearDownModule():
    _NET.stop()


D = date(2015, 6, 16)                     # a Tuesday; London on BST, New York on EDT
S, DEADLINE, CUTOFF = O.session_times("LON", 8, 0, D)
HS = 0.00005                              # half spread
SLIP = 0.00001                            # 0.1 pip on EUR_USD
DSLIP = 0.00002                           # 0.2 pip (doubled run)


def at(k):
    """Epoch of candle k, counted in 5-minute steps from S (k = 0 is the first range candle)."""
    return S + k * O.STEP


def build(rows, hs=HS):
    """rows: {k: (o, h, l, c)} mid prices; bid/ask = mid -/+ hs on every field."""
    c = O.Candles()
    for k in sorted(rows):
        o, h, l, cl = rows[k]
        v = {"mo": o, "mh": h, "ml": l, "mc": cl,
             "bo": o - hs, "bh": h - hs, "bl": l - hs, "bc": cl - hs,
             "ao": o + hs, "ah": h + hs, "al": l + hs, "ac": cl + hs}
        c.append(at(k), v)
    return c


FLAT_IN = (1.1005, 1.1007, 1.1003, 1.1005)           # inside the range
RANGE = {0: (1.1005, 1.1010, 1.1004, 1.1006), 1: (1.1006, 1.1008, 1.1000, 1.1002),
         2: FLAT_IN, 3: FLAT_IN, 4: FLAT_IN, 5: FLAT_IN}  # RH 1.1010, RL 1.1000, stop 1.1005


def base_day(extra=None, until=None, after=(1.1005, 1.1007, 1.1003, 1.1005)):
    """Range + candles to 21:00 UTC (k = 0..167). 'after' fills every k >= 6 not in extra."""
    last = (CUTOFF - S) // O.STEP if until is None else until
    rows = dict(RANGE)
    for k in range(6, last):
        rows[k] = after
    rows.update(extra or {})
    return rows


def long_signal_rows(post=(1.1040, 1.1042, 1.1038, 1.1040)):
    """Breakout at k=6, retest at k=7 (signal, E = 1.1014), entry candle k=8, then 'post'."""
    rows = base_day(after=post)
    rows[6] = (1.1006, 1.1016, 1.1005, 1.1015)   # close 1.1015 > RH 1.1010: breakout
    rows[7] = (1.1015, 1.1016, 1.1008, 1.1014)   # low < RH, close > RH: retest -> signal
    rows[8] = (1.1014, 1.1016, 1.1012, 1.1015)   # entry candle
    return rows


def sim(rows, run="normal", ema_value=None, ema_map=None):
    c = build(rows)
    idx = {c.t[i]: i for i in range(len(c))}
    sig = O.find_signal(c, idx, S, DEADLINE)
    if sig["kind"] != "signal":
        return sig, None, c
    ema = O.ema_series(c)
    if ema_value is not None or ema_map:
        for i in range(len(c)):
            k = (c.t[i] - S) // O.STEP
            if ema_map and k in ema_map:
                ema[i] = ema_map[k]
            elif ema_value is not None:
                ema[i] = ema_value
    return sig, O.simulate_trade(c, ema, sig, DEADLINE, CUTOFF, "EUR_USD", run), c


# ---------------------------------------------------------------- time zones
class TestTimeZones(unittest.TestCase):
    def test_session_times_example(self):
        self.assertEqual(O.iso(S), "2015-06-16T07:00:00Z")          # London 08:00 BST
        self.assertEqual(O.iso(DEADLINE), "2015-06-16T20:45:00Z")   # 16:45 EDT
        self.assertEqual(O.iso(CUTOFF), "2015-06-16T21:00:00Z")
        w = date(2015, 1, 13)
        self.assertEqual(O.iso(O.session_times("NY", 9, 30, w)[0]), "2015-01-13T14:30:00Z")
        self.assertEqual(O.iso(O.session_times("LON", 8, 0, w)[0]), "2015-01-13T08:00:00Z")

    def test_dst_rules_match_tz_database(self):
        try:
            from zoneinfo import ZoneInfo
            zones = {"NY": ZoneInfo("America/New_York"), "LON": ZoneInfo("Europe/London")}
        except Exception:
            self.skipTest("no tz database on this machine")
        d = date(2009, 12, 1)
        while d <= date(2026, 12, 31):
            for zone, hh, mm in (("LON", 8, 0), ("NY", 8, 0), ("NY", 9, 30), ("NY", 16, 45), ("NY", 17, 0)):
                ref = datetime(d.year, d.month, d.day, hh, mm, tzinfo=zones[zone])
                self.assertEqual(O.local_to_epoch(zone, d, hh, mm), int(ref.timestamp()), (zone, d, hh, mm))
            d += timedelta(days=1)


# ---------------------------------------------------------------- session rules
class TestSessionRules(unittest.TestCase):
    def test_SR1_incomplete_range_skips_date(self):
        rows = long_signal_rows()
        del rows[3]
        sig, _, _ = sim(rows)
        self.assertEqual(sig["kind"], "skip")

    def test_SR1_range_high_low_and_stop(self):
        sig, _, _ = sim(long_signal_rows())
        self.assertEqual((sig["RH"], sig["RL"]), (1.1010, 1.1000))
        self.assertAlmostEqual(sig["stop"], 1.1005)

    def test_SR2_first_candidate_is_S_plus_30(self):
        rows = base_day()
        rows[5] = (1.1005, 1.1016, 1.1004, 1.1009)   # inside range (last range candle, high defines RH)
        rows[6] = (1.1009, 1.1022, 1.1008, 1.1021)   # first candidate: breakout
        rows[7] = (1.1021, 1.1022, 1.1012, 1.1020)   # retest (low < RH 1.1016)
        sig, _, _ = sim(rows)
        self.assertEqual(sig["kind"], "signal")
        self.assertEqual(O.iso(at(6)), "2015-06-16T07:30:00Z")

    def test_SR3_breakout_is_strict(self):
        rows = base_day()
        rows[6] = (1.1006, 1.1012, 1.1005, 1.1010)   # close == RH: no breakout
        rows[7] = (1.1010, 1.1012, 1.1008, 1.1011)   # breakout here (close > RH)
        rows[8] = (1.1011, 1.1012, 1.1009, 1.1011)   # retest: low < RH, close > RH
        sig, _, c = sim(rows)
        self.assertEqual(sig["kind"], "signal")
        self.assertEqual(c.t[sig["sig_i"]], at(8))

    def test_SR4_SR5_failure_candle_never_starts_new_breakout(self):
        rows = base_day()
        rows[6] = (1.1006, 1.1016, 1.1005, 1.1015)   # long breakout
        rows[7] = (1.1015, 1.1015, 1.0990, 1.0995)   # fails AND closes below RL: no short breakout here
        rows[8] = (1.0997, 1.0999, 1.0995, 1.0996)   # first chance for short: close < RL -> breakout
        rows[9] = (1.0996, 1.1002, 1.0995, 1.0998)   # retest: high > RL, close < RL -> short signal
        sig, _, c = sim(rows)
        self.assertEqual(sig["kind"], "signal")
        self.assertEqual(sig["dir"], -1)
        self.assertEqual(c.t[sig["sig_i"]], at(9))

    def test_SR4_failure_candle_would_have_been_a_short_retest_but_is_not(self):
        rows = base_day()
        rows[6] = (1.1006, 1.1016, 1.1005, 1.1015)   # long breakout
        rows[7] = (1.1015, 1.1015, 1.0990, 1.0995)   # failure; closes below RL
        rows[8] = (1.0995, 1.1001, 1.0994, 1.0996)   # short breakout here (not a retest of anything)
        sig, _, _ = sim(rows)
        # Had the failure candle started a short breakout, k=8 (high > RL, close < RL) would be a
        # retest and signal. Correctly, k=8 is only the breakout; later candles close inside the
        # range, so the breakout fails and there is no signal at all.
        self.assertEqual(sig["kind"], "none")

    def test_SR6_close_exactly_at_level_continues_breakout(self):
        rows = base_day()
        rows[6] = (1.1006, 1.1016, 1.1005, 1.1015)   # breakout
        rows[7] = (1.1015, 1.1016, 1.1009, 1.1010)   # close == RH: neither failure nor retest
        rows[8] = (1.1010, 1.1013, 1.1009, 1.1012)   # retest
        sig, _, c = sim(rows)
        self.assertEqual(c.t[sig["sig_i"]], at(8))
        self.assertEqual(sig["dir"], 1)

    def test_SR7_retest_low_must_be_strictly_below(self):
        rows = base_day()
        rows[6] = (1.1006, 1.1016, 1.1005, 1.1015)
        rows[7] = (1.1015, 1.1016, 1.1010, 1.1014)   # low == RH: not a retest
        rows[8] = (1.1014, 1.1016, 1.1009, 1.1013)   # retest
        sig, _, c = sim(rows)
        self.assertEqual(c.t[sig["sig_i"]], at(8))

    def test_SR7_no_retest_on_breakout_candle(self):
        rows = base_day()
        rows[6] = (1.1006, 1.1016, 1.1005, 1.1015)   # breakout with low < RH: still not a retest
        sig, _, _ = sim(rows)
        self.assertEqual(sig["kind"], "none")

    def test_SR8_E_is_signal_candle_mid_close(self):
        sig, _, _ = sim(long_signal_rows())
        self.assertEqual(sig["E"], 1.1014)

    def test_SR9_SR10_last_signal_candle_opens_1635(self):
        k1635 = (DEADLINE - 2 * O.STEP - S) // O.STEP
        rows = base_day()
        rows[k1635 - 1] = (1.1006, 1.1016, 1.1005, 1.1015)   # breakout 16:30
        rows[k1635] = (1.1015, 1.1016, 1.1008, 1.1014)       # retest 16:35 -> signal
        sig, tr, c = sim(rows)
        self.assertEqual(sig["kind"], "signal")
        self.assertEqual(O.iso(tr["entry_time"]), "2015-06-16T20:40:00Z")

    def test_SR10_retest_on_1640_candle_is_too_late(self):
        k1640 = (DEADLINE - O.STEP - S) // O.STEP
        rows = base_day()
        rows[k1640 - 1] = (1.1006, 1.1016, 1.1005, 1.1015)
        rows[k1640] = (1.1015, 1.1016, 1.1008, 1.1014)
        sig, _, _ = sim(rows)
        self.assertEqual(sig["kind"], "none")

    def test_SR9_missing_entry_candle_cancels_and_uses_allowance(self):
        k1635 = (DEADLINE - 2 * O.STEP - S) // O.STEP
        rows = base_day()
        rows[k1635 - 1] = (1.1006, 1.1016, 1.1005, 1.1015)
        rows[k1635] = (1.1015, 1.1016, 1.1008, 1.1014)
        del rows[k1635 + 1]                                   # 16:40 candle missing -> next opens 16:45
        sig, tr, _ = sim(rows)
        self.assertEqual(tr["status"], "cancelled")


# ---------------------------------------------------------------- entry, stop, exits
class TestEntryAndExits(unittest.TestCase):
    def test_entry_fill_R_and_entry_candle(self):
        _, tr, _ = sim(long_signal_rows())
        self.assertEqual(O.iso(tr["entry_time"]), O.iso(at(8)))
        self.assertAlmostEqual(tr["entry_fill"], 1.1014 + HS + SLIP)
        self.assertAlmostEqual(tr["R"], 1.1014 + HS + SLIP - 1.1005)

    def test_invalid_entry_when_R_not_positive(self):
        rows = long_signal_rows()
        rows[8] = (1.1003, 1.1006, 1.1002, 1.1004)            # opens below the stop
        _, tr, _ = sim(rows)
        self.assertEqual(tr["status"], "invalid")

    def test_EO4_entry_candle_open_through_stop_on_exit_side(self):
        rows = long_signal_rows()
        rows[8] = (1.10052, 1.1010, 1.1004, 1.1008)           # ask 1.10057 > stop, bid 1.10047 <= stop
        _, tr, _ = sim(rows)
        self.assertEqual(tr["exits"][0]["reason"], "gap_stop")
        self.assertAlmostEqual(tr["exits"][0]["fill"], 1.10052 - HS - SLIP)

    def test_stop_touch_inclusive_during_entry_candle(self):
        rows = long_signal_rows()
        rows[8] = (1.1014, 1.1016, 1.1005 + HS, 1.1015)       # bid low == stop exactly
        _, tr, _ = sim(rows)
        self.assertEqual(tr["exits"][0]["reason"], "stop")
        self.assertAlmostEqual(tr["exits"][0]["fill"], 1.1005 - SLIP)
        self.assertAlmostEqual(tr["r"], (1.1005 - SLIP - tr["entry_fill"]) / tr["R"])

    def test_XR1_gate_below_threshold_no_exit_then_liquidation(self):
        # Trigger true on every candle (close 1.1022 < EMA 1.10250) but the gate is false (0.10% < 0.20%),
        # so no partial ever fires and the trade runs to 16:45.
        rows = long_signal_rows(post=(1.1022, 1.1024, 1.1020, 1.1022))
        _, tr, _ = sim(rows, ema_value=1.1014 * 1.0010)
        self.assertLess(1.1022, 1.1014 * 1.0010)                       # trigger condition holds
        self.assertEqual([x["reason"] for x in tr["exits"]], ["end_of_day"])
        self.assertEqual(O.iso(tr["exits"][0]["time"]), "2015-06-16T20:45:00Z")
        self.assertAlmostEqual(tr["exits"][0]["fill"], 1.1022 - HS - SLIP)

    def test_XR2_trigger_has_no_previous_candle_condition(self):
        # Every candle closes below the EMA (no cross ever happens); gate open from k=9 on.
        rows = long_signal_rows(post=(1.1040, 1.1042, 1.1038, 1.1039))
        rows[8] = (1.1014, 1.1041, 1.1012, 1.1039)
        ema_map = {8: 1.1030}            # gate shut at k=8 (0.145%)
        _, tr, _ = sim(rows, ema_value=1.1040, ema_map=ema_map)
        self.assertEqual(tr["exits"][0]["reason"], "partial_1")
        self.assertEqual(O.iso(tr["exits"][0]["time"]), O.iso(at(10)))   # triggered at k=9 close

    def test_XR3_to_XR6_stages_spacing_one_per_candle_and_adaptive_stop(self):
        rows = long_signal_rows(post=(1.1060, 1.1062, 1.1058, 1.1061))   # closes above EMA: no trigger
        rows[9] = (1.1045, 1.1047, 1.1036, 1.1038)    # close < EMA 1.1040, gate 0.236%: stage 1, P1 = 1.1040
        rows[10] = (1.1038, 1.1046, 1.1037, 1.1043)   # fill 1; close < EMA 1.1045 but spacing 0.045%: no stage
        rows[11] = (1.1043, 1.1051, 1.1042, 1.1048)   # close < EMA 1.1050, spacing 0.0906%: stage 2, P2 = 1.1050
        rows[12] = (1.1048, 1.1061, 1.1047, 1.1058)   # fill 2; close < EMA 1.1060, spacing 0.0905%: stage 3
        e = {9: 1.1040, 10: 1.1045, 11: 1.1050, 12: 1.1060}
        _, tr, _ = sim(rows, ema_value=1.1030, ema_map=e)
        reasons = [(x["reason"], O.iso(x["time"]), x["share"]) for x in tr["exits"]]
        self.assertEqual(reasons, [("partial_1", O.iso(at(10)), 0.90),
                                   ("partial_2", O.iso(at(12)), 0.05),
                                   ("partial_3", O.iso(at(13)), 0.05)])
        fills = [1.1038 - HS - SLIP, 1.1048 - HS - SLIP, 1.1060 - HS - SLIP]
        expect = sum((f - tr["entry_fill"]) * sh for f, sh in zip(fills, (0.90, 0.05, 0.05))) / tr["R"]
        self.assertAlmostEqual(tr["r"], expect)

    def test_XR_adaptive_stop_only_after_stage1_fill(self):
        rows = long_signal_rows(post=(1.1060, 1.1062, 1.1058, 1.1061))
        rows[9] = (1.1045, 1.1047, 1.1036, 1.1038)    # stage 1 trigger
        entry_ask = 1.1014 + HS + SLIP
        rows[11] = (1.1060, 1.1062, entry_ask + HS - 0.00001, 1.1061)   # bid low just below entry fill
        _, tr, _ = sim(rows, ema_value=1.1030, ema_map={9: 1.1040})
        self.assertEqual([x["reason"] for x in tr["exits"]], ["partial_1", "stop"])
        self.assertAlmostEqual(tr["exits"][1]["fill"], tr["entry_fill"] - SLIP)
        self.assertAlmostEqual(tr["exits"][1]["share"], 0.10)

    def test_XR_same_low_before_stage1_fill_does_not_stop(self):
        rows = long_signal_rows(post=(1.1060, 1.1062, 1.1058, 1.1061))
        entry_ask = 1.1014 + HS + SLIP
        rows[9] = (1.1060, 1.1062, entry_ask + HS - 0.00001, 1.1061)    # same low, but no partial filled yet
        _, tr, _ = sim(rows, ema_value=1.1030)
        self.assertEqual([x["reason"] for x in tr["exits"]], ["end_of_day"])

    def test_EO1_gap_stop_before_pending_partial(self):
        rows = long_signal_rows(post=(1.1060, 1.1062, 1.1058, 1.1061))
        rows[9] = (1.1045, 1.1047, 1.1036, 1.1038)          # stage 1 scheduled at this close
        rows[10] = (1.1000, 1.1002, 1.0998, 1.1001)          # next open gaps through the stop
        _, tr, _ = sim(rows, ema_value=1.1030, ema_map={9: 1.1040})   # partial pending from k=9
        self.assertEqual([(x["reason"], x["share"]) for x in tr["exits"]], [("gap_stop", 1.0)])
        control = dict(rows)
        control[10] = (1.1038, 1.1046, 1.1037, 1.1043)       # same day without the gap
        _, tr2, _ = sim(control, ema_value=1.1030, ema_map={9: 1.1040})
        self.assertEqual((tr2["exits"][0]["reason"], O.iso(tr2["exits"][0]["time"])), ("partial_1", O.iso(at(10))))
        self.assertAlmostEqual(tr["exits"][0]["fill"], 1.1000 - HS - SLIP)

    def test_stop_beats_profit_take_in_same_candle(self):
        rows = long_signal_rows(post=(1.1060, 1.1062, 1.1058, 1.1059))
        rows[9] = (1.1040, 1.1042, 1.1004, 1.1020)           # touches the stop and closes below EMA
        _, tr, _ = sim(rows, ema_value=1.1030, ema_map={9: 1.1040})
        self.assertEqual([x["reason"] for x in tr["exits"]], ["stop"])

    def test_no_new_partial_scheduled_into_the_1645_open(self):
        k1640 = (DEADLINE - O.STEP - S) // O.STEP
        rows = long_signal_rows(post=(1.1040, 1.1042, 1.1038, 1.1039))
        _, tr, _ = sim(rows, ema_value=1.1030, ema_map={k1640: 1.1045})   # trigger only at 16:40 close
        self.assertEqual([x["reason"] for x in tr["exits"]], ["end_of_day"])


# ---------------------------------------------------------------- 16:45 clock event
class TestLiquidation(unittest.TestCase):
    def k(self, hhmm_utc):
        h, m = map(int, hhmm_utc.split(":"))
        return (int(datetime(2015, 6, 16, h, m, tzinfo=timezone.utc).timestamp()) - S) // O.STEP

    def test_liquidate_at_1645_open(self):
        _, tr, _ = sim(long_signal_rows(), ema_value=1.1014)
        self.assertEqual(tr["exits"][-1]["reason"], "end_of_day")
        self.assertEqual(O.iso(tr["exit_time"]), "2015-06-16T20:45:00Z")

    def test_missing_1645_candle_uses_1650(self):
        rows = long_signal_rows()
        del rows[self.k("20:45")]
        _, tr, _ = sim(rows, ema_value=1.1014)
        self.assertEqual(O.iso(tr["exit_time"]), "2015-06-16T20:50:00Z")

    def test_no_candle_before_1700_is_unresolved(self):
        rows = long_signal_rows()
        for hm in ("20:45", "20:50", "20:55"):
            del rows[self.k(hm)]
        _, tr, _ = sim(rows, ema_value=1.1014)
        self.assertEqual(tr["status"], "unresolved")

    def test_gap_stop_processed_first_at_delayed_quote(self):
        rows = long_signal_rows()
        del rows[self.k("20:45")]
        rows[self.k("20:50")] = (1.1000, 1.1002, 1.0998, 1.1001)
        _, tr, _ = sim(rows, ema_value=1.1014)
        self.assertEqual(tr["exits"][-1]["reason"], "gap_stop")

    def test_pending_partial_cancelled_by_deadline(self):
        k1635 = self.k("20:35")
        rows = long_signal_rows(post=(1.1040, 1.1042, 1.1038, 1.1039))
        del rows[self.k("20:40")]                 # pending partial from 16:35 close has no 16:40 candle
        _, tr, _ = sim(rows, ema_value=1.1030, ema_map={k1635: 1.1045})
        self.assertEqual([x["reason"] for x in tr["exits"]], ["end_of_day"])
        self.assertEqual(tr["exits"][0]["share"], 1.0)

    def test_verdict_incomplete_when_unresolved(self):
        st = O.stats([])
        self.assertEqual(O.verdict(st, [{"pair": "EUR_USD"}]), "NO VERDICT (INCOMPLETE)")


# ---------------------------------------------------------------- doubled-cost run
class TestDoubledCost(unittest.TestCase):
    def test_DC1_widen_envelope(self):
        # Valid native candle whose spread is wide at the high: raw widened bid high (0.9998)
        # falls below the widened bid open (0.9999) and close (1.0000), so repair is required.
        v = {"mo": 1.0001, "mh": 1.0008, "ml": 0.9996, "mc": 1.0002,
             "bo": 1.0000, "bh": 1.0003, "bl": 0.9995, "bc": 1.0001,
             "ao": 1.0002, "ah": 1.0013, "al": 0.9997, "ac": 1.0003}
        O.validate_candle(raw_from_vals("2015-06-16T07:00:00.000000000Z", v), None)   # native data valid
        w = O.widen(v)
        h = {f: (v["a" + f] - v["b" + f]) / 2 for f in "ohlc"}
        rb = {f: v["b" + f] - h[f] for f in "ohlc"}
        ra = {f: v["a" + f] + h[f] for f in "ohlc"}
        self.assertLess(rb["h"], rb["o"])                               # the case that needs repair
        self.assertAlmostEqual(w["bo"], rb["o"]); self.assertAlmostEqual(w["bc"], rb["c"])
        self.assertAlmostEqual(w["bh"], max(rb.values())); self.assertAlmostEqual(w["bl"], min(rb.values()))
        self.assertAlmostEqual(w["ao"], ra["o"]); self.assertAlmostEqual(w["ac"], ra["c"])
        self.assertAlmostEqual(w["ah"], max(ra.values())); self.assertAlmostEqual(w["al"], min(ra.values()))
        self.assertAlmostEqual(w["bh"], 1.0000); self.assertAlmostEqual(w["bl"], 0.9994)
        self.assertAlmostEqual(w["ah"], 1.0018); self.assertAlmostEqual(w["al"], 0.9998)
        for side in "ba":
            self.assertTrue(w[side + "l"] <= min(w[side + "o"], w[side + "c"]) <= max(w[side + "o"], w[side + "c"]) <= w[side + "h"])
        self.assertEqual((w["mo"], w["mh"], w["ml"], w["mc"]), (1.0001, 1.0008, 0.9996, 1.0002))

    def test_DC_widened_spread_and_slippage_in_fills_and_R(self):
        _, tr, _ = sim(long_signal_rows(), run="double", ema_value=1.1014)
        self.assertAlmostEqual(tr["entry_fill"], 1.1014 + 2 * HS + DSLIP)
        self.assertAlmostEqual(tr["R"], 1.1014 + 2 * HS + DSLIP - 1.1005)

    def test_DC3_DC4_independent_state(self):
        rows = long_signal_rows()
        rows[9] = (1.1040, 1.1042, 1.1005 + HS + 0.00002, 1.1040)   # normal bid low 2 points above stop
        _, normal, _ = sim(rows, run="normal", ema_value=1.1014)
        _, double, _ = sim(rows, run="double", ema_value=1.1014)
        self.assertEqual(normal["exits"][-1]["reason"], "end_of_day")
        self.assertEqual(double["exits"][-1]["reason"], "stop")
        self.assertEqual(O.iso(double["exit_time"]), O.iso(at(9)))


# ---------------------------------------------------------------- data, EMA, statistics
class TestDataAndStats(unittest.TestCase):
    def test_DATA_ema_seed_and_recursion(self):
        rows = {0: (1.0, 1.2, 1.0, 1.1), 1: (1.1, 1.4, 1.2, 1.3), 2: (1.3, 1.3, 1.1, 1.2)}
        e = O.ema_series(build(rows))
        self.assertAlmostEqual(e[0], 1.1)
        self.assertAlmostEqual(e[1], 0.2 * 1.3 + 0.8 * 1.1)
        self.assertAlmostEqual(e[2], 0.2 * 1.2 + 0.8 * e[1])

    def test_DATA_validation_errors(self):
        good = {"time": "2015-06-16T07:00:00.000000000Z", "complete": True,
                "mid": {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.1"},
                "bid": {"o": "1.09", "h": "1.19", "l": "0.99", "c": "1.09"},
                "ask": {"o": "1.11", "h": "1.21", "l": "1.01", "c": "1.11"}}
        O.validate_candle(good, None)
        cases = [({"time": "2015-06-16T07:00:00.500000000Z"}, "fractional"),
                 ({"time": "2015-06-16T07:01:00.000000000Z"}, "aligned"),
                 ({"complete": False}, "incomplete"),
                 ({"bid": {"o": "1.2", "h": "1.25", "l": "0.99", "c": "1.09"}}, "bid above ask"),
                 ({"mid": {"o": "1.1", "h": "1.05", "l": "1.0", "c": "1.1"}}, "OHLC"),
                 ({"mid": {"o": "nan", "h": "1.2", "l": "1.0", "c": "1.1"}}, "finite")]
        for change, msg in cases:
            c = json.loads(json.dumps(good))
            c.update(change)
            with self.assertRaises(O.DataError) as cm:
                O.validate_candle(c, None)
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(O.DataError):
            O.validate_candle(good, O.parse_time(good["time"]))          # duplicate

    def test_PB_stats_halves_pf_drawdown_and_verdict(self):
        tr = []
        for i, r in enumerate([1.0, -0.5, 2.0, -1.0, 0.5]):
            tr.append({"pair": O.PAIRS[i], "entry_time": 100 * i, "exit_time": 100 * i + 50, "r": r})
        st = O.stats(tr)
        self.assertEqual(st["n"], 5)
        self.assertAlmostEqual(st["first_half_avg"], 0.25)          # first floor(5/2) = 2 trades
        self.assertAlmostEqual(st["second_half_avg"], (2.0 - 1.0 + 0.5) / 3)
        self.assertAlmostEqual(st["pf"], 3.5 / 1.5)
        self.assertAlmostEqual(st["max_dd_r"], 1.0)
        self.assertEqual(st["pairs_positive"], 3)
        self.assertEqual(O.verdict(st, []), "FAIL")                  # < 300 trades
        self.assertEqual(O.stats([{"pair": "EUR_USD", "entry_time": 1, "exit_time": 2, "r": 1.0}])["pf"], "infinite")
        self.assertEqual(O.stats([{"pair": "EUR_USD", "entry_time": 1, "exit_time": 2, "r": 0.0}])["pf"], "undefined")

    def test_PB_pass_case(self):
        tr = [{"pair": O.PAIRS[i % 12], "entry_time": i, "exit_time": i, "r": 1.0 if i % 3 else -0.5}
              for i in range(360)]
        self.assertEqual(O.verdict(O.stats(tr), []), "PASS")


# ---------------------------------------------------------------- downloader (mocked HTTP)
def raw(t, mid=1.1, hs=0.0001):
    def q(o):
        return {"o": f"{mid + o:.5f}", "h": f"{mid + 0.001 + o:.5f}", "l": f"{mid - 0.001 + o:.5f}", "c": f"{mid + o:.5f}"}
    return {"time": datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "complete": True, "mid": q(0), "bid": q(-hs), "ask": q(hs)}


def raw_from_vals(time_s, v):
    return {"time": time_s, "complete": True,
            "mid": {f: repr(v["m" + f]) for f in "ohlc"}, "bid": {f: repr(v["b" + f]) for f in "ohlc"},
            "ask": {f: repr(v["a" + f]) for f in "ohlc"}}


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, series, pair="EUR_USD", script=None, page=4):
        self.series, self.pair, self.script, self.calls = series, pair, list(script or []), []
        self.page = page

    def get(self, url, params=None, timeout=None, allow_redirects=True, headers=None):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout,
                           "allow_redirects": allow_redirects})
        if self.script:
            act = self.script.pop(0)
            if isinstance(act, Exception):
                raise act
            if isinstance(act, int):
                return FakeResp(act, {})
            if isinstance(act, dict):
                return FakeResp(200, act)
        frm = O.parse_time(params["from"])
        sel = [t for t in self.series if t >= frm][: int(params["count"])]
        return FakeResp(200, {"instrument": self.pair, "granularity": "M5", "candles": [raw(t) for t in sel]})


class TestDownloader(unittest.TestCase):
    def setUp(self):
        self.p_sleep = mock.patch.object(O, "_sleep", lambda s: None)
        self.p_sleep.start()
        self.start = datetime(2015, 6, 16, 7, 0, tzinfo=timezone.utc)
        self.end = datetime(2015, 6, 16, 8, 0, tzinfo=timezone.utc)
        t0 = int(self.start.timestamp())
        self.series = [t0 + k * 300 for k in range(14) if k != 6]       # one absent period at 07:30; data runs past the end

    def tearDown(self):
        mock.patch.stopall()
        O._session = None

    def dl(self, fake, **kw):
        O._session = fake
        c, n = O.download("EUR_USD", "tok", start=self.start, end=self.end, page=4, log=lambda *a: None, **kw)
        self.requests = n
        return c

    def test_DATA_pagination_params_no_duplicates_no_gaps(self):
        fake = FakeSession(self.series)
        c = self.dl(fake)
        self.assertEqual(list(c.t), [x for x in self.series if x < int(self.end.timestamp())])   # end excluded
        p0 = fake.calls[0]["params"]
        self.assertEqual({k: p0[k] for k in ("granularity", "price", "smooth", "includeFirst", "count")},
                         {"granularity": "M5", "price": "BAM", "smooth": "false", "includeFirst": "true", "count": "4"})
        self.assertEqual(p0["from"], "2015-06-16T07:00:00Z")
        # next page starts one full candle after the last one returned (07:15 -> 07:20)
        self.assertEqual(fake.calls[1]["params"]["from"], "2015-06-16T07:20:00Z")
        self.assertTrue(all(call["allow_redirects"] is False for call in fake.calls))
        self.assertTrue(all(call["url"] == "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
                            for call in fake.calls))
        self.assertTrue(all(call["timeout"] == (10, 30) for call in fake.calls))

    def test_DATA_response_identity_checked(self):
        with self.assertRaises(O.DataError):
            self.dl(FakeSession(self.series, pair="GBP_USD"))

    def test_DATA_auth_failure_stops_after_one_attempt(self):
        fake = FakeSession(self.series, script=[401])
        with self.assertRaises(O.DownloadError):
            self.dl(fake)
        self.assertEqual(len(fake.calls), 1)

    def test_DATA_three_attempts_then_failed_download(self):
        import requests
        fake = FakeSession(self.series, script=[503, requests.ConnectionError("x"), 502])
        with self.assertRaises(O.DownloadError) as cm:
            self.dl(fake)
        self.assertEqual(len(fake.calls), 3)
        self.assertIn("after 3 attempts", str(cm.exception))

    def test_DATA_bad_candle_in_page_stops(self):
        bad = raw(self.series[0])
        bad["complete"] = False
        fake = FakeSession(self.series, script=[{"instrument": "EUR_USD", "granularity": "M5", "candles": [bad]}])
        with self.assertRaises(O.DataError):
            self.dl(fake)

    def test_DATA_empty_page_after_successful_pages_is_failure(self):
        fake = FakeSession(self.series[:6])        # data stops at 07:25; the next page comes back empty
        with self.assertRaises(O.DownloadError) as cm:
            self.dl(fake)
        self.assertIn("empty page", str(cm.exception))
        self.assertGreaterEqual(len(fake.calls), 2)

    def test_DATA_short_page_ending_one_day_early_is_failure(self):
        self.end = datetime(2015, 6, 17, 8, 0, tzinfo=timezone.utc)    # one day later than the data
        with self.assertRaises(O.DownloadError):
            self.dl(FakeSession(self.series[:11]))

    def test_DATA_short_page_then_more_data_continues(self):
        t0 = int(self.start.timestamp())
        first = {"instrument": "EUR_USD", "granularity": "M5", "candles": [raw(t0), raw(t0 + 300)]}   # short page
        fake = FakeSession(self.series, script=[first])
        c = self.dl(fake)
        self.assertEqual(list(c.t), [x for x in self.series if x < int(self.end.timestamp())])
        self.assertEqual(fake.calls[1]["params"]["from"], "2015-06-16T07:10:00Z")

    def test_DATA_gap_records_keep_every_gap(self):
        c = self.dl(FakeSession(self.series))
        self.assertEqual(O.gap_records(c), [(int(self.start.timestamp()) + 6 * 300,
                                             int(self.start.timestamp()) + 7 * 300, 1)])
        long_gap = O.Candles()
        for t in (0, 300, 300 + 3 * 86400):                              # a 3-day gap is reported too
            long_gap.append(t, {f: 1.0 for f in O.FIELDS})
        self.assertEqual(O.gap_records(long_gap)[0][2], 3 * 288 - 1)
        self.assertEqual(O.gap_summary(O.gap_records(long_gap)), {1970: {"gaps": 1, "absent_slots": 863}})


# ---------------------------------------------------------------- whole pipeline on one synthetic day
class TestRunPair(unittest.TestCase):
    def test_run_pair_one_day_both_runs(self):
        c = build(long_signal_rows())
        res = O.run_pair(c, "EUR_USD", sessions=[("London", "LON", 8, 0)], first=D, last=D)
        for r in ("normal", "double"):
            self.assertEqual(len(res["London"]["runs"][r]["trades"]), 1)
        self.assertEqual(res["London"]["diag"]["signals"], 1)


# ---------------------------------------------------------------- review additions
def short_signal_rows(post=(1.0970, 1.0972, 1.0968, 1.0972)):
    """Short breakout at k=6, retest at k=7 (E = 1.0996), entry candle k=8, then 'post'."""
    rows = base_day(after=post)
    rows[6] = (1.1004, 1.1005, 1.0994, 1.0995)   # close 1.0995 < RL 1.1000: short breakout
    rows[7] = (1.0995, 1.1002, 1.0994, 1.0996)   # high > RL, close < RL: retest -> signal
    rows[8] = (1.0996, 1.0998, 1.0994, 1.0995)   # entry candle
    return rows


def kk(hhmm_utc):
    h, m = map(int, hhmm_utc.split(":"))
    return (int(datetime(2015, 6, 16, h, m, tzinfo=timezone.utc).timestamp()) - S) // O.STEP


class TestShortSide(unittest.TestCase):
    def test_short_entry_partial_adaptive_and_liquidation(self):
        rows = short_signal_rows()
        rows[9] = (1.0975, 1.0977, 1.0969, 1.0972)   # close > EMA 1.0970, gate 0.236% below E: stage 1
        sig, tr, _ = sim(rows, ema_value=1.0980, ema_map={9: 1.0970})
        self.assertEqual((sig["dir"], sig["E"]), (-1, 1.0996))
        entry = 1.0996 - HS - SLIP
        self.assertAlmostEqual(tr["entry_fill"], entry)
        self.assertAlmostEqual(tr["R"], 1.1005 - entry)                 # d x (fill - stop), d = -1
        self.assertEqual([x["reason"] for x in tr["exits"]], ["partial_1", "end_of_day"])
        f1 = 1.0970 + HS + SLIP                                         # buy to close at ask + s
        self.assertAlmostEqual(tr["exits"][0]["fill"], f1)
        self.assertAlmostEqual(tr["r"], (-(f1 - entry) * 0.9 - (f1 - entry) * 0.1) / tr["R"])

    def test_short_stop_inclusive_on_ask_high(self):
        rows = short_signal_rows()
        rows[9] = (1.0990, 1.1005 - HS, 1.0985, 1.0995)                 # ask high == stop exactly
        _, tr, _ = sim(rows, ema_value=1.0980)
        self.assertEqual(tr["exits"][0]["reason"], "stop")
        self.assertAlmostEqual(tr["exits"][0]["fill"], 1.1005 + SLIP)
        self.assertAlmostEqual(tr["r"], -(1.1005 + SLIP - tr["entry_fill"]) / tr["R"])


class TestReviewEdgeCases(unittest.TestCase):
    def test_delayed_entry_before_deadline_succeeds(self):
        rows = long_signal_rows()
        del rows[8]                                                      # entry candle missing
        _, tr, _ = sim(rows, ema_value=1.1014)
        self.assertEqual(tr["status"], "closed")
        self.assertEqual(O.iso(tr["entry_time"]), O.iso(at(9)))
        self.assertAlmostEqual(tr["entry_fill"], 1.1040 + HS + SLIP)

    def test_adaptive_stop_hit_at_the_same_open_as_stage1_fill(self):
        rows = long_signal_rows(post=(1.1060, 1.1062, 1.1058, 1.1061))
        rows[9] = (1.1045, 1.1047, 1.1036, 1.1038)                       # stage 1 scheduled
        rows[10] = (1.1012, 1.1015, 1.1011, 1.1013)                      # bid open above old stop, below entry fill
        _, tr, _ = sim(rows, ema_value=1.1030, ema_map={9: 1.1040})
        fill = 1.1012 - HS - SLIP
        self.assertEqual([(x["reason"], O.iso(x["time"])) for x in tr["exits"]],
                         [("partial_1", O.iso(at(10))), ("gap_stop", O.iso(at(10)))])
        self.assertAlmostEqual(tr["exits"][0]["fill"], fill)
        self.assertAlmostEqual(tr["exits"][1]["fill"], fill)
        self.assertAlmostEqual(tr["exits"][1]["share"], 0.10)

    def test_unresolved_after_filled_partial_keeps_record(self):
        rows = long_signal_rows(post=(1.1060, 1.1062, 1.1058, 1.1061))
        rows[9] = (1.1045, 1.1047, 1.1036, 1.1038)
        for hm in ("20:45", "20:50", "20:55"):
            del rows[kk(hm)]
        _, tr, _ = sim(rows, ema_value=1.1030, ema_map={9: 1.1040})
        self.assertEqual(tr["status"], "unresolved")
        self.assertEqual([x["reason"] for x in tr["exits"]], ["partial_1"])
        self.assertAlmostEqual(tr["remaining"], 0.10)
        self.assertNotIn("exit_time", tr)
        self.assertNotIn("r", tr)
        for k in ("entry_time", "entry_fill", "R", "stop0", "E"):
            self.assertIn(k, tr)

    def test_drawdown_uses_exit_order_not_entry_order(self):
        tr = [{"pair": "EUR_USD", "entry_time": 1, "exit_time": 10, "r": -1.0},
              {"pair": "GBP_USD", "entry_time": 2, "exit_time": 5, "r": 2.0},
              {"pair": "USD_JPY", "entry_time": 3, "exit_time": 6, "r": -1.5}]
        self.assertAlmostEqual(O.stats(tr)["max_dd_r"], 2.5)            # by entry order it would be 1.5

    def test_entry_time_ties_resolved_by_pair(self):
        tr = [{"pair": "GBP_USD", "entry_time": 5, "exit_time": 9, "r": -1.0},
              {"pair": "EUR_USD", "entry_time": 5, "exit_time": 9, "r": 1.0}]
        st = O.stats(tr)
        self.assertEqual((st["first_half_avg"], st["second_half_avg"]), (1.0, -1.0))


class TestSummaryAndExport(unittest.TestCase):
    SESS = [("London", "LON", 8, 0)]

    def res_for(self, rows):
        return {"EUR_USD": O.run_pair(build(rows), "EUR_USD", sessions=self.SESS, first=D, last=D)}

    def test_complete_run_status_and_verdicts(self):
        sm = O.build_summary(self.res_for(long_signal_rows()), {"commit": "x"}, pairs=["EUR_USD"], sessions=self.SESS)
        self.assertEqual(sm["status"], "COMPLETE")
        self.assertEqual(sm["sessions"]["London"]["normal"]["status"], "COMPLETE")
        self.assertEqual(sm["sessions"]["London"]["normal"]["verdict"], "FAIL")      # fewer than 300 trades
        self.assertEqual(sm["sessions"]["London"]["double"]["verdict"], "diagnostic only (doubled costs)")

    def test_unresolved_makes_both_runs_incomplete_without_verdict(self):
        rows = long_signal_rows()
        for hm in ("20:45", "20:50", "20:55"):
            del rows[kk(hm)]
        res = self.res_for(rows)
        sm = O.build_summary(res, {}, pairs=["EUR_USD"], sessions=self.SESS)
        self.assertEqual(sm["status"], "INCOMPLETE")
        self.assertEqual(sm["incomplete_runs"], ["London/normal", "London/double"])
        self.assertEqual(sm["sessions"]["London"]["normal"]["verdict"], "NO VERDICT (INCOMPLETE)")
        self.assertEqual(sm["sessions"]["London"]["normal"]["n"], 0)                 # excluded from stats
        self.assertEqual(len(sm["sessions"]["London"]["normal"]["unresolved"]), 1)
        self.assertEqual(list(sm)[0], "status")                                       # stated at the top

    def test_download_failure_summary_is_incomplete(self):
        sm = O.build_summary({}, {}, failure={"type": "DownloadError", "error": "x", "pair": "EUR_USD"},
                             pairs=["EUR_USD"], sessions=self.SESS)
        self.assertEqual(sm["status"], "INCOMPLETE")
        self.assertEqual(sm["failure"]["pair"], "EUR_USD")
        self.assertEqual(sm["sessions"]["London"]["normal"]["verdict"], "NO VERDICT (INCOMPLETE)")

    def test_events_record_skips_cancellations_and_invalid_entries(self):
        rows = long_signal_rows()
        rows[8] = (1.1003, 1.1006, 1.1002, 1.1004)                       # invalid entry (R <= 0)
        ev = self.res_for(rows)["EUR_USD"]["London"]["events"]
        self.assertEqual({(e["kind"], e["run"]) for e in ev}, {("invalid", "normal"), ("invalid", "double")})
        rows = long_signal_rows()
        del rows[3]
        ev = self.res_for(rows)["EUR_USD"]["London"]["events"]
        self.assertEqual(ev[0]["kind"], "skipped")
        self.assertIn("2015-06-16T07:15:00Z", ev[0]["detail"])

    def test_export_round_trip_is_exact(self):
        import tempfile, os
        closed_rows = long_signal_rows(post=(1.1060, 1.1062, 1.1058, 1.1061))
        closed_rows[9] = (1.1045, 1.1047, 1.1036, 1.1038)
        _, closed, _ = sim(closed_rows, ema_value=1.1030, ema_map={9: 1.1040})
        un_rows = dict(closed_rows)
        for hm in ("20:45", "20:50", "20:55"):
            del un_rows[kk(hm)]
        _, unres, _ = sim(un_rows, ema_value=1.1030, ema_map={9: 1.1040})
        for tr in (closed, unres):
            tr.update(session="London", run="normal", date=D.isoformat(), risk_pips=tr["R"] / 0.0001)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.csv")
            O.write_trades(path, [closed, unres])
            back = O.read_trades(path)
        b_closed = next(x for x in back if x["status"] == "closed")
        b_unres = next(x for x in back if x["status"] == "unresolved")
        self.assertEqual(b_closed["entry_fill"], closed["entry_fill"])     # exact, not rounded
        self.assertEqual(O.trade_r(b_closed), closed["r"])                 # saved trade reproduces reported R
        self.assertEqual(b_closed["r"], closed["r"])
        self.assertEqual([x["fill"] for x in b_unres["exits"]], [x["fill"] for x in unres["exits"]])
        self.assertEqual(b_unres["remaining"], 0.1)
        self.assertIsNone(b_unres["exit_time"])
        self.assertIsNone(b_unres["r"])


class TestExportPrecision(unittest.TestCase):
    def test_full_float_precision_survives_export(self):
        import os, tempfile
        tr = {"session": "London", "run": "normal", "status": "closed", "pair": "EUR_USD", "date": "2015-06-16",
              "dir": 1, "signal_time": at(7), "entry_time": at(8), "entry_fill": 1.1014612345678901,
              "stop0": 1.1005, "R": 0.0009612345678901, "risk_pips": 9.612345678901, "E": 1.1014,
              "entry_cost_r": 0.0631, "exit_slippage_r": 0.0104, "remaining": 0.0,
              "exits": [{"time": at(10), "reason": "partial_1", "share": 0.9, "fill": 1.1037387654321098},
                        {"time": at(12), "reason": "stop", "share": 0.1, "fill": 1.1014512345678901}]}
        tr["exit_time"], tr["r"] = at(12), O.trade_r(tr)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.csv")
            O.write_trades(path, [tr])
            b = O.read_trades(path)[0]
        self.assertEqual([x["fill"] for x in b["exits"]], [1.1037387654321098, 1.1014512345678901])
        self.assertEqual((b["entry_fill"], b["R"]), (tr["entry_fill"], tr["R"]))
        self.assertEqual(O.trade_r(b), tr["r"])


class TestStartupAndExport(unittest.TestCase):
    def args(self, **kw):
        import argparse
        base = {"out": "/tmp/x", "expect_sha": "", "expect_test_sha": "", "require_volume": False}
        base.update(kw)
        return argparse.Namespace(**base)

    def test_refuses_on_hash_mismatch(self):
        import os
        here = os.path.dirname(os.path.abspath(O.__file__))
        problems, _, _ = O.preflight(self.args(expect_sha="0" * 64, expect_test_sha="1" * 64), here)
        self.assertEqual(len(problems), 2)
        good, code_sha, test_sha = O.preflight(self.args(), here)
        self.assertEqual(good, [])
        ok, _, _ = O.preflight(self.args(expect_sha=code_sha, expect_test_sha=test_sha), here)
        self.assertEqual(ok, [])

    def test_refuses_without_persistent_volume(self):
        import os, tempfile
        here = os.path.dirname(os.path.abspath(O.__file__))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAILWAY_VOLUME_MOUNT_PATH", None)
            p, _, _ = O.preflight(self.args(require_volume=True), here)
            self.assertIn("not mounted", p[0])
        with tempfile.TemporaryDirectory() as vol, mock.patch.dict(os.environ, {"RAILWAY_VOLUME_MOUNT_PATH": vol}):
            p, _, _ = O.preflight(self.args(require_volume=True, out="/tmp/elsewhere"), here)
            self.assertIn("not on the persistent volume", p[0])
            used = os.path.join(vol, "run1")
            os.makedirs(used)
            open(os.path.join(used, "f"), "w").close()
            p, _, _ = O.preflight(self.args(require_volume=True, out=used), here)
            self.assertIn("not empty", p[0])
            p, _, _ = O.preflight(self.args(require_volume=True, out=os.path.join(vol, "run2")), here)
            self.assertEqual(p, [])

    def test_main_refuses_before_any_request(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, mock.patch.object(O, "_get", side_effect=AssertionError("request")):
            self.assertEqual(O.main(["--out", d, "--expect-sha", "0" * 64]), 3)

    def test_log_export_round_trip_and_tamper_detection(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.csv"), "w") as fh:
                fh.write("x,y\n" + "1.0,2.0\n" * 5000)
            with open(os.path.join(d, "b.json"), "w") as fh:
                fh.write('{"k": 1}')
            lines = []
            with mock.patch.object(O, "_sleep", lambda s: None):
                O.export_dir(d, emit=lines.append)
            files = O.rebuild_from_lines(lines)
            with open(os.path.join(d, "a.csv"), "rb") as fh:
                self.assertEqual(files["a.csv"], fh.read())
            self.assertEqual(files["b.json"], b'{"k": 1}')
        bad = [ln if not ln.startswith("EXPORT-SHA b.json") else "EXPORT-SHA b.json " + "0" * 64 for ln in lines]
        with self.assertRaises(O.ExportError):
            O.rebuild_from_lines(bad)


class TestDSTReference(unittest.TestCase):
    REF = "b0694398013fa5c2be875aeda5fc51e93c37473d91b0da2cc60a335d99a91264"   # official tz database, 31,200 values

    def test_fixed_dst_matches_official_reference_fingerprint(self):
        import hashlib
        vals = []
        d = date(2009, 12, 1)
        while d <= date(2026, 12, 31):
            for z, h, m in (("LON", 8, 0), ("NY", 8, 0), ("NY", 9, 30), ("NY", 16, 45), ("NY", 17, 0)):
                vals.append(str(O.local_to_epoch(z, d, h, m)))
            d += timedelta(days=1)
        self.assertEqual(len(vals), 31200)
        self.assertEqual(hashlib.sha256(",".join(vals).encode()).hexdigest(), self.REF)

    def test_fixed_dst_refuses_other_dates_and_times(self):
        with self.assertRaises(ValueError):
            O.local_to_epoch("NY", date(2030, 1, 2), 8, 0)
        with self.assertRaises(ValueError):
            O.local_to_epoch("LON", date(2015, 1, 2), 9, 0)


class TestExportVerification(unittest.TestCase):
    """verify_export must require a manifest-complete, checksum-exact reconstruction."""

    def make(self):
        import os, tempfile
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        with open(os.path.join(d, "trades.csv"), "w") as fh:
            fh.write("a,b\n" + "".join(f"{i},{(i * 7919) % 1000003 / 7.0!r}\n" for i in range(4000)))   # varied: many chunks
        with open(os.path.join(d, "summary.json"), "w") as fh:
            fh.write('{"status": "COMPLETE"}')
        O.write_manifest(d)
        lines = []
        with mock.patch.object(O, "_sleep", lambda s: None), mock.patch.object(O, "EXPORT_CHUNK", 200):
            O.export_dir(d, emit=lines.append)
        return d, lines

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_valid_export_round_trip(self):
        import os
        d, lines = self.make()
        files = O.verify_export(lines)
        self.assertEqual(set(files), {"trades.csv", "summary.json", "MANIFEST.sha256"})
        with open(os.path.join(d, "trades.csv"), "rb") as fh:
            self.assertEqual(files["trades.csv"], fh.read())

    def test_missing_whole_file_rejected(self):
        _, lines = self.make()
        lines = [ln for ln in lines if " summary.json " not in ln]
        with self.assertRaises(O.ExportError) as cm:
            O.verify_export(lines)
        self.assertIn("not exported", str(cm.exception))

    def test_missing_manifest_rejected(self):
        _, lines = self.make()
        lines = [ln for ln in lines if " MANIFEST.sha256 " not in ln]
        with self.assertRaises(O.ExportError) as cm:
            O.verify_export(lines)
        self.assertIn("MANIFEST.sha256 missing", str(cm.exception))

    def test_missing_chunk_rejected(self):
        _, lines = self.make()
        chunk_lines = [i for i, ln in enumerate(lines) if ln.startswith("EXPORT trades.csv ")]
        self.assertGreater(len(chunk_lines), 2)
        del lines[chunk_lines[1]]
        with self.assertRaises(O.ExportError) as cm:
            O.verify_export(lines)
        self.assertIn("missing export chunks", str(cm.exception))

    def test_conflicting_duplicate_chunk_rejected_identical_accepted(self):
        _, lines = self.make()
        first = next(ln for ln in lines if ln.startswith("EXPORT trades.csv 1/"))
        O.verify_export(lines + [first])                                   # identical repeat: fine
        head, body = first.rsplit(" ", 1)
        altered = head + " " + ("A" if body[0] != "A" else "B") + body[1:]
        with self.assertRaises(O.ExportError) as cm:
            O.verify_export(lines + [altered])
        self.assertIn("conflicting duplicate chunk", str(cm.exception))

    def test_unexpected_file_rejected(self):
        import os
        d, lines = self.make()
        extra_dir = os.path.join(d, "extra")
        os.makedirs(extra_dir)
        with open(os.path.join(extra_dir, "stray.txt"), "w") as fh:
            fh.write("not in the manifest")
        more = []
        with mock.patch.object(O, "_sleep", lambda s: None):
            O.export_dir(extra_dir, emit=more.append)
        with self.assertRaises(O.ExportError) as cm:
            O.verify_export(lines + more)
        self.assertIn("unexpected files", str(cm.exception))

    def test_manifest_checksum_mismatch_rejected(self):
        _, lines = self.make()
        lines = [ln if not ln.startswith("EXPORT-SHA MANIFEST.sha256") else "EXPORT-SHA MANIFEST.sha256 " + "0" * 64
                 for ln in lines]
        with self.assertRaises(O.ExportError):
            O.verify_export(lines)


class TestSpreadMedian(unittest.TestCase):
    def test_even_count_median_is_mean_of_middle_values(self):
        c = O.Candles()
        t0 = int(datetime(2015, 6, 16, tzinfo=timezone.utc).timestamp())
        for k, spread_pips in enumerate((1.0, 2.0, 3.0, 5.0)):           # middle values 2.0 and 3.0
            v = {f: 1.1 for f in O.FIELDS}
            v["bc"], v["ac"] = 1.1, 1.1 + spread_pips * 0.0001
            c.append(t0 + k * O.STEP, v)
        out = O.spread_diagnostics(c, "EUR_USD")[2015]
        self.assertAlmostEqual(out["median"], 2.5)                      # upper-middle bug would give 3.0
        self.assertEqual(out["n"], 4)


if __name__ == "__main__":
    unittest.main()
