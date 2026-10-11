"""Synthetic, offline regression tests: python -m unittest -v test_hypo.py."""
import ast
import contextlib
import io
import json
from dataclasses import replace
from datetime import datetime, timedelta
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import hypo_test as h


def candle(t='2024-01-08T14:00:00Z', o=100, high=100.2, low=99.8, c=100, spread=0):
    t = h.stamp(t)
    mid = (o, high, low, c)
    return h.Candle(t, t+timedelta(minutes=15), mid,
                    tuple(x-spread/2 for x in mid), tuple(x+spread/2 for x in mid))


def plan(c, direction=1, **kwargs):
    return h.Plan('EUR_USD', c.t, direction, 1, 2, **kwargs)


class ExitTests(unittest.TestCase):
    def test_target_gap_precedes_intrabar_stop_both_directions(self):
        for d in (1, -1):
            entry = candle()
            pos = h.Position(plan(entry, d), entry, 'base')
            gap = candle(entry.end, o=100+3*d, high=104, low=96)
            result = pos.step(gap)
            self.assertEqual(result['reason'], 'target_gap')
            self.assertEqual(result['exit_price'], 100+2*d)
            self.assertFalse(result['approximate'])
            self.assertEqual(result['ambiguous'], 0)
            old = h.Position(plan(entry, d), entry, 'old').step(gap)
            self.assertEqual(old['reason'], 'stop')
            self.assertEqual(old['ambiguous'], 1)
            self.assertEqual(h.changed_count([result], [old]), 1)

    def test_stop_gap_both_directions(self):
        for d in (1, -1):
            entry = candle()
            gap = candle(entry.end, o=100-2*d, high=103, low=97)
            result = h.Position(plan(entry, d), entry, 'base').step(gap)
            self.assertEqual(result['reason'], 'stop_gap')
            self.assertEqual(result['exit_price'], 100-2*d)
            self.assertFalse(result['approximate'])

    def test_both_touched_is_stop_and_approximate(self):
        c = candle(high=103, low=98)
        r = h.Position(plan(c), c, 'base').step(c)
        self.assertEqual(r['reason'], 'stop')
        self.assertEqual(r['ambiguous'], 1)
        self.assertTrue(r['approximate'])
        self.assertEqual(r['exit'], h.iso(c.t))

    def test_sixteenth_candle_close_timestamp(self):
        c = candle()
        candles = [replace(c, t=c.t+timedelta(minutes=15*i), end=c.end+timedelta(minutes=15*i)) for i in range(20)]
        rows, _ = h.simulate(candles, [plan(c, bars=16)], c.t, candles[-1].end, 'shock')
        r = rows['base'][0]
        self.assertEqual(r['reason'], 'time')
        self.assertEqual(r['exit'], h.iso(candles[15].end))
        self.assertEqual(r['holding_hours'], 4)
        self.assertFalse(r['approximate'])

    def test_truncate_before_simulation(self):
        c = candle()
        future = candle(c.end, high=500, low=0)
        end = c.end + timedelta(minutes=5)
        records, _ = h.simulate([c, future], [plan(c)], c.t, end, 'shock')
        r = records['base'][0]
        self.assertEqual(r['reason'], 'end')
        self.assertTrue(r['end_forced'])
        self.assertEqual(r['exit'], h.iso(c.end))
        self.assertEqual(r['exit_price'], 100)
        self.assertEqual(list(h.truncate([c, future], end)), [c])

    def test_spread_not_double_counted_and_costs_once(self):
        c = candle(spread=.02)
        for d in (1, -1):
            r = h.Position(plan(c, d), c, 'base').force_end(c)
            self.assertAlmostEqual(r['r'], -.02002)
            self.assertAlmostEqual(r['cost_r'], .02002)
            self.assertAlmostEqual(r['slippage_r'], .00002)

    def test_stress_uses_actual_filter_same_entries(self):
        c = candle(spread=.04)
        rows, skips = h.simulate([c], [plan(c)], c.t, c.end, 'shock')
        self.assertEqual(skips, 0)
        self.assertEqual([len(v) for v in rows.values()], [1, 1, 1])
        self.assertAlmostEqual(rows['stress'][0]['cost_r'], .08004)
        rejected = candle(spread=.06)
        rows, skips = h.simulate([rejected], [plan(rejected)], rejected.t, rejected.end, 'shock')
        self.assertEqual(skips, 1)
        self.assertTrue(all(not v for v in rows.values()))

    def test_monthly_exit_at_open_before_new_entry(self):
        c = candle()
        nxt = candle(c.t+timedelta(days=30), o=100.1)
        p = plan(c, expiry=nxt.t, month='2024-01')
        p2 = plan(nxt, month='2024-02')
        rows, _ = h.simulate([c, nxt], [p, p2], c.t, nxt.end, 'csm')
        self.assertEqual(len(rows['base']), 2)
        self.assertEqual(rows['base'][0]['reason'], 'month')
        self.assertEqual(rows['base'][0]['exit'], rows['base'][1]['entry'])
        self.assertEqual(rows['base'][0]['exit_price'], nxt.bid[0])

    def test_rollover_dst_and_calendar_weekends(self):
        # Friday after rollover to Monday after rollover = Sat/Sun/Mon: 3.
        for a, b in [('2024-03-08T23:00:00Z', '2024-03-11T22:00:00Z'),
                     ('2024-11-01T22:00:00Z', '2024-11-04T23:00:00Z')]:
            self.assertEqual(h.rollovers(h.stamp(a), h.stamp(b)), 3)
        for before, at in [('2024-07-01T20:59:59Z', '2024-07-01T21:00:00Z'),
                           ('2024-01-01T21:59:59Z', '2024-01-01T22:00:00Z')]:
            self.assertEqual(h.rollovers(h.stamp(before), h.stamp(at)), 1)
            self.assertEqual(h.rollovers(h.stamp(at), h.stamp(at)), 0)

    def test_one_entry_per_ny_date_not_utc_date(self):
        first = candle('2024-01-08T23:45:00Z')
        same_ny = candle('2024-01-09T00:15:00Z')
        next_ny = candle('2024-01-09T05:00:00Z')
        candles = [first, same_ny, next_ny]
        rows, _ = h.simulate(candles, [plan(c, bars=1) for c in candles], first.t, next_ny.end, 'shock')
        self.assertEqual([r['entry'] for r in rows['base']], [h.iso(first.t), h.iso(next_ny.t)])

    def test_only_one_open_position(self):
        first = candle()
        second = candle(first.end)
        rows, _ = h.simulate([first, second], [plan(first, bars=16), plan(second, bars=16)], first.t, second.end, 'shock')
        self.assertEqual(len(rows['base']), 1)


class SignalTests(unittest.TestCase):
    def test_usd_base_flip(self):
        self.assertEqual(h.currency_direction('USD_JPY', True), -1)
        self.assertEqual(h.currency_direction('USD_CHF', False), 1)
        self.assertEqual(h.currency_direction('EUR_USD', True), 1)
        self.assertEqual(h.currency_direction('GBP_USD', False), -1)

    def test_csm_63_returns_ranking_and_no_lookahead(self):
        dates = []
        d = h.stamp('2023-09-01T21:00:00Z')
        while d < h.stamp('2024-02-03T00:00:00Z'):
            if d.astimezone(h.NY).weekday() < 5:
                dates.append(d)
            d += timedelta(days=1)
        history = {}
        for pair in h.CSM_PAIRS:
            # USD_JPY rising means JPY is weakest; EUR is strongest.
            rate = .004 if pair in ('EUR_USD', 'USD_JPY') else .0001
            price, rows = 100., []
            for i, end in enumerate(dates):
                price *= math.exp(rate + (.0002 if i % 2 else -.0002))
                c = candle(end-timedelta(days=1), o=price, high=price+1, low=price-1, c=price)
                rows.append(replace(c, end=end))
            history[pair] = rows
        start = h.stamp('2024-01-01T00:00:00Z')
        end = h.stamp('2024-02-02T00:00:00Z')
        plans = h.csm_plans(history, start, end)
        january = [p for p in plans if p.month == '2024-01']
        self.assertEqual({p.pair for p in january}, {'EUR_USD', 'USD_JPY'})
        self.assertTrue(all(p.direction == 1 for p in january))
        self.assertEqual(january[0].when, h.stamp('2024-01-01T14:00:00Z'))
        # Alter only data unavailable at the January entry.
        changed = {p: [replace(c, mid=(1., 900., .01, 500.)) if c.end > january[0].when else c
                       for c in rows] for p, rows in history.items()}
        again = [p for p in h.csm_plans(changed, start, end) if p.month == '2024-01']
        self.assertEqual(january, again)

    def test_wilder_seed_and_recursion(self):
        candles = [candle(high=101, low=99) for _ in range(20)]
        candles.append(candle(high=103, low=97))
        atr = h.wilder(candles)
        self.assertTrue(all(x is None for x in atr[:19]))
        self.assertEqual(atr[19], 2)
        self.assertAlmostEqual(atr[20], 2.2)

    def test_shock_uses_prior_atr_and_close_session(self):
        start = h.stamp('2024-01-08T12:00:00Z')  # closes 08:00 NY
        warm = [replace(candle(start-timedelta(hours=20-i), high=100.5, low=99.5),
                        end=start-timedelta(hours=19-i)) for i in range(20)]
        sig = replace(candle(start, o=100, high=102, low=100, c=102), end=start+timedelta(hours=1))
        rows = warm+[sig]
        plans = h.shock_plans('EUR_USD', rows, start, sig.end+timedelta(hours=1))
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].risk, 1)
        self.assertEqual(plans[0].when, sig.end)
        self.assertEqual(plans[0].direction, 1)
        late = replace(sig, t=h.stamp('2024-01-08T17:00:00Z'), end=h.stamp('2024-01-08T18:00:00Z'))
        self.assertEqual(h.shock_plans('EUR_USD', warm+[late], start, late.end+timedelta(hours=1)), [])

    def test_daily_close_alignment(self):
        for start, expected in [('2024-03-10T21:00:00Z', '2024-03-11T21:00:00Z'),
                                ('2024-11-03T22:00:00Z', '2024-11-04T22:00:00Z')]:
            self.assertEqual(h.close_time(h.stamp(start), 'D'), h.stamp(expected))

    def test_report_month_sum_and_drawdown(self):
        c = candle()
        r = h.Position(plan(c, month='2024-01'), c, 'base').force_end(c)
        rows = [dict(r, r=2), dict(r, r=-1), dict(r, r=-2, month='2024-02', exit='2024-02-01T00:00:00Z')]
        report = h.summarize(rows, 'csm', 5)
        self.assertEqual(report['monthly_R'], {'2024-01': 1, '2024-02': -2})
        self.assertEqual(report['monthly']['months_traded'], 2)
        self.assertEqual(report['monthly']['max_drawdown_R'], 2)
        self.assertEqual(report['spread_filter_skips'], 5)


class SafetyTests(unittest.TestCase):
    def test_no_bot_import_or_trading_endpoints(self):
        source = Path(h.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'bot':
                self.assertLessEqual({x.name for x in node.names}, {'Oanda', 'fetch_range', 'pip'})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ('post', 'put', 'delete', 'patch'))
        self.assertNotIn('/v3/accounts', source)

    def test_offline_main_both_strategies(self):
        class FakeClient:
            def history(self, pair, gran, end):
                rows, price = [], 100.
                start = h.stamp('2023-01-01T00:00:00Z')
                step = timedelta(days=1) if gran == 'D' else timedelta(hours=1)
                t, i = start, 0
                while t + step <= end:
                    i += 1
                    drift = .2 if pair == 'EUR_USD' else -.2
                    closing = price + drift * (1 if i % 2 else 1.1)
                    c = candle(t, o=price, high=max(price, closing)+.1,
                               low=min(price, closing)-.1, c=closing, spread=.0001)
                    rows.append(replace(c, end=t+step))
                    price, t = closing, t+step
                return rows
            def first_m15(self, start, end, pair):
                return start
            def stream(self, pair, start, end):
                # Sparse fixture exercises delayed entries and period truncation.
                t = start
                while t + timedelta(minutes=15) <= end:
                    yield candle(t, spread=.0001)
                    t += timedelta(days=1)
        for strategy in ('csm', 'shock'):
            output = io.StringIO()
            with patch.object(h, 'CandleClient', FakeClient), patch.dict('os.environ',
                    {'HYPO': strategy, 'HYPO_END': '2023-07-01T00:00:00Z'}), contextlib.redirect_stdout(output):
                h.main()
            text = output.getvalue()
            self.assertIn('REPORT evaluation base ', text)
            self.assertIn('REPORT holdout stress ', text)
            self.assertIn('REPORT pooled old ', text)
            self.assertEqual(text.count('PASS_BAR '), 2)
            if strategy == 'csm':
                self.assertIn('previously inspected (related trend-following tests)', text)
                self.assertIn('TRADE ', text)

    def test_client_url_allowlist_and_stream_end(self):
        class Response:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {'candles': []}
        with patch.dict('os.environ', {'OANDA_API_KEY': 'synthetic', 'OANDA_ENV': 'practice'}):
            client = h.CandleClient()
        with patch.object(client.session, 'get', return_value=Response()) as get:
            client.page('EUR_USD', 'D', count=1)
            self.assertEqual(get.call_args.args[0], 'https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles')
            with self.assertRaises(ValueError):
                client.page('../accounts', 'D', count=1)
        c = candle()
        future = candle(c.end, high=500)
        with patch.object(client, 'page', return_value=[c, future]):
            self.assertEqual(list(client.stream('EUR_USD', c.t, c.end)), [c])


if __name__ == '__main__':
    unittest.main()
