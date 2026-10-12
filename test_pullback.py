"""Offline synthetic regression suite: python -m unittest -v test_pullback.py."""
import ast
from collections import Counter
from dataclasses import replace
from datetime import timedelta, date
import contextlib
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import pullback_test as P


def c(at, o=100., h=100.2, l=99.8, close=100., spread=.02):
    at=P.ts(at)
    mid=(o,h,l,close)
    return P.Candle(at,at+P.M15,mid,tuple(x-spread/2 for x in mid),tuple(x+spread/2 for x in mid))


def d(at='2026-01-05T22:00:00Z',close=100.,sma200=90.,sma5=101.,rsi=5.,atr=1.):
    return P.Daily(P.ts(at),close,sma200,sma5,rsi,atr)


def run(days,candles,start='2026-01-05T22:00:00Z',end='2026-01-16T22:00:00Z',split=None,pair='EUR_USD'):
    a,b=P.ts(start),P.ts(end)
    split=P.ts(split) if split else a+(b-a)*2/3
    return P.simulate_pair(pair,days,iter(candles),a,split,b)


def series(a,b,**kw):
    a,b=P.ts(a),P.ts(b)
    result=[]
    while a<b:
        result.append(c(a,**kw)); a+=P.M15
    return result


class Indicators(unittest.TestCase):
    def rows(self,prices):
        at=P.ts('2025-01-01T22:00Z')
        return [replace(c(at+timedelta(days=i),o=x,h=x+1,l=x-1,close=x),
                        end=at+timedelta(days=i+1)) for i,x in enumerate(prices)]

    def test_rsi_seed_and_wilder_recursion(self):
        x=P.indicators(self.rows([100,102,101,103]))
        self.assertIsNone(x[1].rsi)
        self.assertAlmostEqual(x[2].rsi,100-100/(1+1/.5))
        self.assertAlmostEqual(x[3].rsi,100-100/(1+1.5/.25))

    def test_rsi_zero_cases(self):
        for prices,expected in [([100,100,100],50),([100,101,102],100),([100,99,98],0)]:
            self.assertEqual(P.indicators(self.rows(prices))[-1].rsi,expected)

    def test_atr_excludes_first_candle_seeds_twenty_ranges(self):
        rows=self.rows([100]*22)
        rows[0]=replace(rows[0],mid=(100,1000,1,100))
        rows[21]=replace(rows[21],mid=(100,103,97,100))
        x=P.indicators(rows)
        self.assertIsNone(x[19].atr)
        self.assertEqual(x[20].atr,2)
        self.assertAlmostEqual(x[21].atr,2.2)

    def test_atr_uses_previous_close_gaps(self):
        x=P.indicators(self.rows([100]+[110]*20))
        self.assertAlmostEqual(x[20].atr,(11+19*2)/20)

    def test_warmup_and_sma_current_close(self):
        rows=self.rows([100]*199+[102])
        x=P.indicators(rows)
        self.assertFalse(x[198].eligible)
        self.assertTrue(x[199].eligible)
        self.assertAlmostEqual(x[199].sma200,100.01)
        self.assertAlmostEqual(x[199].sma5,100.4)

    def test_equality_and_strict_thresholds(self):
        self.assertEqual(d(sma200=100).signal(),0)
        self.assertEqual(d(rsi=10).signal(),0)
        self.assertEqual(d(close=80,rsi=90).signal(),0)
        self.assertEqual(d(rsi=9.999).signal(),1)
        self.assertEqual(d(close=80,rsi=90.001).signal(),-1)
        self.assertEqual(replace(d(),atr=None).signal(),0)


class Timing(unittest.TestCase):
    def test_friday_to_monday_and_dst_mismatch(self):
        self.assertEqual(P.next_london_open(P.ts('2026-01-09T22:00Z')),P.ts('2026-01-12T08:00Z'))
        self.assertEqual(P.next_london_open(P.ts('2026-03-20T21:00Z')),P.ts('2026-03-23T08:00Z'))
        self.assertEqual(P.next_london_open(P.ts('2026-07-06T21:00Z')),P.ts('2026-07-07T07:00Z'))

    def test_signal_before_common_start_can_enter_inside_window(self):
        x=run([d()], [c('2026-01-06T08:00Z'),c('2026-01-06T09:00Z')],
              start='2026-01-06T02:00Z')
        self.assertEqual(len(x['records']['baseline']),1)
        self.assertEqual(x['records']['baseline'][0]['signal'],P.ts('2026-01-05T22:00Z'))

    def test_friday_signal_enters_monday(self):
        x=run([d('2026-01-09T22:00Z')],[c('2026-01-12T08:00Z'),c('2026-01-12T09:00Z')],start='2026-01-09T22:00Z')
        self.assertEqual(x['records']['baseline'][0]['entry'],P.ts('2026-01-12T08:00Z'))

    def test_market_closed_versus_gap_and_no_carry(self):
        for candles,reason in [([c('2026-01-06T11:00Z'),c('2026-01-07T08:00Z')],'market_closed'),
                               ([c('2026-01-06T07:45Z'),c('2026-01-06T08:15Z'),c('2026-01-07T08:00Z')],'data_gap')]:
            x=run([d()],candles)
            self.assertEqual(x['records']['baseline'],[])
            self.assertEqual(x['skipped'],{reason:1})

    def test_market_presence_after_eight_still_classifies_gap(self):
        x=run([d()],[c('2026-01-06T09:00Z')])
        self.assertEqual(x['skipped'],{'data_gap':1})

    def test_no_entry_from_signal_without_warmup(self):
        x=run([replace(d(),sma200=None)],[c('2026-01-06T08:00Z')])
        self.assertFalse(x['records']['baseline'])

    def test_delayed_recovery_exit(self):
        days=[d(),d('2026-01-06T22:00Z',sma5=99)]
        candles=[c('2026-01-06T08:00Z'),c('2026-01-07T08:15Z'),c('2026-01-07T09:00Z')]
        r=run(days,candles)['records']['baseline'][0]
        self.assertEqual(r['reason'],'delayed')
        self.assertEqual(r['trigger'],'recovery')
        self.assertEqual(r['exit'],P.ts('2026-01-07T08:15Z'))

    def test_recovery_checked_only_after_entry(self):
        # Signal candle already above SMA5 must not schedule an entry-morning exit.
        x=run([d(sma5=99)],[c('2026-01-06T08:00Z'),c('2026-01-06T09:00Z')])
        r=x['records']['baseline'][0]
        self.assertEqual(r['reason'],'end')
        self.assertIsNone(r['scheduled'])

    def test_five_daily_completions_count_entry_day_as_one(self):
        pos=P.open_positions('EUR_USD',d(),c('2026-01-06T08:00Z'))[0]
        pos.daily(d())
        self.assertEqual(pos.held,0)
        for at in ['2026-01-06T22:00Z','2026-01-07T22:00Z','2026-01-08T22:00Z','2026-01-09T22:00Z']:
            pos.daily(d(at))
        self.assertEqual(pos.held,4);self.assertIsNone(pos.scheduled)
        pos.daily(d('2026-01-12T22:00Z'))
        self.assertEqual(pos.held,5)
        self.assertEqual(pos.scheduled,P.ts('2026-01-13T08:00Z'))
        self.assertEqual(pos.trigger,'five_day')

    def test_five_day_exit_executes(self):
        times=['2026-01-05T22:00Z','2026-01-06T22:00Z','2026-01-07T22:00Z',
               '2026-01-08T22:00Z','2026-01-09T22:00Z','2026-01-12T22:00Z']
        candles=[c('2026-01-06T08:00Z'),c('2026-01-13T08:00Z'),c('2026-01-13T09:00Z')]
        r=run([d(t) for t in times],candles)['records']['baseline'][0]
        self.assertEqual(r['reason'],'five_day')
        self.assertEqual(r['held'],5)

    def test_no_same_morning_flip_and_signals_discarded(self):
        days=[d(),d('2026-01-06T22:00Z',sma5=99),d('2026-01-07T22:00Z')]
        candles=[c('2026-01-06T08:00Z'),c('2026-01-07T08:00Z'),c('2026-01-07T08:15Z'),
                 c('2026-01-08T08:00Z'),c('2026-01-08T09:00Z')]
        rows=run(days,candles)['records']['baseline']
        self.assertEqual([r['entry'] for r in rows],[P.ts('2026-01-06T08:00Z'),P.ts('2026-01-08T08:00Z')])

    def test_discarded_signal_not_resurrected_after_stop(self):
        candles=[c('2026-01-06T08:00Z'),c('2026-01-07T06:00Z',l=97),c('2026-01-07T08:00Z')]
        rows=run([d(),d('2026-01-06T22:00Z')],candles)['records']['baseline']
        self.assertEqual(len(rows),1)

    def test_new_signal_must_be_strictly_after_close(self):
        # Stop occurs during candle ending at the daily close: conservative
        # timestamp equality prevents recycling that day's signal.
        candles=[c('2026-01-06T08:00Z'),c('2026-01-06T21:45Z',l=97),c('2026-01-07T08:00Z')]
        rows=run([d(),d('2026-01-06T22:00Z')],candles)['records']['baseline']
        self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]['approximate'])


class Execution(unittest.TestCase):
    def test_entry_open_stop_both_directions(self):
        for signal in [d(),d(close=80,sma200=90,rsi=95)]:
            candles=[c('2026-01-06T08:00Z',spread=6),c('2026-01-06T09:00Z')]
            row=run([signal],candles)['records']['baseline'][0]
            self.assertEqual(row['reason'],'stop_gap')
            self.assertEqual(row['exit'],row['entry'])
            self.assertLess(row['r'],-1)

    def test_scheduled_exit_precedes_later_candle_low(self):
        candles=[c('2026-01-06T08:00Z'),c('2026-01-07T08:00Z',l=80),c('2026-01-07T09:00Z')]
        r=run([d(),d('2026-01-06T22:00Z',sma5=99)],candles)['records']['baseline'][0]
        self.assertEqual(r['reason'],'recovery')
        self.assertAlmostEqual(r['exit_price'],99.99)

    def test_exit_open_gap_precedes_scheduled_exit_both_sides(self):
        for direction in (1,-1):
            signal=d() if direction==1 else d(close=80,rsi=95)
            recovery=d('2026-01-06T22:00Z',sma5=99 if direction==1 else 101)
            price=90 if direction==1 else 110
            candles=[c('2026-01-06T08:00Z'),c('2026-01-07T08:00Z',o=price,h=price+1,l=price-1,close=price),c('2026-01-07T09:00Z')]
            r=run([signal,recovery],candles)['records']['baseline'][0]
            self.assertEqual(r['reason'],'stop_gap')
            self.assertAlmostEqual(r['exit_price'],price-.01*direction)

    def test_weekend_gap_fills_open(self):
        candles=[c('2026-01-09T08:00Z'),c('2026-01-09T21:45Z'),
                 c('2026-01-11T22:00Z',o=95,h=96,l=94,close=95),c('2026-01-12T08:00Z')]
        r=run([d('2026-01-08T22:00Z')],candles,start='2026-01-08T22:00Z')['records']['baseline'][0]
        self.assertEqual(r['reason'],'stop_gap');self.assertAlmostEqual(r['exit_price'],94.99)

    def test_endpoint_truncation_and_forced_close(self):
        end=P.ts('2026-01-06T09:07Z')
        candles=[c('2026-01-06T08:00Z'),c('2026-01-06T08:45Z'),c('2026-01-06T09:00Z',l=1)]
        r=run([d()],candles,end=end)['records']['baseline'][0]
        self.assertEqual(r['reason'],'end')
        self.assertEqual(r['exit'],P.ts('2026-01-06T09:00Z'))

    def test_stress_freezes_ids_risk_and_stop(self):
        candles=[c('2026-01-06T08:00Z'),c('2026-01-06T09:00Z')]
        result=run([d()],candles)
        b,s=[result['records'][v][0] for v in P.VARIANTS]
        for key in ('id','entry','signal','direction','stop','risk','baseline_entry','units_per_cash_R'):
            self.assertEqual(b[key],s[key])
        self.assertLess(s['r'],b['r'])
        self.assertAlmostEqual(s['risk'],2)
        self.assertNotEqual(abs(s['entry_price']-s['stop']),s['risk'])

    def test_stress_earlier_stop_does_not_create_extra_entry(self):
        # Baseline stop=98.01; widened low=98.00 touches, actual low=98.01
        # use slightly larger low so only stress hits.
        candles=[c('2026-01-06T08:00Z'),c('2026-01-06T12:00Z',l=98.025),c('2026-01-07T08:00Z'),c('2026-01-08T08:00Z')]
        result=run([d(),d('2026-01-06T22:00Z')],candles)
        self.assertEqual(len(result['records']['baseline']),1)
        self.assertEqual(len(result['records']['stress']),1)
        self.assertLess(result['records']['stress'][0]['exit'],result['records']['baseline'][0]['exit'])

    def test_spread_cost_attribution_close_vs_open_no_double_deduction(self):
        first=c('2026-01-06T08:00Z')
        last=c('2026-01-06T09:00Z')
        last=replace(last,bid=last.bid[:3]+(99.8,),ask=last.ask[:3]+(100.2,))
        r=run([d()],[first,last])['records']['baseline'][0]
        self.assertAlmostEqual(r['spread_r'],(.01+.2)/2)
        self.assertAlmostEqual(r['r'],(99.8-100.01-.00002)/2)

    def test_usd_base_units_frozen_at_baseline_entry(self):
        result=run([d()],[c('2026-01-06T08:00Z'),c('2026-01-06T09:00Z')],pair='USD_JPY')
        for v in P.VARIANTS:
            self.assertAlmostEqual(result['records'][v][0]['units_per_cash_R'],100.01/2)


class CalendarPortfolio(unittest.TestCase):
    def test_rollovers_both_dst_changes_and_exact_boundary(self):
        for a,b in [('2026-10-30T22:00Z','2026-11-02T23:00Z'),('2027-03-12T22:30Z','2027-03-15T21:30Z')]:
            self.assertEqual(P.nights(P.ts(a),P.ts(b)),3)
        self.assertEqual(P.nights(P.ts('2026-07-01T20:59Z'),P.ts('2026-07-01T21:00Z')),1)
        self.assertEqual(P.nights(P.ts('2026-07-01T21:00Z'),P.ts('2026-07-01T21:00Z')),0)

    def test_full_date_coverage_midnight_and_partial_end(self):
        a,b=P.ts('2026-01-05T05:00Z'),P.ts('2026-01-06T05:00Z')
        self.assertEqual(P.full_weekdays(a,b),{date(2026,1,5)})
        self.assertEqual(P.full_weekdays(a+timedelta(seconds=1),b),set())
        self.assertEqual(P.full_weekdays(a,b-timedelta(seconds=1)),set())

    def test_coverage_and_weekend_gap_exclusion(self):
        self.assertEqual(P.missing_slots(P.ts('2026-01-09T22:00Z'),P.ts('2026-01-11T22:00Z')),0)
        self.assertEqual(P.missing_slots(P.ts('2026-01-05T08:00Z'),P.ts('2026-01-05T09:00Z')),4)

    def test_weekly_exit_side_quotes_swap_and_usd_count(self):
        start,split,end=P.ts('2026-01-08T22:00Z'),P.ts('2026-01-10T10:00Z'),P.ts('2026-01-13T22:00Z')
        candles=[c('2026-01-09T08:00Z'),c('2026-01-09T21:45Z',close=101),c('2026-01-12T08:00Z')]
        results=[P.simulate_pair(pair,[d(start)],iter(candles),start,split,end) for pair in ('EUR_USD','USD_JPY')]
        at=P.ts('2026-01-09T22:00Z')
        mark=results[0]['snapshots']['baseline'][at]
        self.assertAlmostEqual(mark['open_r'],(100.99-100.01-.00002-.00005)/2)
        self.assertEqual(mark['usd'],-1)
        self.assertEqual(results[1]['snapshots']['baseline'][at]['usd'],1)
        self.assertEqual(results[0]['snapshots']['baseline'][split]['stale'],1)
        for variant in P.VARIANTS:
            weeks=P.portfolio(results,variant,start,split,end)
            self.assertAlmostEqual(sum(x['r'] for x in weeks),sum(r['r'] for p in results for r in p['records'][variant]))
            self.assertTrue(any(x['end']==split and x['period']=='evaluation' for x in weeks))
            self.assertTrue(any(x['start']==split and x['period']=='second' for x in weeks))
            self.assertTrue(all(x['usd_risk_count']==0 for x in weeks))

    def test_mark_inside_candle_uses_only_prior_close(self):
        start=P.ts('2026-01-05T22:00Z');split=P.ts('2026-01-06T08:07Z');end=P.ts('2026-01-06T10:00Z')
        candles=[c('2026-01-06T07:45Z'),c('2026-01-06T08:00Z',h=201,close=200),c('2026-01-06T09:00Z')]
        r=P.simulate_pair('EUR_USD',[d()],iter(candles),start,split,end)
        mark=r['snapshots']['baseline'][split]
        self.assertAlmostEqual(mark['open_r'],(99.99-100.01-.00002)/2)
        self.assertAlmostEqual(mark['max_age_hours'],7/60)

    def test_endpoint_quote_staleness_does_not_keep_closed_trade_open(self):
        r=run([d()],[c('2026-01-06T08:00Z')])
        for state in r['snapshots']['baseline'].values():
            if state['count']==0:self.assertEqual(state['open_r'],0)
        self.assertEqual(r['records']['baseline'][0]['exit'],P.ts('2026-01-06T08:15Z'))

    def test_long_gap_flag(self):
        r=run([d()],[c('2026-01-06T08:00Z'),c('2026-01-12T08:00Z')])
        self.assertGreater(r['long_trade_gaps'],0)
        self.assertTrue(r['records']['baseline'][0]['long_gap'])


class SafetyAndReports(unittest.TestCase):
    def test_incomplete_blocks_pass_both_periods(self):
        split=P.ts('2026-01-06T00:00Z')
        rows=[dict(entry=split+timedelta(days=1 if i%2 else -1),r=1) for i in range(300)]
        self.assertEqual(P.verdict(rows,split,[]),'PASS')
        self.assertEqual(P.verdict(rows,split,['thin data']),'INCOMPLETE')
        self.assertEqual(P.verdict(rows[:299],split,[]),'FAIL')

    def test_candle_only_client_allowlist(self):
        with patch.dict('os.environ',{'OANDA_API_KEY':'synthetic'}): client=P.CandleClient()
        class Response:
            def __enter__(self): return io.StringIO('{"candles": []}')
            def __exit__(self,*args): pass
        with patch.object(P,'urlopen',return_value=Response()) as fake:
            client.page('EUR_USD','M15',count=1)
            req=fake.call_args.args[0]
            self.assertTrue(req.full_url.startswith('https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles?'))
            self.assertEqual(req.get_method(),'GET')
            with self.assertRaises(ValueError): client.page('../accounts','M15')
        tree=ast.parse(Path(P.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):self.assertNotEqual(node.module,'bot')

    def test_token_alias_and_missing_token(self):
        with patch.dict('os.environ',{'OANDA_TOKEN':'synthetic'},clear=True):
            self.assertIsInstance(P.CandleClient(),P.CandleClient)
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(ValueError):P.CandleClient()

    def test_fetch_stream_bounded_and_end_truncated(self):
        with patch.dict('os.environ',{'OANDA_API_KEY':'synthetic'}):client=P.CandleClient()
        a,b=P.ts('2026-01-01T00:00Z'),P.ts('2026-02-01T00:07Z')
        requests=[]
        def page(pair,gran,**kw):
            left,right=P.ts(kw['from']),P.ts(kw['to'])
            requests.append((left,right))
            return [c(left),c(right)]
        with patch.object(client,'page',side_effect=page):
            rows=list(client.stream('EUR_USD',a,b))
        self.assertTrue(all(right-left<=timedelta(days=14) for left,right in requests))
        self.assertTrue(all(x.end<=b for x in rows))
        self.assertEqual(len({x.t for x in rows}),len(rows))

    def test_daily_parser_dst_and_incomplete_candles(self):
        raw={'time':'2026-03-08T21:00:00Z','complete':True,
             **{k:dict(o='100',h='101',l='99',c='100') for k in ('mid','bid','ask')}}
        rows=P.parse([raw,dict(raw,complete=False)],'D')
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0].end,P.ts('2026-03-09T21:00Z'))

    def test_discovery_failure_reports_incomplete_without_exception_text(self):
        class Api:
            def daily(self,*args):raise RuntimeError('private-token-do-not-print')
        output=io.StringIO()
        with contextlib.redirect_stdout(output):status=P.execute(Api(),['EUR_USD'],P.FROZEN_END)
        self.assertEqual(status,'INCOMPLETE')
        self.assertNotIn('private-token',output.getvalue())
        self.assertEqual(output.getvalue().count('VERDICT'),2)

    def test_main_requires_exact_endpoint_without_network(self):
        for env in ({},{'PB_END':'2026-10-08T21:00:00Z'}):
            with patch.dict('os.environ',env,clear=True),patch.object(P,'CandleClient') as api,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(P.main(),'INCOMPLETE')
                api.assert_not_called()

    def test_complete_fake_execution_and_report_reconciliation(self):
        # Exercises availability discovery, indicator warm-up, pair processing,
        # summary formatting and all verdicts, with no network access.
        start=P.ts('2025-01-01T22:00Z')
        daily=[replace(c(start+timedelta(days=i),o=100+i*.01,h=101+i*.01,l=99+i*.01,close=100+i*.01),
                       end=start+timedelta(days=i+1)) for i in range(230)]
        end=daily[-1].end
        class Api:
            def daily(self,pair,e):return daily
            def first_m15(self,pair,a,e):return a
            def stream(self,pair,a,e):
                while a+P.M15<=e:
                    yield c(a)
                    a+=timedelta(hours=6)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):status=P.execute(Api(),['EUR_USD'],end)
        self.assertEqual(status,'FAIL')
        text=output.getvalue()
        self.assertIn('REPORT baseline evaluation',text)
        self.assertIn('REPORT stress second',text)
        self.assertIn('OVERALL',text)
        self.assertNotIn('TRADE ',text)
        for line in text.splitlines():json.loads(line[line.index('{'):] if '{' in line and '[' not in line[:line.index('{')] else line[line.index('['):])


if __name__=='__main__':
    unittest.main()
