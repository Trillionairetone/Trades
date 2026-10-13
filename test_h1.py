import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import urllib.parse
import h1_test as h


class ExecutionTests(unittest.TestCase):
    def test_long_stop_before_target_and_gap(self):
        # Both touched: conservatively choose stop; gap fills beyond stop.
        x=np.array([1.0]); hi=np.array([1.2]); lo=np.array([.8])
        k,px,why=h.exit_trade(1,0,1,.1,x,hi,lo,x,x,hi,lo,x)
        self.assertEqual(why,'stop'); self.assertAlmostEqual(px,.9)
        k,px,why=h.exit_trade(1,0,1,.1,lo,hi,lo,x,x,hi,lo,x)
        self.assertAlmostEqual(px,.8)

    def test_short_stop_uses_ask_and_gap(self):
        x=np.array([1.0]); hi=np.array([1.2]); lo=np.array([.8])
        _,px,why=h.exit_trade(-1,0,1,.1,x,hi,lo,x,hi,hi,lo,x)
        self.assertEqual(why,'stop'); self.assertAlmostEqual(px,1.2)

    def test_unclosed_trade_not_a_realized_win(self):
        x=np.array([1.0]); high=np.array([1.01]); low=np.array([.99])
        _,_,why=h.exit_trade(1,0,1,.1,x,high,low,x,x,high,low,x)
        self.assertEqual(why,'open_at_end')
        stats=h.metrics([{'reason':why,'net_r':.2,'exit':'2026-01-02'}])
        self.assertEqual(stats['closed_trades'],0)
        self.assertEqual(stats['open_at_end'],1)
        self.assertAlmostEqual(stats['open_mark_to_market_r'],.2)

    def test_financing_rollover_uses_new_york_dst(self):
        self.assertEqual(h.rollovers(pd.Timestamp('2026-07-01T20:59Z'),pd.Timestamp('2026-07-01T21:01Z')),1)
        self.assertEqual(h.rollovers(pd.Timestamp('2026-01-01T21:59Z'),pd.Timestamp('2026-01-01T22:01Z')),1)

    def fixture(self):
        d=pd.DataFrame({'time':pd.date_range('2026-01-01',periods=43,freq='h',tz='UTC'),
                        'o':1.,'h':1.005,'l':.995,'c':1.})
        d.loc[40,'h']=1.02
        d.loc[41,'l']=.94
        for k in 'ohlc':
            d['b'+k]=d[k]-.0001
            d['a'+k]=d[k]+.0001
        return d

    def test_next_open_and_cost_stress(self):
        d=self.fixture(); start=d.time.iloc[0]; end=d.time.iloc[-1]+pd.Timedelta(hours=1)
        with patch.object(h,'calc_atr',side_effect=lambda d,n:pd.Series(.01,index=d.index)), patch.object(h,'echo_direction',return_value=-1):
            baseline=h.simulate(d,'EUR_USD',start,end)
            stressed=h.simulate(d,'EUR_USD',start,end,cost_mult=2)
            # Signal at 40 is not executable without a subsequent candle.
            self.assertEqual(h.simulate(d.iloc[:41],'EUR_USD',start,end),[])
        self.assertEqual(len(baseline),1)
        self.assertEqual(baseline[0]['entry'],d.time.iloc[41].isoformat())
        self.assertAlmostEqual(baseline[0]['entry_price'],d.bo.iloc[41])
        self.assertEqual(baseline[0]['reason'],'target')
        self.assertLess(stressed[0]['net_r'],baseline[0]['net_r'])

    def test_echo_does_not_read_future(self):
        rng=np.random.default_rng(47); c=1+np.cumsum(rng.normal(0,.001,220))
        before=h.echo_direction(c,180); c[181:]+=100
        self.assertEqual(h.echo_direction(c,180),before)

    def test_bad_timeframe_and_pair_rejected_before_network(self):
        with self.assertRaises(ValueError): h.download('BAD',None,None,'H1')
        with self.assertRaises(ValueError): h.download('EUR_USD',None,None,'M15')


class PaginationTests(unittest.TestCase):
    def test_two_page_download_has_no_duplicate_or_gap(self):
        """Mock OANDA: 'from' returns the candle CONTAINING that time (as the real API did),
        5,000 candles per page. The download must request both pages and return every
        candle exactly once, with nothing skipped at the page boundary."""
        import io, json
        for gran, step in (('H1', pd.Timedelta(hours=1)), ('H4', pd.Timedelta(hours=4))):
            with self.subTest(gran=gran):
                start = pd.Timestamp('2010-01-04', tz='UTC')
                full = [start + i * step for i in range(7000)]
                end = full[-1] + step
                requests_seen = []

                class Opener:
                    def open(self, req, timeout=30):
                        q = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
                        frm = pd.Timestamp(q['from'][0])
                        requests_seen.append(frm)
                        k = next(i for i, t in enumerate(full) if t + step > frm)
                        page = [{'time': t.strftime('%Y-%m-%dT%H:%M:%S.000000000Z'), 'complete': True,
                                 'mid': {c: '1.1' for c in 'ohlc'}, 'bid': {c: '1.0999' for c in 'ohlc'},
                                 'ask': {c: '1.1001' for c in 'ohlc'}} for t in full[k:k + 5000]]
                        return io.BytesIO(json.dumps({'candles': page}).encode())

                with patch.dict('os.environ', {'OANDA_TOKEN': 'test-token'}), \
                     patch('urllib.request.build_opener', return_value=Opener()), patch('time.sleep'):
                    d = h.download('EUR_USD', start, end, gran)
                t = pd.to_datetime(d['time'], utc=True)
                self.assertEqual(len(requests_seen), 2)
                self.assertFalse(t.duplicated().any())
                self.assertEqual(list(t), full)
                h.validate(d, full[2000], end, gran)   # real validator, with 2,000 warm-up candles


if __name__=='__main__': unittest.main()
