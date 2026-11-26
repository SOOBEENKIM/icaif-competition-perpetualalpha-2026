import unittest
import numpy as np
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.common_comparison.models import covariance, timestamp
from replay_lab.macro_development.engine import simulate as existing_simulate
from replay_lab.pairs_rl.verify import independent_accounting
from .data import DAY, HOUR, context, daily_bars, load
from .policy import PaperAgent, choose
from .engine import simulate


class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.z, cls.protocol = load()

    def test_daily_bars_exclude_partial_edges(self):
        d = daily_bars(self.z)
        for n, end in enumerate(d['known']):
            ids = np.flatnonzero((self.z['known'] > end-DAY) & (self.z['known'] <= end))
            self.assertEqual(len(ids), 24)
            np.testing.assert_array_equal(d['open'][n], self.z['open'][ids[0]])
            np.testing.assert_array_equal(d['close'][n], self.z['close'][ids[-1]])
            np.testing.assert_array_equal(d['high'][n], self.z['high'][ids].max(0))
            np.testing.assert_array_equal(d['low'][n], self.z['low'][ids].min(0))
        self.assertLessEqual(d['known'][-1], self.z['known'][-1])

    def test_future_mutation_cannot_change_context(self):
        origin = timestamp('2026-07-01T00:00:00Z')
        altered = {k: v.copy() for k, v in self.z.items()}
        mask = altered['known'] > origin
        for k in ['open', 'high', 'low', 'close', 'volume']:
            altered[k][mask] *= 100
        for step, lookback, horizon in [(DAY,90,10), (HOUR,80,12)]:
            before = daily_bars(self.z) if step == DAY else self.z
            after = daily_bars(altered) if step == DAY else altered
            for a, b in zip(context(before, origin, 0, lookback, horizon, step),
                            context(after, origin, 0, lookback, horizon, step)):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_prefix_only_can_forecast_past_dataset_end(self):
        origin = timestamp('2026-07-01T00:00:00Z')
        prefix = {k: v[self.z['known'] <= origin] for k,v in self.z.items()}
        full = context(daily_bars(self.z), origin, 0, 90, 10, DAY)
        cut = context(daily_bars(prefix), origin, 0, 90, 10, DAY)
        for a,b in zip(full,cut):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_stale_origin_rejected(self):
        with self.assertRaises(ValueError):
            context(daily_bars(self.z), timestamp('2026-07-01T01:00:00Z'), 0, 90, 10, DAY)


class PolicyTests(unittest.TestCase):
    def test_initial_top_two_even_if_all_negative(self):
        s = -np.arange(1,11,dtype=float)
        selected, sell, buy = choose(s, set(), {}, 10*DAY)
        self.assertEqual(selected, {0,1})
        self.assertEqual(buy, {0,1})
        self.assertFalse(sell)

    def test_minimum_holding_blocks_replacement(self):
        s = np.arange(10,dtype=float)
        selected,sell,buy = choose(s,{0,1},{0:7*DAY,1:7*DAY},10*DAY)
        self.assertEqual(selected,{0,1})
        self.assertFalse(sell or buy)

    def test_only_one_replacement_after_five_days(self):
        selected,sell,buy = choose(np.arange(10.),{0,1},{0:5*DAY,1:5*DAY},10*DAY)
        self.assertEqual(selected,{1,9})
        self.assertEqual(sell,{0})
        self.assertEqual(buy,{9})

    def test_ties_stable_by_symbol_order(self):
        self.assertEqual(choose(np.zeros(10),set(),{},0)[0],{0,1})

    def test_restart_and_duplicate_decision(self):
        _,_,p = load()
        t = 100*DAY
        a = PaperAgent({t:np.arange(10.)})
        args=(t,np.ones(10)*100,np.zeros(10),1e6,np.eye(10)*1e-6,np.ones(10)*1e6,np.ones(10),p)
        result = a.decide(*args)
        self.assertIsNotNone(result)
        b=PaperAgent({t:np.arange(10.)},saved=a.snapshot())
        self.assertEqual(a.snapshot(),b.snapshot())
        self.assertIsNone(b.decide(*args))
        with self.assertRaises(ValueError):
            PaperAgent({},name='other',saved=a.snapshot())

    def test_invalid_input_does_not_consume_cursor(self):
        _,_,p=load()
        a=PaperAgent({100*DAY:np.arange(10.)})
        s=a.snapshot()
        self.assertIsNone(a.decide(100*DAY,np.ones(10),np.zeros(10),1e6,np.eye(10)*np.nan,np.ones(10),np.ones(10),p))
        self.assertEqual(s,a.snapshot())

    def test_unfilled_target_does_not_create_holding_age(self):
        _,_,p=load(); t=100*DAY
        a=PaperAgent({t:np.arange(10.)})
        a.decide(t,np.ones(10),np.zeros(10),1e6,np.eye(10)*1e-6,np.ones(10),np.ones(10),p)
        self.assertFalse(a.entries)
        qty=np.zeros(10);qty[9]=1
        a.decide(t+HOUR,np.ones(10),qty,1e6,np.eye(10)*1e-6,np.ones(10),np.ones(10),p)
        self.assertEqual(a.entries,{9:t+HOUR})


class LedgerParity(unittest.TestCase):
    def test_stateful_adapter_matches_existing_event_order(self):
        m,z,p=load();start=timestamp('2026-09-01T00:00:00Z');end=start+7*DAY
        sigma=np.full((len(m.times),10,10),np.nan)
        for i in range(m.exact(start),m.exact(end)+1):sigma[i]=covariance(m,i,p['risk'])[1]
        w=np.zeros_like(m.close);w[:,0]=.025;w[:,2]=-.025
        class Replay:
            name='ledger_parity'
            def __init__(self):self.controller=Controller();self.selections=[]
            def decide(self,t,prices,qty,equity,s,volume,trades,protocol,halted=False):
                return self.controller.propose(t,prices,qty,equity,w[m.exact(t)],s,volume,trades,protocol,buffer=.2,halted=halted)
            def snapshot(self):return {}
        for scenario in ['base','stress']:
            old=existing_simulate(m,{'target_weight':w},'ledger_parity',p,scenario,start,end,buffer=.2,sigmas=sigma)
            new=simulate(m,Replay(),p,scenario,start,end,sigma)
            self.assertEqual(old['metrics'],new['metrics'])
            self.assertEqual(old['fills'],new['fills'])
            self.assertEqual(old['daily'],new['daily'])
            self.assertTrue(independent_accounting(new,m)['passed'])


if __name__ == '__main__':
    unittest.main()
