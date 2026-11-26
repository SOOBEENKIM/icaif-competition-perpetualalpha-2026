import json
import unittest
from pathlib import Path
import numpy as np

from .signals import build,formation,pair_vector,relative_momentum,channel_breakout
from .research import BlendController
from experiments.cost_horizon_20261016.research import data

CONFIG=json.loads(Path(__file__).with_name('protocol.json').read_text())


def sample(n=2300):
    rng=np.random.default_rng(97);x=11+np.cumsum(rng.normal(0,.002,n));residual=np.zeros(n)
    for i in range(1,n):residual[i]=.98*residual[i-1]+rng.normal(0,.005)
    log=np.column_stack([x,1.1*x-4+residual,5+np.cumsum(rng.normal(0,.003,n)),
                         4+np.cumsum(rng.normal(0,.002,(n,7)),axis=0)])
    return dict(close=np.exp(log),market_times=np.arange(n)*3600,funding=np.zeros((n,10)))


class SignalTests(unittest.TestCase):
    def test_future_tail_cannot_change_any_signal(self):
        a=sample();full,_=build(a,CONFIG);cut=1900
        truncated,_=build({k:v[:cut].copy() for k,v in a.items()},CONFIG)
        corrupted={k:v.copy() for k,v in a.items()};corrupted['close'][cut:]*=100
        disturbed,_=build(corrupted,CONFIG)
        for name in full:
            np.testing.assert_array_equal(full[name][:cut],truncated[name])
            np.testing.assert_array_equal(full[name][:cut],disturbed[name][:cut])

    def test_formation_excludes_current_and_future(self):
        a=sample();log=np.log(a['close']);fit=formation(log,1600,CONFIG['pair'])
        log[1600:]*=2;other=formation(log,1600,CONFIG['pair'])
        self.assertEqual(fit,other);self.assertEqual(fit['last_source_index'],1599)

    def test_pair_hedge_and_cost_units(self):
        for beta in (.5,1.,2.):
            v=pair_vector(beta,1.,10)
            self.assertAlmostEqual(abs(v).sum(),1.)
            self.assertAlmostEqual(v[0]+beta*v[1],0.)
            self.assertAlmostEqual(2*.0019*abs(v).sum(),.0038)
            self.assertTrue(np.all(v[2:]==0))

    def test_no_fabricated_formation_on_flat_prices(self):
        log=np.full((2000,10),5.)
        self.assertIsNone(formation(log,1600,CONFIG['pair']))

    def test_pair_entries_cover_declared_two_leg_cost(self):
        _,diag=build(sample(),CONFIG)
        events=[e for name in ('pair_ou','pair_kalman') for e in diag[name]['events'] if e['action']=='enter']
        self.assertGreater(len(events),0)
        for e in events:self.assertGreater(e['predicted_convergence_per_gross'],e['required_edge'])

    def test_relative_momentum_is_dollar_neutral_and_persistent(self):
        a=sample();w,d=relative_momentum(a,CONFIG)
        np.testing.assert_allclose(w.sum(1),0,atol=0)
        times=[e['time'] for e in d['events']]
        self.assertTrue(np.all(np.diff(times)>=CONFIG['relative_momentum']['minimum_hold_hours']*3600))
        self.assertTrue(np.all(w[:,3:]==0))

    def test_breakout_excludes_current_price_from_channel(self):
        a=sample(800);a['close'][:]=100.;a['close'][672:]=101.
        w,_=channel_breakout(a,CONFIG)
        self.assertTrue(np.all(w[:672]==0));self.assertTrue(np.all(w[672]>0))

    def test_shared_account_ensemble_nets_opposing_experts(self):
        _,a,p,m=data();t=next(t for t in m.times if t%21600==0 and t>m.times[200]);i=m.exact(t)
        base=np.zeros_like(m.close);expert=base.copy();base[:,0]=.04;expert[:,0]=-.12
        net=.75*base+.25*expert
        c=BlendController(m.times,np.ones_like(base)*.5,base,net)
        args=(t,m.close[i],np.zeros(10),1e6,net[i],a['sigma'][i],m.volume[i],m.trades[i],p)
        action=c.propose(*args)
        self.assertTrue(np.all(np.array(action['target_qty'])==0))
        self.assertIsNone(c.propose(*args))


if __name__=='__main__':unittest.main()
