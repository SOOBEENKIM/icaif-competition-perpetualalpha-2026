import copy,unittest
import numpy as np
from .replay import simulate

class HistoricalReplayTests(unittest.TestCase):
    def case(self,n=12):
        start=1792411200000;t=start+np.arange(1,n+1)*60000;p=np.full((n,1),100.)
        cfg={'symbols':['BTC'],'start_ms':start,'cutoff_ms':int(t[-1])}
        d=dict(times=t,close=p.copy(),high=p.copy(),low=p.copy(),active=np.ones_like(p,bool),volume=np.full_like(p,1000),age_ms=np.zeros_like(p),funding_events=[])
        events=[dict(time=start,weights=[.01],units='weights'),dict(time=start+240000,weights=[-.01],units='weights')]
        return cfg,d,events
    def test_future_price_changes_do_not_change_earlier_fills(self):
        c,d,e=self.case();a=simulate(c,d,e,impact_bps=10,participation=.01)
        changed=copy.deepcopy(d)
        for key in ('close','high','low'):changed[key][9:]*=2
        b=simulate(c,changed,e,impact_bps=10,participation=.01)
        self.assertEqual([x for x in a['fills'] if x['time']<d['times'][9]],[x for x in b['fills'] if x['time']<d['times'][9]])
    def test_no_activity_cannot_fill_and_participation_is_respected(self):
        c,d,e=self.case();d['active'][:]=False
        a=simulate(c,d,e,impact_bps=10,participation=.01);self.assertEqual(a['fills'],[])
        d['active'][:]=True;d['volume'][:]=50
        a=simulate(c,d,e,impact_bps=10,participation=.01)
        self.assertTrue(a['fills']);self.assertTrue(all(abs(x['signed_qty'])<=.5+1e-8 for x in a['fills']))
        self.assertTrue(all(x['decision_time']<x['time'] for x in a['fills']))
    def test_confirmed_flat_precedes_opposite_entry(self):
        c,d,e=self.case();r=simulate(c,d,e,impact_bps=10,participation=.01)
        shorts=[x for x in r['fills'] if x['signed_qty']<0 and not x['reduce_only']]
        self.assertTrue(shorts)
        first=shorts[0];prior=[x for x in r['fills'] if x['time']<first['time']][-1]
        self.assertAlmostEqual(prior['quantity_after'],0.)
        self.assertLess(r['metrics']['accounting_error'],1e-7)
if __name__=='__main__':unittest.main()
