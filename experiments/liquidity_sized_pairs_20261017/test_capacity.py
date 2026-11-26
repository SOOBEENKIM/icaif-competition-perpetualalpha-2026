import copy,unittest
import numpy as np
from experiments.conditional_completion_pairs_20261017.test_completion import ConditionalTests as Fixture
from experiments.conditional_completion_pairs_20261017.portfolio import replay as previous_replay
from .policy import capacity_volume,prepare,proposals
from .portfolio import replay

class CapacityTests(unittest.TestCase):
    def setUp(self):
        f=Fixture();f.setUp();self.d,self.de,self.cfg,self.p=f.d,f.de,f.cfg,f.p
        self.cfg['entry_capacity']={'window_bars':12};self.t=self.d['times']
    def run_arm(self,engine=replay):
        return engine(self.d,self.de,None,self.cfg,self.p,'pair_dense_hourlyhedge_direct_24','base',int(self.t[0]),int(self.t[-1]))
    def test_one_spike_cannot_set_capacity_and_current_volume_is_still_a_bound(self):
        v=np.ones((15,2))*10;tr=np.ones_like(v);v[12,0]=100000.;v[13,1]=2.
        cap=capacity_volume(v,tr,12)
        self.assertTrue(np.isnan(cap[:11]).all());self.assertEqual(cap[12,0],10.)
        self.assertEqual(cap[13,1],2.)
    def test_inactive_observations_stay_in_median(self):
        v=np.zeros((12,1));tr=np.zeros_like(v);v[-1]=10000;tr[-1]=100
        self.assertEqual(capacity_volume(v,tr,12)[-1,0],0.)
    def test_missing_observation_suppresses_instead_of_filling_with_zero(self):
        v=np.ones((24,1));tr=v.copy();v[8,0]=np.nan
        cap=capacity_volume(v,tr,12)
        self.assertTrue(np.isnan(cap[11:20]).all());self.assertEqual(cap[20,0],1.)
    def test_future_volume_and_trades_cannot_change_earlier_capacity(self):
        rng=np.random.default_rng(17);v=rng.lognormal(size=(100,3));tr=np.ones_like(v)
        a=capacity_volume(v,tr,12);v[60:]=1e9;tr[60:]=0
        np.testing.assert_array_equal(a[:60],capacity_volume(v,tr,12)[:60])
    def test_original_market_volume_and_trades_are_not_rewritten(self):
        before=self.d['par_volume'].copy();tr=self.d['par_trades'].copy();new=prepare(self.d,self.cfg)
        self.assertIs(new['par_volume'],self.d['par_volume'])
        np.testing.assert_array_equal(self.d['par_volume'],before);np.testing.assert_array_equal(self.d['par_trades'],tr)
        self.assertNotIn('entry_capacity_volume',self.d)
    def test_disabled_policy_is_exact_parent(self):
        self.cfg.pop('entry_capacity');a=self.run_arm();b=self.run_arm(previous_replay)
        for key in ['fills','events','metrics','snapshots']:self.assertEqual(a[key],b[key],key)
    def test_recomputed_pair_size_and_fill_capacity_are_separate(self):
        self.d['par_volume'][:]=10.;i=int(np.searchsorted(self.t,self.t[0]+6*3600))
        self.d['par_volume'][i:i+2,:]=1e6
        r=self.run_arm();intent=next(e for e in r['events'] if e['event']=='pair_intent')
        self.assertEqual(intent['time'],int(self.t[i]))
        np.testing.assert_array_equal(intent['entry_capacity_volume_units'],[10.,10.])
        self.assertTrue(all(abs(q)<=1.+1e-10 for q in intent['target']))
        orders=[o for o in r['orders'] if o['decision_time']==intent['time']]
        self.assertTrue(orders);self.assertTrue(all(o['capacity']==100000. for o in orders))
        self.assertTrue(all(o['filled_qty']<=1.+1e-10 for o in orders))
    def test_unknown_history_blocks_pair_without_an_invented_minimum_lot(self):
        d=prepare(self.d,self.cfg);p=proposals(0,d,self.de,None,24,self.cfg,self.p,self.cfg['scenarios']['base'],
            np.zeros(10),1e6,np.zeros(10),[1],'pair_dense_hourlyhedge_direct_24')
        self.assertEqual(p,[])
    def test_zero_arrival_liquidity_still_forces_real_failure(self):
        self.d['par_volume'][:]=10000.;i=int(np.searchsorted(self.t,self.t[0]+6*3600))
        self.d['par_volume'][i+1,0]=0.
        r=self.run_arm();first_close=next(e for e in r['events'] if e['event']=='closed')
        self.assertTrue(first_close['aborted']);self.assertFalse(first_close['balanced'])
        fills=[f for f in r['owner_fills']['pair'] if f['time']<=first_close['time']]
        self.assertEqual(len(fills),2);self.assertTrue(all(f['fee']>0 for f in fills))
    def test_future_data_changes_preserve_entire_account_prefix(self):
        a=self.run_arm();cut=int(self.t[90]);self.d['par_volume'][91:]=1e8;self.d['par_close'][91:]*=2
        b=self.run_arm()
        for key in ['events','fills','snapshots']:
            self.assertEqual([r for r in a[key] if r['time']<=cut],[r for r in b[key] if r['time']<=cut],key)

if __name__=='__main__':unittest.main()
