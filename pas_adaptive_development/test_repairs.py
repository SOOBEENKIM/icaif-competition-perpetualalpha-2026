import copy
import gzip
import json
import unittest
from pathlib import Path

import numpy as np
from . import study
from .all_market_ema import extend
from .risk_trim import trim_targets, cap_base_addition, soft_limits


class UniverseContracts(unittest.TestCase):
    def data(self, n=2400):
        from replay_lab.ema_development.lifecycle import Policy
        from dataclasses import asdict
        t=np.arange(n)*3600
        slope=np.array([1,-1,1,-1,1,-1,1,-1,1,-1])*.0003
        close=100*np.exp(np.arange(n)[:,None]*slope)
        d=dict(known=t,close=close,high=close*1.002,low=close*.998)
        return d,dict(policy=asdict(Policy())),np.full((n,10),.2)

    def test_directional_rule_reaches_all_assets_without_identity_bias(self):
        d,p,v=self.data();a=extend(d,p,v,True)
        self.assertTrue(np.all(a['signal'][-1]==np.array([1,-1]*5)))
        order=[8,5,0,9,2,7,3,1,6,4]
        permuted={k:x[:,order] if x.ndim==2 else x for k,x in d.items()}
        b=extend(permuted,p,v[:,order],True)
        np.testing.assert_array_equal(a['target_weight'][:,order],b['target_weight'])

    def test_future_or_removed_rows_cannot_change_earlier_targets(self):
        d,p,v=self.data();a=extend(d,p,v,True)
        prefix={k:x[:2100] for k,x in d.items()}
        b=extend(prefix,p,v[:2100],True)
        np.testing.assert_array_equal(a['target_weight'][:2100],b['target_weight'])
        for k in ['close','high','low']:d[k][2100:]*=10
        c=extend(d,p,v,True)
        np.testing.assert_array_equal(a['target_weight'][:2100],c['target_weight'][:2100])


class RiskRepairContracts(unittest.TestCase):
    def test_shared_capacity_is_reserved_and_reductions_are_retained(self):
        from competition_live.profile import load_profile
        from replay_lab.common_comparison.engine import risk_breaches
        p=load_profile()['trading'];price=np.full(10,100.);sigma=np.eye(10)*1e-8
        base=np.zeros(10);base[0]=10;base[2]=10
        pair=np.zeros(10);pair[0]=50;pair[1]=-50
        desired=base.copy();desired[0]=30;desired[2]=0
        target=cap_base_addition(base,desired,pair,price,50000.,sigma,p)
        self.assertEqual(target[2],0)
        self.assertLess(target[0],desired[0])
        self.assertGreaterEqual(target[0],base[0])
        self.assertFalse(risk_breaches(target+pair,50000.,price,sigma,soft_limits(p['risk'])))

    def test_paid_reduction_preserves_episode_loss_and_stop_anchor(self):
        from competition_live.profile import load_profile
        from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
        from experiments.paid_rehedge_pairs_20261017.policy import anchor_stop
        p=load_profile()['trading'];a=SharedAccount(50000.,p['symbols'])
        for j,q in [(0,70.),(1,-70.)]:
            a.physical_fill(j,q,100.,9.,1)
            a.owner_fill('pair',j,q,100.,9.,1,'external_fill')
        owner=a.owners['pair'];mark=np.full(10,100.)
        pair=dict(ids=[0,1],risk_quantities=[70.,-70.],risk_entry_prices=[100.,100.])
        anchor_stop(pair,owner,mark);origin=pair['stop_price_origin']
        mark[0]=90.;mark[1]=110.
        targets=trim_targets(a.owners['base'].qty,owner.qty,mark,a.reconcile(mark),
                             np.eye(10)*1e-8,p,dict(pair_balance_tolerance=.05))
        self.assertIsNotNone(targets);old_fee=a.physical.fees
        for j in [0,1]:
            q=targets[1][j]-owner.qty[j]
            self.assertLess(q*owner.qty[j],0.)
            a.physical_fill(j,q,mark[j],9.,2)
            a.owner_fill('pair',j,q,mark[j],9.,2,'external_fill')
        a.reconcile(mark);anchor_stop(pair,owner,mark)
        self.assertEqual(pair['stop_price_origin'],origin)
        self.assertAlmostEqual(owner.equity(mark)+owner.fees-owner.funding-origin,-1400.)
        self.assertGreater(a.physical.fees,old_fee)

    def test_one_leg_trim_failure_requests_paid_unwind_in_actual_engine(self):
        from competition_live.profile import BUNDLE,load_profile
        from .guarded_portfolio import replay
        from .sleeve_budget import RATIO
        root=study.ROOT/'reports/pas_guarded_allocation_20261021'
        with gzip.open(root/'reference/guarded_both_continuous_base.json.gz','rt') as f:
            original=json.load(f)
        event=next(e for e in original['events'] if e['event']=='risk_trim_requested')
        d=dict(np.load(BUNDLE/'data/reference.npz'));z=dict(np.load(study.OUT/'targets.npz'))
        ids=np.searchsorted(z['times'],d['times']);d['first']=z['size_only'][ids]
        state=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
        idx=int(np.searchsorted(d['times'],event['time']+3600))
        legs=np.flatnonzero(event['pair_target'])
        d['par_volume']=d['par_volume'].copy();d['par_volume'][idx,legs[1]]=0.
        profile=load_profile();cfg=copy.deepcopy(profile['configs']['reference'])
        cfg['conditional_completion']={'risk_aversion':0.};cfg['pair_max_gross']*=RATIO
        r=replay(d,state,None,cfg,profile['trading'],profile['candidate'],'base',original['start'],int(d['times'][idx+4]))
        failed=[e for e in r['events'] if e['event']=='risk_trim_partial_unwind']
        self.assertTrue(failed)
        self.assertEqual(failed[0]['time'],int(d['times'][idx]))
        unwind=[f for f in r['owner_fills']['pair'] if f['time']>failed[0]['time'] and f['reduce_only']]
        self.assertTrue(unwind)
        self.assertTrue(any(f['fee']>0 for f in unwind))
        self.assertLess(r['metrics']['maximum_accounting_error'],1e-6)


if __name__=='__main__':unittest.main()
