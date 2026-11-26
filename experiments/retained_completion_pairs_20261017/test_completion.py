import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
from experiments.retained_sizing_pairs_20261017.test_sizing import RetainedSizingTests as Fixture
from experiments.retained_sizing_pairs_20261017.portfolio import replay as previous_replay
from experiments.shared_execution_pairs_20261017.test_account import AccountTests
from .portfolio import replay
from .policy import reconcile_adjustment, completion_plan


class RetainedCompletionTests(unittest.TestCase):
    def setUp(self):
        self.f=Fixture();self.f.setUp()
        self.d,self.de,self.cfg,self.p=self.f.d,self.f.de,self.f.cfg,self.f.p
        self.i,self.t,self.change=self.f.i,self.f.t,self.f.change

    def direct_partial(self):
        pair,a=self.f.direct();pair['retained_completion_enabled']=True
        plan=self.f.plan(pair,a);pair.update(rehedging=True,rehedge_plan=plan)
        j=pair['ids'][0];z=plan['delta'][0]
        a.physical_fill(j,z,99.9,9,self.change+300)
        a.owner_fill('pair',j,z,99.9,9,self.change+300,'external_fill')
        return pair,a,plan

    def reconcile(self,pair,a,i=None,known=True):
        i=self.i+1 if i is None else i
        return reconcile_adjustment(pair,a.owners['pair'],i,self.d,self.de,self.cfg,self.p,
            self.cfg['scenarios']['base'],a.physical.qty,a.physical.equity(self.d['par_close'][i]),all_outcomes_observed=known)

    def execute_remaining(self,pair,a,fraction=1.):
        plan=pair['rehedge_plan']
        for j,z in zip(pair['ids'],plan['delta']):
            if abs(z)<1e-10:continue
            a.physical_fill(j,z*fraction,100.1,9,self.change+600)
            a.owner_fill('pair',j,z*fraction,100.1,9,self.change+600,'external_fill')

    def run_arm(self,arm='pair_dense_utilitycomplete64_bias_24',engine=replay):
        return engine(self.d,self.de,None,self.cfg,self.p,arm,'base',int(self.t[0]),int(self.t[-1]))

    def test_v1_and_hourly_controls_reproduce_exactly(self):
        for arm in ['first_strategy','second_strategy','first_dense_renewhedge_bias_24','pair_dense_utility64_bias_24']:
            a=self.run_arm(arm);b=self.run_arm(arm,previous_replay)
            for key in ['fills','events','snapshots','metrics','decisions']:
                def encode(value):return json.dumps(value,sort_keys=True,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
                self.assertEqual(encode(a[key]),encode(b[key]),(arm,key))

    def test_completion_uses_only_remaining_reductions_and_preserves_source(self):
        pair,a,old=self.direct_partial();origin=pair['stop_price_origin'];fees=a.physical.fees
        result=self.reconcile(pair,a);new=pair['rehedge_plan']
        self.assertEqual(result['outcome'],'completion_pending');self.assertNotIn('renewal_maturity',pair)
        self.assertEqual(new['delta'],[0.,2.5]);self.assertEqual(new['target'],old['target'])
        for key in ['terminal_prices','new_maturity','new_source','hard_holding_deadline']:
            self.assertEqual(new[key],old[key],key)
        self.assertEqual(pair['stop_price_origin'],origin);self.assertEqual(a.physical.fees,fees)
        self.assertGreaterEqual(new['minimum_advantage_at_limits'],new['buffer_cash']-1e-8)
        self.assertGreaterEqual(new['minimum_utility_advantage_at_limits'],new['utility_buffer_cash']-1e-8)
        self.assertEqual(set(new['entry_limits']),{'0'})

    def test_success_requires_paid_final_fill_before_renewal(self):
        pair,a,old=self.direct_partial();self.reconcile(pair,a)
        before=a.physical.fees;self.execute_remaining(pair,a)
        result=self.reconcile(pair,a,self.i+2)
        self.assertEqual(result['outcome'],'renewed');self.assertEqual(result['retained_completion_attempt'],1)
        self.assertEqual(pair['renewal_maturity'],old['new_maturity']);self.assertGreater(a.physical.fees,before)
        np.testing.assert_array_equal(a.owners['pair'].qty[pair['ids']],old['target'])
        a.reconcile(self.d['par_close'][self.i+2])

    def test_zero_completion_fill_unwinds_existing_imbalance(self):
        pair,a,_=self.direct_partial();self.reconcile(pair,a);qty=a.physical.qty.copy()
        result=self.reconcile(pair,a,self.i+2)
        self.assertEqual(result['reason'],'retained_completion_zero_fill');self.assertEqual(result['outcome'],'unwind')
        self.assertTrue(pair['closing']);self.assertNotIn('renewal_maturity',pair)
        np.testing.assert_array_equal(a.physical.qty,qty)

    def test_second_partial_fill_gets_no_third_attempt(self):
        pair,a,_=self.direct_partial();self.reconcile(pair,a);self.execute_remaining(pair,a,.2)
        result=self.reconcile(pair,a,self.i+2)
        self.assertEqual(result['outcome'],'unwind');self.assertEqual(result['retained_completion_attempt'],1)
        self.assertTrue(pair['closing']);self.assertNotIn('rehedge_plan',pair)

    def test_unknown_initial_result_never_creates_completion(self):
        pair,a,old=self.direct_partial();result=self.reconcile(pair,a,known=False)
        self.assertEqual(result['outcome'],'awaiting_reconciliation');self.assertEqual(pair['rehedge_plan'],old)

    def test_unknown_completion_keeps_same_pending_plan(self):
        pair,a,_=self.direct_partial();self.reconcile(pair,a);plan=copy.deepcopy(pair['rehedge_plan'])
        result=self.reconcile(pair,a,self.i+2,known=False)
        self.assertEqual(result['outcome'],'awaiting_reconciliation');self.assertEqual(pair['rehedge_plan'],plan)
        self.assertNotIn('renewal_maturity',pair)

    def test_late_known_completion_never_extends_its_deadline(self):
        pair,a,_=self.direct_partial();self.reconcile(pair,a);self.execute_remaining(pair,a)
        result=self.reconcile(pair,a,self.i+3)
        self.assertEqual(result['outcome'],'unwind');self.assertEqual(result['reason'],'retained_completion_deadline_elapsed')
        self.assertTrue(pair['closing']);self.assertTrue(np.any(a.owners['pair'].qty))

    def test_exhausted_remaining_edge_rejects_completion_and_keeps_actual_fill(self):
        # The short has already moved below its fixed terminal forecast. There
        # is no remaining convergence gain to justify retaining it.
        pair,a,_=self.direct_partial();qty=a.physical.qty.copy();self.d['par_close'][self.i+1,0]=95.
        result=self.reconcile(pair,a)
        self.assertEqual(result['outcome'],'unwind');self.assertTrue(pair['closing'])
        np.testing.assert_array_equal(a.physical.qty,qty);self.assertGreater(a.physical.fees,0)

    def test_cannot_extend_original_exit_to_complete(self):
        pair,a,old=self.direct_partial();pair['has_rehedged']=True
        old['old_maturity']=int(self.t[self.i+2])
        result=completion_plan(pair,old,a.owners['pair'],self.i+1,self.d,self.cfg,self.p,
             self.cfg['scenarios']['base'],a.physical.qty,9999.)
        self.assertEqual(result['reason'],'completion_would_miss_original_deadline')

    def test_no_completion_for_exposure_increasing_resize(self):
        pair,a,_=self.direct_partial()
        # This intentionally invalidated originating plan must never authorize
        # an additional reduction packet under the reduction-only extension.
        pair['rehedge_plan']['old_qty']=[1.,-1.]
        result=self.reconcile(pair,a)
        self.assertEqual(result['outcome'],'unwind');self.assertNotEqual(result['event'],'retained_completion_pending')

    def test_received_book_partial_then_missing_completion_blocks_duplicate(self):
        pair,a=self.f.direct();pair['retained_completion_enabled']=True
        plan=self.f.plan(pair,a);pair.update(rehedging=True,rehedge_plan=plan)
        target=a.owners['pair'].qty.copy();target[pair['ids']]=plan['target']
        sc=dict(self.cfg['scenarios']['base'],depth_fraction=1.,latency_ms=1000)
        def packet(at,plan):
            return a.seal(dict(decision_time=at,source_close=at,source_available=at,source_prices=self.d['par_close'][self.i].tolist(),
                risk_forced=False,owner_targets={'base':np.zeros(10).tolist(),'pair':target.tolist()},
                owner_limits={'pair':plan['entry_limits']}),self.p,sc)
        limits=dict(book_max_age_ms=15000,context_max_age_ms=3000,future_exit_observation_grace_ms=5000)
        book=AccountTests().book
        streams={'ETH':book(at=self.change*1000+1000,size=10.),'BTC':book(at=self.change*1000+1000,size=.5)}
        a.execute_books(packet(self.change,plan),streams,self.d['par_close'][self.i],self.d['sigma'][self.i],self.p,sc,limits)
        first=self.reconcile(pair,a);self.assertEqual(first['outcome'],'completion_pending')
        pending=packet(self.change+300,pair['rehedge_plan'])
        a.execute_books(pending,{},self.d['par_close'][self.i+1],self.d['sigma'][self.i+1],self.p,sc,limits)
        result=self.reconcile(pair,a,self.i+2,known=not a.unknown)
        self.assertEqual(result['outcome'],'awaiting_reconciliation')
        with self.assertRaises(RuntimeError):a.execute_books(pending,{},self.d['par_close'][self.i+1],self.d['sigma'][self.i+1],self.p,sc,limits)

    def test_full_portfolio_completion_waits_one_step_and_conserves_cash(self):
        self.d['par_volume'][self.i,0]=0.
        self.d['funding'][:,1]=.00001;self.d['known_hourly_funding'][:,1]=.00001
        run=self.run_arm();pending=next(e for e in run['events'] if e['event']=='retained_completion_pending')
        done=next(e for e in run['events'] if e.get('outcome')=='renewed' and e.get('retained_completion_attempt')==1)
        self.assertEqual(done['time'],pending['time']+300)
        m=run['metrics'];self.assertEqual(m['retained_completion_attempts'],1)
        self.assertAlmostEqual(m['net_pnl'],m['price_move_pnl']-m['execution_cost']-m['fees']+m['funding'],places=7)
        self.assertLess(m['maximum_ownership_error'],1e-7)

    def test_competing_base_target_cannot_strand_partial_inventory(self):
        self.d['par_volume'][self.i,0]=0.;self.p['risk']['daily_volatility']=.1
        at=self.change+300;target=np.zeros(10);target[0]=-14.
        class PendingBase:
            def propose(self,time,*args,**kwargs):
                return dict(target_qty=target.tolist()) if time==at else None
        with patch('experiments.retained_completion_pairs_20261017.portfolio.Controller',PendingBase):
            run=self.run_arm('first_dense_utilitycomplete64_bias_24')
        pending=next(e for e in run['events'] if e['event']=='retained_completion_pending')
        rejected=next(e for e in run['events'] if e['event']=='adjustment_packet_rejected' and e['time']==pending['time'])
        closed=next(e for e in run['events'] if e['event']=='closed' and e['time']>rejected['time'])
        self.assertEqual(closed['time'],rejected['time']+300)
        self.assertFalse(any(e.get('outcome')=='renewed' and pending['time']<e['time']<closed['time'] for e in run['events']))

    def test_future_changes_do_not_change_earlier_completion_decision(self):
        self.d['par_volume'][self.i,0]=0.;a=self.run_arm();cut=self.change+300
        self.d['par_close'][self.t>cut]*=1.2;self.d['sigma'][self.t>cut]*=10;self.de['mu'][self.t>cut]*=-2
        b=self.run_arm()
        for key in ['events','fills','snapshots']:
            self.assertEqual([e for e in a[key] if e['time']<=cut],[e for e in b[key] if e['time']<=cut],key)


if __name__=='__main__':unittest.main()
