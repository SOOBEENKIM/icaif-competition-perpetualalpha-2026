"""Inventory transition invariants shared by research timing policies."""
from unittest.mock import patch
import numpy as np
from competition_live.profile import load_profile
from pas_economic_allocation.lifecycle import LifecycleController
from .inventory import EntryController
from .timing import TimingController
from .ensemble_timing import EnsembleTimingController
from .soft_timing import SoftTimingController

CASES=[(EntryController,'entry_pullback6',0.),(TimingController,'entry_pullback_3_0',0.),
       (EnsembleTimingController,'pullback_any',0.),(SoftTimingController,'pullback_any3',0.),
       (SoftTimingController,'pullback_any_soft',.25)]

def run_case(cls,rule,qty,target,increasing=True,future_poison=False):
    protocol=load_profile()['trading'];times=np.arange(0,49*3600,3600);now=24*3600
    path=np.exp(np.arange(49)*(.01 if increasing else -.01))[:,None]*np.ones((1,10))*100
    if future_poison:path[times>now]*=1000
    f=np.zeros((49,10));c=cls(times,f,f,f,np.zeros_like(f,bool),5.,price_times=times,prices=path,entry_rule=rule)
    def parent(self,*args,**kwargs):self.trace.append({});return {'target_qty':target.tolist()}
    with patch.object(LifecycleController,'propose',parent):
        action=c.propose(now,np.full(10,100.),qty,50000.,np.zeros(10),np.eye(10)*1e-8,np.ones(10),np.ones(10),protocol)
    return np.array(action['target_qty'])

def test_entry_filter_does_not_liquidate_surviving_position():
    for cls,rule,residual in CASES:
        q=np.zeros(10);q[0]=3.;target=q.copy();target[0]=5.
        got=run_case(cls,rule,q,target)
        np.testing.assert_allclose(got[0],3+2*residual,atol=1e-5)

def test_exit_and_risk_reduction_always_survive():
    for cls,rule,residual in CASES:
        q=np.zeros(10);q[:2]=[10.,3.];target=np.zeros(10);target[0]=5.
        got=run_case(cls,rule,q,target)
        np.testing.assert_allclose(got[:2],[5.,0.],atol=1e-5)

def test_short_reversal_and_future_prefix():
    for cls,rule,residual in CASES:
        q=np.zeros(10);q[0]=3.;target=np.zeros(10);target[0]=-5.
        # Falling prices are an unfavourable new short entry for pullback rules.
        got=run_case(cls,rule,q,target,increasing=False)
        np.testing.assert_allclose(got[0],-5*residual,atol=1e-5)
        poisoned=run_case(cls,rule,q,target,increasing=False,future_poison=True)
        np.testing.assert_array_equal(got,poisoned)

def test_all_ten_markets_are_eligible_both_sides():
    for cls,rule,residual in CASES:
        q=np.zeros(10);long=np.ones(10)
        np.testing.assert_allclose(run_case(cls,rule,q,long,increasing=False),long,atol=.001)
        np.testing.assert_allclose(run_case(cls,rule,q,-long,increasing=True),-long,atol=.001)

if __name__=='__main__':
    checks=[(k,v) for k,v in globals().copy().items() if k.startswith('test_') and callable(v)]
    for name,check in checks:check();print('PASS',name,flush=True)
    print(len(checks),'checks across',len(CASES),'policies passed')
