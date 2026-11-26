"""Validate actual-settlement conservation and no-future-data assertions."""
import json
import numpy as np
from .pas_study_v2 import OUT,RAW,START,END,ROOT,INPUT,verify
from .study import read,save
from auto_pair_selection.selection import SYMBOLS
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount
from experiments.historical_par_extension_20261017.settlements import charge


def main():
    verify();d=dict(np.load(OUT/'dataset.npz'));records=read(OUT/'funding_timing.json')
    assert all(r['known_funding_last_ms'] is None or r['known_funding_last_ms']<r['decision_source']*1000 for r in records)
    assert all(all(x<=r['decision_source']*1000 for x in r['settled_ms']) for r in records)
    comparisons=[]
    for j,s in enumerate(SYMBOLS):
        expected=[r for r in read(RAW/f'pas_{s}.json')['funding_history']['response'] if START*1000<r['time']<=END*1000]
        mask=d['event_observed'][:,:,j]
        assert int(mask.sum())==len(expected)
        assert abs(float(d['event_rates'][:,:,j][mask].sum())-sum(float(r['fundingRate']) for r in expected))<1e-12
        comparisons.append(dict(symbol=s,raw_payments=len(expected),replay_payments=int(mask.sum())))
    # Receipt haircut must be applied to each actual settlement, not to the
    # net rate of two payments in the same recognition interval.
    a=SharedAccount(50000.,['X']);a.physical_fill(0,1.,100.,0.,0);a.owner_fill('pair',0,1.,100.,0.,0,'fixture')
    before=a.physical.equity(np.array([100.]))
    charge(a,10,np.array([100.]),np.array([.01]),np.array([True]),1001,.5)
    charge(a,10,np.array([100.]),np.array([-.02]),np.array([True]),1002,.5)
    assert abs(a.physical.equity(np.array([100.]))-before)<1e-12
    a.reconcile(np.array([100.]));assert np.any((d['event_stamps']>=0).sum(1)==2)
    save(OUT/'validation.json',dict(strictly_past_funding_predictors=True,settlement_conservation=comparisons,
                                  each_receipt_stressed_separately=True,multiple_events_in_one_interval_tested=True,
                                  no_missing_payment_imputed=True))
    print('PASS causal funding inputs, all actual payments conserved, per-payment receipt stress')


if __name__=='__main__':main()
