import json
import numpy as np
from .funding import forecast,filtered_anchor
from .study import ROOT,masks,live_files,BASE

def observations(sign=1):
    return [dict(symbol=s,known=t+60,rate=sign*(j+1)*1e-5) for t in range(0,48*3600,3600) for j,s in enumerate(['A','B'])]

def test_future_funding_never_changes_prefix():
    t=np.arange(0,49*3600,3600);o=observations();v,n=forecast(t,o,['A','B'])
    cut=24*3600
    altered=[dict(r,rate=r['rate']*1e6) if r['known']>cut else dict(r) for r in o]
    poisoned,_=forecast(t,altered,['A','B'])
    prefix,_=forecast(t[t<=cut],o,['A','B'])
    np.testing.assert_array_equal(poisoned[t<=cut],v[t<=cut]);np.testing.assert_array_equal(prefix,v[t<=cut])

def test_settlement_delay_missing_data_and_staleness():
    o=observations();t=np.array([5*3600+59,5*3600+60,49*3600+61,70*3600])
    v,n=forecast(t,o,['A','B'])
    assert not v[0].any() and np.all(v[1]>0)
    assert not v[2:].any() # Do not extrapolate through stale observations.
    gap=[r for r in o if r['known']<10*3600 or r['known']>30*3600]
    v,n=forecast(np.array([31*3600+60]),gap,['A','B']);assert not v.any()

def test_funding_no_named_asset_or_side_bias():
    t=np.arange(0,49*3600,3600);o=observations();v,_=forecast(t,o,['A','B'])
    flipped,_=forecast(t,observations(-1),['A','B']);np.testing.assert_allclose(flipped,-v)
    permuted,_=forecast(t,o,['B','A']);np.testing.assert_array_equal(permuted,v[:,::-1])

def test_cost_gate_is_symmetric_and_cannot_expand_risk():
    a=np.array([[.1,-.1,0,.04]]);p=np.array([[True,False,False,True]])
    mu=np.array([[.03,-.03,.03,-.03]]);carry=np.array([[.01,-.01,0,0]])
    target,guard=filtered_anchor(a,p,mu,carry,'venue_funding')
    inverse,g2=filtered_anchor(-a,p,-mu,-carry,'venue_funding')
    np.testing.assert_array_equal(inverse,-target);np.testing.assert_array_equal(guard,g2)
    assert np.all(abs(target)<=abs(a));assert np.all(guard[p]);assert target[0,3]==0 and guard[0,3]

def test_reference_filter_prefix_no_future_prices():
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'))
    p=json.loads((ROOT/'reports/universe_repair_20261022/protocol.json').read_text())
    d=dict(np.load(ROOT/p['input']['inputs']));f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'))
    all_masks=masks(d,f['forecast'],f['carry'],z[BASE]);n=len(z['times'])-100
    prefix=masks({k:v[:n] for k,v in d.items()},f['forecast'][:n],f['carry'][:n],z[BASE][:n])
    for k in all_masks:np.testing.assert_allclose(prefix[k],all_masks[k][:n],equal_nan=True,atol=0,rtol=0)

def test_no_live_mutation_and_frozen_fit_before_evaluation():
    p=json.loads((ROOT/'reports/pas_performance_research_20261022/protocol.json').read_text())
    assert live_files()==p['live_files']
    fit=json.loads((ROOT/'reports/pas_economic_allocation_20261022/fit.json').read_text())
    assert fit['last_training_maturity']<fit['last_calibration_maturity']<1785585600

def test_retained_weekly_fit_ignores_post_calibration_labels():
    from pas_economic_allocation.model import train
    p=json.loads((ROOT/'reports/pas_economic_allocation_20261022/protocol.json').read_text())
    d=dict(np.load(ROOT/p['pointer']['inputs']))
    fit=json.loads((ROOT/'reports/pas_economic_allocation_20261022/fit.json').read_text())
    altered={k:v.copy() for k,v in d.items()}
    altered['close'][d['market_times']>=p['calibration_end']]*=1000
    again=train(altered,p)
    for key in ('beta','center','scale','attenuation','last_training_maturity','last_calibration_maturity'):
        np.testing.assert_allclose(again[key],fit[key],atol=1e-14,rtol=1e-14)

if __name__=='__main__':
    checks=[(k,v) for k,v in globals().copy().items() if k.startswith('test_') and callable(v)]
    for name,check in checks:
        check();print('PASS',name,flush=True)
    print(len(checks),'checks passed')
