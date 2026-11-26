import copy,gzip,json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .data import ROOT,OUT,read,save,sha,verify as verify_sources
from .replay import replay
from competition_live.profile import load_profile,BUNDLE
from replay_lab.common_comparison.models import Market,covariance
from pas_allocation_study.study import summarize


def profile():
    p=load_profile();trading=copy.deepcopy(p['trading']);trading['symbols']=['BTC','ETH'];trading['execution']['decimals']=trading['execution']['decimals'][:2]
    cfg=copy.deepcopy(p['configs']['reference']);cfg['dense_pairs']=[['BTC','ETH']];cfg['conditional_completion']={'risk_aversion':0.}
    return cfg,trading


def freeze():
    verify_sources()
    if (OUT/'adapter_freeze.json').exists():raise FileExistsError('Already frozen')
    # New adapters can only change array dimension / translate the historical
    # schema. Source trading algorithms were frozen before data acquisition.
    paths=[*Path(__file__).parent.glob('*.py')]
    save(OUT/'adapter_freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files={str(p.relative_to(ROOT)):sha(p) for p in paths}))


def verify():
    verify_sources()
    for p,h in read(OUT/'adapter_freeze.json')['files'].items():assert sha(ROOT/p)==h,p


def prepare():
    verify();raw=dict(np.load(OUT/'raw_inputs.npz'));cfg,p=profile();t=raw['market_times'];price=raw['close']
    m=Market(t,price,raw['volume'],raw['trades'],raw['funding'],raw['funding_observed'],('BTC','ETH'))
    sigma=np.zeros((len(t),2,2))
    for i in range(168,len(t)):_,sigma[i]=covariance(m,i,p['risk'])
    d=dict(times=t,par_close=price,reference_close=price,par_volume=raw['volume'],par_trades=raw['trades'],
        funding=raw['funding'],funding_observed=raw['funding_observed'],known_hourly_funding=raw['known_hourly_funding'],
        sigma=sigma,first=np.zeros_like(price),second=np.zeros_like(price),retention=np.zeros_like(price))
    np.savez_compressed(OUT/'execution_dataset.npz',**d)
    save(OUT/'evaluation_inputs.json',{p.name:sha(p) for p in [OUT/'raw_inputs.npz',OUT/'execution_dataset.npz',OUT/'candidate72_state.npz',OUT/'incumbent24_state.npz']})


def parity():
    """Before2022 outcomes: mechanical adapter matches exposed10asset engine."""
    from pair_regime_research.replay import replay as original
    from .models import candidate,incumbent
    from pair_value_learning.study import INPUT
    full=dict(np.load(BUNDLE/'data/reference.npz'));n=int(np.searchsorted(full['times'],1785931200))+1
    raw=dict(np.load(INPUT));data={k:raw[k][:n,:2].copy() for k in ['close','volume','trades']};data['market_times']=raw['market_times'][:n]
    states={};checks={}
    for name,fn in [('candidate72',candidate),('incumbent24',incumbent)]:
        states[name],_=fn(data,1785542400)
    prior=dict(np.load(ROOT/'reports/auto_pair_selection_20261023/states.npz'))
    adaptive=dict(np.load(ROOT/'reports/pair_model_research_20261023/wave9/adaptive_hedge_72.npz'))
    mask=(data['market_times']>=1785542400)
    for key in ['mu','beta','vol']:
        np.testing.assert_allclose(states['candidate72'][key][mask,0],adaptive[key][:n][mask,0],atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_allclose(states['incumbent24'][key][mask,0],prior[key][:n][mask,0],atol=1e-12,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(states['incumbent24']['economic_action'][mask,0],prior['gate_action'][:n][mask,0])
    cfg,p=profile();p10=load_profile()['trading'];dd={k:v[:n].copy() for k,v in full.items()}
    two={k:(v[:,:2,:2].copy() if k=='sigma' else v[:,:2].copy() if v.ndim==2 else v.copy()) for k,v in dd.items()}
    for name in states:
        s=states[name];s.pop('times');s10={k:v.copy() for k,v in s.items()};s10['source_prices']=dd['reference_close'].copy()
        h=72 if name=='candidate72' else 24;arm=f'pairs_dense_hourlyhedge_individual_{h}'
        a=replay(two,s,None,cfg,p,arm,'base',1785585600,1785931200)
        b=original(dd,s10,None,cfg,p10,arm,'base',1785585600,1785931200)
        assert a['metrics']==b['metrics']
        assert a['fills']==b['fills'] and a['events']==b['events']
        for x,y in zip(a['snapshots'],b['snapshots']):
            assert x['equity']==y['equity'] and x['qty']==y['qty'][:2] and not any(y['qty'][2:])
        checks[name]=dict(forecast_parity=True,full_path_equity_fills_events_parity=True)
    save(OUT/'adapter_parity.json',checks);print('Pretest2026exposed-input parity passed',flush=True)


def job(name,scenario):
    verify();cfg,p=profile();protocol=read(OUT/'protocol.json')
    for filename,h in read(OUT/'evaluation_inputs.json').items():assert sha(OUT/filename)==h
    d=dict(np.load(OUT/'execution_dataset.npz'));s=dict(np.load(OUT/f'{name}_state.npz'));s.pop('times')
    horizon=72 if name=='candidate72' else 24
    r=replay(d,s,None,cfg,p,f'pairs_dense_hourlyhedge_individual_{horizon}',scenario,protocol['test_start'],protocol['test_end'])
    row=summarize(r,name,'new_2022_external_test',scenario)
    assert row['timing_pass'] and row['accounting_error']<1e-6
    path=OUT/'runs'/f'{name}_{scenario}.json.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
    save(OUT/'rows'/f'{name}_{scenario}.json',row)
    return row


def run():
    verify();assert (OUT/'adapter_parity.json').exists()
    if (OUT/'evaluation_started.json').exists():raise FileExistsError('One-use test already started; do not tune/reselect')
    save(OUT/'evaluation_started.json',dict(time=datetime.now(timezone.utc).isoformat(),adapter_freeze_sha256=sha(OUT/'adapter_freeze.json'),inputs=read(OUT/'evaluation_inputs.json')))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,n,s) for n in ['candidate72','incumbent24'] for s in ['base','stress']]):
            row=f.result();rows.append(row);print(json.dumps({k:row[k] for k in ['arm','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    save(OUT/'results.json',rows);verify()


if __name__=='__main__':globals()[sys.argv[1]]()
