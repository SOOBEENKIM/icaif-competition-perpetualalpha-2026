import copy
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from competition_live.profile import ROOT,BUNDLE,load_profile,verify_bundle,verify_deployment
from competition_live.scoring import evaluate
from experiments.decision_gate_pairs_20261017.portfolio import replay
from replay_lab.common_comparison.engine import project
from .model import candidates

OUT=ROOT/'reports/submission_compare_20261021'
OLD=ROOT/'reports/isolated_repairs_20261020'
AGREE=ROOT/'reports/all_market_alpha_wave3_20261020'


def native(value):
    return value.tolist() if hasattr(value,'tolist') else value.item()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=native))


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Comparison already registered')
    spec=json.loads((OLD/'protocol.json').read_text())
    saved=ROOT/spec['pointer']['run']/'targets.npz'
    z=dict(np.load(OLD/'targets.npz',allow_pickle=False))
    agreement=dict(np.load(AGREE/'targets.npz',allow_pickle=False))
    forecast=dict(np.load(saved,allow_pickle=False))
    np.testing.assert_array_equal(z['times'],agreement['times'])
    weights=candidates(z['incumbent'],z['short_only'],forecast['chronos_forecast'],
                       forecast['frozen_regime_probability'],agreement['agreement'])
    profile=load_profile()
    for w in weights.values():
        for i in range(168,len(z['times'])):
            w[i]=project(w[i],profile['trading']['initial'],z['sigma'][i],profile['trading']['risk'])
        w[:168]=0.
    np.savez_compressed(OUT/'targets.npz',times=z['times'],incumbent=z['incumbent'],**weights)
    sources=[OLD/'targets.npz',AGREE/'targets.npz',saved,OUT/'targets.npz',
        BUNDLE/'data/reference.npz',BUNDLE/'models/reference_gate_state.npz',
        ROOT/'submission_compare/model.py',ROOT/'submission_compare/run.py']
    protocol=dict(at=time.time(),purpose='Compare candidates now using existing historical backtests.',
        arms=['incumbent',*weights],new_hypotheses={
            'confirmed_short':'Use symmetric target only when target and frozen Chronos are negative and frozen JM bull probability <0.4; otherwise incumbent.',
            'forecast_filter':'Flatten incumbent crypto exposure when its sign conflicts with frozen Chronos; no resizing of survivors.',
            'agreement_confirmed_short':'Use the same confirmed short on top of the existing agreement allocation.'},
        universe='Changes confined to BTC/ETH/HYPE. Other base sleeves and pair policy unchanged.',
        comparison='Same 50k allocation, 50k reserve, source timing, fees, slippage, risk caps, and delayed execution.',
        phases=['August','September 1 through October 15','continuous August through October 15'],
        scenarios=['base','stress'],historical_independence=False,
        prior_history_seen=True,parameter_search=False,
        decision='Report all results, long/short attribution, recent and continuous PnL/Q/drawdown. '
            'A future observation period is NOT a prerequisite for a present submission recommendation.',
        live_submission=False,orders_sent=0,new_background_evaluation=False,
        deployment_hashes=verify_deployment()['files'],bundle_hash=verify_bundle(),
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    save(OUT/'protocol.json',protocol)


def run():
    spec=json.loads((OUT/'protocol.json').read_text())
    for name,digest in spec['hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    profile=load_profile()
    data=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    state=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    weights=dict(np.load(OUT/'targets.npz',allow_pickle=False))
    ids=np.searchsorted(weights['times'],data['times'])
    np.testing.assert_array_equal(weights['times'][ids],data['times'])
    np.testing.assert_allclose(weights['incumbent'][ids],data['first'],rtol=1e-11,atol=1e-13)
    windows=[r for r in json.loads((BUNDLE/'expected.json').read_text())['runs']
        if r['venue']=='reference' and r['arm']==profile['candidate']]
    old=json.loads((OLD/'reference_results.json').read_text())
    rows=[];(OUT/'reference').mkdir(exist_ok=True)
    for window in windows:
        for name in spec['arms']:
            d=dict(data);d['first']=weights[name][ids]
            cfg=copy.deepcopy(profile['configs']['reference'])
            cfg['conditional_completion']={'risk_aversion':0.}
            result=replay(d,state,None,cfg,profile['trading'],profile['candidate'],
                          window['scenario'],window['start'],window['end'])
            m=result['metrics']
            daily=[dict(time=s['time'],equity=s['equity']+50000) for s in result['snapshots']
                   if s['time']%86400==43200]
            sc=evaluate(daily,trading_days=len({f['time']//86400 for f in result['fills']}))
            curve=np.array([s['equity']+50000 for s in result['snapshots']])
            row=dict(variant=name,phase=window['phase'],scenario=window['scenario'],
                start=window['start'],end=window['end'],net_pnl=m['net_pnl'],
                terminal_adjusted_pnl=m['pnl_after_liquidation_reserve'],
                Q=sc['Q'],volatility=sc['volatility'],return_fraction=sc['return_fraction'],
                max_drawdown=float(-np.min(curve/np.maximum.accumulate(curve)-1)),
                mean_notional=float(np.mean([s['gross']*s['equity'] for s in result['snapshots']])),
                fees=m['fees'],execution_cost=m['execution_cost'],funding=m['funding'],
                base_pnl=m['base_owned_pnl'],pair_pnl=m['pair_owned_pnl'],
                fills=m['fills'],completed_pairs=m['completed_pairs'],
                risk_breach_bars=m['risk_breach_bars'],accounting_error=m['maximum_accounting_error'],
                timing_pass=all(f['source_available']<=f['decision_time']<f['time'] for f in result['fills']))
            assert row['timing_pass'] and row['accounting_error']<1e-6
            if name=='incumbent':
                expected=next(r for r in old if r['variant']==name and r['phase']==row['phase'] and r['scenario']==row['scenario'])
                assert abs(row['net_pnl']-expected['net_pnl'])<1e-7
                assert abs(row['Q']-expected['Q'])<1e-12
            with gzip.open(OUT/'reference'/f"{name}_{row['phase']}_{row['scenario']}.json.gz",'wt') as f:
                json.dump(result,f,default=native)
            rows.append(row);save(OUT/'results.json',rows)
            print(json.dumps(row),flush=True)
    save(OUT/'completed.json',dict(at=time.time(),rows=len(rows),
        incumbent_reproduced=True,deployment_unchanged=verify_deployment()['files']==spec['deployment_hashes'],
        bundle_hash=verify_bundle(),orders_sent=0))


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    else:raise ValueError('Unknown action')
