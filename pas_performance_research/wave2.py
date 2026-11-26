"""Second declared wave: economic core checks with actual venue funding."""
import copy,gzip,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,BASE,save,sha,live_files
from .funding import forecast,filtered_anchor

OUT=ROOT/'reports/pas_performance_research_20261022/wave2'
ARMS=[BASE,'venue_funding','venue_funding_decay','venue_funding_no_price']

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,
        hypothesis='External funding forecasts do not represent PAS funding. Require economic edge on core positions, and feed the existing allocator lagged forecasts of the venue actually traded.',
        primary='venue_funding',ablation='no_price sets the retained weekly price prediction to zero; decay uses48h funding half-life instead of constant weekly rates.',
        model_refit=False,history_reused=True,independent_test=False,live_changed=False,orders_sent=0,
        live_files=live_files(),fixed='Same baseline,10eligible markets,symmetric sides,weekly price model,pairs,budget,risk,fees,delayed partialfills. No peer copying or future cashflow inputs.',
        funding='Completed observed rates become available60s later.6h/24h EWMA minimum signed magnitude,minimum6observations,confidence n/24,capped1. Full payments/half receipts in expected utility,actual settlements separately.',
        evaluation='Same six reference and three PAS windows; candidate must beat baseline full/later reference and PAS base/stress PnL,Q; report active assets,shorts,exposure,terminalreserve,all failed arms.',
        limitations=['Weekly persistence is an assumption, challenged by48hdecay.','PAS only48hours; conditional trade tape is not historical L2 execution.','New funding rows were collected retrospectively; imposed60s publication delay is an assumption, not a contemporaneous reception log.'])
    save(OUT/'protocol.json',p)
    files=[Path(__file__),Path(__file__).with_name('funding.py'),Path(__file__).with_name('study.py'),OUT/'protocol.json']
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})

def verify():
    for f,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/f)==h,f
    assert live_files()==json.loads((OUT/'protocol.json').read_text())['live_files']

def controller(times,z,ix,f,fi,carry,arm):
    from pas_economic_allocation.lifecycle import LifecycleController
    if arm==BASE:return LifecycleController(times,f['forecast'][fi],f['carry'][fi],z[BASE][ix],z[BASE+'_protected'][ix],5.)
    price=f['forecast'][fi] if arm!='venue_funding_no_price' else np.zeros_like(carry)
    anchor,guard=filtered_anchor(z[BASE][ix],z[BASE+'_protected'][ix],price,carry,arm)
    return LifecycleController(times,price,carry,anchor,guard,5.)

def reference():
    from competition_live.profile import load_profile,BUNDLE
    from pas_economic_allocation.replay import replay
    from pas_allocation_study.study import summarize
    verify();p=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times'])
    source=json.loads((ROOT/'reports/universe_repair_20261022/protocol.json').read_text())['input']['inputs']
    raw=dict(np.load(ROOT/source));symbols=p['trading']['symbols']
    observations=[dict(known=int(t)+60,symbol=s,rate=float(raw['funding'][i,j])) for i,t in enumerate(raw['market_times']) for j,s in enumerate(symbols) if raw['observed'][i,j]]
    carries={a:forecast(d['times'],observations,symbols,decay=a=='venue_funding_decay')[0] for a in ARMS[1:]}
    cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==p['candidate']]
    rows=[]
    for w in windows:
        for arm in ARMS:
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz"
            if file.exists():
                with gzip.open(file,'rt') as fp:r=json.load(fp)
            else:
                c=controller(d['times'],z,ix,f,fi,carries.get(arm),arm)
                r=replay(d,states,None,cfg,p['trading'],p['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
                file.parent.mkdir(exist_ok=True)
                with gzip.open(file,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
            row=summarize(r,arm,w['phase'],w['scenario']);rows.append(row)
            assert row['timing_pass'] and row['accounting_error']<1e-6
            save(OUT/'reference_results.json',rows)
            print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q','risk_breach_bars']}),flush=True)
    verify()

def pas():
    from competition_live.profile import load_profile
    from pas_counterfactual_20261022.engine import simulate
    from pas_counterfactual_20261022.prepare import END
    from pas_strategy_audit.data import load
    verify();old=ROOT/'reports/pas_counterfactual_20261022';read=lambda n:json.loads((old/n).read_text())
    assert all(f['economic_available'] and f['economic_action']==0 for row in read('pair_forecasts.json') for f in row['forecasts'] if f['model']=='decision_gate')
    cfg,d=load(ROOT/'reports/pas_allocation_study_20261021/pas');cfg['cutoff_ms']=END*1000
    p=load_profile();events=read('events.json');tape=read('trade_tape.json');funding=read('funding_tape.json')
    known={(r['coin'],r['time']//3600000) for r in funding}
    for r in d['funding_events']:
        coin='pas:'+cfg['symbols'][r['j']]
        if (coin,r['time']//3600000) not in known:funding.append(dict(coin=coin,time=r['time'],rate=r['rate'],cash_per_unit=None))
    f=dict(np.load(old/'candidate_inputs.npz'));z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));ix=np.searchsorted(z['times'],f['times']);fi=np.arange(len(f['times']))
    obs=[dict(known=r['time']/1000+60,symbol=r['coin'][4:],rate=r['rate']) for r in funding]
    carries={a:forecast(f['times'],obs,p['trading']['symbols'],decay=a=='venue_funding_decay')[0] for a in ARMS[1:]}
    np.savez_compressed(OUT/'pas_carry.npz',times=f['times'],**carries)
    rows=[]
    for name,sc in read('protocol.json')['scenarios'].items():
        for arm in ARMS:
            c=controller(f['times'],z,ix,f,fi,carries.get(arm),arm)
            r=simulate(cfg,d,copy.deepcopy(events),tape,funding,p['trading'],sc,controller=c)
            row=dict(arm=arm,scenario=name,**r['metrics']);rows.append(row)
            assert row['timing_pass'] and row['maximum_accounting_error']<1e-6
            file=OUT/'pas'/f'{arm}_{name}.json.gz';file.parent.mkdir(exist_ok=True)
            with gzip.open(file,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
            save(OUT/'pas_results.json',rows)
            print(json.dumps({k:row[k] for k in ['arm','scenario','net_pnl','Q','risk_breach_events','fills']}),flush=True)
    verify()

if __name__=='__main__':globals()[sys.argv[1]]()
