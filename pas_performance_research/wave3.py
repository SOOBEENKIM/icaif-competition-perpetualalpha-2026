"""Venue-direction confirmation, no future labels or asset-name rules."""
import copy,gzip,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,BASE,save,sha,live_files
from .extension import OUT as EXT,START,END,SPLIT,SYMBOLS

OUT=ROOT/'reports/pas_performance_research_20261022/wave3'
ARMS=[BASE,'venue_6h','venue_24h','venue_consensus']

def direction_mask(times,price_times,prices,anchor,forecast,carry,arm,age=None):
    h=[6] if arm=='venue_6h' else [24] if arm=='venue_24h' else [6,24]
    # An otherwise empty slot inherits the allocator's intended direction.
    direction=np.where(anchor!=0,np.sign(anchor),np.sign(forecast-.75*carry))
    result=np.zeros_like(anchor)
    for hours in h:
        now=np.searchsorted(price_times,times,side='right')-1
        then=np.searchsorted(price_times,times-hours*3600,side='right')-1
        valid=(now>=0)&(then>=0);a=np.maximum(now,0);b=np.maximum(then,0)
        p=prices[a];q=prices[b]
        good=valid[:,None]&np.isfinite(p)&np.isfinite(q)&(p>0)&(q>0)
        if age is not None:good&=(age[a]<=3600)&(age[b]<=3600)
        change=np.log(np.where(good,p/q,1.))
        result+=good*(direction*change>0)/len(h)
    return result

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,live_files=live_files(),orders_sent=0,live_changed=False,
        history_reused=True,independent_test=False,model_refit=False,
        reason='Wave2 passed original48h but failed later23h and continuous71h. Record rejection. Test whether external directional decisions agree with completed actual venue prices.',
        rules='6h,24h,and equal consensus direction masks. If venue prices missing/stale more than1h, disallow additional target. Retain hard risk reductions. Same eligibility and both sides. No name/date/forced utilization rules.',
        primary='venue_consensus',selection='Report all declared arms; no claim that selecting after these reused outcomes is independent validation.',
        fixed='Same50k+50k reserve,all10symbols,sharedrisk,fees,pairs,rebalance6h,15minexecutionwindow.',
        evaluation='Six reference conditions, original48htrade-tape3conditions, full71hminute-candle3conditions with earlier48h/later23h decomposition.'))
    files=[Path(__file__),OUT/'protocol.json']
    save(OUT/'freeze.json',{str(f.relative_to(ROOT)):sha(f) for f in files})

def verify():
    for f,h in json.loads((OUT/'freeze.json').read_text()).items():assert sha(ROOT/f)==h,f
    assert live_files()==json.loads((OUT/'protocol.json').read_text())['live_files']

def build(f,z,ix,times,price_times,prices,arm,age=None):
    from pas_economic_allocation.lifecycle import LifecycleController
    anchor=z[BASE][ix];guard=z[BASE+'_protected'][ix]
    if arm!=BASE:
        g=direction_mask(times,price_times,prices,anchor,f['forecast'],f['carry'],arm,age)
        anchor=anchor*g;guard=guard|(g<1)
    return LifecycleController(times,f['forecast'],f['carry'],anchor,guard,5.)

def reference():
    from competition_live.profile import load_profile,BUNDLE
    from pas_economic_allocation.replay import replay
    from pas_allocation_study.study import summarize
    verify();p=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times']);f={k:v[fi] for k,v in f.items()}
    source=json.loads((ROOT/'reports/universe_repair_20261022/protocol.json').read_text())['input']['inputs'];raw=dict(np.load(ROOT/source))
    cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    windows=[w for w in json.loads((BUNDLE/'expected.json').read_text())['runs'] if w['venue']=='reference' and w['arm']==p['candidate']];rows=[]
    for w in windows:
        for arm in ARMS:
            c=build(f,z,ix,d['times'],raw['market_times'],raw['close'],arm)
            r=replay(d,states,None,cfg,p['trading'],p['candidate'],w['scenario'],w['start'],w['end'],base_policy=c)
            row=summarize(r,arm,w['phase'],w['scenario']);rows.append(row)
            file=OUT/'reference'/f"{arm}_{w['phase']}_{w['scenario']}.json.gz";file.parent.mkdir(exist_ok=True)
            with gzip.open(file,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
            save(OUT/'reference_results.json',rows);print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q']}),flush=True)
    verify()

def pas():
    from competition_live.profile import load_profile
    from pas_strategy_audit.data import load
    from pas_counterfactual_20261022.engine import simulate
    verify();p=load_profile();cfg,d=load(EXT)
    assert json.loads((EXT/'pair_audit.json').read_text())['can_use_base_only']
    events=json.loads((EXT/'events.json').read_text());f=dict(np.load(EXT/'candidate_inputs.npz'))
    z={BASE:f['anchor'],BASE+'_protected':f['protected']};ix=np.arange(len(f['times']))
    tape=[];funding=[]
    for s in SYMBOLS:
        for b in json.loads((EXT/'raw'/f'pas_candles_{s}.json').read_text())['data']:
            at=int(b['t'])+60000
            if float(b['v'])>0 and b['n']>0 and START*1000<=at<=END*1000:tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
        for r in json.loads((EXT/'raw'/f'pas_funding_{s}.json').read_text())['data']:funding.append(dict(time=r['time'],coin='pas:'+s,rate=float(r['fundingRate']),cash_per_unit=None))
    protocol=json.loads((EXT/'protocol.json').read_text());rows=[]
    runs=[('earlier48h',START,SPLIT,tape,funding),('later23h',SPLIT,END,tape,funding),('continuous71h',START,END,tape,funding)]
    old=ROOT/'reports/pas_counterfactual_20261022';real_tape=json.loads((old/'trade_tape.json').read_text());real_funding=json.loads((old/'funding_tape.json').read_text())
    known={(r['coin'],r['time']//3600000) for r in real_funding}
    real_funding+=[r for r in funding if (r['coin'],r['time']//3600000) not in known]
    runs.append(('trade_tape48h',START,SPLIT,real_tape,real_funding))
    for phase,start,end,tt,ff in runs:
        for scenario,sc in protocol['scenarios'].items():
            for arm in ARMS:
                cc=copy.deepcopy(cfg);cc.update(start_ms=start*1000,cutoff_ms=end*1000)
                c=build(f,z,ix,f['times'],d['times']/1000,d['close'],arm,d['age_ms']/1000)
                r=simulate(cc,d,copy.deepcopy(events),tt,ff,p['trading'],sc,controller=c)
                row=dict(arm=arm,phase=phase,scenario=scenario,**r['metrics']);rows.append(row)
                file=OUT/'pas'/f'{arm}_{phase}_{scenario}.json.gz';file.parent.mkdir(exist_ok=True)
                with gzip.open(file,'wt') as fp:json.dump(r,fp,default=lambda x:x.tolist(),allow_nan=False)
                save(OUT/'pas_results.json',rows);print(json.dumps({k:row[k] for k in ['arm','phase','scenario','net_pnl','Q']}),flush=True)
    verify()

if __name__=='__main__':globals()[sys.argv[1]]()
