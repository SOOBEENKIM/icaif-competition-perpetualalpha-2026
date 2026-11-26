import json,sys,gzip
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .data import ROOT,OUT,START,DEV_END,SYMBOLS,read,save,sha
from .wave3 import verify as verify_prior
from .policy_v2 import VenueController
from .economic_execution import simulate
from .runner import persist
from pas_strategy_audit.data import load

WAVE=OUT/'wave4'
ARMS={'incumbent_ttl1h':('incumbent',3600),'reversion12_quote1h':('pas_reversion12',3600),'reversion12_quote3h':('pas_reversion12',10800),'funding168_quote1h':('funding_persistent168',3600)}

def prepare():
    verify_prior()
    if (WAVE/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(WAVE/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,
        hypothesis='Fixed15minute intentions can expire before sparse PAS trades; extension must preserve original forecast economics. Reversion12h earned price gains but liquidity-constrained intended quantities rarely filled.',
        change='Precommit quantityandmaximum buy/minimum sell from source terminal forecast, actual fee, adverse exitimpact, conservative funding,and5bpbuffer.1h primary/3h diagnostic lifetime. Every childfill uses next observed minute worstprice+impact and same participationcap. Exit requestsremain untilfilled. No inferredmakerqueue or visible historicaldepth.',
        control='Current incumbent with1hlifetime andoriginal5percent limit to separate persistence from newalpha; original15min controls remain wave1/wave3. Identical account/risk/costs. Higherutilization measured, not automatically called betteralpha.',
        selection='Development only, all arms/partitions/scenarios reported. Uninspected4hPAS tail still not acquired.'))
    paths=[ROOT/'pas_cashflow_research/economic_execution.py',ROOT/'pas_cashflow_research/wave4.py',WAVE/'protocol.json']
    save(WAVE/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})

def verify():
    verify_prior()
    for p,h in read(WAVE/'freeze.json').items():assert sha(ROOT/p)==h,p

def job(arm,scenario,phase,start,end):
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller as incumbent
    verify();directory=OUT/'development';cfg,d=load(directory);cfg.update(start_ms=start*1000,cutoff_ms=end*1000)
    p=load_profile()['trading'];sc=read(OUT/'protocol.json')['fees'][scenario];model,ttl=ARMS[arm]
    z=dict(np.load(OUT/'wave3/forecasts.npz'));f=dict(np.load(directory/'candidate_inputs.npz'))
    if model=='incumbent':c=incumbent(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    else:
        h=12 if model=='pas_reversion12' else 168;pred=z[model];valid=z['fresh']&np.isfinite(pred)
        c=VenueController(z['times'],np.nan_to_num(pred,nan=0.),z[f'carry{h}'],valid,model,horizon=h,receipt=.5,impact_bps=sc['impact_bps'])
    tape=[]
    for symbol in SYMBOLS:
        for b in read(directory/'raw'/f'pas_candles_{symbol}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and start*1000<=at<=end*1000:
                tape.append(dict(time=at,coin='pas:'+symbol,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    r=simulate(cfg,d,read(directory/'events.json'),tape,read(directory/'funding.json'),p,sc,controller=c,entry_lifetime=ttl,economic_limits=model!='incumbent',sticky_exits=True)
    row=dict(arm=arm,scenario=scenario,phase=phase,**r['metrics']);r['controller_trace']=c.trace;persist(WAVE,row,r);return row

def run():
    verify()
    if (WAVE/'started.json').exists():raise FileExistsError('Already run')
    save(WAVE/'started.json',dict(at=datetime.now(timezone.utc).isoformat()))
    args=[(a,s,ph,lo,hi) for a in ARMS for s in ['base','stress','no_receipts'] for ph,lo,hi in [('selection48h',START,START+48*3600),('later44h',START+48*3600,DEV_END),('continuous92h',START,DEV_END)]]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*a) for a in args]):
            r=f.result();rows.append(r);save(WAVE/'results.json',rows);print({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','risk_breach_events']},flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
