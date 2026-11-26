"""Post-exposure sensitivity, never relabelled as independent selection."""
import sys,json
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .data import ROOT,OUT,SYMBOLS,read,save,sha
from .tail_eval import HERE,verify
from .strict_execution import simulate
from .runner import persist
from pas_strategy_audit.data import load

DEST=OUT/'liquidity_audit'
SCENARIOS={
    'larger_flow_base':dict(fee_bps=9.,impact_bps=10.,participation=.10,receipt_multiplier=1.),
    'larger_flow_stress':dict(fee_bps=12.,impact_bps=20.,participation=.025,receipt_multiplier=.5),
    'saturation_base':dict(fee_bps=9.,impact_bps=10.,participation=1.,receipt_multiplier=1.),
    'saturation_stress':dict(fee_bps=12.,impact_bps=20.,participation=1.,receipt_multiplier=.5),
}

def run_one(arm,name):
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller
    verify();cfg,d=load(HERE/'dataset');sc=SCENARIOS[name];p=load_profile()['trading'];f=dict(np.load(HERE/'dataset/candidate_inputs.npz'))
    c=controller(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    tape=[]
    for symbol in SYMBOLS:
        for b in read(HERE/'dataset/raw'/f'pas_candles_{symbol}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+symbol,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    extended=arm=='incumbent_ttl1h'
    r=simulate(cfg,d,read(HERE/'dataset/events.json'),tape,read(HERE/'dataset/funding.json'),p,sc,controller=c,entry_lifetime=3600 if extended else 900,sticky_exits=extended)
    row=dict(arm=arm,scenario=name,phase='exposed97h',**r['metrics']);persist(DEST,row,r);return row

def run():
    verify();assert not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),scenarios=SCENARIOS,arms=['incumbent','incumbent_ttl1h'],
        intent='Test whether expiry-control gains depend on conservative tape participation. Larger/saturated flows are capacity sensitivity, NOT claims of available queue priority. All outcomes now exposed; no independent-test or new-winner claim.',
        source_hash=sha(__file__)))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run_one,a,s) for a in ['incumbent','incumbent_ttl1h'] for s in SCENARIOS]):
            r=f.result();rows.append(r);save(DEST/'results.json',rows);print({k:r[k] for k in ['arm','scenario','net_pnl','Q','mean_notional']},flush=True)

if __name__=='__main__':run()
