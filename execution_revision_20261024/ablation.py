"""Post-exposure mechanism decomposition; no new nominee selection."""
import gzip,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
from .study import ROOT,OUT,BOUNDARY,read,save,sha,verify,SYMBOLS
from .handover_engine import simulate
from pas_strategy_audit.data import load
from pas_cashflow_research.runner import persist

def job(arm,scenario):
    from competition_live.profile import load_profile
    from pas_performance_research.selected import controller
    verify();cfg,d=load(OUT/'dataset');f=dict(np.load(OUT/'dataset/candidate_inputs.npz'))
    c=controller(f['times'],f['forecast'],f['carry'],f['anchor'],f['protected'],price_times=d['times']/1000,prices=d['close'],age=d['age_ms']/1000)
    sc=read(OUT/'protocol.json')['scenarios'][scenario];tape=[]
    for s in SYMBOLS:
        for b in read(OUT/'dataset/raw'/f'pas_candles_{s}.json')['data']:
            at=b['t']+60000
            if b['n']>0 and float(b['v'])>0 and cfg['start_ms']<=at<=cfg['cutoff_ms']:
                tape.append(dict(time=at,coin='pas:'+s,qty=float(b['v']),vwap=float(b['c']),buy_price=float(b['h']),sell_price=float(b['l'])))
    ttl,sticky={'ttl_only':(3600,False),'reductions_only':(900,True)}[arm]
    r=simulate(cfg,d,read(OUT/'dataset/events.json'),tape,read(OUT/'dataset/funding.json'),load_profile()['trading'],sc,controller=c,entry_lifetime=ttl,sticky_exits=sticky)
    a=next(s for s in r['snapshots'] if s['time']==BOUNDARY*1000);b=r['snapshots'][-1]
    row=dict(arm=arm,scenario=scenario,phase='mechanism120h',**r['metrics'],tail_pnl=b['equity']-a['equity'])
    persist(OUT/'ablation',row,r);return row

def main():
    path=OUT/'ablation/protocol.json';assert not path.exists()
    save(path,dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Post-exposure mechanism attribution; do not replace primary by winning ablation.',
        code_sha256=sha(__file__),arms=['ttl_only','reductions_only'],scenarios=['base','stress'],orders_sent=0))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,s) for a in ['ttl_only','reductions_only'] for s in ['base','stress']]):
            row=f.result();rows.append(row);save(OUT/'ablation/results.json',rows)
            print({k:row[k] for k in ['arm','scenario','net_pnl','Q','tail_pnl']},flush=True)

if __name__=='__main__':main()
