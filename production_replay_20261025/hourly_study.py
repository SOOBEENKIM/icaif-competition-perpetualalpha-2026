import gzip,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from competition_live.profile import load_profile
from pas_strategy_audit.data import load
from .engine import run
from .protocol import ROOT,OUT,DATA,save,sha

DEST=OUT/'hourly_fresh'
SWITCH=1792767600

def job(arm,scenario,switch):
    cfg,d=load(DATA);p=load_profile();sc=json.loads((OUT/'protocol.json').read_text())['scenarios'][scenario]
    source=json.loads((OUT/'packets.json').read_text());funding=json.loads((DATA/'funding.json').read_text())
    r=run(d,p,sc,funding,source,cfg['start_ms']//1000,cfg['cutoff_ms']//1000,arm,SWITCH if switch else None)
    first=next(x for x in r['snapshots'] if x['time']==SWITCH);last=r['snapshots'][-1]
    row=dict(arm=arm,scenario=scenario,switch=switch,**r['metrics'],
             tail_pnl=last['team_equity']-first['team_equity'],
             tail_exit_reserve_pnl=last['exit_reserve_equity']-first['exit_reserve_equity'],
             fills=len(r['fills']),risk_violations=len(r['audit']['risk_violations']),
             decisions=sum(e['kind']=='hourly_fresh_decision' for e in r['events']))
    with gzip.open(DEST/f'{arm}_{scenario}_{"switch" if switch else "full"}.json.gz','wt') as f:
        json.dump(dict(row=row,result=r),f,allow_nan=False,default=lambda x:x.tolist())
    return row

def main():
    if (DEST/'protocol.json').exists():raise FileExistsError('Preserve original runs')
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='hourly_fresh',
        hypothesis='Signal producer already refreshes hourly. Recompute optimal actual-inventory target using each fresh hour, not only6h boundaries. Preserve exact alpha, reference model, all10symbols, sizing, protected cuts, economic costs, sharedhardlimits and15minintent. New entries fixed to each new decision mark+/-10bp. Missing/late hourly publication cannot be backfilled. No forced trades or short percentage.',
        comparisons='Unchanged current6h control. Same production account/risk/quote code and same initial50k own+50kreserve. Fullpath plus same inventory/cash/funding prefix throughOct23 15UTC. All old history development, later first-use data required.',
        source_hashes={p:sha(ROOT/p) for p in ['production_replay_20261025/hourly_portfolio.py','production_replay_20261025/engine.py']},orders_sent=0))
    rows=[];tasks=[(a,s,w) for a in ['incumbent','hourly_fresh'] for s in ['base','stress','no_receipts'] for w in ([False] if a=='incumbent' else [False,True])]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*x) for x in tasks]):
            r=f.result();rows.append(r);save(DEST/'results.json',rows);print(r,flush=True)

if __name__=='__main__':main()
