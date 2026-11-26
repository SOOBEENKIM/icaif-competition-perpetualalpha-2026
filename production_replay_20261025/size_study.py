"""Separate capital utilization from prediction/execution changes.

Frozen 2x anchor budget, same risk-retention ratios and hard portfolio limits.
No model refit, extra symbols, return-target selection or claim of better alpha.
"""
import copy,gzip,json,time
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from competition_live.runtime import fingerprint
from .protocol import ROOT,OUT,DATA,save,sha
from .engine import run

DEST=OUT/'anchor_budget'
SWITCH=1792767600

class ScaledPacket:
    def __init__(self,multiplier,start):self.multiplier=multiplier;self.start=start;self.cache={}
    def __call__(self,packet,now):
        if now<self.start:return packet
        key=packet['sha256']
        if key not in self.cache:
            p=copy.deepcopy(packet);body=p['body'];f=body['frame']
            f['anchor_weights']=(np.asarray(f['anchor_weights'])*self.multiplier).tolist()
            body['research_budget_multiplier']=self.multiplier
            p['sha256']=fingerprint(body);self.cache[key]=p
        return self.cache[key]

def job(multiplier,scenario,switch):
    from competition_live.profile import load_profile
    from pas_strategy_audit.data import load
    cfg,d=load(DATA);profile=load_profile();protocol=json.loads((OUT/'protocol.json').read_text())
    source=json.loads((OUT/'packets.json').read_text());funding=json.loads((DATA/'funding.json').read_text())
    start=cfg['start_ms']//1000;end=cfg['cutoff_ms']//1000
    r=run(d,profile,protocol['scenarios'][scenario],funding,source,start,end,
          packet_transform=ScaledPacket(multiplier,SWITCH if switch else start))
    beginning=next(x for x in r['snapshots'] if x['time']==SWITCH);last=r['snapshots'][-1]
    row=dict(multiplier=multiplier,scenario=scenario,switch=switch,**r['metrics'],
             tail_pnl=last['team_equity']-beginning['team_equity'],
             tail_exit_reserve_pnl=last['exit_reserve_equity']-beginning['exit_reserve_equity'],
             fills=len(r['fills']),risk_violations=len(r['audit']['risk_violations']))
    with gzip.open(DEST/f'{multiplier:g}_{scenario}_{"switch" if switch else "full"}.json.gz','wt') as f:
        json.dump(dict(row=row,result=r),f,allow_nan=False,default=lambda x:x.tolist())
    return row

def main():
    if (DEST/'protocol.json').exists():raise FileExistsError('Preserve experiment')
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary_multiplier=2.,
        control_multiplier=1.,
        definition='Scale every10-market anchor by2 before unchanged jointrisk projection, turnover, protectedmodel caps, lot, exchange minimum and execution. Protected caps scale with intended raw strategy budget, retaining original shrink ratios; never undo a within-run protected reduction. Existing50kcapital,asset7500,gross.6,net.4,dailyvol.005 limits unchanged. Existing15minentry and price guard unchanged.',
        hypothesis='Audit capital utilization only. Not a new prediction model and not risk-matched alpha. Higher PnL from larger exposure must be reported with increased volatility/drawdown and Q. No scale grid or winner selection.',
        validation='Same production policy/account under candle-limited conservative base/stress/zero receipts. Fullpath plus exact same-inventory switch. Historical development, incomplete L2 assumptions. Need later first-use data before promotion.',
        source_hash=sha(__file__),orders_sent=0))
    rows=[];tasks=[(m,s,w) for m in [1.,2.] for s in ['base','stress','no_receipts'] for w in ([False] if m==1. else [False,True])]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*x) for x in tasks]):
            r=f.result();rows.append(r);save(DEST/'results.json',rows);print(r,flush=True)

if __name__=='__main__':main()
