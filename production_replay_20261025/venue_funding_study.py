import gzip,json
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
from pas_strategy_audit.data import load
from competition_live.profile import load_profile
from .protocol import ROOT,OUT,DATA,save,sha
from .engine import run
from .venue_funding_packet import VenueFundingPacket

DEST=OUT/'venue_funding_production'
SWITCH=1792767600


def job(arm,scenario,switch):
    cfg,d=load(DATA);profile=load_profile();sc=json.loads((OUT/'protocol.json').read_text())['scenarios'][scenario]
    source=json.loads((OUT/'packets.json').read_text());funding=json.loads((DATA/'funding.json').read_text())
    start,end=cfg['start_ms']//1000,cfg['cutoff_ms']//1000
    transform=None if arm=='incumbent' else VenueFundingPacket(funding,SWITCH if switch else start,filter_core=arm=='cost_core')
    r=run(d,profile,sc,funding,source,start,end,packet_transform=transform)
    beginning=next(x for x in r['snapshots'] if x['time']==SWITCH);last=r['snapshots'][-1]
    row=dict(arm=arm,scenario=scenario,switch=switch,**r['metrics'],
        tail_pnl=last['team_equity']-beginning['team_equity'],tail_exit_reserve_pnl=last['exit_reserve_equity']-beginning['exit_reserve_equity'],
        fills=len(r['fills']),risk_violations=len(r['audit']['risk_violations']),
        ranking_eligible=bool(r['fills']) and r['metrics']['completed_daily_returns']>=2,
        note='Q is a conditional numerical formula; no-trade arms are not ranking eligible.')
    with gzip.open(DEST/f'{arm}_{scenario}_{"switch" if switch else "full"}.json.gz','wt') as fp:json.dump(dict(row=row,result=r),fp,allow_nan=False,default=lambda x:x.tolist())
    return row


def main():
    assert not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),primary='cost_core',diagnostic='replace_only',control='incumbent',
        rationale='Port the previously frozen Oct22 venue_funding rule to the actual production decision/accounting path. Current frame uses reference funding to allocate PAS capital. Compare replacement of funding input alone and the original cost filter; do not relabel an older failed PAS trial as a new untested idea.',
        preserved='All10assets,retained pair forecasts,50kcapital,hard risk limits,6hdecision,15minuteentry,originalpullback_any3. Same pre-switch actual simulated inventory. No fitting, parameter grid, or current rank selection.',
        data='Exposed Oct19-Oct24 12UTC PAS histories; exact reconstructed settlement cash and separately lagged rates. Conservative synthetic trade-candle book is conditional, not actual L2.',
        acceptance='Primary positivePnL and betterPnL/Q/terminal reserve bothfull and sameinventorytail underbase/stress; no risk/accounting/timing failures. Zero-trade cash is not an eligible leaderboard improvement.',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'production_replay_20261025/venue_funding_packet.py',ROOT/'production_replay_20261025/venue_funding_study.py']},
        orders_sent=0,live_changed=False))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks=[(a,s,w) for a in ['incumbent','cost_core','replace_only'] for s in ['base','stress','no_receipts'] for w in ([False] if a=='incumbent' else [False,True])]
        for f in as_completed([pool.submit(job,*x) for x in tasks]):
            r=f.result();rows.append(r);save(DEST/'results.json',rows)
            print({k:r[k] for k in ['arm','scenario','switch','net_pnl','tail_pnl','fills','risk_violations']},flush=True)

if __name__=='__main__':main()
