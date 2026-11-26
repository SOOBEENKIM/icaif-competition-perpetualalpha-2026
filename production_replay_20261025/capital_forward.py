"""Separately frozen prospective capital-usage comparison; no signed orders."""
import argparse,gzip,json,time
from datetime import datetime,timezone
from .protocol import ROOT,OUT,save,sha

DEST=OUT/'capital_forward'

def freeze():
    DEST.mkdir(exist_ok=True)
    if (DEST/'protocol.json').exists():raise FileExistsError('One predeclared comparison')
    now=time.time();start=(int(now)//3600+1)*3600;end=1792915500
    if start>=end:raise ValueError('No future assessment window')
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),start=start,end=end,
        primary='size_only_2',control='incumbent',diagnostic='hourly_fresh',
        rule='Predefined2x anchor capital budget, unchanged6h schedule/15minentry/costforecast/risk-retention ratios and all hardportfolio caps. Same copied real own inventory/cash/fees/owner state. The1h cadence arm is diagnostic only, not a substitute winner. No model refit or current PnL used in selection.',
        reason='Conservative candle capacity makes1x/2x identical and does not represent actual~10k live own positions. Direct received-depth comparison can measure capacity and costs. Scaling can improve PnL by increasing risk, not proof of better predictive alpha.',
        acceptance='Positive incrementalPnL and improvement versuscontrol after exit-cost reserve in base andstress; no missing funding oracle, accounting/risk/time violations. Commonperiod starts/ends; first-use fixed window, not long-term efficacy. Report all3arms; no retuning on this window.',
        funding='Settled rates with recent pre-event observed oracle; disclosed approximation. Missing oracle invalidates, never hidden zero cash.',
        hashes={p:sha(ROOT/p) for p in ['production_replay_20261025/size_study.py','production_replay_20261025/hourly_portfolio.py','production_replay_20261025/book_replay.py']},
        seed_sha=sha(OUT/'forward_seed.pkl'),orders_sent=0))

def run(wait):
    p=json.loads((DEST/'protocol.json').read_text())
    if wait:
        while time.time()<p['end']+300:
            save(DEST/'status.json',dict(at=time.time(),state='waiting_for_frozen_end',start=p['start'],end=p['end'],orders_sent=0))
            time.sleep(min(30,max(1,p['end']+300-time.time())))
    if time.time()<p['end']+300:raise ValueError('No prospective peeking')
    for path,digest in p['hashes'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen capital candidate/evaluator changed: '+path)
    if sha(OUT/'forward_seed.pkl')!=p['seed_sha']:raise ValueError('Seed changed')
    if (DEST/'started.json').exists():raise FileExistsError('Keep first opening')
    save(DEST/'started.json',dict(at=time.time()))
    from competition_live.profile import load_profile
    from competition_live.runtime import private_load
    from .book_replay import load_books,run_book
    profile=load_profile();funding=[]
    for s in profile['trading']['symbols']:
        path=OUT/'forward_evaluation'/(s+'_funding.json')
        if not path.exists():raise ValueError('Settled funding read not completed; do not replace with predicted funding')
        for row in json.loads(path.read_text())['rows']:
            if p['start']*1000<row['time']<=p['end']*1000:funding.append(dict(row,coin='pas:'+s))
    streams,audit=load_books((OUT/'capture').glob('*.jsonl.gz'),p['end']);save(DEST/'capture_audit.json',audit)
    source=sorted([json.loads(f.read_text()) for f in (OUT/'forward_signals').glob('*.json')],key=lambda x:x['received_at'])
    seed=private_load(OUT/'forward_seed.pkl')['portfolio'];rows=[]
    for scenario,sc in dict(base=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=1.),
                            stress=dict(fee_bps=12.,depth_fraction=.5,latency_ms=3000,receipt_multiplier=.5)).items():
        for arm in ['incumbent','size_only_2','hourly_fresh']:
            r=run_book(streams,source,profile,sc,p['start'],p['end'],seed,arm,funding)
            with gzip.open(DEST/f'{arm}_{scenario}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda x:x.tolist())
            row={k:r[k] for k in ['arm','net_pnl','fee_increment','funding_increment','price_pnl','start_equity','end_equity','exposure_times']}
            first,last=r['snapshots'][0],r['snapshots'][-1]
            row.update(scenario=scenario,exit_reserve_increment=r['net_pnl']-(last['notional']-first['notional'])*(sc['fee_bps']+10)/10000,
                mean_notional=sum(x['notional'] for x in r['snapshots'])/len(r['snapshots']),orders=len(r['orders']),
                fills=len(r['fills']),failures=len(r['failures']),missing_marks=len(r['missing_mark_times']),
                risk_violations=len(r['risk_violations']),accounting_error=r['accounting_error'],
                unknown=r['unknown'],pending_orders=r['pending_orders'],fatal=r['fatal'])
            rows.append(row);save(DEST/'results.json',rows);print(row,flush=True)
    save(DEST/'status.json',dict(at=time.time(),state='finished',orders_sent=0))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);a.add_argument('--wait',action='store_true');args=a.parse_args()
    freeze() if args.action=='freeze' else run(args.wait)
