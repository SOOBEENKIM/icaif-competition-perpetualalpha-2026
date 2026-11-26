"""Finish the frozen read-only prospective comparison once its clock ends."""
import argparse,gzip,json,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from competition_live.profile import load_profile
from competition_live.runtime import private_load
from .protocol import ROOT,OUT,save,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    protocol=json.loads((OUT/'forward_contract.json').read_text())
    start=datetime.fromisoformat(protocol['start_utc'].replace('Z','+00:00')).timestamp()
    end=datetime.fromisoformat(protocol['end_utc'].replace('Z','+00:00')).timestamp()
    for rel,h in protocol['candidate_hashes'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Frozen forward candidate changed')
    if args.wait:
        while time.time()<end+65:
            save(OUT/'forward_job_status.json',dict(at=time.time(),state='collecting_until_fixed_end',end=end,orders_sent=0))
            time.sleep(min(30,max(1,end+65-time.time())))
    if time.time()<end+65:raise ValueError('Do not peek at prospective result early')
    from .book_replay import load_books,run_book
    dest=OUT/'forward_evaluation';dest.mkdir(exist_ok=True)
    if (dest/'started.json').exists():raise FileExistsError('Forward opened once; keep original result')
    save(dest/'started.json',dict(at=time.time(),start=start,end=end,candidate='fixed_entry_1h',orders_sent=0))
    profile=load_profile();funding=[]
    for symbol in profile['trading']['symbols']:
        request=dict(type='fundingHistory',coin='pas:'+symbol,startTime=int(start*1000),endTime=int(end*1000))
        req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as response:rows=json.load(response)
        save(dest/(symbol+'_funding.json'),dict(request=request,received_at=time.time(),rows=rows))
        funding.extend(dict(r,coin='pas:'+symbol) for r in rows if start*1000<r['time']<=end*1000)
        time.sleep(2)
    streams,audit=load_books((OUT/'capture').glob('*.jsonl.gz'),end)
    save(dest/'capture_audit.json',audit)
    source=sorted([json.loads(p.read_text()) for p in (OUT/'forward_signals').glob('*.json')],key=lambda r:r['received_at'])
    seed=private_load(OUT/'forward_seed.pkl')['portfolio'];results=[]
    for scenario,sc in dict(base=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=1.),
                            stress=dict(fee_bps=12.,depth_fraction=.5,latency_ms=3000,receipt_multiplier=.5)).items():
        for arm in ['incumbent','fixed_entry_1h']:
            r=run_book(streams,source,profile,sc,start,end,seed,arm,funding)
            with gzip.open(dest/f'{arm}_{scenario}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda x:x.tolist())
            row={k:r[k] for k in ['arm','net_pnl','fee_increment','funding_increment','price_pnl','start_equity','end_equity','exposure_times']}
            row.update(scenario=scenario,orders=len(r['orders']),fills=len(r['fills']),failures=len(r['failures']),missing_marks=len(r['missing_mark_times']),
                risk_violations=len(r['risk_violations']),accounting_error=r['accounting_error'],unknown=r['unknown'],pending_orders=r['pending_orders'],fatal=r['fatal'])
            results.append(row);save(dest/'results.json',results);print(row,flush=True)
    save(OUT/'forward_job_status.json',dict(at=time.time(),state='evaluation_complete',results=str(dest/'results.json'),orders_sent=0))

if __name__=='__main__':main()
