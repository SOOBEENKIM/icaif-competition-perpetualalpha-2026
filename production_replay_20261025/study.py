import argparse,gzip,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pas_strategy_audit.data import load
from competition_live.profile import load_profile
from .protocol import ROOT,OUT,DATA,save,sha
from .engine import packets,run

def prepare():
    p=load_profile();items=packets(DATA,p)
    save(OUT/'packets.json',items)
    save(OUT/'input_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in [*DATA.glob('*.json'),*DATA.glob('*.npz'),*list((DATA/'raw').glob('*.json'))]})
    print('Prepared',len(items),'historical envelopes',flush=True)

def job(arm,scenario,switch):
    cfg,d=load(DATA);profile=load_profile();protocol=json.loads((OUT/'protocol.json').read_text())
    sc=protocol['scenarios'][scenario];source=json.loads((OUT/'packets.json').read_text())
    funding=json.loads((DATA/'funding.json').read_text())
    t=time.time();r=run(d,profile,sc,funding,source,cfg['start_ms']//1000,cfg['cutoff_ms']//1000,arm,switch)
    row=dict(arm=arm,scenario=scenario,switch=switch,**r['metrics'],run_seconds=time.time()-t,
        fill_fragments=len(r['fills']),orders=len(r['orders']),risk_violations=len(r['audit']['risk_violations']))
    path=OUT/'development'/f'{arm}_{scenario}_{switch or "full"}.json.gz';path.parent.mkdir(exist_ok=True)
    with gzip.open(path,'wt') as f:json.dump(r,f,allow_nan=False,default=lambda x:x.tolist())
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--one',action='store_true');args=p.parse_args()
    if args.action=='prepare':return prepare()
    files=[*Path(__file__).parent.glob('*.py'),ROOT/'candidate_handoff/portfolio.py',ROOT/'candidate_handoff/risk_priority.py',ROOT/'competition_live/coordinator.py',ROOT/'competition_live/live_account.py']
    save(OUT/'development_source_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    tasks=[('incumbent','base',None)] if args.one else [(a,s,w) for a in ['incumbent','refresh_entry_1h','fixed_entry_1h'] for s in ['base','stress','no_receipts'] for w in ([None] if a=='incumbent' else [None,1792767600])]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,*x) for x in tasks]):
            row=f.result();rows.append(row);save(OUT/'development'/('smoke.json' if args.one else 'results.json'),rows);print(row,flush=True)

if __name__=='__main__':main()
