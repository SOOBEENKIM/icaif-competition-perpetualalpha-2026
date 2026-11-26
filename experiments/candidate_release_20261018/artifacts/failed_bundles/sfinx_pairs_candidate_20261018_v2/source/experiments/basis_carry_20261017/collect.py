"""Public /info queries only. Retain requests, responses and provenance."""
import json,time,hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scripts.refresh_public_wallets import api

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'artifacts'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
def ts(s):return int(np.datetime64(s,'ms').astype('int64'))

def main():
    OUT.mkdir(exist_ok=True);raw=OUT/'raw';raw.mkdir(exist_ok=True)
    if (raw/'manifest.json').exists():raise FileExistsError('Public inputs already frozen')
    config=json.loads((HERE/'protocol.json').read_text());start=ts(config['funding_study']['start']);end=int(time.time()*1000)
    # Freeze the hypothesis before collecting its outcome data.
    save(raw/'plan.json',dict(at_ms=end,protocol_sha256=sha(HERE/'protocol.json'),start=start,end=end,
        source_sha256=sha(__file__),orders_sent=0,sealed_streams_opened=False))
    meta=api({'type':'metaAndAssetCtxs','dex':'par'});save(raw/'metadata.json',meta)
    dexes=api({'type':'perpDexs'});save(raw/'perp_dexes.json',dexes)
    def funding(symbol):
        pages=[];cursor=start;rows={}
        while True:
            page=api({'type':'fundingHistory','coin':'par:'+symbol,'startTime':cursor,'endTime':end});pages.append(page)
            values=page.get('response')
            if not isinstance(values,list):raise RuntimeError(page)
            for row in values:rows[row['time']]=row
            if len(values)<500:break
            nxt=max(r['time'] for r in values)+1
            if nxt<=cursor:raise RuntimeError('Funding pagination stalled')
            cursor=nxt
        output=dict(pages=pages,rows=[rows[t] for t in sorted(rows)])
        save(raw/f'funding_{symbol}.json',output)
        return dict(symbol=symbol,observations=len(rows),first=min(rows) if rows else None,last=max(rows) if rows else None)
    with ThreadPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(funding,config['symbols']))
    save(raw/'manifest.json',dict(start_ms=start,end_ms=end,receipts=receipts,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in raw.glob('*.json')},orders_sent=0))
    print(json.dumps(receipts,indent=2),flush=True)

if __name__=='__main__':main()
