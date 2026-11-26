"""Read-only public funding receipts for the ten mainnet reference markets."""
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
from ..model import SYMBOLS
from ..v3.public import REFERENCES
from ..common_comparison.models import load_inputs, Market
from ..common_comparison.experiment import ROOT, write, digest, stamp

DEST = ROOT/'replay_lab/output/pairs_rl/20261013-v1'


def fetch():
    _, original, _ = load_inputs(ROOT)
    start, end = int(original['known'][0])*1000, int(original['known'][-1])*1000+999
    for symbol in SYMBOLS:
        coin=REFERENCES[symbol]
        path=DEST/'funding_receipts'/f'{symbol}.json'
        if path.exists():
            continue
        batches=[]; cursor=start; error=None
        for page in range(16):
            payload=dict(type='fundingHistory',coin=coin,startTime=cursor,endTime=end)
            request=urllib.request.Request('https://api.hyperliquid.xyz/info',data=json.dumps(payload).encode(),
                headers={'Content-Type':'application/json'},method='POST')
            try:
                with urllib.request.urlopen(request,timeout=25) as response:
                    rows=json.load(response)
                if not isinstance(rows,list):
                    raise ValueError('Unexpected public response')
                for r in rows:
                    if r['coin']!=coin or not cursor<=int(r['time'])<=end:
                        raise ValueError('Wrong coin or out-of-range funding event')
                batches.append(dict(request=payload,response=rows))
                if not rows:
                    break
                cursor=max(int(r['time']) for r in rows)+1
                if cursor>end:
                    break
                time.sleep(.12)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                break
        write(path,dict(network='mainnet',endpoint='https://api.hyperliquid.xyz/info',symbol=symbol,
            fetched_at=stamp(),start=start,end=end,batches=batches,error=error))
        print(symbol,sum(len(b['response']) for b in batches),'funding rows',error,flush=True)


def load_market():
    _, original, manifest=load_inputs(ROOT)
    times=original['known']; rates=np.zeros_like(original['close']); observed=np.zeros_like(rates,dtype=bool)
    mapping={int(t):i for i,t in enumerate(times)}; receipts={}
    for j,symbol in enumerate(SYMBOLS):
        path=DEST/'funding_receipts'/f'{symbol}.json'
        if not path.exists():
            continue
        receipts[symbol]=digest(path)
        data=json.loads(path.read_text())
        if data['network']!='mainnet' or data['symbol']!=symbol:
            raise ValueError('Incorrect funding source')
        for b in data['batches']:
            for r in b['response']:
                raw=int(r['time']); t=raw//3600000*3600
                if r['coin']!=REFERENCES[symbol] or raw%3600000>=1000:
                    raise ValueError('Incorrect funding coin/hour')
                if t not in mapping:
                    continue
                i=mapping[t]; value=float(r['fundingRate'])
                if not np.isfinite(value):
                    raise ValueError('Nonfinite funding')
                if observed[i,j] and rates[i,j]!=value:
                    raise ValueError('Funding conflict')
                rates[i,j]=value; observed[i,j]=True
    market=Market(times,original['close'],original['volume'],original['trades'],rates,observed,tuple(SYMBOLS))
    return market,original,dict(data=manifest,funding_receipts=receipts,
        observed_hours={s:int(observed[:,j].sum()) for j,s in enumerate(SYMBOLS)})


if __name__=='__main__':
    fetch()
