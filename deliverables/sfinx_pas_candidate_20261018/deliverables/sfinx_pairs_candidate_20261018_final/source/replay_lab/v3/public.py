"""Allowlisted public /info data only. No credentials, user fields or /exchange path."""
import json
import time
import urllib.request
from pathlib import Path

from ..model import SYMBOLS, Candle

URLS={'testnet':'https://api.hyperliquid-testnet.xyz/info','mainnet':'https://api.hyperliquid.xyz/info'}
REFERENCES=dict(zip(SYMBOLS,('BTC','ETH','HYPE','xyz:SP500','xyz:EWY','xyz:EWJ','xyz:BRENTOIL','xyz:GOLD','xyz:EUR','xyz:JPY')))


def public_info(network,payload):
    kind=payload.get('type')
    if network not in URLS:raise ValueError('Unknown public network')
    if kind in ('meta','metaAndAssetCtxs','perpDexLimits'):
        valid=network=='testnet' and payload=={'type':kind,'dex':'par'}
    elif kind=='l2Book':
        valid=network=='testnet' and set(payload)=={'type','coin'} and payload['coin'] in {'par:'+s for s in SYMBOLS}
    elif kind=='candleSnapshot':
        req=payload.get('req',{})
        valid=(network=='mainnet' and set(payload)=={'type','req'} and
               set(req)=={'coin','interval','startTime','endTime'} and req['coin'] in REFERENCES.values() and req['interval']=='1h')
    elif kind=='fundingHistory':
        valid=(network=='testnet' and set(payload)=={'type','coin','startTime','endTime'} and payload['coin'] in {'par:'+s for s in SYMBOLS})
    else:valid=False
    if not valid:raise ValueError('Only allowlisted public market-data queries are permitted')
    request=urllib.request.Request(URLS[network],data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(request,timeout=20) as response:
        return json.load(response)


def venue_snapshot():
    started=time.time()
    meta,contexts=public_info('testnet',{'type':'metaAndAssetCtxs','dex':'par'})
    mapping={asset['name']:(asset,context) for asset,context in zip(meta['universe'],contexts)}
    markets={}
    for s in SYMBOLS:
        asset,context=mapping['par:'+s]
        markets[s]={'spec':asset,'context':context}
    limits=public_info('testnet',{'type':'perpDexLimits','dex':'par'})
    books={}
    for s in SYMBOLS:
        book=public_info('testnet',{'type':'l2Book','coin':'par:'+s})
        if book.get('coin')!='par:'+s:raise ValueError('Wrong market book')
        books[s]={'received_at':time.time(),'book':book}
    return {'started_at':started,'received_at':time.time(),'network':'testnet','dex':'par',
            'markets':markets,'limits':limits,'books':books}


def references(now=None,hours=80):
    now=now or time.time();end=int(now//3600)*3600;data={}
    for s,coin in REFERENCES.items():
        raw=public_info('mainnet',{'type':'candleSnapshot','req':{'coin':coin,'interval':'1h',
                         'startTime':(end-hours*3600)*1000,'endTime':end*1000-1}})
        rows=[]
        for c in raw:
            if c['s']!=coin or c['i']!='1h':raise ValueError('Wrong reference candle')
            if int(c['t'])//1000+3600>end:continue
            rows.append(Candle(int(c['t'])//1000,*[float(c[k]) for k in ('o','h','l','c','v')],int(c['n'])))
        rows=sorted(rows,key=lambda c:c.time)
        if len(rows)<49 or any(b.time-a.time!=3600 for a,b in zip(rows,rows[1:])):
            raise ValueError('Missing or insufficient hourly reference candles: '+s)
        if rows[-1].time+3600!=end:raise ValueError('Stale reference candles: '+s)
        data[s]=rows
    # Different instruments may return one extra starting candle.
    common=set.intersection(*(set(c.time for c in rows) for rows in data.values()))
    return {s:[c for c in rows if c.time in common] for s,rows in data.items()}


if __name__=='__main__':
    folder=Path(__file__).resolve().parents[1]/'output/paper_forward_v3'
    folder.mkdir(exist_ok=True)
    snap=venue_snapshot()
    path=folder/'initial_public_snapshot.json';path.write_text(json.dumps(snap,separators=(',',':')))
    print('Saved public par snapshot:',path)
    print('Limits:',json.dumps(snap['limits']))
    print('Markets:',[(s,v['spec']['szDecimals']) for s,v in snap['markets'].items()])
