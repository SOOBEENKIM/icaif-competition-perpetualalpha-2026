"""Unsigned public reads only; fixed cutoff, saved raw receipts."""
import json
import time
import urllib.request
from .protocol import OUT, freeze, save


def main():
    p=freeze();target=OUT/'pas'
    cfg=dict(start_ms=p['pas_start']*1000,cutoff_ms=p['pas_end']*1000,
        warmup_start_ms=(p['pas_start']-6*3600)*1000,symbols=p['symbols'])
    save(target/'dataset.json',cfg)
    for s in p['symbols']:
        requests=[('pas_candles_'+s,dict(type='candleSnapshot',req=dict(coin='pas:'+s,
                interval='1m',startTime=cfg['warmup_start_ms'],endTime=cfg['cutoff_ms']-1))),
            ('pas_hourly_'+s,dict(type='candleSnapshot',req=dict(coin='pas:'+s,
                interval='1h',startTime=(p['pas_start']-7*86400)*1000,endTime=cfg['cutoff_ms']-1))),
            ('pas_funding_'+s,dict(type='fundingHistory',coin='pas:'+s,
                startTime=(p['pas_start']-7*86400)*1000,endTime=cfg['cutoff_ms']))]
        for name,body in requests:
            path=target/'raw'/(name+'.json')
            if path.exists():
                assert json.loads(path.read_text())['request']==body
                continue
            started=time.time()
            req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json','User-Agent':'SFINX-research-readonly'})
            with urllib.request.urlopen(req,timeout=30) as r: data=json.load(r)
            if not isinstance(data,list) or len(data)>=5000: raise ValueError('Incomplete pagination')
            save(path,dict(request=body,started_ms=int(started*1000),received_ms=int(time.time()*1000),data=data))
            print(name,len(data),flush=True)
            time.sleep(.8)


if __name__=='__main__': main()
