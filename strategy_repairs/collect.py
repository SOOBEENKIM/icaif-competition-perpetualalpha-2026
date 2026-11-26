import json
import time
import urllib.request
from .protocol import OUT, freeze
from all_market_strategy.signals import SYMBOLS

URL='https://api.hyperliquid-testnet.xyz/info'


def main():
    spec=freeze();out=OUT/'pas';(out/'raw').mkdir(parents=True,exist_ok=True)
    cfg=dict(start_ms=spec['pas_start_ms'],cutoff_ms=spec['pas_cutoff_ms'],
             warmup_start_ms=spec['pas_start_ms']-6*3600000,symbols=list(SYMBOLS))
    (out/'dataset.json').write_text(json.dumps(cfg,indent=2))
    for s in SYMBOLS:
        for name,body in [
            ('pas_candles_'+s,dict(type='candleSnapshot',req=dict(coin='pas:'+s,interval='1m',
                 startTime=cfg['warmup_start_ms'],endTime=cfg['cutoff_ms']-1))),
            ('pas_funding_'+s,dict(type='fundingHistory',coin='pas:'+s,
                 startTime=cfg['warmup_start_ms'],endTime=cfg['cutoff_ms']))]:
            path=out/'raw'/f'{name}.json'
            if path.exists():
                assert json.loads(path.read_text())['request']==body;continue
            req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            started=time.time()
            with urllib.request.urlopen(req,timeout=30) as f:data=json.load(f)
            assert isinstance(data,list) and len(data)<5000
            path.write_text(json.dumps(dict(url=URL,request=body,started_ms=int(started*1000),
                received_ms=int(time.time()*1000),data=data)))
            print(name,len(data),flush=True);time.sleep(1.2)


if __name__=='__main__':
    main()
