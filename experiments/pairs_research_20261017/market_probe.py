"""Unsigned public par quotes; never imports the trading wallet or Exchange."""
import json,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .research import save

ENDPOINT='https://api.hyperliquid-testnet.xyz/info'
HERE=Path(__file__).resolve().parent

def query(payload):
    assert payload['type'] in ('metaAndAssetCtxs','l2Book')
    begin=time.time();req=urllib.request.Request(ENDPOINT,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=25) as r:body=json.load(r)
    return dict(request=payload,sent_at=begin,received_at=time.time(),response=body)

def vwap(levels,quantity):
    left=quantity;value=0.;filled=0.
    for level in levels:
        n=min(left,float(level['sz']));value+=n*float(level['px']);filled+=n;left-=n
        if left<1e-10:break
    return dict(vwap=value/filled if filled else None,filled_qty=filled,complete=left<1e-8)

def main():
    raw=HERE/'public_probe';raw.mkdir(exist_ok=True)
    symbols=['BTC','ETH','HYPE','SP500','EWY','EWJ']
    requests=[dict(type='metaAndAssetCtxs',dex='par')]+[dict(type='l2Book',coin='par:'+s) for s in symbols]
    with ThreadPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(query,requests))
    save(raw/'receipts.json',receipts)
    meta,ctx=receipts[0]['response'];contexts={u['name']:c for u,c in zip(meta['universe'],ctx)}
    rows=[]
    for receipt in receipts[1:]:
        b=receipt['response'];bid,ask=b['levels'];coin=b['coin'];c=contexts[coin]
        if not bid or not ask:
            rows.append(dict(coin=coin,tradable=False));continue
        mid=(float(bid[0]['px'])+float(ask[0]['px']))/2
        sizes=[]
        for notional in (500.,10000.,60000.):
            quantity=notional/mid;buy=vwap(ask,quantity);sell=vwap(bid,quantity)
            sizes.append(dict(notional=notional,buy=buy,sell=sell,
                instantaneous_roundtrip_cost_bps=(10000*(buy['vwap']-sell['vwap'])/mid+9*(buy['vwap']+sell['vwap'])/mid)
                    if buy['complete'] and sell['complete'] else None))
        rows.append(dict(coin=coin,tradable=True,oracle=float(c['oraclePx']),mark=float(c['markPx']),mid=mid,
            hourly_funding=float(c['funding']),mid_basis_bps=10000*(mid/float(c['oraclePx'])-1),
            book_time=b['time'],received_at=receipt['received_at'],book_age_ms=receipt['received_at']*1000-b['time'],
            sizes=sizes))
    save(raw/'summary.json',dict(at=time.time(),rows=rows,live_orders_sent=0,
        limitations=['Sequential snapshots are not an atomic pair fill','A present book is not a future exit guarantee','Instant roundtrip quote friction is diagnostic, not a predicted trade return']))
    for row in rows:print(row['coin'],row.get('mid_basis_bps'),row.get('sizes'),flush=True)

if __name__=='__main__':main()
