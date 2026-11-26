"""Verify public trade side against the venue's crossed user-fill field."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request
from experiments.pairs_research_20261017.research import save

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    history = ROOT / 'experiments/pairs_research_20261017/par_history/20261016T173824Z'
    examples = {}
    for symbol in ('BTC','ETH','HYPE'):
        rows = json.loads((history/f'recentTrades_{symbol}.json').read_text())['response']
        for r in rows:
            key = symbol,r['side']
            if key not in examples and len(r.get('users',[]))==2: examples[key] = r
    requests = []
    for trade in examples.values():
        for user in trade['users']:
            requests.append((trade,user,dict(type='userFillsByTime',user=user,startTime=trade['time'],endTime=trade['time']+1,aggregateByTime=False)))
    def fetch(item):
        trade,user,payload=item; sent=time.time()
        request=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=20) as response: rows=json.load(response)
        matched=[r for r in rows if r['coin']==trade['coin'] and r.get('tid')==trade['tid'] and r['time']==trade['time']]
        return dict(trade=trade,user=user,request=payload,sent_at=sent,received_at=time.time(),matched=matched)
    with ThreadPoolExecutor(max_workers=3) as pool: receipts=list(pool.map(fetch,requests))
    checks=[]
    for trade in examples.values():
        fills=[f for r in receipts if r['trade']==trade for f in r['matched']]
        takers=[f for f in fills if f['crossed']]; makers=[f for f in fills if not f['crossed']]
        checks.append(dict(coin=trade['coin'],time=trade['time'],tid=trade['tid'],trade_side=trade['side'],
            taker_side=[f['side'] for f in takers],maker_side=[f['side'] for f in makers],
            passed=len(takers)==len(makers)==1 and takers[0]['side']==trade['side'] and makers[0]['side']!=trade['side']))
    save(HERE/'trade_side_verification.json',dict(at=time.time(),checks=checks,receipts=receipts,
        verified=bool(checks) and all(c['passed'] for c in checks),orders_sent=0))
    print(json.dumps(checks,indent=2))
    assert checks and all(c['passed'] for c in checks)


if __name__=='__main__':main()
