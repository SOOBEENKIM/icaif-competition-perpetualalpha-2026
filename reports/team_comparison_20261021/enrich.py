"""Additional public account-mode and order-type evidence, after main collection."""
import concurrent.futures as cf
import json
from pathlib import Path
from collect import RAW, api, save, now

def one(path):
    w=json.loads(path.read_text());a=w['address'];out=RAW/'extra'/(a+'.json')
    if out.exists():return
    data=dict(address=a,started_kst=now(),mode=api(dict(type='userAbstraction',user=a)))
    fills=[f for f in w['fills']['rows'] if f['coin'].startswith('pas:')]
    if fills:
        data['historical_orders']=api(dict(type='historicalOrders',user=a))
    data['completed_kst']=now();save(out,data)
    print(a,flush=True)

if __name__=='__main__':
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one,sorted((RAW/'wallets').glob('*.json'))))
