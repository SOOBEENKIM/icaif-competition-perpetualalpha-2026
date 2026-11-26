"""Public data only. Does not import the trading bot or open credentials."""
import argparse
import gzip
import datetime as dt
import json
import time
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
START = 1792411200000
USER = '0x125778065a578d598d88bddaa3fdd635280d0e47'
FUNDER = '0x304199b126a2bb77dedda4c608a9f1dadfada521'
S = requests.Session()
last = 0

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    payload=json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode()
    tmp.write_bytes(gzip.compress(payload, compresslevel=6) if path.suffix=='.gz' else payload)
    tmp.replace(path)

def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()) if p.exists() else {}

def at(v):
    return int(dt.datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp()*1000) if v else 0

def post(req):
    global last
    for attempt in range(3):
        time.sleep(max(0, 1.8-(time.monotonic()-last)))
        last = time.monotonic()
        r = S.post('https://api.hyperliquid-testnet.xyz/info', json=req, timeout=25)
        if r.status_code == 429:
            time.sleep(30*(attempt+1)); continue
        r.raise_for_status()
        return {'data': r.json(), 'at': int(time.time()*1000)}
    raise RuntimeError('API rate limit: collection paused')

def official(name):
    r = S.get('https://d23m2l9bjnbgul.cloudfront.net/public/'+name+'.json', params={'t': int(time.time()*1000)}, timeout=25)
    r.raise_for_status()
    return {'data': r.json(), 'at': int(time.time()*1000)}

def merge(a, b):
    values = {json.dumps(x, sort_keys=True): x for x in a+b}
    return sorted(values.values(), key=lambda x:x.get('time', 0))

def history(kind, address, old):
    old = old or {'rows': [], 'through': START-1, 'complete': True}
    end = int(time.time()*1000)
    cursor = max(START-600000, old.get('through', START-1)-1000)
    rows = old.get('rows', [])
    limit = 2000 if kind == 'userFillsByTime' else 500
    complete = False
    for page in range(12):
        req = dict(type=kind, user=address, startTime=cursor, endTime=end)
        if kind == 'userFillsByTime': req['aggregateByTime'] = False
        data = post(req)['data']
        if not isinstance(data, list): raise ValueError('Unexpected history response')
        previous = len(rows); rows = merge(rows, data)
        if len(data)<limit: complete = True; break
        nxt = max(x['time'] for x in data)
        if nxt <= cursor and len(rows)==previous: break
        cursor = nxt
    return {'rows': rows, 'through': end if complete else cursor, 'complete': complete and old.get('complete', True),
            'retention': old.get('retention', False) or (kind=='userFillsByTime' and len(rows)>=10000), 'at': int(time.time()*1000)}

def bootstrap():
    manifest = read(ROOT/'reports/top10_live_review_20261024/raw/manifest.json')
    seed = dict(wallets={}, official={}, phase_start=START, initial_capital=100000, generated_at=int(time.time()*1000))
    for address, grants in manifest['grant_wallets'].items():
        out = dict(address=address, grants=grants, errors=[])
        for folder in ['team_comparison_20261021','market_candidate_audit_20261023','top10_live_review_20261024']:
            path = ROOT/'reports'/folder/'raw'
            w = read(path/'wallets'/(address+'.json'))
            extra = read(path/'extra'/(address+'.json'))
            for key, oldkey in [('position','positions'),('spot','spot'),('openOrders','orders')]:
                v = w.get(oldkey)
                if isinstance(v, dict) and 'response' in v: out[key] = {'data':v['response'], 'at':at(v.get('received_kst'))}
            for key, oldkey in [('mode','mode'),('orders','historical_orders')]:
                v = extra.get(oldkey)
                if isinstance(v, dict) and 'response' in v: out[key] = {'data':v['response'], 'at':at(v.get('received_kst'))}
            for key in ['fills','funding','ledger']:
                v = w.get(key)
                if isinstance(v, dict) and 'rows' in v:
                    end = v.get('end_ms') or w.get('history_end_ms') or at(w.get('scan_completed_kst')) or START-1
                    previous = out.get(key, {})
                    out[key] = {'rows':merge(previous.get('rows',[]),v['rows']), 'through':end,
                                'complete':v.get('pagination_complete',False), 'retention':v.get('retention_limit_possible',False), 'at':end}
        seed['wallets'][address] = out
    for key in ['chart','leaderboard']:
        old=read(ROOT/'reports/top10_live_review_20261024'/(key+'.json'))
        seed['official'][key]={'data':old['response'],'at':at(old['received_kst'])}
    return seed

def main():
    p=argparse.ArgumentParser();p.add_argument('--bootstrap-only',action='store_true');p.add_argument('--limit',type=int);args=p.parse_args()
    file=HERE/'public/seed.json.gz'
    seed=read(file) or bootstrap()
    write(file,seed)
    if args.bootstrap_only: print('Bootstrap',len(seed['wallets']));return
    for key in ['chart','leaderboard']:seed['official'][key]=official(key)
    funding=history('userNonFundingLedgerUpdates',FUNDER,{})
    for r in funding['rows']:
        d=r['delta']
        if d.get('user','').lower()==FUNDER and d.get('token')=='USDP' and float(d.get('amount',0))==100000:
            a=d['destination'].lower();seed['wallets'].setdefault(a,dict(address=a,grants=[r],errors=[]))
    seed['market']=post(dict(type='metaAndAssetCtxs',dex='pas'))
    write(file,seed)
    def priority(a):
        w=seed['wallets'][a];s=w.get('spot',{}).get('data',{})
        value=next((float(x['total']) for x in s.get('balances',[]) if x['coin']=='USDP'),0)
        return (a!=USER,-value)
    for i,a in enumerate(sorted(seed['wallets'],key=priority)[:args.limit]):
        w=seed['wallets'][a];w['errors']=[]
        for key,typ in [('mode','userAbstraction'),('position','clearinghouseState'),('spot','spotClearinghouseState'),('openOrders','frontendOpenOrders'),('orders','historicalOrders')]:
            try:
                req=dict(type=typ,user=a)
                if key in ['position','openOrders']:req['dex']='pas'
                w[key]=post(req)
            except Exception as e:w['errors'].append(key+': '+str(e))
        for key,typ in [('fills','userFillsByTime'),('funding','userFunding'),('ledger','userNonFundingLedgerUpdates')]:
            try:w[key]=history(typ,a,w.get(key))
            except Exception as e:w['errors'].append(key+': '+str(e))
        seed['generated_at']=int(time.time()*1000);write(file,seed)
        print(json.dumps(dict(index=i+1,total=len(seed['wallets']),address=a,fills=len(w.get('fills',{}).get('rows',[])),errors=w['errors'])),flush=True)
    for key in ['chart','leaderboard']:seed['official'][key]=official(key)
    seed['market']=post(dict(type='metaAndAssetCtxs',dex='pas'));seed['generated_at']=int(time.time()*1000);write(file,seed)
    print('COMPLETE',seed['generated_at'],flush=True)

if __name__=='__main__':main()
