"""Public, read-only PAS account survey. No signing clients or exchange actions."""
import concurrent.futures as cf
import datetime as dt
import json
import math
from pathlib import Path
import threading
import time
import urllib.request

OUT = Path(__file__).resolve().parent
RAW = OUT / 'raw'
START = 1792411200000
FUNDER = '0x304199b126a2bb77dedda4c608a9f1dadfada521'
USER = '0x125778065a578d598d88bddaa3fdd635280d0e47'
LOCK = threading.Lock()
SPENT = []

def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)

def reserve(weight):
    while True:
        with LOCK:
            t = time.monotonic()
            SPENT[:] = [r for r in SPENT if t-r[0] < 61]
            if sum(r[1] for r in SPENT)+weight <= 600:
                slot=[t,weight];SPENT.append(slot);return slot
        time.sleep(.2)

def api(body):
    base=2 if body['type'] in ('clearinghouseState','spotClearinghouseState','l2Book','orderStatus') else 20
    for attempt in range(3):
        slot=reserve(base+100 if body['type'] in ('userFillsByTime','userFunding','historicalOrders') else base)
        sent=now()
        try:
            req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=25) as res:data=json.load(res)
            with LOCK:slot[1]=base+(math.ceil(len(data)/20) if isinstance(data,list) and body['type'] in ('userFillsByTime','userFunding','recentTrades','historicalOrders') else 0)
            return dict(request=body,sent_kst=sent,received_kst=now(),response=data)
        except Exception as e:
            if attempt==2:return dict(request=body,sent_kst=sent,received_kst=now(),error=str(e))
            time.sleep(2*(attempt+1))

def history(kind,address,end,start=START):
    cursor=start;rows={};pages=[];complete=False;note=''
    for _ in range(30):
        b=dict(type=kind,user=address,startTime=cursor,endTime=end)
        if kind=='userFillsByTime':b['aggregateByTime']=False
        p=api(b);pages.append(p);data=p.get('response')
        if not isinstance(data,list):note='request_failed';break
        before=len(rows)
        for row in data:
            key=(row.get('coin'),row['time'],row.get('tid')) if kind=='userFillsByTime' else json.dumps(row,sort_keys=True)
            rows[key]=row
        limit=2000 if kind=='userFillsByTime' else 500
        if len(data)<limit:complete=True;break
        nxt=max(r['time'] for r in data)
        if nxt<=cursor and len(rows)==before:note='pagination_stalled';break
        cursor=nxt
    return dict(rows=sorted(rows.values(),key=lambda r:r['time']),pages=pages,pagination_complete=complete,note=note,
                retention_limit_possible=kind=='userFillsByTime' and len(rows)>=10000)

def collect(address,grants,extra=False):
    path=RAW/'wallets'/(address+'.json')
    if path.exists():return json.loads(path.read_text())
    w=dict(address=address,grants=grants,extra_observed_wallet=extra,started_kst=now())
    w['positions']=api(dict(type='clearinghouseState',user=address,dex='pas'))
    w['spot']=api(dict(type='spotClearinghouseState',user=address))
    end=int(time.time()*1000);w['history_end_ms']=end
    w['fills']=history('userFillsByTime',address,end)
    pas=[f for f in w['fills']['rows'] if f['coin'].startswith('pas:')]
    active=bool(pas or w['positions'].get('response',{}).get('assetPositions'))
    if active:
        w['funding']=history('userFunding',address,end)
        w['orders']=api(dict(type='frontendOpenOrders',user=address,dex='pas'))
    else:
        w['funding']=dict(rows=[],pages=[],pagination_complete=True,note='no PAS fills or positions')
        w['orders']=dict(response=[],note='not queried: no PAS fills or positions')
    w['ledger']=history('userNonFundingLedgerUpdates',address,end,start=START-600000)
    w['completed_kst']=now();save(path,w)
    print(json.dumps(dict(address=address,pas_fills=len(pas),positions=len(w['positions'].get('response',{}).get('assetPositions',[])),history_complete=w['fills']['pagination_complete']),ensure_ascii=False),flush=True)
    return w

def main():
    cutoff=int(time.time()*1000)
    p=RAW/'funder_ledger.json'
    if p.exists():ledger=json.loads(p.read_text())
    else:ledger=history('userNonFundingLedgerUpdates',FUNDER,cutoff,start=START-600000);save(p,ledger)
    grants={}
    for row in ledger['rows']:
        d=row['delta']
        if d.get('token')=='USDP' and d.get('user','').lower()==FUNDER and float(d.get('amount',0))==100000:
            grants.setdefault(d['destination'].lower(),[]).append(row)
    if USER not in grants:raise RuntimeError('SFINX grant missing from discovery; investigate pagination')
    market=json.loads((RAW/'market.json').read_text())['response']
    recent={}
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        pairs=list(pool.map(lambda u:(u['name'],api(dict(type='recentTrades',coin=u['name']))),market[0]['universe']))
    recent=dict(pairs);save(RAW/'recent_trades.json',recent)
    extras={a.lower() for r in recent.values() for f in r.get('response',[]) for a in f.get('users',[]) if a and int(a,16)}-set(grants)
    manifest=dict(created_kst=now(),phase_start_ms=START,grant_wallets=grants,extra_observed_wallets=sorted(extras),team_name_mapping='Only SFINX confirmed; official public feed does not expose addresses.',funder_ledger_complete=ledger['pagination_complete'])
    save(RAW/'manifest.json',manifest)
    print('DISCOVERED '+json.dumps(dict(granted=len(grants),extras=len(extras))),flush=True)
    todo=sorted(grants,key=lambda a:(a!=USER,a))+sorted(extras)
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda a:collect(a,grants.get(a,[]),a not in grants),todo))
    manifest['completed_kst']=now();save(RAW/'manifest.json',manifest)

if __name__=='__main__':main()
