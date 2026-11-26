"""Read-only public PAS survey. No signing or live strategy imports."""
import json
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import requests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'reports/market_candidate_audit_20261023/raw'
RAW = OUT / 'raw'
START = 1792411200000
USER = '0x125778065a578d598d88bddaa3fdd635280d0e47'
FUNDER = '0x304199b126a2bb77dedda4c608a9f1dadfada521'
SESSION = requests.Session()
LAST = 0.0

def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))

def api(req):
    global LAST
    time.sleep(max(0, 1.8 - (time.monotonic() - LAST)))
    sent = now()
    LAST = time.monotonic()
    r = SESSION.post('https://api.hyperliquid-testnet.xyz/info', json=req, timeout=25)
    if r.status_code != 200:
        save(RAW / 'error.json', dict(request=req, status=r.status_code, text=r.text[:600], at=now()))
        r.raise_for_status()
    return dict(request=req, sent_kst=sent, received_kst=now(), response=r.json())

def public(name, suffix=''):
    url = 'https://d23m2l9bjnbgul.cloudfront.net/public/' + name + '.json'
    r = SESSION.get(url, params={'t': int(time.time()*1000)}, timeout=25)
    r.raise_for_status()
    d = dict(url=url, received_kst=now(), headers=dict(r.headers), response=r.json())
    save(OUT / (name + suffix + '.json'), d)
    return d

def history(kind, address, end, prior=None):
    prior = prior or {}
    cursor = max(START-600000, prior.get('end_ms', START-600000)-1000)
    rows = {json.dumps(r, sort_keys=True): r for r in prior.get('rows', [])}
    pages = []
    complete = False
    limit = 2000 if kind == 'userFillsByTime' else 500
    for _ in range(25):
        req = dict(type=kind, user=address, startTime=cursor, endTime=end)
        if kind == 'userFillsByTime':
            req['aggregateByTime'] = False
        p = api(req)
        pages.append(p)
        data = p['response']
        if not isinstance(data, list):
            raise ValueError((kind, data))
        before = len(rows)
        for row in data:
            rows[json.dumps(row, sort_keys=True)] = row
        if len(data) < limit:
            complete = True
            break
        nxt = max(r['time'] for r in data)
        if nxt <= cursor and before == len(rows):
            break
        cursor = nxt
    return dict(rows=sorted(rows.values(), key=lambda r:r['time']), pages=pages,
                pagination_complete=complete and prior.get('pagination_complete', True),
                retention_limit_possible=prior.get('retention_limit_possible', False) or (kind=='userFillsByTime' and len(rows)>=10000),
                end_ms=end, prior_source=prior.get('source'))

def prior_wallet(a):
    p = OLD / 'wallets' / (a+'.json')
    return json.loads(p.read_text()) if p.exists() else {}

def old_history(w, key, a):
    d = dict(w.get(key, {}))
    d['end_ms'] = w.get('history_end_ms', START-600000)
    d['source'] = str(OLD / 'wallets' / (a+'.json')) if w else None
    return d

def scan(a, grants):
    path = RAW / 'wallets' / (a+'.json')
    if path.exists():
        return json.loads(path.read_text())
    w = dict(address=a, grants=grants, started_kst=now())
    mode = api(dict(type='userAbstraction', user=a))
    save(RAW/'extra'/(a+'.json'), dict(address=a, mode=mode))
    w['positions'] = api(dict(type='clearinghouseState', user=a, dex='pas'))
    w['spot'] = api(dict(type='spotClearinghouseState', user=a))
    end = int(time.time()*1000)
    w['ledger'] = history('userNonFundingLedgerUpdates', a, end, old_history(prior_wallet(a),'ledger',a))
    w['scan_completed_kst'] = now()
    spot = next((float(x['total']) for x in w['spot']['response']['balances'] if x['coin']=='USDP'), 0.)
    if mode['response']=='unifiedAccount':
        equity = spot
    elif mode['response'] in ('disabled','default','standard'):
        equity = spot + float(w['positions']['response']['marginSummary']['accountValue'])
    else:
        equity = None
    flows=[]; unknown=[]
    for row in w['ledger']['rows']:
        d=row['delta']
        if d.get('token')!='USDP': continue
        sender=d.get('user','').lower(); dest=d.get('destination','').lower()
        if sender==dest==a: continue
        amount=float(d.get('amount',0))
        if dest==a or d.get('type')=='spotGenesis': flows.append(amount)
        elif sender==a: flows.append(-amount)
        else: unknown.append(d)
    old=prior_wallet(a)
    active=bool(w['positions']['response']['assetPositions'] or any(f['coin'].startswith('pas:') for f in old.get('fills',{}).get('rows',[])) or (equity is not None and abs(equity-sum(flows))>1e-7))
    w['scan']=dict(equity=equity, flows=flows, net_inflows=sum(flows), single_grant_100k=len(flows)==1 and abs(flows[0]-100000)<1e-7,
                   pnl=equity-sum(flows) if equity is not None and not unknown else None, unknown_flows=unknown, active=active)
    save(path,w)
    print(json.dumps(dict(stage='scan',address=a,at=now(),**w['scan'])), flush=True)
    return w

def detail(a):
    p=RAW/'wallets'/(a+'.json'); w=json.loads(p.read_text())
    if 'completed_kst' in w:return
    old=prior_wallet(a); end=int(time.time()*1000)
    w['history_end_ms']=end
    for key,typ in [('fills','userFillsByTime'),('funding','userFunding')]:
        w[key]=history(typ,a,end,old_history(old,key,a))
    w['orders']=api(dict(type='frontendOpenOrders',user=a,dex='pas'))
    ex=json.loads((RAW/'extra'/(a+'.json')).read_text())
    ex['historical_orders']=api(dict(type='historicalOrders',user=a))
    save(RAW/'extra'/(a+'.json'),ex)
    # Re-read positions and balance after histories. Flag any concurrent fills later.
    w['positions_initial']=w['positions'];w['spot_initial']=w['spot']
    w['positions']=api(dict(type='clearinghouseState',user=a,dex='pas'))
    w['spot']=api(dict(type='spotClearinghouseState',user=a))
    w['completed_kst']=now();save(p,w)
    print(json.dumps(dict(stage='detail',address=a,at=now(),fills=len(w['fills']['rows']),positions=len(w['positions']['response']['assetPositions']))),flush=True)

def main():
    manifest_path=RAW/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
    else:
        end=int(time.time()*1000)
        fund=history('userNonFundingLedgerUpdates',FUNDER,end)
        save(RAW/'funder.json',fund)
        grants={}
        for r in fund['rows']:
            d=r['delta']
            if d.get('token')=='USDP' and d.get('user','').lower()==FUNDER and float(d.get('amount',0))==100000:
                grants.setdefault(d['destination'].lower(),[]).append(r)
        assert USER in grants
        manifest=dict(started_kst=now(),grant_wallets=grants,funder_pagination_complete=fund['pagination_complete'],phase_start_ms=START)
        save(manifest_path,manifest)
    public('chart','_initial');public('leaderboard','_initial')
    wallets=[scan(a,manifest['grant_wallets'][a]) for a in sorted(manifest['grant_wallets'],key=lambda a:(a!=USER,a))]
    ranked=sorted([w for w in wallets if w['scan']['active'] and w['scan']['pnl'] is not None and w['scan']['single_grant_100k']],key=lambda w:-w['scan']['pnl'])
    # Four extra wallets protect the top-10 boundary from movements during sequential reads.
    selected=list(dict.fromkeys([w['address'] for w in ranked[:14]]+[USER]))
    save(OUT/'selected.json',dict(selected=selected,scan_ranked=[dict(address=w['address'],**w['scan']) for w in ranked]))
    print('SCAN_COMPLETE '+json.dumps(dict(wallets=len(wallets),selected=selected)),flush=True)
    for a in selected:detail(a)
    save(RAW/'market.json',api(dict(type='metaAndAssetCtxs',dex='pas')))
    meta=json.loads((RAW/'market.json').read_text())['response'][0]
    for u in meta['universe']:
        coin=u['name'];save(RAW/'recent_trades'/(coin.replace(':','_')+'.json'),api(dict(type='recentTrades',coin=coin)))
    public('chart');public('leaderboard')
    manifest['completed_kst']=now();save(manifest_path,manifest)
    print('COMPLETE '+now(),flush=True)

if __name__=='__main__':main()
