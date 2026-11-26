import concurrent.futures, datetime, json, math, os, pathlib, threading, time, urllib.request
ROOT=pathlib.Path('/tmp/par-leaderboard-20261014')
RAW=ROOT/'wallets'; RAW.mkdir(exist_ok=True)
OLD=pathlib.Path('/home/ssd-990/soobeenkim/icaif_competition_2026/reports/wallet_review_20261014/raw')
START=int(datetime.datetime(2026,9,1,tzinfo=datetime.timezone.utc).timestamp()*1000)
END=int(time.time()*1000)
LOCK=threading.Lock(); SPENT=[]
sources={}; old={}
def add(a,why):
 if isinstance(a,str) and a.startswith('0x') and len(a)==42 and a!='0x'+'0'*40:
  sources.setdefault(a.lower(),set()).add(why)
for p in OLD.glob('W*.json'):
 d=json.load(open(str(p)));old[d['address']]=d;add(d['address'],'previous observed par trades')
for name in ['funding_team','funding_other']:
 d=json.load(open(str(ROOT/(name+'.json'))))
 for r in d['response']:
  if r['delta'].get('token')=='USDPR':
   add(r['delta'].get('user'),name+' sender');add(r['delta'].get('destination'),name+' recipient')
add('0x125778065a578d598D88bddaA3FdD635280d0E47','user supplied address')
recent_path=ROOT/'recent_trades_refresh.json'
if recent_path.exists():
 for item in json.load(open(str(recent_path))).values():
  for trade in item.get('response',[]):
   for a in trade.get('users',[]):add(a,'refreshed public par or USDPR spot trade')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def reserve(w):
 while True:
  with LOCK:
   t=time.monotonic();SPENT[:]=[x for x in SPENT if t-x[0]<61]
   if sum(x[1] for x in SPENT)+w<=900:
    e=[t,w];SPENT.append(e);return e
   delay=max(.1,61-(t-SPENT[0][0]))
  time.sleep(min(delay,1))
def api(p):
 k=p['type'];base=2 if k in ('clearinghouseState','spotClearinghouseState','exchangeStatus') else 20
 extra=100 if k in ('userFillsByTime','userFunding','historicalOrders') else 0
 slot=reserve(base+extra)
 for retry in range(3):
  try:
   req=urllib.request.Request('https://api.hyperliquid-testnet.xyz/info',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=25) as f:r=json.load(f)
   with LOCK:slot[1]=base+(math.ceil(len(r)/20) if extra and isinstance(r,list) else 0)
   return {'request':p,'received_at_utc':now(),'response':r}
  except Exception as e:
   err=str(e)
   if retry<2:time.sleep(2*(retry+1))
 return {'request':p,'received_at_utc':now(),'error':err}
def key(r,kind):return (r['coin'],r['time'],r['tid']) if kind=='userFillsByTime' else json.dumps(r,sort_keys=True)
def history(kind,a,oldset=None):
 cursor=max(START,old[a]['history_end_ms']-1) if oldset is not None else START
 rows={key(r,kind):r for r in oldset or []};pages=[];complete=False;note=''
 for _ in range(80):
  p=api({'type':kind,'user':a,'startTime':cursor,'endTime':END});pages.append(p);rs=p.get('response')
  if not isinstance(rs,list):note='request failed';break
  before=len(rows)
  for r in rs:rows[key(r,kind)]=r
  if not rs:complete=True;break
  nxt=max(r['time'] for r in rs)
  if kind=='userFillsByTime' and len(rs)<2000:complete=True;break
  if kind!='userFillsByTime' and len(rs)<500:complete=True;break
  if nxt==cursor and len(rows)==before:note='pagination stalled';break
  cursor=nxt
 else:note='pagination bound reached'
 return {'pages':pages,'rows':sorted(rows.values(),key=lambda r:r['time']),'pagination_complete':complete,'note':note,'cached_history_included':oldset is not None}

def collect(a):
 path=RAW/(a+'.json')
 if path.exists():return json.load(open(str(path)))
 d={'address':a,'discovery_sources':sorted(sources[a]),'history_start_ms':START,'history_end_ms':END,'started_at_utc':now()}
 d['fills']=history('userFillsByTime',a,old[a]['fills']['rows'] if a in old else None)
 par=[f for f in d['fills']['rows'] if f['coin'].startswith('par:')]
 d['par_fill_count']=len(par)
 d['positions']=api({'type':'clearinghouseState','user':a,'dex':'par'})
 d['orders']=api({'type':'frontendOpenOrders','user':a,'dex':'par'})
 # Include every candidate in the ledger graph, even inactive accounts.
 d['ledger']=history('userNonFundingLedgerUpdates',a,old[a]['ledger']['rows'] if a in old else None)
 if par or d['positions'].get('response',{}).get('assetPositions') or d['orders'].get('response'):
  d['mode']=api({'type':'userAbstraction','user':a})
  d['spot']=api({'type':'spotClearinghouseState','user':a})
  d['funding']=history('userFunding',a,old[a]['funding']['rows'] if a in old else None)
  d['portfolio']=api({'type':'portfolio','user':a})
  d['historical_orders']=api({'type':'historicalOrders','user':a})
 else:
  d['funding']={'rows':[],'pagination_complete':False,'note':'not queried: no par fills, positions, or open orders'}
 d['completed_at_utc']=now();path.write_text(json.dumps(d,indent=2))
 print(json.dumps({'address':a,'par_fills':len(par),'positions':len(d['positions'].get('response',{}).get('assetPositions',[])),'orders':len(d['orders'].get('response',[])) if isinstance(d['orders'].get('response'),list) else None,'ledger_rows':len(d['ledger']['rows'])}),flush=True)
 return d

processed=set();rounds=[]
for wave in range(8):
 todo=sorted(set(sources)-processed)
 if not todo:break
 print('Discovery wave '+str(wave+1)+': '+str(len(todo))+' candidate wallets',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: results=list(pool.map(collect,todo))
 processed.update(todo)
 for d in results:
  for r in d['ledger']['rows']:
   de=r['delta']
   if de.get('token')=='USDPR':
    add(de.get('user'),'USDPR ledger graph');add(de.get('destination'),'USDPR ledger graph')
 rounds.append({'wave':wave+1,'new_candidates':len(todo),'known_after_wave':len(sources)})
 (ROOT/'manifest.json').write_text(json.dumps({'started_at_ms':END,'history_start_ms':START,'history_end_ms':END,'candidate_count':len(sources),'processed_count':len(processed),'rounds':rounds,'sources':{a:sorted(v) for a,v in sources.items()},'unprocessed':sorted(set(sources)-processed)},indent=2))
else:print('Discovery wave bound reached; remaining candidates recorded',flush=True)
print('Collection complete: '+str(len(processed))+' wallets',flush=True)
