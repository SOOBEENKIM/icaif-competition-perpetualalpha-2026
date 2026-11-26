"""Build reproducible public wallet/team return observations; no identity assertions."""
import datetime as dt,gzip,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
START=1792411200000
seed=json.load(gzip.open(HERE/'public/seed.json.gz','rt'))
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def ms(s):
 if isinstance(s,(float,int)):return int(s)
 return int(dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()*1000) if s else 0
wallets={a:[] for a in seed['wallets']};teams={};seen={};sources=[]
def add_official(raw,source):
 d=raw.get('response',raw);generated=ms(d.get('generated_at'))
 if not isinstance(d,dict) or ms(d.get('start') or d.get('from'))!=START:return
 sources.append(source)
 for t in d.get('teams',[]):
  name=t['name'];arr=teams.setdefault(name,{})
  if 'values' in t:
   for i,v in enumerate(t['values']):
    if v is None:continue
    at=START+i*int(d['step_s'])*1000;key=(name,at)
    if generated>=seen.get(key,0):arr[at]={'at':at,'returnPct':v*100,'kind':'chart','source':source};seen[key]=generated
   if t.get('pnl') is not None and t.get('last_at'):arr[ms(t['last_at'])]={'at':ms(t['last_at']),'returnPct':t['pnl']*100,'kind':'latest','source':source}
  elif t.get('return') is not None:
   # Daily standing uses the latest completed 12:00 UTC snapshot, not publication time.
   moment=dt.datetime.fromtimestamp(generated/1000,dt.timezone.utc);at=moment.replace(hour=12,minute=0,second=0,microsecond=0)
   if at>moment:at-=dt.timedelta(days=1)
   timestamp=ms(at.isoformat());key=(name,timestamp)
   if generated>=seen.get(key,0):arr[timestamp]={'at':timestamp,'returnPct':t['return']*100,'kind':'daily','source':source};seen[key]=generated
for p in sorted((ROOT/'reports').glob('**/*.json')):
 if p.name not in ('chart.json','leaderboard.json') or '2026102' not in str(p):continue
 add_official(read(p),str(p.relative_to(ROOT)))
for key in ('chart','leaderboard'):
 add_official(seed['official'][key]['data'],'public/seed.json.gz:'+key)
 add_official(read(HERE/'work/team-name-sources'/(key+'.json')),'live-official:'+key)

def sample(w,a,mode,source):
 spot=w.get('spot',{});pos=w.get('positions',{});s=spot.get('response');p=pos.get('response');t=ms(spot.get('received_kst'));pt=ms(pos.get('received_kst'))
 if not s or not p or t<START or abs(pt-t)>120000:return
 coin=next((x for x in s['balances'] if x['coin']=='USDP'),None);balance=float(coin['total']) if coin else 0;perp=float(p['marginSummary']['accountValue'])
 if mode=='unifiedAccount':eq=balance
 elif mode in ('disabled','standard','default'):eq=balance+perp
 elif abs(perp)<1e-9:eq=balance
 else:return
 ledger=w.get('ledger',{});net=0;other=0;grants=0
 for row in ledger.get('rows',[]):
  d=row.get('delta',{})
  if row['time']>t or d.get('token')!='USDP':continue
  src=d.get('user','').lower();dst=d.get('destination','').lower()
  if src==dst:continue
  value=float(d['amount']);sign=1 if dst==a else -1 if src==a else 0
  net+=sign*value
  if dst==a and src=='0x304199b126a2bb77dedda4c608a9f1dadfada521' and value==100000:grants+=1
  elif sign:other+=1
 if grants!=1 or other or not ledger.get('pagination_complete'):return
 wallets[a].append({'at':t,'returnPct':(eq-net)/100000*100,'source':source})
for p in sorted((ROOT/'reports').glob('**/wallets/*.json')):
 a=p.stem
 if a not in wallets or '2026102' not in str(p):continue
 w=read(p);raw=p.parent.parent;extra=read(raw/'extra'/p.name);mode=extra.get('mode',{}).get('response')
 sample(w,a,mode,str(p.relative_to(ROOT)))
 # Initial and final snapshot from a long scan are distinct public observations.
 if 'spot_initial' in w and 'positions_initial' in w:sample({**w,'spot':w['spot_initial'],'positions':w['positions_initial']},a,mode,str(p.relative_to(ROOT))+':initial')
for a,w in seed['wallets'].items():
 sample({'spot':{'response':w['spot']['data'],'received_kst':w['spot']['at']},'positions':{'response':w['position']['data'],'received_kst':w['position']['at']},'ledger':{'rows':w['ledger']['rows'],'pagination_complete':w['ledger']['complete']}},a,w['mode']['data'],'public/seed.json.gz')
for a,rows in wallets.items():wallets[a]=sorted({r['at']:r for r in rows}.values(),key=lambda r:r['at'])
result={'generated_at':int(dt.datetime.now().timestamp()*1000),'wallets':wallets,'teams':{n:sorted(v.values(),key=lambda r:r['at']) for n,v in teams.items()},'method':'Nearest official sample within 30 minutes; max one comparison per 30-minute wallet interval; comparable single-grant accounts only; mean absolute return error (percentage points).','sources':sources}
(HERE/'public/team-evidence.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
print('wallet points',sum(map(len,wallets.values())),'team points',sum(len(t) for t in result['teams'].values()),'teams',len(teams))
print('wallet samples',sorted([(a,len(v)) for a,v in wallets.items()],key=lambda x:x[1]))
