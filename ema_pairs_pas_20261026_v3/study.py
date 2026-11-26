"""Same production portfolio, venue-specific forecasts, strictly offline venue."""
import copy,gzip,json,sys,time,bisect
from collections import Counter
from unittest.mock import patch
import numpy as np
from .protocol import *
from . import data,model

def expanded_profile():
 from competition_live.profile import load_profile
 p=load_profile();p['pairs']=[[SYMBOLS[j],SYMBOLS[k]] for j,k in model.PAIRS]
 return p

def prepare(later=False):
 from competition_live.runtime import fingerprint
 out,cfg,d=data.assemble(later);packets=data.packets(out,later)
 refpath=ROOT/('practice_live/updates/20261026T120007825231Z/inputs.npz' if later else json.loads((OLD/'protocol.json').read_text())['input'])
 ref=dict(np.load(refpath));times=np.array([p['body']['frame']['source_time'] for p in packets])
 carry,counts,funding=data.funding_inputs(out,times)
 save(out/'funding.json',funding);np.savez_compressed(out/'carry.npz',times=times,carry=carry,counts=counts)
 arms=[json.loads((OUT/'selection.json').read_text())['candidate']] if later else ARMS[1:]
 for arm in arms:
  forecasts={};audits=[]
  for i,p in enumerate(packets):
   t=p['body']['frame']['source_time']
   f,a=model.estimates(t,ref,d,carry[i],arm)
   for row in f:row['available_time']=max(row['available_time'],p['body']['published_at']+5)
   forecasts[str(t)]=f;audits.extend(a)
  save(out/(arm+'_forecasts.json'),forecasts);save(out/(arm+'_audit.json'),audits)
 save(out/'manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [refpath,*out.glob('*.json'),*out.glob('*.npz'),*(out/'raw').glob('*.json')]})
 print('prepared',out.name,len(packets),flush=True)

def run_one(arm,scenario,later=False,switch=False):
 from pas_strategy_audit.data import load
 from production_replay_20261025 import engine
 from competition_live.runtime import fingerprint
 from competition_live.profile import load_profile
 out=OUT/('later_data' if later else 'development_data');profile=expanded_profile()
 cfg,d=load(out);source=json.loads((out/'packets.json').read_text());funding=json.loads((out/'funding.json').read_text())
 fcasts={} if arm=='incumbent' else json.loads((out/(arm+'_forecasts.json')).read_text())
 c=dict(np.load(out/'carry.npz'));known=c['times'];carry=c['carry'];original_build=engine.build;built=[]
 sc=SCENARIOS[scenario]
 class Venue(engine.SyntheticVenue):
  def stream(self,symbol):
   r=super().stream(symbol);i=bisect.bisect_right(known,self.now)-1
   if r['ctx'] and i>=0:r['ctx'][-1][2]['funding']=str(carry[i,SYMBOLS.index(symbol)]/24)
   return r
  def settle(self,at_ms,account):
   while self.fidx<len(self.funding) and self.funding[self.fidx]['time']<=at_ms:
    f=self.funding[self.fidx];self.fidx+=1;j=self.symbols.index(f['coin'][4:]);at=f['time']/1000
    # Positions immediately before settlement, including serial-leg execution.
    qty=sum(x['signed_qty'] for x in account.physical.fills if x['asset']==j and x['time']<at)
    if not qty:continue
    unit=f.get('cash_per_unit');proxy=unit is None
    if proxy:unit=-self.marks(at)[j]*f['rate']
    cash=unit*qty
    if cash>0:cash*=self.sc['receipt_multiplier']
    account.apply_actual_funding([dict(time=f['time'],delta=dict(coin=f['coin'],usdc=str(cash),szi=str(qty)),synthetic=True)])
    self.receipts.append(dict(time=at,coin=f['coin'],qty=qty,cash=cash,oracle_proxy=proxy,synthetic=True))
 def build(v,a):
  p=original_build(v,a);p.cfg['pair_max_gross']=6000
  p.cfg['renewal_policy']['maximum_holding_hours']=24
  p.sc.update(fee_bps=sc['fee_bps'],exit_impact_bps=sc['impact_bps'])
  built.append(p);return p
 def transform(packet,now):
  p=copy.deepcopy(packet);b=p['body'];o=b['original_signal'];t=b['frame']['source_time']
  b['profile_hash']=o['profile_hash']=fingerprint(profile)
  if arm!='incumbent' and (not switch or now>=SPLIT):o['forecasts']=copy.deepcopy(fcasts.get(str(t),[]))
  p['sha256']=fingerprint(b);return p
 # Change profile identity in immutable copies, even before any candidate activation.
 source=[transform(p,START-1 if switch else END+1) for p in source]
 start=START if not later or switch else SPLIT;end=END if later else SPLIT
 # Keep source-original copies for a same-inventory switch, so old forecasts survive pre-switch.
 if switch:
  source=json.loads((out/'packets.json').read_text())
  for p in source:
   p['body']['profile_hash']=p['body']['original_signal']['profile_hash']=fingerprint(profile)
   p['sha256']=fingerprint(p['body'])
 with patch.object(engine,'build',build):
  r=engine.run(d,profile,sc,funding,source,start,end,venue_class=Venue,
    packet_transform=transform if switch else None)
 p=built[0];a=p.account;prices=d['close'][-1]
 sessions=[*p.sessions]+([p.pair] if p.pair is not None else [])
 r['pair_sessions']=[dict(pair=s.intent['pair'],source=s.intent['forecast_source'],state=s.state,
    first_fill=s.first_fill,was_balanced=s.was_balanced,closed_reason=s.closed_reason,events=s.events) for s in sessions]
 r['owner_fills']={k:v.fills for k,v in a.owners.items()}
 r['owner_funding']={k:v.funding for k,v in a.owners.items()}
 r['metrics'].update(pair_pnl=a.owners['pair'].equity(prices),base_pnl=a.owners['base'].equity(prices)-50000,
    pair_fills=len(a.owners['pair'].fills),completed_pairs=sum(s.state=='closed' and s.was_balanced for s in sessions),
    pair_sessions=len(sessions),max_accounting_error=max(a.maximum_error,a.physical.max_error),
    funding_proxy_events=sum(f['oracle_proxy'] for f in r['funding']),
    timing_pass=all(o['sent']<o['arrival'] for o in r['orders']),
    entry_reasons=dict(Counter(e['reason'] for e in r['events'] if e['kind']=='entry_check')))
 name=f'{arm}_{scenario}_{"switch" if switch else "later" if later else "development"}'
 path=OUT/'runs'/(name+'.json.gz');path.parent.mkdir(exist_ok=True)
 with gzip.open(path,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
 row=dict(arm=arm,scenario=scenario,phase='switch' if switch else 'later' if later else 'development',**r['metrics'],audit=r['audit'])
 save(OUT/'rows'/(name+'.json'),row)
 print(json.dumps({k:row[k] for k in ['arm','scenario','phase','net_pnl','Q','pair_fills','completed_pairs','pair_pnl']}),flush=True)
 return row

def select():
 rows=[json.loads(p.read_text()) for p in (OUT/'rows').glob('*_development.json')]
 base={r['scenario']:r for r in rows if r['arm']=='incumbent'};rank=[]
 for arm in ARMS[1:]:
  rr=[r for r in rows if r['arm']==arm];assert len(rr)==3
  delta={r['scenario']:r['net_pnl']-base[r['scenario']]['net_pnl'] for r in rr}
  clean=all(r['max_accounting_error']<1e-6 and r['timing_pass'] and not r['audit']['risk_violations'] and not r['audit']['unknown'] and not r['audit']['pending'] for r in rr)
  rank.append(dict(candidate=arm,min_delta=min(delta['base'],delta['stress']),delta=delta,
   min_Q_delta=min(r['Q']-base[r['scenario']]['Q'] for r in rr if r['scenario']!='no_receipts'),
   eligible=clean and min(delta['base'],delta['stress'])>0 and all(r['pair_fills']>0 for r in rr)))
 rank.sort(key=lambda r:(r['min_delta'],r['min_Q_delta']),reverse=True)
 choice=rank[0];save(OUT/'selection.json',dict(at=time.time(),**choice,all_candidates=rank,
  selected_before_later_outcome_evaluation=True,no_independence_claim=True))
 print(json.dumps(choice),flush=True)

def freeze_source():
 if (OUT/'source_freeze.json').exists():raise FileExistsError('Preserve the original freeze; use an explicit audited correction')
 paths=list(Path(__file__).resolve().parent.glob('*.py'))+[OUT/'protocol.json']
 save(OUT/'source_freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})

def verify():
 from adaptive_leverage.study import live_manifest
 corrected=OUT/'source_freeze_corrected_after_evaluation.json'
 path=corrected if corrected.exists() else OUT/'source_freeze.json'
 frozen=json.loads(path.read_text())
 expected={str(p.relative_to(ROOT)) for p in Path(__file__).resolve().parent.glob('*.py')}
 assert expected<=set(frozen),'Freeze does not cover this version of the source'
 for f,h in frozen.items():assert sha(ROOT/f)==h,f
 assert live_manifest()==json.loads((OUT/'protocol.json').read_text())['live_manifest']

def main():
 cmd=sys.argv[1]
 if cmd=='prepare':prepare('--later' in sys.argv)
 elif cmd=='freeze':freeze_source()
 elif cmd=='select':select()
 elif cmd=='run':
  verify();run_one(sys.argv[2],sys.argv[3],later='--later' in sys.argv,switch='--switch' in sys.argv);verify()

if __name__=='__main__':main()
