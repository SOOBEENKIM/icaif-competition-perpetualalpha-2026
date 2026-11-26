import copy,json
import numpy as np
from .protocol import *

def assemble(later=False):
 from pas_strategy_audit.data import load
 out=OUT/('later_data' if later else 'development_data');end=END if later else SPLIT
 for s in SYMBOLS:
  candles={int(r['t']):r for r in json.loads((OLD/'raw'/f'pas_candles_{s}.json').read_text())['data'] if r['T']<SPLIT*1000}
  funding={int(r['time']):r for r in json.loads((OLD/'raw'/f'pas_funding_{s}.json').read_text())['data'] if r['time']<=SPLIT*1000}
  if later:
   if not (OUT/'selection.json').exists():raise ValueError('Select before opening later outcomes')
   for p in sorted((OUT/'collection').glob(f'{s}_*.json')):
    if 'funding' in p.name:
     for r in json.loads(p.read_text())['data']:funding[int(r['time'])]=r
    else:
     for r in json.loads(p.read_text())['data']:
      if r['t']>=SPLIT*1000 and r['T']<end*1000:candles[int(r['t'])]=r
  save(out/'raw'/f'pas_candles_{s}.json',dict(data=[candles[t] for t in sorted(candles)]))
  save(out/'raw'/f'pas_funding_{s}.json',dict(data=[funding[t] for t in sorted(funding) if t<=end*1000]))
 cfg=json.loads((OLD/'dataset.json').read_text());cfg['cutoff_ms']=end*1000;save(out/'dataset.json',cfg)
 cfg,d=load(out)
 save(out/'coverage.json',dict(start=cfg['start_ms'],end=cfg['cutoff_ms'],
  active_minutes={s:int(d['active'][:,j].sum()) for j,s in enumerate(SYMBOLS)},
  last_price_age={s:float(d['age_ms'][-1,j]/1000) for j,s in enumerate(SYMBOLS)}))
 return out,cfg,d

def packets(out,later=False):
 """Reuse audited historical envelopes; build later core from immutable sources."""
 from production_replay_20261025.engine import packets as old_packets
 from competition_live.profile import load_profile
 from competition_live.runtime import fingerprint
 from universe_repair.core import calculate,REVISION
 from pas_economic_allocation import model
 from pas_economic_allocation.anchored import RATIO
 from candidate_handoff.entry_policy import summarize_candles
 from datetime import datetime
 profile=load_profile();p=old_packets(OLD,profile)
 if later:
  refpath=ROOT/'practice_live/updates/20261026T120007825231Z/inputs.npz'
  ref=dict(np.load(refpath));z=dict(np.load(refpath.with_name('targets.npz')))
  anchor,gate,parts=calculate(ref,z['chronos_forecast'],z['frozen_regime_probability'],REVISION)
  fit=json.loads((ROOT/'reports/pas_economic_allocation_20261022/fit.json').read_text())
  x,u,carry=model.features(ref);price=model.predict(x,u,fit)*fit['attenuation']
  raw={s:json.loads((out/'raw'/f'pas_candles_{s}.json').read_text())['data'] for s in SYMBOLS}
  from competition_live.candidate import build_candidate
  adapter=build_candidate(profile,lambda *args:None)['signals']
  for previous in p:
   st=previous['body']['frame']['source_time']
   if st<SPLIT-12*3600:continue
   stop=np.searchsorted(ref['market_times'],st,side='right')
   adapter.ingest(previous['body']['original_signal']['publication'],{k:v[:stop] for k,v in ref.items()})
  pubs={}
  for f in (ROOT/'practice_live/updates').glob('*/signals.json'):
   r=json.loads(f.read_text());source=int(datetime.fromisoformat(r['input_completed_bar_utc']).timestamp())
   at=datetime.fromisoformat(r['created_at_utc']).timestamp()
   if SPLIT<=source<END and (source not in pubs or at<pubs[source][0]):pubs[source]=(at,f)
  audit=[]
  for source,(at,f) in sorted(pubs.items()):
   i=int(np.searchsorted(ref['market_times'],source));saved=dict(np.load(f.parent/'inputs.npz'))
   # Recomputed core is allowed only when the actual published prefix is identical.
   for key in ['close','volume','funding','trades']:
    np.testing.assert_array_equal(saved[key],ref[key][:i+1],err_msg=f'Historical revision:{source}:{key}')
   target=dict(np.load(f.parent/'targets.npz'));available=max(at,source+120)+120
   item=copy.deepcopy(p[-1]);body=item['body'];frame=body['frame'];old=body['original_signal'];env=old['publication']
   digest=sha(f.parent/'inputs.npz')
   frame.update(source_time=source,source_hash=digest,anchor_weights=(anchor[i]*RATIO*2).tolist(),protected=gate['protected'][i].tolist(),
    price_forecast_168h=price[i].tolist(),reference_funding_forecast_168h=carry[i].tolist(),sigma=parts['sigma'][i].tolist(),volume=ref['volume'][i].tolist(),trades=ref['trades'][i].tolist())
   frame['venue_history']=summarize_candles(raw,SYMBOLS,source);frame['venue_history']['received_at']=available
   ef=env['data']['frame'];ef.update(source_time=source,source_hash=digest,published_time=available,weights=target['default'][-1].tolist(),
    sigma=frame['sigma'],volume=frame['volume'],trades=frame['trades'])
   from experiments.published_pair_lifecycle_20261017.signals import FRAME_KEYS
   import hashlib
   env['received_ms']=int(available*1000)+1
   env['data'].update(incumbent=ef['weights'],candidate=ef['weights'],publication_sha256=hashlib.sha256(json.dumps({k:ef[k] for k in FRAME_KEYS},sort_keys=True,allow_nan=False).encode()).hexdigest())
   body['published_at']=available+.001;old['published_at']=available+.001
   old['forecasts']=adapter.ingest(env,{k:v[:i+1] for k,v in ref.items()})
   for fc in old['forecasts']:fc['available_time']=max(fc['available_time'],old['published_at'])
   item['sha256']=fingerprint(body);p.append(item)
   audit.append(dict(source=source,available=available,input_sha256=digest,
    retained_pair_actions=[dict(pair=fc['pair'],action=fc.get('economic_action')) for fc in old['forecasts'] if fc['model']=='decision_gate' and fc.get('economic_action')]))
  save(out/'publication_audit.json',audit)
 save(out/'packets.json',p)
 return p

def funding_inputs(out,times):
 from pas_performance_research.funding import forecast
 rows=[];settled=[]
 old={(r['coin'],r['time']//3600000):r for r in json.loads((OLD/'funding.json').read_text())}
 for s in SYMBOLS:
  for r in json.loads((out/'raw'/f'pas_funding_{s}.json').read_text())['data']:
   rows.append(dict(known=r['time']/1000+60,symbol=s,rate=float(r['fundingRate'])))
   settled.append(dict(time=r['time'],coin='pas:'+s,rate=float(r['fundingRate']),
                       cash_per_unit=old.get(('pas:'+s,r['time']//3600000),{}).get('cash_per_unit')))
 carry,count=forecast(times,rows,SYMBOLS,horizon=24)
 return carry,count,settled
