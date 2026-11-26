"""Post-evaluation audit. This does not manufacture a pre-test freeze."""
import gzip,json,time
from pathlib import Path
import numpy as np
from .study import ROOT,OUT,PARENT,save,sha

def read(p):
 return json.load(gzip.open(p,'rt')) if str(p).endswith('.gz') else json.loads(p.read_text())

def main():
 from adaptive_leverage.study import live_manifest
 from ema_pairs_pas_20261026_v3 import model,study
 from ema_pairs_pas_20261026_v3.protocol import OLD,SPLIT,ARMS
 from pas_strategy_audit.data import load
 protocol=read(OUT/'protocol.json')
 source_checks={f:sha(ROOT/f)==h for f,h in protocol['sources'].items()}
 assert all(ok or f.endswith('/test_repair.py') for f,ok in source_checks.items())
 assert live_manifest()==protocol['live_manifest']
 correction=PARENT/'source_freeze_corrected_after_evaluation.json'
 if not correction.exists():
  paths=[*Path(study.__file__).resolve().parent.glob('*.py'),PARENT/'protocol.json']
  save(correction,{str(p.relative_to(ROOT)):sha(p) for p in paths})
 study.verify()
 note=dict(at=time.time(),original_freeze_preserved=True,
  problem='v3 source-freeze helper incorrectly hashed the v1 source directory. This was discovered after evaluation.',
  remedy='Helper now hashes its own directory and rejects incomplete source coverage. Current v3 sources have an explicitly post-evaluation freeze. Existing result files are preserved; forecasts independently regenerated and compared below.',
  independent_test=False,retroactive_pretest_guarantee=False,
  test_correction='Price-limit test counted submitted orders as fills. Expected three submissions, two fills, one zero-fill continuation, unchanged original limit and balanced 0.2/-0.2 virtual quantities. Strategy/execution source hashes unchanged.')
 save(OUT/'audit_corrections.json',note)
 reconstruction=[];manifest_checks=[]
 for later in [False,True]:
  dpath=PARENT/('later_data' if later else 'development_data');cfg,d=load(dpath)
  rp=ROOT/'practice_live/updates/20261026T120007825231Z/inputs.npz' if later else ROOT/read(OLD/'protocol.json')['input']
  ref=dict(np.load(rp));packets=read(dpath/'packets.json');carry=dict(np.load(dpath/'carry.npz'))
  assert np.array_equal(carry['times'],[p['body']['frame']['source_time'] for p in packets])
  for arm in ['pas_spread_reversion'] if later else ARMS[1:]:
   saved=read(dpath/(arm+'_forecasts.json'));fresh={}
   for i,p in enumerate(packets):
    t=p['body']['frame']['source_time'];f,_=model.estimates(t,ref,d,carry['carry'][i],arm)
    for row in f:row['available_time']=max(row['available_time'],p['body']['published_at']+5)
    fresh[str(t)]=f
   assert fresh==saved,(later,arm)
   reconstruction.append(dict(phase='later' if later else 'development',arm=arm,sources=len(packets),exact_match=True))
  for f,h in read(dpath/'manifest.json').items():
   if f.endswith('/manifest.json'):continue
   manifest_checks.append(dict(path=f,match=sha(ROOT/f)==h))
 assert all(x['match'] for x in manifest_checks)
 prefix=[]
 for sc in ['base','stress','no_receipts']:
  a=read(PARENT/'runs'/f'incumbent_{sc}_switch.json.gz')
  b=read(OUT/'native/runs'/f'pas_spread_reversion_{sc}_switch.json.gz')
  fields=['time','equity','qty','fees','funding','notional']
  aa=[{k:r[k] for k in fields} for r in a['snapshots'] if r['time']<SPLIT]
  bb=[{k:r[k] for k in fields} for r in b['snapshots'] if r['time']<SPLIT]
  assert aa==bb,sc
  prefix.append(dict(scenario=sc,equal_snapshots=len(aa),identical_before_switch=True))
 results=read(OUT/'results.json');assert len(results)==12
 for r in results:
  assert r['valid'] and not r['failures'] and not r['risk_violations'] and not r['unknown'] and not r['pending']
  assert r['accounting_error']<1e-6
  run=(OUT/'runs'/f"execution_repaired_{r['scenario']}_{r['phase']}.json.gz" if r['arm']=='execution_repaired'
       else PARENT/'quotes'/f"{r['arm']}_{r['scenario']}_{r['phase']}.json.gz")
  v=read(run);assert v['actual_orders_sent']==0
  assert all(o['arrival']>o['time'] for o in v['orders'])
 assert live_manifest()==protocol['live_manifest']
 save(OUT/'final_audit.json',dict(at=time.time(),forecast_reconstruction=reconstruction,
  immutable_data_files_checked=len(manifest_checks),all_data_hashes_match=True,
  same_inventory_switch=prefix,original_runtime_sources_unchanged=source_checks,
  live_sources_unchanged=True,live_orders_sent_by_study=0,quote_conditions_valid=4,
  maximum_accounting_error=max(r['accounting_error'] for r in results),
  current_manifest={str(p.relative_to(ROOT)):sha(p) for folder in ['ema_pairs_pas_20261026_v3','pair_execution_repair_20261026','pair_execution_sequence_20261026'] for p in (ROOT/folder).glob('*.py')}))
 print('Audit passed: exact forecast reconstruction, immutable data, same-inventory prefix, accounting, live unchanged.')

if __name__=='__main__':main()
