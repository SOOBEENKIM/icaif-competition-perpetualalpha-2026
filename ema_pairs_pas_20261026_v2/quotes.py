"""Selected algorithm on actual captured books and verified funding cash."""
import copy,gzip,json
from contextlib import ExitStack
from unittest.mock import patch
from .protocol import *
from .study import expanded_profile,verify

def main():
 from competition_live.runtime import private_load,fingerprint
 from production_replay_20261025 import book_replay,cash_replay,data_aware_book_replay,lot_sweep,liquidation_audit
 from production_replay_20261025.exact_settlement import run_exact,read_records
 from production_replay_20261025.delayed_forward_review import delayed_submit
 verify();archive=ROOT/'reports/quote_history_replication_20261025'
 cfg=json.loads((archive/'protocol.json').read_text());start,end=cfg['start'],cfg['end']
 primary=json.loads((OUT/'selection.json').read_text())['candidate'];profile=expanded_profile()
 fs=json.loads((OUT/'later_data'/(primary+'_forecasts.json')).read_text())
 streams,manifest=book_replay.load_books([ROOT/'reports/production_replay_20261025/sealed_forward_capture/received_through_fixed_end.jsonl.gz',archive/'sealed_forward_capture/received_through_fixed_end.jsonl.gz'],end)
 save(OUT/'quote_manifest.json',manifest)
 source=sorted([json.loads(p.read_text()) for p in (archive/'forward_signals').glob('*.json')],key=lambda r:r['received_at'])
 funding=json.loads((archive/'funding_schedule.json').read_text());units=json.loads((archive/'cash_units.json').read_text())
 records=read_records(archive/'funding_cash');seed=private_load(archive/'forward_seed.pkl')['portfolio']
 groups=[]
 for phase in [0,45]:
  for scenario,sc in cfg['scenarios'].items():
   results={}
   for arm in ['incumbent',primary]:
    p=copy.deepcopy(seed);p.account.profile=profile
    p.cfg['pair_max_gross']=6000;p.cfg['renewal_policy']['maximum_holding_hours']=24
    p.sc.update(fee_bps=sc['fee_bps'],exit_impact_bps=10. if scenario=='base' else 20.)
    packets=copy.deepcopy(source)
    for row in packets:
     packet=row['packet'];b=packet['body'];o=b['original_signal'];t=b['frame']['source_time']
     b['profile_hash']=o['profile_hash']=fingerprint(profile)
     if arm!='incumbent':
      o['forecasts']=copy.deepcopy(fs.get(str(t),[]))
      for f in o['forecasts']:f['available_time']=max(f['available_time'],o['published_at'])
     packet['sha256']=fingerprint(b)
    with ExitStack() as stack:
     stack.enter_context(patch.object(book_replay,'run_book',data_aware_book_replay.run_book_data_aware))
     stack.enter_context(patch.object(data_aware_book_replay,'PHASE_SECONDS',phase))
     stack.enter_context(patch.object(cash_replay.CashVenue,'submit',delayed_submit))
     stack.enter_context(patch.object(book_replay,'sweep',lot_sweep.sweep))
     r=run_exact(streams,packets,profile,sc,start,end,p,'incumbent',funding,cash_units=units,records=records)
    r['arm']=arm;results[arm]=r
    dest=OUT/'quotes'/f'{arm}_{scenario}_{phase}.json.gz';dest.parent.mkdir(exist_ok=True)
    with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
   with patch.object(liquidation_audit,'sweep',lot_sweep.sweep):
    rows=liquidation_audit.compare_at_common_end(streams,profile,sc,results,start,end)
   for row in rows:
    r=results[row['arm']];row.update(phase=phase,scenario=scenario,fees=r['fee_increment'],funding=r['funding_increment'],
      failures=r['failures'],risk_violations=r['risk_violations'],accounting_error=r['accounting_error'],
      fills=len(r['fills']),unknown=r['unknown'],pending=r['pending_orders'])
   groups.extend(rows);save(OUT/'quote_results.json',groups)
   print('quotes',phase,scenario,[(r['arm'],r.get('net_pnl'),r.get('net_after_terminal_exit'),r.get('valid')) for r in rows],flush=True)
 verify()

if __name__=='__main__':main()
