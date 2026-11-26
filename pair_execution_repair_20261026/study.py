import copy,gzip,json,time
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from ema_pairs_pas_20261026_v3.protocol import ROOT,save,sha
from ema_pairs_pas_20261026_v3.study import expanded_profile
OUT=ROOT/'reports/pair_execution_repair_20261026'
PARENT=ROOT/'reports/ema_pairs_pas_20261026_v3'

def freeze():
 from adaptive_leverage.study import live_manifest
 if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already exists')
 save(OUT/'protocol.json',dict(at=time.time(),candidate='pas_spread_reversion_with_execution_repair',
  unchanged='Same frozen PAS spread forecast,50k own capital,6000pair limit,all45 candidates,24h holding,risk limits and original economic limits.',
  observed_failure='Native adapter closes opposite physical ETH inventory first; pair state machine then sees both legs partially filled and trims the larger GOLD leg. Extra buyback fees observed.90s polling at45s phase also never meets now%900<30, despite valid signals.',
  changes=['One additional original-limit IOC only after confirmed physical flat at new-pair entry; preserve actual virtual owners and independent risk/receipt checks. No retry after unknown receipt, partial close or ordinary liquidity shortfall.','Quarter-hour evaluation once per actual bucket, independent of polling phase; no requirement to land in its first30s.'],
  scope='Offline replay only; newly inspected quote window is now development. No independent-test claim, no new forecast fit, no threshold optimization.',
  comparisons='Same archived actual inventory and quote capture25Oct09:00-15:05UTC; phases0/45,base/stress, unchanged incumbent and prerepair candidate preserved. Compare fees,whole-sleeve net and terminal-depth exit PnL. No promotion on favorable phase alone.',
  live_manifest=live_manifest(),orders_sent=0,live_changed=False,
  sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}))

def main():
 from competition_live.runtime import private_load,fingerprint
 from candidate_handoff import risk_priority
 from .coordinator import Portfolio
 from .account import PairOpeningAccount
 from production_replay_20261025 import book_replay,cash_replay,data_aware_book_replay,lot_sweep,liquidation_audit
 from production_replay_20261025.exact_settlement import run_exact,read_records
 from production_replay_20261025.delayed_forward_review import delayed_submit
 archive=ROOT/'reports/quote_history_replication_20261025';cfg=json.loads((archive/'protocol.json').read_text());start,end=cfg['start'],cfg['end']
 profile=expanded_profile();fs=json.loads((PARENT/'later_data/pas_spread_reversion_forecasts.json').read_text())
 streams,manifest=book_replay.load_books([ROOT/'reports/production_replay_20261025/sealed_forward_capture/received_through_fixed_end.jsonl.gz',archive/'sealed_forward_capture/received_through_fixed_end.jsonl.gz'],end)
 source=sorted([json.loads(p.read_text()) for p in (archive/'forward_signals').glob('*.json')],key=lambda r:r['received_at'])
 funding=json.loads((archive/'funding_schedule.json').read_text());units=json.loads((archive/'cash_units.json').read_text());records=read_records(archive/'funding_cash')
 seed=private_load(archive/'forward_seed.pkl')['portfolio'];groups=[]
 for phase in [0,45]:
  for scenario,sc in cfg['scenarios'].items():
   results={}
   for arm in ['incumbent','pas_spread_reversion']:
    with gzip.open(PARENT/'quotes'/f'{arm}_{scenario}_{phase}.json.gz','rt') as fp:results[arm]=json.load(fp)
   p=copy.deepcopy(seed);p.account.__class__=PairOpeningAccount;p.account.profile=profile
   p.cfg['pair_max_gross']=6000;p.cfg['renewal_policy']['maximum_holding_hours']=24
   p.sc.update(fee_bps=sc['fee_bps'],exit_impact_bps=10. if scenario=='base' else 20.)
   packets=copy.deepcopy(source)
   for row in packets:
    packet=row['packet'];b=packet['body'];o=b['original_signal'];t=b['frame']['source_time']
    b['profile_hash']=o['profile_hash']=fingerprint(profile);o['forecasts']=copy.deepcopy(fs.get(str(t),[]))
    for f in o['forecasts']:f['available_time']=max(f['available_time'],o['published_at'])
    packet['sha256']=fingerprint(b)
   with ExitStack() as stack:
    stack.enter_context(patch.object(risk_priority,'Coordinator',Portfolio))
    stack.enter_context(patch.object(book_replay,'run_book',data_aware_book_replay.run_book_data_aware))
    stack.enter_context(patch.object(data_aware_book_replay,'PHASE_SECONDS',phase))
    stack.enter_context(patch.object(cash_replay.CashVenue,'submit',delayed_submit))
    stack.enter_context(patch.object(book_replay,'sweep',lot_sweep.sweep))
    r=run_exact(streams,packets,profile,sc,start,end,p,'incumbent',funding,cash_units=units,records=records)
   arm='execution_repaired';r['arm']=arm;results[arm]=r
   dest=OUT/'runs'/f'{arm}_{scenario}_{phase}.json.gz';dest.parent.mkdir(exist_ok=True)
   with gzip.open(dest,'wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
   with patch.object(liquidation_audit,'sweep',lot_sweep.sweep):
    rows=liquidation_audit.compare_at_common_end(streams,profile,sc,results,start,end)
   for row in rows:
    v=results[row['arm']];row.update(phase=phase,scenario=scenario,fees=v['fee_increment'],funding=v['funding_increment'],
     fills=len(v['fills']),failures=v['failures'],risk_violations=v['risk_violations'],accounting_error=v['accounting_error'],unknown=v['unknown'],pending=v['pending_orders'])
   groups.extend(rows);save(OUT/'results.json',groups)
   print('repair',phase,scenario,[(r['arm'],r.get('net_pnl'),r.get('net_after_terminal_exit'),r.get('valid')) for r in rows],flush=True)
 from adaptive_leverage.study import live_manifest
 assert live_manifest()==json.loads((OUT/'protocol.json').read_text())['live_manifest']

if __name__=='__main__':
 import sys
 freeze() if sys.argv[1]=='freeze' else main()
