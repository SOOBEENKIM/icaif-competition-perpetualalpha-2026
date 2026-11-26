"""Unchanged repaired algorithm under the long OHLC-proxy scenarios too."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from unittest.mock import patch
from .study import ROOT,OUT,PARENT,save,sha

def job(scenario,phase):
 from production_replay_20261025 import engine
 from candidate_handoff import risk_priority
 from ema_pairs_pas_20261026_v3 import study
 from .coordinator import Portfolio
 from .account import PairOpeningAccount
 original=engine.build
 def build(v,a):
  p=original(v,a);p.account.__class__=PairOpeningAccount;return p
 with patch.object(study,'OUT',OUT/'native'),patch.object(engine,'build',build),patch.object(risk_priority,'Coordinator',Portfolio):
  r=study.run_one('pas_spread_reversion',scenario,later=phase!='development',switch=phase=='switch')
 return r

def main():
 dest=OUT/'native';dest.mkdir(exist_ok=True)
 for name in ['development_data','later_data']:
  p=dest/name
  if not p.exists():p.symlink_to(PARENT/name,target_is_directory=True)
 save(dest/'contract.json',dict(algorithm_unchanged=True,source_sha=sha(__file__),
  strategy_source=json.loads((OUT/'protocol.json').read_text())['sources'],
  purpose='Long-span consistency and risk audit; all intervals are now development, no new independent evaluation claim.'))
 with ProcessPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(job,s,p) for p in ['development','later','switch'] for s in ['base','stress','no_receipts']]
  for f in as_completed(fs):f.result()

if __name__=='__main__':main()
