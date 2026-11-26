"""Refresh our public state after the roster scan; preserve initial observations."""
import json,shutil
from pathlib import Path
import collect as c
out=c.OUT;root=c.RAW
manifest=json.loads((root/'manifest.json').read_text())
for folder in ('wallets','extra'):
 p=root/folder/(c.USER+'.json');shutil.copy2(p,root/(folder+'_sfinx_initial.json'))
(root/'wallets'/(c.USER+'.json')).unlink()
c.wallet(c.USER,manifest)
c.public('leaderboard');c.public('chart')
for name in ('ownership.json','dispatch.json','status.json','producer_status.json'):
 p=c.ROOT/'.runtime/pas_budget_lifecycle'/name
 if p.exists():shutil.copy2(p,out/'runtime'/name)
ledger=json.loads((root/'current/funder.json').read_text())['response']
new={r['delta']['destination'].lower() for r in ledger if r['delta'].get('token')=='USDP' and float(r['delta'].get('amount',0))==100000}
manifest.update(latest_funder_check=str(root/'current/funder.json'),latest_funder_grant_count=len(new),latest_grants_match_prior=(new==set(manifest['grant_wallets'])),funder_history_under_page_limit=len(ledger)<500,collection_complete_kst=c.now());c.save(root/'manifest.json',manifest)
print('COMPLETE',c.now())
