import json,math
from pathlib import Path
p=Path(__file__).resolve().parent;d=json.loads((p/'analysis.json').read_text());s=d['summary'];m=json.loads((p/'raw/manifest.json').read_text())
checks={
 'all_grant_recipients_collected':len(d['wallets'])==len(m['grant_wallets']),
 'inventory_and_history_checks':not s['issues'],
 'all_SFINX_fill_owners_identified':not s['unknown_own_fills'],
 'SFINX_only_initial_100k_net_transfer':s['sfinx']['single_grant_100k'],
 'SFINX_owner_pnl_conservation_within_1_cent':abs(sum(v['net_pnl'] for v in s['virtual_owner_accounting'].values())-s['sfinx']['trade_net_pnl'])<.01,
 'all_10_market_quotes_collected':len(d['market'])==10,
 'official_Q_recomputed':all(t['q'] is None or math.isclose(t['q'],(max(t['return'],0)+.005)/(abs(t['return'])+1.5*t['volatility_horizon']+.05),abs_tol=1e-12) for t in d['official']),
 'all_verified_grants_are_100k_no_extra_USDP_transfers':all(r['single_grant_100k'] for r in d['wallets']),
}
result=dict(checks=checks,all_pass=all(checks.values()),balance_pnl_reconstruction_differences=[{k:r[k] for k in ['address','accounting_snapshot_difference','snapshot_kst']} for r in d['wallets'] if abs(r['accounting_snapshot_difference'] or 0)>.01],note='Non-atomic balance/position differences are not hidden or asserted to reconcile. Only inventory/history and explicit invariants above are checked.')
(p/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
assert result['all_pass']
