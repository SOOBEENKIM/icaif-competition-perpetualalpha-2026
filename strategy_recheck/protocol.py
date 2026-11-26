"""Record choices before examining this comparison's results."""
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from competition_live.profile import ROOT, load_profile, verify_deployment, verify_bundle

OUT = ROOT / 'reports/strategy_recheck_20261020'
PRIOR = ROOT / 'reports/pas_strategy_audit_20261020'
VARIANTS = {
    'incumbent': ('incumbent_published', 1.),
    'universe_only': ('all_markets_long_only', 1.),
    'signed': ('all_markets_symmetric_daily', 1.),
    'incumbent_original_size': ('incumbent_published', None),
    'signed_original_size': ('all_markets_symmetric_daily', None),
}


def freeze():
    OUT.mkdir(exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    p = load_profile()
    original = 1 / json.loads((ROOT / 'strategy/profiles.json').read_text())['default']['relative_to_original']
    variants = {k: {'target': v[0], 'multiplier': original if v[1] is None else v[1]}
                for k, v in VARIANTS.items()}
    hashes = {}
    for f in [PRIOR / 'reference_targets.npz', PRIOR / 'reference_pointer.json',
              ROOT / 'all_market_strategy/signals.py', ROOT / 'pas_strategy_audit/execution.py']:
        hashes[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'historical_independence': False,
        'classification': 'reused-history development comparison, not untouched holdout',
        'team_capital': 100000., 'user_capital': 50000., 'reserved_capital': 50000.,
        'variants': variants, 'risk': p['trading']['risk'],
        'sizing': 'Existing downside scale versus original scale (1/0.6938788714042263). '
                  'No optimization, forced full investment, or 2000 notional cap. Pair allocation unchanged.',
        'reference_windows': ['validation', 'comparison', 'continuous'],
        'reference_engine': 'Unchanged selected pair engine; only baseline first weights replaced. '
                            'Stored past-only rolling model states reused; no new model fitting.',
        'pas_start_ms': 1792411200000,
        'pas_cutoff_ms': int(time.time() // 60) * 60000,
        'pas_scenarios': {
            'base': {'fee_bps': 9., 'impact_bps': 10., 'participation': .01, 'receipt_multiplier': 1.},
            'stress': {'fee_bps': 12., 'impact_bps': 20., 'participation': .0025, 'receipt_multiplier': .5},
        },
        'pas_execution': 'Completed trade candles, subsequent active minute high/low plus impact; '
                         'source price economic bounds, participation, lot/minimum, original risk controller. '
                         'Historical L2/marks/oracles absent. Not exact live fills. No fictitious pair trades.',
        'promotion_rule': 'Candidate must improve net PnL AND team Q in both scenarios of both reference '
                          'windows and continuous run; improve PAS terminal-cost-adjusted PnL in both scenarios; '
                          'no accounting/timing/order defects. PAS Q is unavailable if fewer than two returns. '
                          'Passing development results alone is not proof of independent future profitability.',
        'source_hashes': hashes,
        'deployment_hashes': verify_deployment()['files'],
        'bundle_hash': verify_bundle(),
        'orders_sent_by_recheck': 0,
    }
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    print(json.dumps(freeze(), indent=2))
