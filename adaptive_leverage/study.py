"""Reused PAS history: incumbent versus one declared adaptive sizing overlay.

No parameter search or promotion. Both use the same original risk caps, 50k
capital, forecasts, fees and fill engine. Results are development diagnostics.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from .controller import AdaptiveController
from .policy import Settings
from pas_performance_research.soft_timing import SoftTimingController

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/adaptive_leverage_20261026'


def live_manifest():
    policy = json.loads((ROOT / '.runtime/pas_budget_lifecycle/execution_policy.json').read_text())
    files = [*policy['source_sha256'], '.runtime/pas_budget_lifecycle/execution_policy.json',
             '.runtime/pas_budget_lifecycle/team_execution_contract.json', 'competition_live/config.json',
             '.runtime/active_pas_strategy.json']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}


def main():
    from pas_performance_research import wave3
    OUT.mkdir(exist_ok=True)
    protocol_path = OUT / 'protocol.json'
    if protocol_path.exists():
        raise FileExistsError('Do not overwrite a completed or frozen experiment')
    original = live_manifest()
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        settings=vars(Settings()), arms=['pullback_any3', 'adaptive_leverage'],
        hypothesis='Amplify only unprotected, entry-approved incumbent targets whose forecast signed edge exceeds costs; maximize the existing mean-variance utility on a predeclared grid.',
        risk='Original limits unchanged; own gross cap 0.6 of 50k equity. Target multiplier is NOT exchange leverage.',
        history_reused=True, independent_test=False, model_refit=False,
        window='2026-10-19T12:00:00Z through 2026-10-22T11:00:00Z; earlier, later, continuous and trade-tape diagnostics',
        limitation='Historical queues/L2 unavailable; conditional next-minute extrema or recorded trade participation. Funding uses a price proxy where actual oracle unavailable. Not the exact current account or official score.',
        orders_sent=0, live_changed=False, live_manifest=original)
    protocol_path.write_text(json.dumps(protocol, indent=2))
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / 'adaptive_leverage').glob('*.py')}
    (OUT / 'source_freeze.json').write_text(json.dumps(source, indent=2))

    def verify():
        assert live_manifest() == original, 'Live code/config changed during comparison'
        for p, sha in source.items():
            assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha

    def build(f, z, ix, times, price_times, prices, arm, age=None):
        from pas_performance_research.study import BASE
        cls = AdaptiveController if arm == 'adaptive_leverage' else SoftTimingController
        return cls(times, f['forecast'], f['carry'], z[BASE][ix], z[BASE + '_protected'][ix], 5.,
                   price_times=price_times, prices=prices, entry_rule='pullback_any3', age=age)

    wave3.OUT = OUT
    wave3.ARMS = protocol['arms']
    wave3.build = build
    wave3.verify = verify
    wave3.pas()
    rows = json.loads((OUT / 'pas_results.json').read_text())
    comparisons = []
    for row in rows:
        assert row['timing_pass'] and row['maximum_accounting_error'] < 1e-6
        if row['arm'] != 'adaptive_leverage':
            continue
        base = next(b for b in rows if b['arm'] == 'pullback_any3' and b['phase'] == row['phase'] and b['scenario'] == row['scenario'])
        comparisons.append(dict(phase=row['phase'], scenario=row['scenario'],
                                baseline_pnl=base['net_pnl'], candidate_pnl=row['net_pnl'],
                                pnl_change=row['net_pnl'] - base['net_pnl'],
                                baseline_Q=base['Q'], candidate_Q=row['Q'],
                                risk_breach_events=row['risk_breach_events']))
    (OUT / 'comparison.json').write_text(json.dumps(comparisons, indent=2))
    verify()


if __name__ == '__main__': main()
