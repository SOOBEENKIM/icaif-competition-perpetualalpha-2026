"""Validate the prepared candidate and report conditions still requiring activation."""
import json
import time
import re
from pathlib import Path
from .profile import load_profile,verify_bundle
from .venue import account_equity,validate_metadata,phase_state


def main():
    root=Path(__file__).parent
    profile=load_profile()
    if verify_bundle()!=profile['source_manifest_sha256']:raise ValueError('Candidate source changed')
    snapshot=json.loads((root/'artifacts/public_preflight/snapshot.json').read_text())
    venue=validate_metadata(snapshot,profile)
    account=account_equity(snapshot,profile)
    phase=phase_state(snapshot['queried_at_unix'],profile)
    tests=(root/'artifacts/tests.log').read_text()
    matched=re.search(r'Ran (\d+) tests',tests)
    if not matched or int(matched[1])<28 or not tests.rstrip().endswith('OK'):
        raise ValueError('PAS integration tests have not passed')
    count=int(matched[1])
    live_input=json.loads((root/'artifacts/live_input_smoke.json').read_text())
    if not live_input['marks_available'] or live_input['decision_gate_forecasts']!=2 or live_input['orders_sent']!=0:
        raise ValueError('Actual PAS input / selected signal connection not verified')
    backtest=json.loads((root/'artifacts/backtest_100k/results.json').read_text())
    if not backtest['all_pnl_Q_improved']:raise ValueError('100k replay lost the claimed improvement')
    report=dict(checked_at_unix=time.time(),market_snapshot_at_unix=snapshot['queried_at_unix'],
        current_snapshot_is_not_an_activation_authorization=True,
        profile=profile['candidate'],network='testnet',dex='pas',collateral='USDP',initial_capital=100000,
        configuration_verified=True,tests_passed=count,backtest_accounts=18,
        selected_signal_and_PAS_public_input_verified=live_input,
        venue=venue,account=account,phase_at_snapshot=phase,
        unexpired_api_approvals=sum(a.get('validUntil',0)>snapshot['queried_at_unix']*1000 for a in snapshot['agents']),
        orders_sent=0,existing_practice_service_changed=False,live_service_activated=False,
        fresh_pre_send_checks=['PAS metadata and USDP token','actual USDP funding and mode-specific equity',
            'reconciled inventory','market and total OI','availableToTrade/maxTradeSzs','capital/risk limits','phase clock'],
        remaining_before_live_operation=['Actual funding at phase start and fresh preflight',
            'Activate the prepared independent producer and single-writer runner after the practice writer stops; no daemon was started by this preparation.'],
        paper_deadline={'date':profile['paper_deadline_date'],'time':None,'timezone':None})
    (root/'artifacts/preflight.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
