"""A second, prospectively frozen quote-history comparison. No order client.

The primary was selected using the first five-candidate test. That selection
is development; this later interval is a separate, unused confirmation.
"""
import json
import sys
import time
from datetime import datetime, timezone

from .protocol import ROOT, OUT as PREVIOUS, save, sha

OUT = ROOT / 'reports/quote_history_replication_20261025'
START = int(datetime(2026, 10, 25, 9, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 10, 25, 15, 5, tzinfo=timezone.utc).timestamp())


def freeze():
    if time.time() >= START or (OUT/'protocol.json').exists():
        raise ValueError('Must freeze once before the unused interval')
    from competition_live.profile import load_profile
    profile = load_profile()
    sources = set()
    for folder in ['production_replay_20261025', 'candidate_handoff',
                   'competition_live', 'execution_revision_20261024',
                   'pas_economic_allocation', 'universe_repair', 'bot',
                   'replay_lab/common_comparison', 'strategy_final_risk']:
        sources.update((ROOT/folder).rglob('*.py'))
    sources.update([ROOT/'competition_live/strategy_plan.json',
                    ROOT/'.runtime/pas_budget_lifecycle/execution_policy.json',
                    ROOT/'.runtime/pas_budget_lifecycle/strategy_revision.json'])
    previous = PREVIOUS/'qualification_review/results.json'
    save(OUT/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        start=START, end=END, primary='quote_inputs',
        controls=['incumbent','frozen_side_quote'],
        selection_evidence=dict(path=str(previous),sha256=sha(previous),family_size=5,
            disclosure='First five overlapping tests all failed the strict gate. Quote_inputs had small positive paired raw-PnL differences across both phases/costs. Selection used that test; do not call it independent confirmation.'),
        initial_condition='One sanitized actual owner checkpoint before09UTC, shared by every arm. Existing own inventory and own cash, fixed teammate reserve. No real orders.',
        own_initial_capital=50000, team_initial_capital=100000,
        profile=profile, phases_seconds=[0,45], loop_seconds=90,
        scenarios=dict(base=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=1.,pre_send_work_seconds=90),
                       stress=dict(fee_bps=12.,depth_fraction=.5,latency_ms=3000,receipt_multiplier=.5,pre_send_work_seconds=120)),
        acceptance='Unchanged prior gate: candidate positive net and positive end-liquidated PnL, greater than BOTH controls after end liquidation in BOTH phases and BOTH costs. No unresolved accounting, risk, pending or unknown orders; complete funding cash/exit depth/OI capacity. Data gaps must be bracketed by valid observations within90seconds. Advantage versus incumbent above actual incumbent replay discrepancy plus cash rounding. No favorable phase or scenario selection.',
        execution='Lot-exact fills, serial90/120s pre-send work then1/3s observed-book arrival;90s loop after work. Fees, depth depletion and positive funding receipt stress. Published packets use first received time; all features as-of. Never use future funding or prices in signals.',
        historical_features='First sealed capture through08:05UTC plus new separate capture from before09UTC. Missing quote history stays missing. No revised trade candle substituted for a quote.',
        no_peeking='No performance computation until15:05UTC plus65seconds; no candidate or threshold changes using this interval. Operational data coverage may be checked without outcome evaluation.',
        interpretation='Six-hour conditional paper confirmation, not live profit, not official daily volatility/Q or proof of no overfitting. No automatic deployment.',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)},
        historical_capture_sha=sha(PREVIOUS/'sealed_forward_capture/received_through_fixed_end.jsonl.gz'),
        orders_sent=0, live_changed=False))
    save(OUT/'forward_contract.json',dict(start_utc='2026-10-25T09:00:00Z',
        end_utc='2026-10-25T15:05:00Z',primary='quote_inputs',orders_sent=0))


def check():
    p=json.loads((OUT/'protocol.json').read_text())
    for f,h in p['hashes'].items():
        if sha(ROOT/f)!=h:raise ValueError('Frozen source changed: '+f)
    if sha(PREVIOUS/'sealed_forward_capture/received_through_fixed_end.jsonl.gz')!=p['historical_capture_sha']:
        raise ValueError('Historical feature archive changed')
    return p


def seed():
    from competition_live.runtime import private_load, save_coordinator
    p=check()
    while time.time()<START-120:
        time.sleep(min(20,START-120-time.time()))
    if time.time()>=START or (OUT/'forward_seed.pkl').exists():
        raise ValueError('Seed must be captured once, before start')
    source=ROOT/'.runtime/pas_budget_lifecycle/coordinator.pkl'
    for _ in range(8):
        before=sha(source);saved=private_load(source);after=sha(source)
        state=saved['portfolio']
        if before==after and not state.fatal and not state.halted and not state.account.unknown and not state.account.pending_orders:
            break
        time.sleep(2)
    else:raise ValueError('Could not capture a reconciled initial state')
    if time.time()>=START:raise ValueError('Checkpoint arrived after fixed start')
    # The production serializer explicitly removes every execution callback;
    # this is a research copy, never a write to the live checkpoint.
    save_coordinator(OUT/'forward_seed.pkl',state,p['profile'],'research_replication_seed')
    clean=private_load(OUT/'forward_seed.pkl')['portfolio']
    assert clean.settlement is None
    for attr in ['executor','checkpoint','collect_fills','refresh_market']:
        assert getattr(clean.account,attr) is None
    save(OUT/'seed_manifest.json',dict(at=time.time(),source_sha=after,
        seed_sha=sha(OUT/'forward_seed.pkl'),last_decision=clean.last_decision,
        own_qty=clean.account.physical.qty,teammate_qty=clean.account.reserved.qty,
        source_fingerprint=saved['fingerprint'],orders_sent=0,live_changed=False))


def capture():
    check()
    from . import capture as collector
    collector.OUT=OUT/'capture'
    sys.argv=['public_replication_capture','--seconds',str(max(1,int(END+65-time.time())))]
    collector.main()


def signals():
    check()
    from . import archive_signals
    archive_signals.OUT=OUT
    archive_signals.main()


if __name__=='__main__':
    globals()[sys.argv[1]]()
