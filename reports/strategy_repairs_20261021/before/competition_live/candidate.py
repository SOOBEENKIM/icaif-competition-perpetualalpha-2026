"""Selected candidate factory with PAS capital, schedule and execution boundary.

No old practice runner is reused: that runner would omit the pair overlay.
The caller provides an actual-fill account and settlement callback for live use;
the default book account is a simulation and cannot transmit orders.
"""
import copy
import json
from pathlib import Path

from .coordinator import Portfolio
from experiments.decision_gate_publication_20261018.signals import DecisionGateSignals
from .profile import ROOT, BUNDLE, verify_bundle


def build_candidate(profile, settlement, *, account=None):
    if verify_bundle()!=profile['source_manifest_sha256']:
        raise ValueError('Selected model source differs from the scored profile')
    source=Path(__file__).parent/'strategy_plan.json'
    old=json.loads(source.read_text())
    # Only copied strategy configuration is read; no reserved observations.
    plan={k:copy.deepcopy(old[k]) for k in ['book_protocol','direct_spec','bias_spec',
        'model_latency_seconds','step_seconds']}
    plan.update(trading_protocol=copy.deepcopy(profile['trading']),
        paper_start_unix=profile['start_unix'],capture_end_unix=profile['end_unix'],
        end_liquidation_seconds=profile['execution']['end_liquidation_seconds'])
    plan['book_protocol']['pair_max_gross']=profile['configs']['reference']['pair_max_gross']
    models={(tuple(pair),24):json.loads((BUNDLE/'models'/('_'.join(pair)+'_24.json')).read_text())
            for pair in profile['pairs']}
    signals=DecisionGateSignals(profile['trading']['symbols'],profile['pairs'],models,
        plan['direct_spec'],plan['bias_spec'],profile['gate_spec'],plan['model_latency_seconds'])
    portfolio=Portfolio('first_plus_decision_gate','base',plan,settlement)
    if account is not None:
        if not getattr(account,'actual_fills_only',False):
            raise ValueError('Live coordinator requires an actual-fill account')
        if account.physical.initial!=profile['initial_capital']:
            raise ValueError('Live initial capital differs from scored profile')
        portfolio.account=account
    return dict(profile=profile,plan=plan,signals=signals,portfolio=portfolio,
        mode='actual_fill_adapter' if account is not None else 'simulation_only')
