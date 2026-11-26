"""Separate, later-start registration; never edits the original sealed study."""
import argparse
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np

from experiments.published_pair_lifecycle_20261017.forward import (
    TapeWindow, FundingClock, raw_events, assert_ready, metrics, sha, save, CAPTURE)
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.complementary_alpha_20261017.research import protected
from .signals import DecisionGateSignals
from .portfolio import Portfolio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'artifacts'
ORIGINAL = ROOT / 'experiments/published_pair_lifecycle_20261017/artifacts/plan.json'
ARMS = ['first_strategy', 'second_strategy', 'first_plus_direct', 'first_plus_decision_gate']


def prepare():
    if (OUT / 'plan.json').exists():
        raise FileExistsError('Sidecar already registered; do not overwrite')
    now = time.time()
    original = json.loads(ORIGINAL.read_text())
    verify(original['sources'])
    assert sha(CAPTURE / 'plan.json') == original['capture_plan_sha256']
    # A genuinely future, common 12:00 UTC start; no retroactive October 17 arm.
    start = (int(now - 43200) // 86400 + 1) * 86400 + 43200
    if start <= now + 3600:
        start += 86400
    if start < original['paper_start_unix'] or original['capture_end_unix'] - start < 2*86400:
        raise ValueError('No future common window with two full daily returns')
    tests = (HERE / 'test_results.txt').read_text()
    if not ('Ran 9 tests' in tests and tests.rstrip().endswith('OK')):
        raise ValueError('Complete adapter and release-guard tests before registration')
    gate = ROOT / 'experiments/decision_gate_pairs_20261017'
    ablation = ROOT / 'experiments/graph_filter_ablation_20261018/artifacts/conclusion_frozen.json'
    sources = json.loads(ablation.read_text())['sources'].copy()
    sources.update(original['sources'])
    paths = [ORIGINAL, ablation, gate / 'protocol.json', *HERE.glob('*.py'),
             HERE / 'protocol.json', HERE / 'test_results.txt', OUT / 'old_book_audit.json']
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    verify(sources)
    plan = copy.deepcopy(original)
    plan.update(created_at_unix=now, paper_start_unix=start, arms=ARMS, sources=sources,
        original_plan=str(ORIGINAL.relative_to(ROOT)), original_plan_sha256=sha(ORIGINAL),
        registration='separate_later_start_sidecar_original_four_arms_unchanged',
        gate_spec=json.loads((gate / 'protocol.json').read_text()),
        historical_selection=dict(retained='first_dense_hourlyhedge_decisiongate_24',
            observed_reference_pnl_q_improvement=True, independent_validation=False),
        candidate_status='historically_improved_candidate_new_book_evaluation_pending',
        forecast_rule=('Use the original publication digest, receipt and source-midnight rules for unchanged direct '
            'means and the retained SPO+ entry gate. Preserve separate action/score fields. Missing midnight cannot be '
            'backfilled. Missing/flat/opposite gate blocks only new entries, not holding forecasts or exits. Fixed '
            '5-second allowance is shared with the control; measured historical adapter calls were below 0.1 seconds.'),
        comparison_rule=('First, second, ungated direct, retained decision-gated candidate, identical capital and '
            'received books. Common later start, settled funding, paid completion/exits and exact daily samples. '
            'No candidate reselection from these outcomes. Old original four-arm results remain a separate study.'),
        success_rule=dict(positive_net_pnl=True, pnl_and_q_above_both_native_arms_both_scenarios=True,
            no_fatal_data_or_execution_errors=True, no_residual_inventory=True,
            evidence_strength='Report balanced completed pairs and concentration; no invented official trade-count threshold.'),
        limitations=[
            'Hypothetical received-book fills; not actual exchange orders or a model of own market impact.',
            'Historical reference improvement is established; par improvement remains unobserved.',
            'No reconnect markers in the existing capture; enforce observed freshness without inventing connectivity.',
            'The par practice market may be inactive during this later window; absent data or exits invalidate the result.',
            'The independent scored market is not this par paper evaluation. No venue switch is implied.',
            'One future window, especially with few pairs, is not proof of general profitability.'],
        protected=protected(), orders_sent=0, automatic_live_replacement=False,
        reserved_raw_read_during_preparation=False, original_plan_mutated=False)
    # All work is finished before the plan is sealed, and both clocks are future-safe.
    if time.time() >= start:
        raise ValueError('Registration must finish before evaluation starts')
    OUT.mkdir(exist_ok=True)
    save(OUT / 'plan.json', plan)
    return plan


def ready(plan, now):
    # Original guard runs first, before any path, network or outcome inspection.
    assert_ready(plan, now)
    if sha(ROOT / plan['original_plan']) != plan['original_plan_sha256']:
        raise ValueError('Original registered comparison changed')


def evaluate(plan):
    ready(plan, time.time())
    if (OUT / 'results.json').exists():
        raise FileExistsError('Sidecar already evaluated')
    from experiments.cost_horizon_20261016.forward import info
    start, end = plan['paper_start_unix'], plan['capture_end_unix']
    rates = {}
    for symbol in plan['trading_protocol']['symbols']:
        data = info(dict(type='fundingHistory', coin='par:' + symbol, startTime=start*1000, endTime=end*1000))
        if not isinstance(data, list) or len(data) >= 500:
            raise ValueError('Incomplete settled funding response')
        save(OUT / f'funding_{symbol}.json', data)
        for row in data:
            at = int(row['time']); bucket = at // 3600000 * 3600000
            if not 0 <= at-bucket < 1000 or symbol in rates.setdefault(bucket, {}):
                raise ValueError('Unexpected or duplicate funding settlement')
            rates[bucket][symbol] = float(row['fundingRate'])
    tape = TapeWindow(raw_events(ROOT / plan['capture_directory']))
    bias = plan['bias_spec']
    models = {(tuple(pair), 24): json.loads((ROOT / bias['model_directory'] / ('_'.join(pair) + '_24.json')).read_text())
              for pair in bias['pairs']}
    signals = DecisionGateSignals(plan['trading_protocol']['symbols'], bias['pairs'], models,
        plan['direct_spec'], bias, plan['gate_spec'], plan['model_latency_seconds'])
    arms = {}
    for scenario, sc in plan['book_protocol']['scenarios'].items():
        for name in plan['arms']:
            clock = FundingClock(start, tape, rates, plan['trading_protocol']['symbols'], plan['book_protocol'],
                                 sc['funding_receipt_multiplier'])
            arms[name, scenario] = Portfolio(name, scenario, plan, clock)
    publication = None; forecasts = []
    for now in range(start, end+1, plan['step_seconds']):
        streams = tape.ensure(now*1000)
        # Includes causal model warm-up publications before the common account
        # start, only after the original reserved-outcome release.
        for event in tape.publications(now*1000):
            forecasts = signals.ingest_file(ROOT, event); publication = event
        for arm in arms.values():
            arm.step(now, streams, publication, forecasts)
    rows = []
    for (name, scenario), arm in arms.items():
        result = metrics(arm, start, end)
        rows.append(dict(arm=name, scenario=scenario, metrics=result))
        sessions = [*arm.sessions, *([arm.pair] if arm.pair else [])]
        payload = dict(metrics=result, snapshots=arm.snapshots, events=arm.events,
            fills=arm.account.physical.fills, owner_fills={k: o.fills for k, o in arm.account.owners.items()},
            funding=arm.account.physical.funding_events,
            pair_sessions=[dict(intent=s.intent, state=s.state, events=s.events, closed_reason=s.closed_reason) for s in sessions])
        with gzip.open(OUT / f'{name}-{scenario}.json.gz', 'wt') as handle:
            json.dump(payload, handle, allow_nan=False,
                      default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())
    lookup = {(r['arm'], r['scenario']): r['metrics'] for r in rows}
    checks = []
    for scenario in plan['book_protocol']['scenarios']:
        candidate = lookup['first_plus_decision_gate', scenario]
        for other in ['first_strategy', 'second_strategy', 'first_plus_direct']:
            control = lookup[other, scenario]
            valid = candidate['valid'] and control['valid']
            q, cq = candidate['interpretable_financial_score'], control['interpretable_financial_score']
            checks.append(dict(scenario=scenario, control=other, valid=valid,
                pnl_improved=bool(valid and candidate['net_pnl'] > control['net_pnl']),
                q_improved=bool(valid and q is not None and cq is not None and q > cq),
                candidate_positive=bool(valid and candidate['net_pnl'] > 0),
                completed_pairs=candidate['completed_pairs']))
    observed = all(c['valid'] and c['pnl_improved'] and c['q_improved'] and c['candidate_positive']
                   for c in checks if c['control'] in ['first_strategy', 'second_strategy'])
    ready(plan, time.time())
    save(OUT / 'results.json', dict(rows=rows, checks=checks, observed_pnl_q_improvement=observed,
        protected=protected(), actual_exchange_executions=False, orders_sent=0,
        automatic_live_replacement=False, original_plan_mutated=False))
    save(OUT / 'fit_audit.json', dict(direct=signals.audit, gate=signals.gate_audit))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare()
        print(json.dumps(dict(prepared=True, start=plan['paper_start_unix'], end=plan['capture_end_unix'],
                              sources=len(plan['sources']))))
    else:
        evaluate(json.loads((OUT / 'plan.json').read_text()))
