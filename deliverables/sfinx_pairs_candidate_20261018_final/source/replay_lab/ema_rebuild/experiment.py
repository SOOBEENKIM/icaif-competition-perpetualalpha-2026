"""Run the predeclared 36-route comparison, seal selection, then audit reused history."""
import copy
import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from . import guard
from .models import build
from .engine import simulate, select
from .agent import OfflineEMAAgent
from ..common_comparison.models import load_inputs, timestamp
from ..common_comparison.experiment import protected_files, audit_run, write, digest, json_value

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def configure():
    spec = json.loads((SOURCE/'protocol.json').read_text())
    common = json.loads((ROOT/'replay_lab/common_comparison/protocol.json').read_text())
    # Only the explicit execution/risk definitions below are inherited. Do not
    # carry obsolete 4h clocks or old selection descriptions into the bundle.
    protocol = {k: copy.deepcopy(common[k]) for k in ('initial', 'symbols', 'risk', 'execution', 'scenarios')}
    protocol['risk']['sizing_headroom'] = spec['sizing_headroom']
    protocol['clock'] = dict(decision_hours=6, source_delay_hours=0, execution_delay_hours=1)
    protocol['funding_assumption'] = common['funding'].split(' Decision reserve')[0]
    protocol['capacity_assumption'] = common['capacity']
    protocol['historical_independence'] = False
    return spec, protocol


def model_hash(model):
    return hashlib.sha256(np.ascontiguousarray(model['target_weight']).tobytes()).hexdigest()


def report(out, result):
    cols = ['phase', 'policy', 'scenario', 'ROI %', 'daily STD %', 'MDD %', 'fills', 'short opens', 'fees', 'funding', 'gates']
    rows = []
    for row in result['routes']:
        m = row['metrics']
        values = [row['phase'], row['policy'], row['scenario'], f"{m['roi']*100:.5f}",
            f"{m['daily_vol']*100:.5f}", f"{m['max_drawdown']*100:.5f}", str(m['fills']),
            str(m['short_opening_fills']), f"{m['fees']:.2f}", f"{m['funding']:.2f}", str(row['gates'])]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    message = ('설정 선택 구간에서 개선 조건을 통과했습니다.' if result['selection']['improvement_on_selection']
               else '설정 선택 구간에서 사전 개선 조건을 통과한 후보가 없어 EMA 기준을 유지합니다.')
    conclusion = result['conclusion']
    doc = '<!doctype html><meta charset="utf-8"><title>EMA 재구축 비교</title><style>body{font:16px system-ui;margin:36px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:7px;text-align:right}th{background:#eef}pre{white-space:pre-wrap}a{color:#136}</style>'
    doc += '<h1>EMA 재구축: 구현·검증·36회 제한 비교</h1><p>'+html.escape(message)+'</p>'
    doc += '<p>선택: <b>'+html.escape(result['selection']['selected'])+'</b>. '+html.escape(conclusion)+'</p>'
    doc += '<p>모든 수치는 이미 연구에 사용한 과거 mainnet 참고 가격의 비용 후 시뮬레이션입니다. 실제 대회 par 손익, oracle 평가, 독립 검증 성과가 아닙니다. 세 자산만 사용했습니다. 미관측 funding은 기본 0이고 스트레스는 매시간 gross notional의 1bp를 추가 차감합니다.</p>'
    doc += '<p>기존 EMA10/60·ATR20×3·규모를 고정했습니다. 추가 요소는 하락 추세의 신규 short 진입, train-only Gaussian VAE 3개 고정 seed ensemble의 OOD 위험 축소입니다. FineFT 전체 재현이 아닙니다. 학습: 7월 1일 전, OOD 보정: 7월 1–29일, 선택: 7월 29일–8월 26일, 재사용 회고 평가: 8월 26일–10월 5일. 원래 보존 전략은 변경하지 않았습니다.</p>'
    doc += '<p>봉 종가는 봉 종료 시각에 입력되고 주문은 다음 시간 종가+비용에 체결됩니다. 전략 판단은 6시간, 위험 점검은 매시간입니다. 여유폭 90%, Decimal 수량, 감축 우선, 부분체결·잔량취소, 초기자본 기준 원장과 UTC 완전 일자 표준편차를 공통 적용했습니다. 위험 초과가 다음 감축 기회 후에도 남으면 실패 처리했습니다.</p>'
    doc += '<p>공식 점수 가중치는 만들지 않았습니다. 수익률과 일간 표준편차를 별도로 비교합니다. 선택 후 회고 성과로 후보를 바꾸지 않았고, 아래에는 탈락 후보와 비용 스트레스 결과도 모두 포함합니다.</p>'
    doc += '<p>실행·회계 감사: <a href="verification.json">verification.json</a> · 전체 결과: <a href="results.json">results.json</a> · 고정 계획: <a href="sealed_protocol.json">sealed_protocol.json</a> · 동결 후보: <a href="selection.json">selection.json</a> · 오프라인 bundle: <a href="bundle.json">bundle.json</a></p>'
    doc += '<table><thead><tr>'+''.join('<th>'+v+'</th>' for v in cols)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    doc += '<h2>검증 증거</h2><pre>'+html.escape(json.dumps(result['verification'], ensure_ascii=False, indent=2))+'</pre>'
    doc += '<p>출처: <a href="https://perpetual-alpha26.github.io/about">대회 공식 평가 설명</a>, <a href="https://personal.ntu.edu.sg/boan/papers/KDD26_FineFT.pdf">FineFT</a>. FineFT 원논문 수익률을 이 전략 기대수익으로 이전하지 않았습니다.</p>'
    (out/'report.html').write_text(doc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-training', type=Path, help='Reuse identically configured training after a non-model output failure')
    parser.add_argument('--prior-aborted-route-evaluations', type=int, default=0)
    args = parser.parse_args()
    spec, protocol = configure()
    out = ROOT/'replay_lab/output/ema_rebuild'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    protected_before, manifest = protected_files()
    sources = {str(p.relative_to(ROOT)): digest(p) for p in SOURCE.glob('*') if p.is_file()}
    sources.update({str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'replay_lab/common_comparison').glob('*.py')})
    previous = None
    if args.reuse_training:
        previous = json.loads((args.reuse_training/'sealed_protocol.json').read_text())
        assert previous['specification'] == spec and previous['execution'] == protocol
        assert previous['protected'] == protected_before
        for name in ('guard.py', 'models.py', 'engine.py'):
            key = str((SOURCE/name).relative_to(ROOT))
            assert previous['sources'][key] == sources[key]
    seal = dict(created=now(), specification=spec, execution=protocol, sources=sources,
                protected=protected_before, no_outcome_opened=previous is None,
                restart_from=str(args.reuse_training) if previous else None,
                restart_reason='JSON numpy boolean serialization only; models and selection unchanged' if previous else None,
                original_plan_sha256=digest(args.reuse_training/'sealed_protocol.json') if previous else None)
    write(out/'sealed_protocol.json', seal)
    print(json.dumps(dict(status='sealed', output=str(out)), ensure_ascii=False), flush=True)
    market, original, data_manifest = load_inputs(ROOT)
    selected_ema = json.loads((ROOT/'replay_lab/ema_development/selected_policy.json').read_text())
    periods = {k: [timestamp(t) for t in v] for k, v in spec['periods'].items()}
    x = guard.inputs(market)
    state = (json.loads((args.reuse_training/'vae_state.json').read_text()) if previous else
             guard.train(x, market.times, spec['vae'], *periods['train'], *periods['calibrate']))
    write(out/'vae_state.json', state)
    scores = guard.score(x, state)
    flags = guard.defensive_flags(scores, market.times, state)
    models = build(original, manifest, selected_ema)
    verification = dict(protected_files=len(protected_before), observed_assets=list(market.symbols),
        train_rows=len(state['train_indices']), calibration_rows=len(state['calibration_indices']),
        last_train_time=state['last_train_time'], last_calibration_time=state['last_calibration_time'],
        training_before_calibration=state['last_train_time'] < periods['calibrate'][0],
        calibration_before_selection=state['last_calibration_time'] < periods['select'][0],
        threshold_enter=state['threshold_enter'], threshold_exit=state['threshold_exit'],
        training_reused=previous is not None, new_training_fits=0 if previous else len(spec['vae']['seeds']),
        earlier_aborted_route_evaluations=args.prior_aborted_route_evaluations,
        model_hashes={k: model_hash(v) for k, v in models.items()}, live_orders_sent=0, routes=[])
    assert verification['training_before_calibration'] and verification['calibration_before_selection']
    np.testing.assert_allclose(state['mean'], x[state['train_indices']].mean(0), rtol=0, atol=0)
    for cut in (periods['select'][0], periods['audit_reused'][0]):
        idx = market.exact(cut)
        changed = copy.deepcopy(market)
        changed.close[idx+1:] *= np.array([1.8, .4, 2.2])
        changed.volume[idx+1:] *= 9
        mutated = guard.inputs(changed)
        np.testing.assert_array_equal(x[:idx+1], mutated[:idx+1])
    verification['feature_future_mutation_checks'] = 2
    # A fresh load of weights and scalers must yield identical inference.
    np.testing.assert_array_equal(scores, guard.score(x, json.loads((out/'vae_state.json').read_text())))
    print(json.dumps(dict(status='trained', train_rows=verification['train_rows'], calibration_rows=verification['calibration_rows'])), flush=True)
    rows, stored = [], {}
    def run(phase, name, model=None):
        for scenario in ('base', 'stress'):
            model = models[name] if model is None else model
            guarded = name in ('ema_vae', 'ema_signed_vae')
            r = simulate(market, model, name, protocol, scenario, *periods[phase],
                         flags=flags if guarded else None,
                         buffer=spec['rebalance_fraction']['retained' if name in spec['controls'] else 'ema'])
            check = audit_run(r)
            filename = f'{phase}__{name}__{scenario}.json.gz'
            with gzip.open(out/filename, 'wt') as handle:
                json.dump(r, handle, allow_nan=False, default=json_value)
            row = {k: r[k] for k in ('policy', 'scenario', 'metrics', 'gates', 'start', 'end')}
            row.update(phase=phase, file=filename, sha256=digest(out/filename))
            rows.append(row); stored[phase, name, scenario] = r
            verification['routes'].append(dict(phase=phase, policy=name, scenario=scenario, **check))
            print(json.dumps(dict(status='route', phase=phase, name=name, scenario=scenario,
                roi=r['metrics']['roi'], volatility=r['metrics']['daily_vol'], gates=r['gates']), default=json_value), flush=True)
    names = spec['candidates']+spec['controls']
    for name in names:
        run('select', name)
    selection = select(rows)
    diagnostic_scales = {}
    for name, parent in [('ema_vae', 'ema_base'), ('ema_signed_vae', 'ema_signed')]:
        a, b = stored['select', name, 'base'], stored['select', parent, 'base']
        diagnostic_scales[name] = min(1., a['metrics']['mean_gross']/max(b['metrics']['mean_gross'], 1e-12))
    selection.update(frozen_at=now(), diagnostic_scales=diagnostic_scales,
        protocol_sha256=digest(out/'sealed_protocol.json'), vae_sha256=digest(out/'vae_state.json'))
    write(out/'selection.json', selection)
    selection_hash = digest(out/'selection.json')
    for name in names:
        run('audit_reused', name)
    for name, parent in [('ema_vae', 'ema_base'), ('ema_signed_vae', 'ema_signed')]:
        model = copy.deepcopy(models[parent]); model['target_weight'] *= diagnostic_scales[name]
        run('audit_reused', name+'_constant_exposure_diagnostic', model)
    assert len(rows) == spec['budget']['total_routes']
    assert digest(out/'selection.json') == selection_hash
    # Replay the exact accounting snapshots through the deliverable controller,
    # including a blocked-input attempt and snapshot/restore on every decision.
    chosen = stored['audit_reused', selection['selected'], 'base']
    adapter = OfflineEMAAgent(selection['selected'], protocol, selected_ema, state)
    adapter.prepare_history(market, original, manifest)
    decisions = {d['decision_time']: d for d in chosen['decisions']}
    parity_count = 0
    for snap in chosen['snapshots'][:-1]:
        before = adapter.snapshot()
        assert adapter.decide(snap['time'], snap['qty'], snap['equity'], valid_input=False) is None
        assert adapter.snapshot() == before
        actual = adapter.decide(snap['time'], snap['qty'], snap['equity'])
        expected = decisions.get(snap['time'])
        assert (actual is None) == (expected is None)
        if actual:
            for k in ('target_qty', 'risk_forced', 'defensive'):
                assert actual[k] == expected[k], (k, actual, expected)
            parity_count += 1
        restored = OfflineEMAAgent(selection['selected'], protocol, selected_ema, state, adapter.snapshot())
        restored.market, restored.model, restored.flags = adapter.market, adapter.model, adapter.flags
        adapter = restored
    verification['agent_replay_matched_decisions'] = parity_count
    verification['agent_blocked_input_and_restore_checks'] = len(chosen['snapshots'])-1
    verification['selected_name_unchanged_after_audit'] = True
    verification['protected_unchanged'] = protected_files()[0] == protected_before
    assert verification['protected_unchanged']
    assert all(digest(ROOT/p) == expected for p, expected in sources.items())
    verification['source_unchanged_during_experiment'] = True
    verification['counterfactual_action_changes'] = {}
    for phase in ('select', 'audit_reused'):
        base = stored[phase, 'ema_base', 'base']
        for name in spec['candidates'][1:]:
            r = stored[phase, name, 'base']
            a = {d['decision_time']: d['target_qty'] for d in base['decisions']}
            changes = sum(d['target_qty'] != a.get(d['decision_time']) for d in r['decisions'])
            verification['counterfactual_action_changes'][phase+'/'+name] = dict(
                changed_targets=changes, short_opening_fills=r['metrics']['short_opening_fills'],
                defensive_decisions=r['metrics']['defensive_decisions'])
    write(out/'verification.json', verification)
    baseline = stored['audit_reused', 'ema_base', 'base']
    audit_improves = (selection['improvement_on_selection'] and all(chosen['gates'].values())
        and chosen['metrics']['roi'] > baseline['metrics']['roi']
        and chosen['metrics']['daily_vol'] <= baseline['metrics']['daily_vol'])
    conclusion = ('동결 후보가 재사용 회고 구간에서도 EMA 대비 수익률 개선·변동성 비증가를 보였습니다. 독립 검증은 아직 없습니다.'
        if audit_improves else '두 구간에 걸쳐 수익률 개선·변동성 비증가를 확인한 새 후보는 없습니다. 회고 결과로 재선택하지 않습니다.')
    bundle = dict(created=now(), policy=selection['selected'], execution_mode='offline_only', live_orders_enabled=False,
        ema_configuration=selected_ema, execution=protocol, vae_file='vae_state.json',
        vae_sha256=digest(out/'vae_state.json'), sources=sources, selection_sha256=selection_hash,
        adapter='replay_lab.ema_rebuild.agent.OfflineEMAAgent', independent_validation=False,
        prospective_start='After a separately recorded final freeze; no forward observer started',
        limitations=spec['limits'])
    write(out/'bundle.json', bundle)
    result = dict(completed=now(), selection=selection, conclusion=conclusion, audit_improves=bool(audit_improves),
        routes=rows, verification=verification, data_manifest=data_manifest, prospective_runs=0, live_orders=0)
    write(out/'results.json', result)
    report(out, result)
    write(out.parent/'latest.json', dict(directory=str(out), status='complete', selection=selection,
                                       report=str(out/'report.html'), conclusion=conclusion))
    print(json.dumps(dict(status='complete', output=str(out), selected=selection['selected'], conclusion=conclusion), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
