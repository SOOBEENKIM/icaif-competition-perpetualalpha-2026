"""Prespecified shared-factor experiment on retained execution and cost rules."""
import copy
import csv
import gzip
import html
import json
import time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.adaptive_residual_pairs_20261017.model import frozen_predictions
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.model import design, labels, map_state
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as PAR_INPUT, save, sha
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.conditional_completion_pairs_20261017.portfolio import replay
from experiments.rolling_feature_pairs_20261017.model import rolling as direct_rolling
from .model import prepare, rolling

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'
PARENT = ROOT / 'experiments/conditional_completion_pairs_20261017/artifacts'
DIRECT = ROOT / 'experiments/rolling_feature_pairs_20261017/artifacts'
EXT = ROOT / 'experiments/historical_par_extension_20261017/artifacts_v3'
MATCHED = 'first_dense_hourlyhedge_cash_direct_24'
ARMS = {kind: f'first_dense_hourlyhedge_factor_{kind}_24' for kind in ['combined', 'market', 'residual']}


def path_for(venue, phase, scenario):
    return PARENT / 'runs' / f'{venue}-{phase}-{MATCHED}-{scenario}.json.gz'


def render(result):
    dest = ROOT / 'reports/shared_factor_pairs_20261017'
    dest.mkdir(exist_ok=True)
    keys = ['venue', 'phase', 'arm', 'scenario', 'net_pnl', 'roi', 'financial_score', 'volatility_horizon', 'max_drawdown', 'completed_pairs', 'unbalanced_minutes']
    with (dest / 'results.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in result['rows']: w.writerow({k: r.get(k, r['metrics'].get(k)) for k in keys})
    page = '''<!doctype html><meta charset="utf-8"><title>공유 BTC 요인과 페어 잔차 비교</title><style>body{font:15px system-ui;margin:32px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:7px;text-align:right}td:first-child{text-align:left}pre{background:#f5f5f5;padding:16px;white-space:pre-wrap}</style>
<h1>공유 BTC 방향 예측＋페어별 잔차 예측</h1><p>개선된 연구 기준은 cash_direct이다. 새 combined 후보만 선정 대상이고 market/residual은 구성 요소의 영향을 보는 진단용이다. 기존 헤지·수량·위험 한도·조건부 헤지 완성·보유 연장·독립 체결·수수료·펀딩·청산 가정은 유지했다. 실거래 교체는 하지 않았다.</p>
<p>BTC 예측기는 BTC 자료만 사용하며 하루 하나를 학습해 두 페어가 공유한다. 각 페어는 원래 20개 특징으로 시장 노출을 제거한 나머지 수익률을 학습한다. 시장 노출은 판단 시점까지의 단순 수익률로 추정하고, 잔차의 정규화 단위는 원래 페어 변동성을 유지한다. 분리 전후 실제 목표 수익률의 항등식과 미래 자료 차단을 검사했다. residual 신호만 쓰더라도 실제 포지션을 시장 중립으로 바꾼 것은 아니다.</p>
<p>두 기준 전략보다 개선 여부, 이미 개선된 cash_direct보다 개선 여부, 표본 충분성을 구분한다. 완료 페어 3건은 내부 참고 기준이지 공식 제출 요건이 아니다. 초기 1M, 기본/불리한 비용 조건 각각 비교한다. 위험 한도는 같지만 실제 투자 비중은 달라질 수 있다.</p>
<p>모두 재사용 과거 자료의 개발 백테스트다. reference는 대회 호가 체결이 아니며, par의 캔들 거래량은 실제 호가 수량과 다르다. 한쪽 체결과 유료 정리 과정을 포함하지만 그 소요 시간은 모의 가정이다. 1일 단독 Q/변동성은 표본 부족으로 표시하지 않는다.</p>'''
    page += '<h2>8월 사전 판정</h2><pre>' + html.escape(json.dumps(result['selection'], ensure_ascii=False, indent=2)) + '</pre>'
    for venue, phase in result['windows']:
        page += f'<h2>{venue} / {phase}</h2><table><tr><th>전략</th><th>비용</th><th>순손익</th><th>수익률</th><th>Q</th><th>기간 변동성</th><th>MDD</th><th>완료 페어</th><th>미헤지 분</th></tr>'
        for r in result['rows']:
            if (r['venue'], r['phase']) != (venue, phase): continue
            m = r['metrics']; q = '—' if m['financial_score'] is None else f"{m['financial_score']:.8f}"
            v = '—' if m['volatility_horizon'] is None else f"{100*m['volatility_horizon']:.6f}%"
            page += f"<tr><td>{html.escape(r['arm'])}</td><td>{r['scenario']}</td><td>{m['net_pnl']:+.6f}</td><td>{100*m['roi']:+.6f}%</td><td>{q}</td><td>{v}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{m['completed_pairs']}</td><td>{m['unbalanced_minutes']:.0f}</td></tr>"
        page += '</table>'
    (dest / 'report.html').write_text(page)
    return str((dest / 'report.html').relative_to(ROOT))


def main():
    OUT.mkdir(exist_ok=False); (OUT / 'runs').mkdir()
    spec = json.loads((HERE / 'protocol.json').read_text())
    tests = (HERE / 'test_results.txt').read_text()
    assert 'Ran 9 tests' in tests and tests.rstrip().endswith('OK')
    parent = json.loads((PARENT / 'results.json').read_text()); windows = parent['windows']
    rows = [r for r in parent['rows'] if r['arm'] in ['first_strategy', 'second_strategy', MATCHED]]
    assert len(rows) == 36
    cfgs = {v: json.loads((ROOT / f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference', 'par']}
    cfgs['par_extended'] = copy.deepcopy(cfgs['par'])
    refs = {'reference': ROOT / 'experiments/risk_preserving_20261016/artifacts/inputs.npz',
            'par_extended': ROOT / 'practice_live/updates/20261017T120009814481Z/inputs.npz'}
    expert = json.loads((ROOT / 'experiments/hourly_refresh_pairs_20261017/model_protocol.json').read_text())
    models = {}; model_paths = []
    for pair in expert['pairs']:
        path = ROOT / expert['model_directory'] / ('_'.join(pair) + '_24.json')
        models[tuple(pair), 24] = json.loads(path.read_text()); model_paths.append(path)
    sources = json.loads((PARENT / 'frozen.json').read_text())['sources'].copy()
    paths = [*HERE.glob('*.py'), HERE / 'protocol.json', HERE / 'test_results.txt', PARENT / 'results.json', PARENT / 'frozen.json',
             *refs.values(), *model_paths, ROOT / spec['prior_artifact'], ROOT / 'experiments/pairs_development_status_20261017/status.json']
    paths += [path_for(v, phase, sc) for v, phase in windows for sc in ['base', 'stress']]
    sources.update({str(path.relative_to(ROOT)): sha(path) for path in paths}); verify(sources)
    prospective = json.loads((ROOT / 'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources']
    verify(prospective)
    save(OUT / 'frozen.json', dict(at_unix=time.time(), sources=sources, protected=protected(), prospective_source_count=len(prospective),
        previous_goal_turn='progress_reference_improvement_status_and_submission_rules_clarified', historical_independence=False))
    rd, _, p = inputs(cfgs['reference']); assert p['symbols'][0] == 'BTC'
    data = {'reference': rd, 'par': load(PAR_INPUT / 'inputs.npz'), 'par_extended': load(EXT / 'inputs.npz')}
    controls = {v: load(DIRECT / f'{v}_direct_24_state.npz') for v in ['reference', 'par']}
    controls['par_extended'] = load(EXT / 'direct_state.npz')
    states, preparation = {}, []
    for venue, path in refs.items():
        ref = load(path); pairs = cfgs[venue]['dense_pairs']
        f = design(ref['market_times'], ref['close'], ref['volume'], ref['trades'], p['symbols'], pairs)
        raw = frozen_predictions(f, models, pairs, 24)
        y, maturity = labels(ref['market_times'], ref['close'], f, 24)
        terminal = np.zeros_like(f['fresh']); terminal[:-24] = f['fresh'][24:]
        old_spec = {**spec, 'candidates': ['direct'], 'diagnostic_variants': []}
        old, _, _ = direct_rolling(ref['market_times'], f, y, maturity, raw, terminal, old_spec)
        factor = prepare(ref, f, spec['horizon'], spec['factor_window'])
        np.testing.assert_allclose(factor['actual'], y*f['vol']*np.sqrt(24), rtol=1e-12, atol=1e-15)
        predictions, btc, components, stamps, records = rolling(ref['market_times'], f, factor, raw, spec)
        np.savez_compressed(OUT / f'{venue}_issued_forecasts.npz', times=ref['market_times'], target=y, maturity=maturity,
            unit=f['vol']*np.sqrt(24), prediction_direct=old['direct'], fit_stamps=stamps, fresh=f['fresh'], terminal_fresh=terminal,
            btc_prediction=btc, btc_target=factor['market_return'], gamma=factor['gamma'], residual_target=factor['residual_target']*factor['pair_unit'],
            **{f'prediction_{k}': v for k, v in predictions.items()}, **{f'component_{k}': v for k, v in components.items()})
        save(OUT / f'{venue}_fits.json', records)
        for v in (['reference'] if venue == 'reference' else ['par', 'par_extended']):
            exact = map_state(data[v]['times'], ref, f, old['direct'], f['pairs'], source_hours=1, lifetime_hours=1)
            for key in controls[v]: np.testing.assert_array_equal(exact[key], controls[v][key], err_msg=f'{v}-{key}')
            states[v] = {}
            for kind, value in predictions.items():
                state = map_state(data[v]['times'], ref, f, value, f['pairs'], source_hours=1, lifetime_hours=1)
                for key in ['beta', 'vol', 'source_prices', 'signal_times']: np.testing.assert_array_equal(state[key], controls[v][key])
                np.savez_compressed(OUT / f'{v}_{kind}_state.npz', **state); states[v][kind] = state
        preparation.append(dict(venue=venue, models=len(records), market_models=sum(r['role']=='market' for r in records),
            strict_maturity=all(r['last_maturity'] < r['fit_time'] for r in records), last_source=int(ref['market_times'][-1])))
        print('PREPARED', venue, preparation[-1], flush=True)
    save(OUT / 'preparation.json', preparation)
    verified = []
    for venue, phase in windows:
        for sc in ['base', 'stress']:
            with gzip.open(path_for(venue, phase, sc), 'rt') as f: old = json.load(f)
            cfg = copy.deepcopy(cfgs[venue]); cfg['conditional_completion'] = {'risk_aversion': 0.}
            run = replay(data[venue], controls[venue], None, cfg, p, MATCHED, sc, old['start'], old['end'])
            for key in ['events', 'fills', 'metrics', 'snapshots']: assert run[key] == old[key], (venue, phase, sc, key)
            verified.append(dict(venue=venue, phase=phase, scenario=sc))
        print('EXACT', venue, phase, flush=True)
    lookup = {(r['venue'], r['phase'], r['arm'], r['scenario']): r['metrics'] for r in rows}
    def record(venue, phase, kind, sc):
        cfg = copy.deepcopy(cfgs[venue]); cfg['conditional_completion'] = {'risk_aversion': 0.}
        with gzip.open(path_for(venue, phase, sc), 'rt') as f: old = json.load(f)
        arm = ARMS[kind]
        run = replay(data[venue], states[venue][kind], None, cfg, p, arm, sc, old['start'], old['end'])
        with gzip.open(OUT / 'runs' / f'{venue}-{phase}-{arm}-{sc}.json.gz', 'wt') as f:
            json.dump(run, f, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())
        m = run['metrics']; lookup[venue, phase, arm, sc] = m
        rows.append(dict(venue=venue, phase=phase, arm=arm, scenario=sc, diagnostic_only=kind!='combined', metrics=m,
                         pnl_advantage_matched=m['net_pnl']-lookup[venue, phase, MATCHED, sc]['net_pnl']))
        print(venue, phase, kind, sc, 'PnL', round(m['net_pnl'], 6), 'Q', m['financial_score'], 'pairs', m['completed_pairs'], flush=True)
    for kind in ARMS:
        for sc in ['base', 'stress']: record('reference', 'validation', kind, sc)
    comparisons = []
    for sc in ['base', 'stress']:
        m = lookup['reference', 'validation', ARMS['combined'], sc]
        comparisons.append(dict(scenario=sc, positive_pnl=m['net_pnl'] > 0, no_risk_breach=m['risk_breach_bars']==0,
            above_native_pnl_q=all(m['net_pnl']>lookup['reference','validation',b,sc]['net_pnl'] and m['financial_score']>lookup['reference','validation',b,sc]['financial_score'] for b in ['first_strategy','second_strategy']),
            above_matched_pnl_q=m['net_pnl']>lookup['reference','validation',MATCHED,sc]['net_pnl'] and m['financial_score']>lookup['reference','validation',MATCHED,sc]['financial_score'],
            completed_pairs=m['completed_pairs'], internal_sample_flag=m['completed_pairs']>=3))
    selection = dict(candidate=ARMS['combined'], checks=comparisons,
        historical_improvement_over_natives=all(r['above_native_pnl_q'] for r in comparisons),
        historical_improvement_over_retained=all(r['above_matched_pnl_q'] for r in comparisons),
        selected_for_further_research=all(r['positive_pnl'] and r['no_risk_breach'] and r['above_native_pnl_q'] and r['above_matched_pnl_q'] for r in comparisons),
        sample_flag_is_not_official_requirement=True, frozen_before_later_replays=True, historical_independence=False)
    save(OUT / 'selection.json', selection); decision_hash = sha(OUT / 'selection.json')
    for venue, phase in windows[1:]:
        for kind in ARMS:
            for sc in ['base', 'stress']: record(venue, phase, kind, sc)
    assert sha(OUT / 'selection.json') == decision_hash
    verify(sources); verify(prospective)
    result = dict(rows=rows, windows=windows, selection=selection, new_account_replays=36, exact_control_replays=verified,
        tests_passed=9, protected=protected(), orders_sent=0, live_strategy_changed=False, reserved_raw_read=False, historical_independence=False)
    save(OUT / 'results.json', result); report = render(result)
    save(HERE / 'work_state.json', dict(status='progress_shared_factor_accounts_completed', report=report, selection=selection,
        new_account_replays=36, exact_controls=12, tests_passed=9, orders_sent=0, live_strategy_changed=False, reserved_raw_read=False,
        historical_independence=False, goal_complete=False, no_progress_streak=0))
    print('DONE', report, flush=True)


if __name__ == '__main__': main()
