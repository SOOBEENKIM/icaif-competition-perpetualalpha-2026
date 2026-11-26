"""Record the fixed-universe result without changing any tested strategy."""
import csv
import json
import time
from collections import Counter
from pathlib import Path

from .research import ROOT, HERE, OUT, save, sha, verify, protected


def main():
    result = json.loads((OUT / 'results.json').read_text())
    books = json.loads((OUT / 'books/results.json').read_text())
    diagnostics = json.loads((OUT / 'diagnostics.json').read_text())
    for name in ['frozen.json', 'books/frozen.json', 'diagnostics_frozen.json']:
        verify(json.loads((OUT / name).read_text())['sources'])
    prospective = json.loads((ROOT / 'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources']
    verify(prospective)
    assert result['new_account_replays'] == 36 and len(result['exact_controls']) == 12
    assert result['tests_passed'] == 12 and result['selection']['selected'] is None
    assert books['exact_prior_trials'] == 4 and books['exact_prior_summaries'] == 20
    assert len(diagnostics['rows']) == 12
    assert books['full_portfolio_return'] is None and books['financial_score'] is None
    assert (HERE / 'diagnostics.log').read_text().rstrip().endswith('DONE pair and risk attribution')

    control = 'first_dense_hourlyhedge_decisiongate_24'
    labels = {
        'first_strategy': '기존 1번', 'second_strategy': '기존 2번',
        control: '유지: 두 페어 비용 민감 진입판',
        'first_dense_hourlyhedge_decision_crypto_24': '후속: 세 암호화폐 페어',
        'first_dense_hourlyhedge_decision_sector_24': '후속: 여섯 페어',
        'first_dense_hourlyhedge_decision_ungated_24': '진단용: 여섯 페어 필터 없음',
    }
    chosen = [r for r in result['rows'] if r['arm'] in labels]
    book_summary = []
    for model in sorted({r['model'] for r in books['summaries']}):
        trials = [r for r in books['trials'] if r['model'] == model]
        book_summary.append(dict(model=model, trials=len(trials),
                                 statuses=dict(Counter(r['status'] for r in trials)),
                                 unique_tape_time_pairs=len({(r['tape'], r['time'], tuple(r['intent']['pair'])) for r in trials})))
    risk = []
    for row in diagnostics['rows']:
        for b in row['risk_breaches']:
            assert b['forced_order_requested']
            assert all(q == 0 for q in b['next_snapshot']['qty'])
            risk.append(dict(phase=row['phase'], scenario=row['scenario'], kind=row['kind'],
                             time=b['time'], limits=b['limits'],
                             daily_volatility=b['account_snapshot']['daily_volatility'],
                             next_time=b['next_snapshot']['time'], next_gross=b['next_snapshot']['gross']))
    next_action = (
        'Retain the established decisiongate improvement over native strategies. '
        'Stop this fixed six-pair expansion: the same decision-cost proxy does not reliably rank '
        'additional pair lifecycle cash value. Before another model, inspect prior expert-selection '
        'work and assess strictly matured realized-lifecycle targets using unchanged execution. '
        'Do not remove ETH/HYPE after observing these outcomes, claim independent validation, '
        'read reserved raw data, or replace live automatically.'
    )
    conclusion = dict(
        at_unix=time.time(), classification='progress_fixed_universe_gate_evaluated_no_further_gain',
        retained_improved_candidate=control, additional_pnl_improvement=False,
        selection=result['selection'], new_account_replays=36, exact_controls=12, tests_passed=12,
        model_fits=sum(p['mean_fits'] + p['gate_fits'] for p in result['preparation']),
        reference_pair_attribution=[{k:r[k] for k in ['phase','scenario','kind','by_pair']} for r in diagnostics['rows']],
        risk_events=risk, book_summary=book_summary, old_book_trials_exact=4,
        protected=protected(), prospective_sources=len(prospective),
        no_official_submission_ban=True, historical_independence=False,
        orders_sent=0, live_strategy_changed=False, reserved_raw_read=False,
        goal_complete=False, no_progress_streak=0, next_action=next_action,
    )
    save(OUT / 'conclusion.json', conclusion)
    dest = ROOT / 'reports/decision_sector_pairs_20261018'
    dest.mkdir(exist_ok=False)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 확장 비교</title>
<style>body{max-width:1180px;margin:40px auto;font:16px/1.7 sans-serif;color:#172634;padding:0 18px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}</style>
<h1>비용 민감 진입 모델의 페어 확장 비교</h1>
<p><strong>기존 두 페어 개선판을 유지한다.</strong> 동일 모델을 세 암호화폐 페어 및 여섯 페어로 확장했지만 추가 PnL·Q 개선을 얻지 못했다. 이 실패는 기존 1·2번 대비 이미 확인한 개선판의 성과를 취소하지 않는다.</p>
<p>초기자본 1M. 기준시장 과거 가격과 모의 체결·수수료·펀딩을 반영한 계좌 평가이며 종료 시 미청산 포지션의 평가손익을 포함한다. 두 기간 경계는 12UTC(한국시간 21시)다. 반복 사용한 과거 개발 결과이며 독립 미래 검증이 아니다. 8월 판정을 먼저 고정했으며 이후 기간에서 후보를 재선택하지 않았다.</p>
<h2>변경 범위</h2><p>기존 BTC/ETH·BTC/HYPE의 예측과 실행을 정확히 재현했다. 기존 일별 릿지 수익 예측과 별도의 SPO+ 비용 민감 방향 분류를 ETH/HYPE, SP500/EWY, SP500/EWJ, EWY/EWJ에도 같은 규칙으로 적용했다. 계좌 자본·위험 한도와 한 번에 한 페어만 보유하는 규칙은 같다. 이전에 실패한 동일 여섯 페어의 단순 예측판도 대조했다. 특정 손실 페어를 사후 제외하지 않았다.</p>'''
    csv_rows = []
    for phase in ['validation', 'comparison']:
        for scenario in ['base', 'stress']:
            title = '8월 1일~9월 1일' if phase == 'validation' else '9월 1일~10월 15일'
            page += f'<h2>{title} · {"기본 비용" if scenario == "base" else "불리한 체결 조건"}</h2><table><tr><th>전략</th><th>순손익</th><th>수익률</th><th>기간 σ</th><th>Q</th><th>최대낙폭</th><th>완료 페어</th></tr>'
            for arm, label in labels.items():
                row = next(r for r in chosen if r['venue'] == 'reference' and r['phase'] == phase and r['scenario'] == scenario and r['arm'] == arm)
                m = row['metrics']
                page += f"<tr><td>{label}</td><td>{m['net_pnl']:+,.2f}</td><td>{100*m['roi']:+.6f}%</td><td>{100*m['volatility_horizon']:.6f}%</td><td>{m['financial_score']:.8f}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{m['completed_pairs']}</td></tr>"
                csv_rows.append(dict(phase=phase, scenario=scenario, arm=arm, **{k:m[k] for k in ['net_pnl','roi','volatility_horizon','financial_score','max_drawdown','completed_pairs']}))
            page += '</table>'
    page += '''<h2>원인과 실제 호가에서 확인한 범위</h2>
<p>세 페어 후보의 이후 기간 기본 조건에서 BTC/HYPE의 페어 손익은 +5,260.80이지만 ETH/HYPE가 −5,362.48로 상쇄했다. 두 종목 방향 점수가 비용 문턱을 넘는 것만으로 새로운 페어의 전체 보유·청산 손익을 충분히 구분하지 못했다. 이는 관측한 결과이며 원인 전체를 입증하는 인과 추정은 아니다.</p>
<p>8월 불리한 조건에서 세 후보 모두 동일 시점에 추정 일별 변동성 0.500365%로 내부 한도 0.5%를 잠깐 초과했다. 그 시점의 강제 청산 목표와 다음 시간봉의 실제 모의 청산 완료를 별도로 확인했고 다음 기록은 전량 현금이다. 주문 당시부터 위반 주문을 허용한 버그라고 단정하지 않는다.</p>
<p>par 캔들·펀딩 평가에서는 확장한 진입 필터 후보의 완료 페어가 0건이었다. 필터 없는 진단판의 완료 1건은 서로 겹치는 평가 창에서 반복 관측되므로 독립 거래 두 건으로 세지 않는다. 이후 확장 창 순손익은 세 페어 −186.35/−111.19, 여섯 페어 −186.41/−111.30으로 기존 개선판 −186.31/−111.16보다 낮았다(기본/불리한 조건).</p>
<p>사용 허용된 과거 호가 테이프 다섯 개에서도 두 페어와 세 페어 필터판은 신규 진입 0건이다. 여섯 페어 필터판은 두 관측 구간을 기본·불리한 조건으로 각각 평가한 진입 시험 4개를 만들었다. 같은 자료의 관련 모델끼리 시험이 겹치며, 24시간 뒤 실제 청산 호가를 관측하지 못했다. 따라서 이 시험에서 전체 수익률이나 Q를 계산해 제시하지 않는다.</p>
<h2>검증 및 다음 판단</h2><p>새 계좌 재생 36개, 기존 계좌와 정확히 같은 재생 12개, 검사 12개와 일별 평균·방향 모델 적합 3,084개를 완료했다. 원래 두 페어 상태 및 기존 호가 시험 4개의 결과도 정확히 같았다. 보호 소스와 예약 평가 의존 파일의 해시를 확인했다. 실제 주문이나 운용 교체는 하지 않았다.</p>
<p>새로운 페어 수를 늘리는 이 후보군은 채택하지 않는다. 다음 연구는 기존 전문가 선택 실험을 먼저 확인한 뒤 진입의 고정 24시간 가격 변화가 아니라 동일 실행 규칙에서 실제로 종료된 거래의 비용 차감 손익을 엄격한 시점 기준으로 학습할 수 있는지 검토한다. 현재 한계나 내부 승격 기준은 공식 제출 금지 규정이 아니다.</p>
<p><a href="../decision_gate_pairs_20261017/report.html">보존한 개선판의 상세 성과</a> · <a href="../decision_gate_robustness_20261017/report.html">위험 규모를 맞춘 비교</a></p></html>'''
    (dest / 'report.html').write_text(page)
    with (dest / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    state = dict(conclusion)
    state.pop('reference_pair_attribution')
    state['report'] = str((dest / 'report.html').relative_to(ROOT))
    save(HERE / 'work_state.json', state)
    cross = ROOT / 'experiments/pairs_development_status_20261017/comparisons/decision_sector_review.json'
    assert not cross.exists()
    save(cross, state)
    files = [Path(__file__), OUT/'results.json', OUT/'selection.json', OUT/'conclusion.json',
             OUT/'diagnostics.json', OUT/'books/results.json', cross, dest/'report.html', dest/'results.csv']
    save(OUT/'conclusion_frozen.json', dict(at_unix=time.time(), sources={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DONE fixed-universe audit; retain established improvement; no live changes', flush=True)


if __name__ == '__main__':
    main()
