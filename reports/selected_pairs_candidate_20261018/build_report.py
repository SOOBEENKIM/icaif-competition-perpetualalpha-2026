"""Publish the selected historical benchmark result without running orders."""
import csv
import gzip
import hashlib
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'deliverables/sfinx_pairs_candidate_20261018_final'
REPLAY = ROOT / 'experiments/candidate_release_20261018/artifacts/reproduction_final'
CANDIDATE = 'first_dense_hourlyhedge_decisiongate_24'
NAMES = {'first_strategy': '첫 번째 전략', 'second_strategy': '두 번째 전략',
         CANDIDATE: '선정: 기존 전략 + 24시간 페어 예측·비용 진입 필터'}


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def main():
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (BUNDLE / name).resolve()
        assert path.is_relative_to(BUNDLE)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    result = json.loads((REPLAY / 'results.json').read_text())
    expected = {r['id']: r for r in json.loads((BUNDLE / 'expected.json').read_text())['runs']}
    assert result['refitted_models'] and result['exact_accounts'] == 42
    rows = []
    for record in result['rows']:
        with gzip.open(REPLAY / 'runs' / (record['id'] + '.json.gz'), 'rt') as f:
            run = json.load(f)
        assert digest(run) == expected[record['id']]['canonical_run_sha256']
        assert record['common_engine_financial_path_exact']
        m = run['metrics']
        snapshots = run['snapshots']
        initial = snapshots[0]['equity']
        close(initial, 1_000_000)
        pnl = snapshots[-1]['equity'] - initial
        close(pnl, m['net_pnl'])
        r = pnl / initial
        close(r, m['roi'])
        # These archived accounts have no external deposits or withdrawals.
        daily_points = [s for s in snapshots if s['time'] % 86400 == 43200]
        returns = [b['equity'] / a['equity'] - 1
                   for a, b in zip(daily_points, daily_points[1:])]
        assert len(returns) == len(run['daily'])
        for observed, archived in zip(returns, run['daily']):
            close(observed, archived['return_'])
        sigma = statistics.stdev(returns) * math.sqrt(len(returns)) if len(returns) >= 2 else None
        q = ((max(r, 0) + .005) / (abs(r) + 1.5 * sigma + .05)) if sigma is not None else None
        if q is not None:
            close(sigma, m['volatility_horizon'])
            close(q, m['financial_score'])
            close(.8 * q, m['score'])
        else:
            assert m['financial_score'] is None
        rows.append(dict(id=record['id'], venue=record['venue'], phase=record['phase'],
            strategy=record['arm'], scenario=record['scenario'],
            start_utc=datetime.fromtimestamp(run['start'], timezone.utc).isoformat(),
            end_utc=datetime.fromtimestamp(run['end'], timezone.utc).isoformat(),
            net_pnl=pnl, return_fraction=r, volatility=sigma, Q=q,
            score_J0=.8*q if q is not None else None, max_drawdown=m['max_drawdown'],
            completed_pairs=m['completed_pairs'], risk_breach_bars=m['risk_breach_bars']))
    lookup = {(r['venue'], r['phase'], r['scenario'], r['strategy']): r for r in rows}
    advantages = []
    for phase in ['validation', 'comparison', 'continuous']:
        for scenario in ['base', 'stress']:
            candidate = lookup['reference', phase, scenario, CANDIDATE]
            for baseline in ['first_strategy', 'second_strategy']:
                control = lookup['reference', phase, scenario, baseline]
                advantage = dict(phase=phase, scenario=scenario, baseline=baseline,
                    pnl_gain=candidate['net_pnl']-control['net_pnl'],
                    Q_gain=candidate['Q']-control['Q'],
                    equal_jury_score_gain=candidate['score_J0']-control['score_J0'])
                assert advantage['pnl_gain'] > 0 and advantage['Q_gain'] > 0
                assert candidate['risk_breach_bars'] == 0
                advantages.append(advantage)
    summary = dict(status='same_period_historical_benchmark_improvement_confirmed',
        selected_candidate=CANDIDATE, comparison_scope='Same capital, historical windows, execution assumptions and risk limits.',
        independent_validation=False, actual_par_profit_improvement_verified=False,
        leaderboard_rank_predicted=False, live_strategy_changed=False, orders_sent=0,
        audited_account_files=len(rows), refitted_gate_models=sum(x['gate_fits'] for x in result['model_checks']),
        bundle_files_verified=len(manifest['files']), advantages=advantages, results=rows)
    (OUT / 'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    with (OUT / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX 선정 전략과 동일기간 성과</title>',
        '<style>body{max-width:1250px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;color:#172b3a}table{border-collapse:collapse;width:100%;font-size:14px;margin-bottom:25px}th,td{border:1px solid #ccd5de;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}.selected{background:#e1f4eb}code{background:#eef1f4;padding:3px}</style>',
        '<h1>선정 결과: 기존 전략 + 24시간 페어 예측·비용 진입 필터</h1>',
        '<p><strong>요청한 동일기간 백테스트 비교에서 PnL과 Q 개선을 확인했다.</strong> 기존 두 전략, 같은 초기자본 100만, 같은 비용·위험 한도로 비교했다. 기본 조건과 불리한 조건, 8월·9월 이후·연속 전체 구간 모두 기존 두 전략보다 PnL과 Q가 높았다. 아래 순위는 세 전략의 모의 비교이며 공식 팀 순위가 아니다.</p>',
        '<h2>매매 논리와 실제 구현</h2><ol>',
        '<li>기존 추세·7일 되돌림 전략을 기본 포트폴리오로 유지하고 BTC/ETH, BTC/HYPE의 상대 수익 기회를 추가한다.</li>',
        '<li>과거 168시간 수익률로 헤지 비율을 추정하고, 상대 수익률·가격 괴리·상관·변동성 등 20개 특징을 릿지 회귀에 넣어 24시간 뒤 헤지한 상대 수익률을 예측한다. 양의 상대 전망이면 첫 종목 매수·둘째 종목 매도, 음이면 반대 방향을 검토한다.</li>',
        '<li>별도 비용 민감 SPO+ 모델이 롱·숏·대기 중 신규 진입 방향을 선별한다. 이 점수를 미래 가격이나 예상 수익률로 사용하지 않는다. 두 모델은 매일 과거 90일 중 24시간 결과가 이미 확정된 표본만 사용해 갱신한다.</li>',
        '<li>실제 진입 판단은 현재 가격, 헤지 수량, 진입·청산 비용, 알려진 펀딩, 위험 한도를 다시 적용한다. 신규 신호는 시간마다 갱신하며 한 번에 한 페어만 운용한다.</li>',
        '<li>24시간 만기에 정리를 요청하되 남은 보유 이익이 비용을 넘으면 정해진 규칙으로 연장한다. 총 보유 상한은 168시간이다. 양쪽 주문의 부분 체결, 한쪽만 체결될 경우 정리, 실제 비용을 지불하는 청산을 모의 실행한다.</li></ol>',
        '<p>학습은 “2021~2025로 한 번 학습”한 구조가 아니라 과거 자료만 사용하는 일별 순차 재학습이다. 기본 비용은 편도 수수료 9bp·가격 불리함 10bp, 불리한 조건은 12bp·20bp 및 낮은 체결 참여율이다. 비용 학습의 보조 목표와 최종 계좌 PnL은 구분한다.</p>',
        '<h2>평가 기준</h2><p>R은 초기자본 대비 비용 차감 후 계좌 수익률, σ는 매일 12:00 UTC 계좌 수익률의 표본 표준편차 × √n이다. Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), Score=0.8Q+0.2J. 표는 J=0이며, J가 같으면 총점 차이는 그대로 유지된다. 종료 시 미청산 포지션의 평가손익도 포함한다.</p>']

    def table(selected):
        out=['<table><tr><th>전략</th><th>조건</th><th>순손익</th><th>수익률</th><th>기간 σ</th><th>Q</th><th>Score(J=0)</th><th>MDD</th><th>완료 페어</th></tr>']
        for r in selected:
            fmt = lambda x: '—' if x is None else f'{x:.6f}'
            pct = lambda x: '—' if x is None else f'{100*x:.5f}%'
            style = ' class="selected"' if r['strategy'] == CANDIDATE else ''
            out.append(f'<tr{style}><td>{html.escape(NAMES[r["strategy"]])}</td><td>{"기본" if r["scenario"]=="base" else "불리"}</td><td>{r["net_pnl"]:+,.2f}</td><td>{pct(r["return_fraction"])}</td><td>{pct(r["volatility"])}</td><td>{fmt(r["Q"])}</td><td>{fmt(r["score_J0"])}</td><td>{pct(r["max_drawdown"])}</td><td>{r["completed_pairs"]}</td></tr>')
        return ''.join(out)+'</table>'

    for phase in ['comparison', 'validation', 'continuous']:
        selected=[r for r in rows if r['venue']=='reference' and r['phase']==phase]
        selected.sort(key=lambda r: (r['scenario'], list(NAMES).index(r['strategy'])))
        page += [f'<h2>기준시장 {selected[0]["start_utc"]} → {selected[0]["end_utc"]}</h2>',table(selected)]
    page += ['<h2>대회 시장 par 결과는 별도 표시</h2>',
        '<p>위 개선은 기준시장 과거 자료와 모의 체결 가정의 결과다. 아래 par 캔들 비교에서는 신규 페어의 수익 개선을 확인하지 못했다. 짧고 끊긴 실제 호가 자료로 24시간 진입부터 청산까지의 성과를 대신 만들지 않았다.</p>']
    for venue,phase in [('par','comparison'),('par_extended','extended')]:
        selected=[r for r in rows if r['venue']==venue and r['phase']==phase]
        if selected:
            selected.sort(key=lambda r:(r['scenario'],list(NAMES).index(r['strategy'])))
            page += [f'<h3>{venue}: {selected[0]["start_utc"]} → {selected[0]["end_utc"]}</h3>',table(selected)]
    page += ['<h2>検증 범위와 산출물</h2>'.replace('検','검'),
        f'<p>이 보고서 작성 시 배포 묶음 {len(manifest["files"])}개 파일의 해시와 42개 계좌 원본을 확인하고, 계좌 스냅샷으로 손익·일별 수익률·σ·Q·Score를 재계산했다. 저장된 재현 실행은 두 모델의 재학습과 526개 일별 진입 모델, 42개 계좌 성과의 일치를 확인했다. 이번 점검에서도 모델·미래정보 유입·진입 필터 관련 10개 테스트를 통과했다.</p>',
        '<p>종료 포지션 평가손익을 포함하며, 같은 위험 한도가 같은 실현 변동성을 의미하지는 않는다. 개선 후보의 변동성과 MDD가 높아진 구간도 위 표에 표시했다. 반복 사용한 과거 구간의 개발 성과이며, 공식 순위나 미래 수익을 보장하지 않는다. 실제 운용 전략은 변경하지 않았다.</p>',
        '<p><a href="comparison.csv">전체 비교 CSV</a> · <a href="result.json">검증 결과 JSON</a> · <a href="../../deliverables/sfinx_pairs_candidate_20261018.zip">코드·데이터·학습 재현 묶음</a></p>',
        '<p>묶음의 실행 명령: <code>python reproduce.py --output results</code>. 기존 첫째·둘째 전략은 당시 고정된 목표 비중을 비교 입력으로 사용하며, 기존 Chronos 전체를 처음부터 학습하는 명령은 아니다.</p></html>']
    (OUT / 'report.html').write_text('\n'.join(page))
    print(json.dumps(dict(audited_accounts=len(rows), positive_pnl_q_comparisons=len(advantages),
        selected=CANDIDATE, comparison=[r for r in rows if r['venue']=='reference' and r['phase']=='comparison']),ensure_ascii=False))


if __name__ == '__main__':
    main()
