"""Present the completed development comparison; never reads forward observations."""
from datetime import datetime
from html import escape
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent


def kst(ms):
    return datetime.fromtimestamp(ms / 1000, ZoneInfo('Asia/Seoul')).strftime('%m-%d %H:%M:%S')


def main():
    r = json.loads((HERE / 'development_results.json').read_text())
    d = json.loads((HERE / 'development_details.json').read_text())
    reservation = json.loads((HERE / 'forward_reservation.json').read_text())
    rows, summary = [], []
    for s in r['summary']:
        selected = [x for x in r['episodes'] if x['scenario'] == s['scenario'] and x['arm'] == s['arm']
                    and x['episode'] in r['matched_valid_episodes'][s['scenario']]]
        exposures, remaining = [], []
        for row in selected:
            series = d[row['episode'] + ':' + row['scenario']][row['arm']]['equity']
            ts = np.array([x['time'] for x in series])
            g = np.array([x['gross'] for x in series])
            exposures.append(float(np.sum(np.diff(ts) * g[:-1]) / sum(np.diff(ts)) * 100))
            remaining.append(series[-1]['gross'] * series[-1]['mark_equity'])
        s.update(average_return_percent=s['average_probe_pnl'] / 1e6 * 100,
                 average_episode_drawdown_percent=float(np.mean([x['max_drawdown_percent'] for x in selected])),
                 worst_episode_drawdown_percent=min(x['max_drawdown_percent'] for x in selected),
                 average_gross_exposure_percent=float(np.mean(exposures)),
                 residual_notional_usdpr=remaining,
                 residual_under_10_usdpr_episodes=sum(0 < x < 10 for x in remaining),
                 residual_at_least_10_usdpr_episodes=sum(x >= 10 for x in remaining))
        summary.append(s)
        label = ('기본' if s['scenario'] == 'base' else '불리한 조건') + ' / ' + ('기존' if s['arm'] == 'baseline' else '수정')
        cells = [label, f"{s['average_target_completion_percent']:.2f}%", f"{s['average_probe_pnl']:+.2f}",
                 f"{s['average_return_percent']:+.6f}%", f"{s['average_fees']:.2f}",
                 f"{s['average_gross_exposure_percent']:.3f}%", f"{s['average_episode_drawdown_percent']:.5f}%",
                 f"{s['fully_closed_episodes']}/{s['valid']}", '산출 불가']
        rows.append('<tr>' + ''.join('<td>' + escape(x) + '</td>' for x in cells) + '</tr>')
    episode_rows = []
    for row in r['episodes']:
        cells = [kst(row['start_ms']), row['scenario'], row['arm'], f"{row['mark_net_pnl']:+.2f}",
                 f"{row['oracle_net_pnl']:+.2f}", f"{row['fees']:.2f}", f"{row['funding']:+.4f}",
                 json.dumps(row['residual_positions']), ' / '.join(row['fatal']) or '통과']
        episode_rows.append('<tr>' + ''.join('<td>' + escape(x) + '</td>' for x in cells) + '</tr>')
    plan = reservation['plan']
    decision = {
        'decision': 'HOLD_EXISTING', 'scope': 'bounded retry execution candidate, not a new alpha/model',
        'comparison_kind': 'historical execution probes; not independent full-strategy backtests',
        'summary': summary,
        'reasons': ['Candidate average net PnL negative in both cost scenarios.',
                    'Stress candidate PnL worse than baseline.',
                    'Candidate mean episode drawdown worsened in both scenarios.',
                    'Candidate fully flattened 0/6 base and 1/6 stress probes.',
                    'Greater target completion also increased gross exposure; no risk-matched alpha claim.',
                    'No completed independent evaluation or measurable daily volatility yet.'],
        'tests_passed': 18, 'protected_files_unchanged': 324, 'orders_sent_by_experiment': 0,
        'production_strategy_changed': False, 'prospective_trial_id': reservation['trial_id'],
        'prospective_start': plan['test_start'], 'prospective_end': plan['test_end'],
        'prospective_status': 'raw collection started; no partial performance review',
        'prospective_limitation': 'At most two complete UTC daily returns; three-day promotion gate cannot pass.'}
    (HERE / 'decision.json').write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>실행 개선판 교체 판단</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1440px;margin:40px auto;padding:0 24px;color:#172c35;background:#f5f8fa}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{padding:10px 12px;border:1px solid #dce4e8;text-align:right}}th:first-child,td:first-child{{text-align:left}}section{{background:white;padding:20px;margin:20px 0;border-radius:8px}}.decision{{border-left:6px solid #d49727}}.scroll{{overflow:auto}}a{{color:#086983}}</style>
<h1>실행 개선판 검증: 현재는 기존판 유지</h1>
<section class="decision"><b>판단: HOLD_EXISTING — 실제 운용판 교체하지 않음.</b>
<p>추세＋7일 되돌림 하방위험 보정판의 신호·자본·위험 한도는 유지하고, 목표에 못 미친 IOC를 재시도하는 실행 정책만 비교했다.
수정판은 목표 수량을 더 많이 채웠지만 비용·청산·위험 조건을 함께 통과하지 못했다. 이 결론은 기존판의 장기 수익성이 검증됐다는 뜻은 아니다.</p></section>
<h2>완료한 비교</h2><p>10월 15일의 저장된 실제 par 호가·mark·oracle 자료에서 비교 가능한 6개 구간.
각 구간은 가상 자본 100만 USDPR, 포지션 0에서 시작한다. 동일한 보존 전략의 당시 목표 비중을 적용하고,
20분 후 청산을 요청하여 5분간 관찰했다. 이는 <b>청산 가능성을 확인하는 25분 실행 실험</b>이며 실제 전략의 보유 규칙을 그대로 재현한 전체 백테스트가 아니다.
BRENT 호가가 없는 4개 시작점은 양쪽 모두 제외했다. 기존/수정 모두 유효한 같은 6개 구간만 비교한다.</p>
<p>기존: 종목·진입/축소 단계별 IOC 1회. 수정: 최대 6회, 30초 간격, 최초 호가 대비 불리한 가격 추격 10bp 제한.
기본 조건: 도착 지연 1초·표시 호가 수량 100%·편도 수수료 9bp.
불리한 조건: 3초·50%·12bp. 같은 호가 수량을 반복 사용하지 않는다. 실제 지급 시각의 펀딩도 반영했다.</p>
<div class="scroll"><table><tr><th>조건 / 실행판</th><th>3분 내 목표 충족률 평균</th><th>구간 순손익 평균 (USDPR)</th><th>구간 수익률 평균</th><th>수수료 평균</th><th>평균 총투자 비중</th><th>구간 최대낙폭 평균</th><th>완전 청산</th><th>일별 변동성</th></tr>{''.join(rows)}</table></div>
<p><b>순손익에는 끝까지 청산되지 않은 포지션의 mark 평가손익이 포함된다.</b> 각 행은 독립적으로 초기화한 구간의 평균이며 실제 지갑 손익이나 하루 수익률이 아니다.
일별 변동성은 짧고 끊어진 자료에서 계산하지 않았다. 구간 최대낙폭을 일별 변동성으로 대체하지 않았다.</p>
<section><h2>교체 기준에 대한 판정</h2><ul>
<li>같은 자본·신호·위험 한도: 적용. 그러나 실제 총투자 비중은 달라졌다. 기본 조건에서 약 1.47% → 2.19%로 증가했다. 이를 예측력 개선으로 해석할 수 없다.</li>
<li>비용 차감 후 수익 개선: 기본 조건에서는 평균 손실이 줄었지만 양쪽 모두 음수. 불리한 조건에서는 수정판 손실이 더 컸다. 통과하지 못함.</li>
<li>위험 악화 없음: 수정판의 구간 최대낙폭 평균은 두 조건 모두 악화. 일별 변동성은 관측 부족. 통과하지 못함.</li>
<li>청산 가능성: 수정판의 완전 청산은 기본 0/6, 불리한 조건 1/6. 기본 조건 6개 중 4개는 10 USDPR 미만의 작은 잔여분이고, 나머지 2개는 약 4,274 및 10,737 USDPR가 남았다. 불리한 조건에서는 3개 구간에서 약 3,117~8,292 USDPR가 남았다. 소액 잔여와 유의미한 청산 미완료를 구분해야 한다. 남은 포지션을 허구의 가격으로 매도 처리하지 않았다. 통과하지 못함.</li>
<li>새 기간에서 재현: 아직 결과 없음. 과거 6개 구간은 개발 자료로만 취급.</li></ul></section>
<h2>새 자료에 대한 고정 검증</h2>
<p>한국시간 <b>{kst(plan['test_start'] * 1000)} ~ {kst(plan['test_end'] * 1000)}</b>.
공개 호가·체결·가격·실제 지갑 주문 가능 수량과, 실시간에 공개된 고정 전략의 신호를 별도 작업으로 저장한다.
기존/수정 각각 가상 100만 USDPR에서 시작하고 같은 정상 6시간 판단 시각과 위험 한도를 적용한다.
기간 종료 후 한 번 평가하며 중간 성과를 보고 규칙을 바꾸지 않는다. 평가 후에도 실제 봇을 자동 교체하지 않는다.</p>
<p>최소 3개 완전한 UTC 일별 수익률, 8개 6시간 비교 구간, 4개 실제로 끝난 포지션 보유 과정,
두 비용 조건에서 양의 순손익 및 기존판 초과, 변동성·최대낙폭 악화 없음, 비교 불확실성 구간이 0 초과를 사전에 요구했다.
<b>남은 연습 기간은 완전한 UTC 일별 수익률을 최대 2개만 제공한다.</b> 따라서 이번 관측에서 이익이 나더라도 최소 표본 기준을 충족한 교체 확정 결과로 발표할 수 없다.</p>
<h2>검증 범위와 한계</h2><ul><li>주문 도착 시 보이는 호가로 체결을 가정하는 공개자료 시뮬레이션이다. 우리 주문으로 바뀔 이후 호가·경쟁 주문·실제 서버 거절은 정확히 복원할 수 없다.</li>
<li>기존 첫 실제 매수와 추가 매수 시간대의 L2·주문 가능 수량 기록이 없다. 그날 첫 주문을 크게 넣었으면 얼마를 벌었을지 단정하지 않았다.</li>
<li>미래 비교의 실제 지갑 주문 가능 수량은 양쪽에 적용하는 보수적인 상한 대용치다. 가상 계좌별 거래소 증거금 계산의 완전한 재현은 아니다.</li>
<li>18개 회계·체결·시간 경계·전체 평가 경로 테스트 통과. 기존 보존 파일 324개 해시 일치. 실험이 보낸 실제 주문은 0개.</li>
<li>공식 전체 대회 점수는 참가자 전체 자료와 심사위원 점수가 필요하므로 산출하지 않았다. <a href="https://perpetual-alpha26.github.io/about">공식 평가 규정</a>과 위 자체 교체 기준은 구분한다.</li></ul>
<details><summary>각 구간의 실제 계산값</summary><div class="scroll"><table><tr><th>시작 (KST)</th><th>조건</th><th>실행판</th><th>mark 순손익</th><th>oracle 순손익</th><th>수수료</th><th>펀딩</th><th>남은 포지션</th><th>자료 검증</th></tr>{''.join(episode_rows)}</table></div></details>
<p>근거 파일: <a href="development_results.json">구간 집계</a> · <a href="development_details.json">모의 체결 원장</a> · <a href="protocol.json">고정 기준</a> · <a href="forward_protocol.json">새 기간 평가 규칙</a> · <a href="forward_reservation.json">평가 예약 및 파일 해시</a> · <a href="test_results.txt">테스트</a></p></html>'''
    (HERE / 'report.html').write_text(html)
    print(json.dumps({'decision': decision['decision'], 'report': str(HERE / 'report.html')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
