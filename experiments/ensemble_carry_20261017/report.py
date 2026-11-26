"""Human-facing record of implementation and prospectively frozen comparison."""
from datetime import datetime
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from experiments.cost_horizon_20261016.research import sha, save

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]


def kst(epoch): return datetime.fromtimestamp(epoch, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')


def main():
    p = json.loads((HERE / 'artifacts/plan.json').read_text())
    source_changes = [path for path, digest in p['sources'].items() if sha(ROOT / path) != digest]
    if source_changes: raise ValueError(source_changes)
    verification = (HERE / 'verification.log').read_text()
    if 'Ran 16 tests' not in verification or not verification.rstrip().endswith('OK'):
        raise ValueError('Expected test evidence is missing')
    links = [(HERE / 'engine.py', '공유 계좌·범위별 주문 실행 코드'),
        (HERE / 'forward.py', '공통 비교와 성과 판정 코드'),
        (HERE / 'artifacts/plan.json', '평가 전에 고정한 규칙과 430개 의존 파일 해시'),
        (HERE / 'verification.log', '통합·검증 테스트 16개 결과'),
        (ROOT / 'reports/paired_funding_20261017/report.html', '펀딩 예측·현재 호가와 별도 72시간 시험'),
        (ROOT / 'reports/incumbent_pairs_20261017/report.html', '기존 페어 예측 후보의 실패와 실행 단계 개선 결과')]
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SFINX 기존 전략＋펀딩 페어 앙상블</title><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:0 22px;color:#183143;background:#fafcfe}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #cbd5df;text-align:left;padding:12px}}th{{background:#e8f0f5}}.notice{{padding:18px;background:#fff1d6;border-left:5px solid #b07008}}h2{{margin-top:32px}}a{{color:#00649f}}</style>
<h1>기존 전략을 유지한 펀딩 페어 앙상블</h1>
<p>등록: {kst(p['created_at'])} · 연구 실주문 0건 · 실거래 자동 교체 없음</p>
<div class="notice"><b>구현 테스트는 통과했습니다. 기존 두 전략보다 실현 PnL·Q가 높다는 결과는 아직 없습니다.</b> 앞으로 동일한 기간의 실제 par 공개 자료로 비교합니다. 예약 등록이나 테스트 통과를 수익성 증명으로 표시하지 않습니다.</div>
<h2>수익을 개선하려는 방법</h2><p>두 번째 추세＋되돌림 전략의 기존 수익원을 유지하고, 당시의 펀딩 차이와 오라클 대비 가격 괴리로 비용을 넘길 가능성이 있을 때 EWY 숏·SP500 롱 페어를 추가합니다. 168시간 과거 수익률의 헤지 비율을 사용하며, 진입 때 정한 수량은 예정된 72시간 동안 유지합니다. 계좌 위험, 자료 지연 또는 기존 전략의 해당 종목 수요가 발생하면 조기 청산합니다.</p>
<p>10월 17일 04:24 KST의 호가에서는 양쪽 진입·청산 비용을 차감한 예상 순손익이 기본 +38.20달러, 불리한 비용 조건 +9.90달러였습니다. 미래 펀딩 수입을 절반으로 줄였지만, 미래 가격·펀딩·청산 유동성에 대한 가정이 남아 있습니다. 이 수치는 실현손익도, 앞으로 보장되는 수익도 아닙니다. 앞선 역사적 펀딩 예측에서 선택한 릿지는 이후 72시간 비교 구간에서 0 예측보다 오차가 컸으므로 우수 모델로 채택하지 않았습니다.</p>
<h2>실행 구조를 어떻게 고쳤는가</h2><table><tr><th>문제</th><th>새 연구용 실행 구조</th></tr>
<tr><td>페어 청산이 계좌 전체 주문 작업을 지울 수 있음</td><td>페어는 EWY·SP500 작업만 취소합니다. 기존 전략의 다른 주문과 도착 중인 주문은 보존합니다.</td></tr>
<tr><td>각 전략이 같은 계좌 위험 여유를 따로 사용</td><td>현금·보유·수수료·펀딩 원장은 하나입니다. 두 전략의 목표와 도착 중인 주문을 합쳐 위험을 예약합니다.</td></tr>
<tr><td>두 종목이 모두 체결된 최종 비중만 안전</td><td>한쪽만 체결된 경우를 포함한 가능한 부분 체결 조합도 총노출·순노출·종목·공분산 위험 한도를 검사합니다.</td></tr>
<tr><td>기존 전략이 페어 종목을 필요로 할 때 충돌</td><td>페어 두 종목을 먼저 청산해 실제 잔여 수량을 확인한 뒤 기존 전략에 넘깁니다. 기존 전략의 위험 예산이 부족해져도 페어 청산이 우선입니다.</td></tr>
<tr><td>체결되지 않은 진입이 계속 대기</td><td>주문이 실제로 제출되지 못했어도 진입 제한 시간을 적용합니다. 실제 양쪽 진입 후 청산한 경우만 완료된 페어로 집계합니다.</td></tr></table>
<h2>검증한 내용</h2><p>16개 테스트가 통과했습니다. 한 계좌에서 기존 포지션과 페어를 함께 보유하고, 펀딩·수수료를 정산하고, 양쪽 청산 후 현금을 독립 계산해 대조했습니다. 페어 한쪽의 유동성이 사라진 경우, 도착 중인 주문, 가격 변경 후 위험 재검사, 주문 소유권 인계, 평가 전 결과 접근 차단도 검사했습니다. 페어 기회가 없는 합성 자료에서는 기존 두 번째 전략과 체결 시각·가격·수량·수수료·계좌 가치가 같았습니다. 합성 자료 테스트는 수익성 평가에 포함하지 않습니다.</p>
<h2>비교 설계</h2><p><b>{kst(p['paper_start_unix'])} 시작 → {kst(p['capture_end_unix'])} 종료.</b> 이미 수집 중인 7일 자료를 종료 후 사용합니다. 이 기간의 중간 결과를 읽거나 후보를 수정하지 않았습니다. 네 계좌 각각 100만 USDPR, 같은 호가·펀딩·위험 한도를 사용하고 기본·불리한 비용 조건을 각각 계산합니다.</p>
<table><tr><th>비교군</th><th>차이</th></tr><tr><td>첫 번째 전략</td><td>추세＋7일 되돌림의 기존 하방위험 보정판</td></tr><tr><td>두 번째 전략</td><td>현재 기준이 되는 위험 감축 보존판</td></tr><tr><td>두 번째 전략 규모 확대</td><td>원래 신호를 {p['scale_control']:.9f}배. 등록 전 목표 공분산 진단으로 고정했습니다.</td></tr><tr><td>두 번째 전략＋펀딩 페어</td><td>기존 신호를 유지하고 사용하지 않는 두 종목에 조건부로 페어를 추가합니다.</td></tr></table>
<p>규모 확대 비교군은 최초 진단 시점의 예상 위험을 맞춘 대조군입니다. 미래 기간의 실현 변동성이 정확히 같아지는 것은 아니므로 실제 변동성·최대낙폭·평균 노출·거래회전·펀딩 기여를 함께 보고합니다. 모든 계좌에 같은 새 실행 기준을 적용하므로 과거 실제 계좌 성적을 새로운 모의 계좌 결과와 섞지 않습니다.</p>
<p>개선 판정은 기본·불리한 비용 조건 모두에서 순손익이 양수이고, 나머지 세 비교군보다 순손익과 Q가 높으며, 양쪽 진입·청산을 완료한 페어 2회 이상, 일별 수익률 5개 이상, 자료·원장 오류와 미해결 주문·잔여 보유가 없는 경우로 고정했습니다. Q는 12:00 UTC 계좌 샘플로 계산한 모의 비교값이며 공식 순위나 심사 점수 J를 대신하지 않습니다.</p>
<p>자동 계산 예약: 10월 24일 21:01 KST, <code>icaif-ensemble-carry-evaluate.timer</code>. 같은 7일 수집 자료를 사용하는 기존 시험과 독립적인 반복 실험으로 세지 않습니다. 10월 20일 종료되는 짧은 펀딩 시험 결과도 이 후보의 고정 규칙을 바꾸는 데 사용하지 않습니다. par 시장의 자료가 끊기거나 유동성이 사라져 검증 기준을 충족하지 못하면 실패·불충분으로 기록합니다.</p>
<h2>원자료·코드·등록 기록</h2><ul>'''
    body += ''.join(f'<li><a href="{path}">{html.escape(label)}</a></li>' for path, label in links)
    body += '</ul></html>'
    out = ROOT / 'reports/ensemble_carry_20261017'; out.mkdir(exist_ok=True)
    (out / 'report.html').write_text(body)
    save(HERE / 'work_state.json', dict(updated_at=datetime.now().timestamp(),
        status='implemented_and_registered_future_comparison', tests_passed=16,
        unchanged_frozen_dependencies=len(p['sources']), frozen_source_changes=source_changes,
        future_outcomes_inspected=False, validated_pnl_improvement=False, goal_complete=False,
        candidate='Second strategy plus scoped EWY-short/SP500-long 72h basis/funding carry',
        control_scale=p['scale_control'], common_start=p['paper_start_unix'], release=p['outcome_release_unix'],
        evaluation_timer='icaif-ensemble-carry-evaluate.timer at 2026-10-24 21:01 KST',
        live_strategy_changed=False, research_live_orders_sent=0,
        prior_failure_summary='64 initial pair runs, 48 forecast runs, horizon/execution repairs, maker-flow study and 56 incumbent-relative pair runs have not passed replacement gates. Current funding ridge selected historically also underperformed zero at 72h. Do not rescan those outcomes as independent evidence.',
        pending_evidence='One fixed 72h mechanism probe releases October20 04:54 KST; seven-day comparison releases October24 21:00 KST. Never inspect their raw outcomes early.',
        next_work='Audit candidate accounting and data coverage without looking at sealed outcomes. If exploring any additional hypothesis, use a separate disclosed experiment; do not change these frozen candidates or silently select the best of many trials.',
        report=str((out / 'report.html').relative_to(ROOT))))
    print(out / 'report.html')


if __name__ == '__main__': main()
