from datetime import datetime
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from experiments.cost_horizon_20261016.research import save, sha

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]


def kst(epoch): return datetime.fromtimestamp(epoch, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')


def main():
    out = HERE / 'artifacts'; p = json.loads((out / 'plan.json').read_text())
    d = json.loads((out / 'hedge_diagnostic.json').read_text()); r = json.loads((out / 'results.json').read_text())
    changed = [path for path, digest in p['sources'].items() if sha(ROOT / path) != digest]
    if changed: raise ValueError(changed)
    log = (HERE / 'verification.log').read_text()
    if 'Ran 6 tests' not in log or not log.rstrip().endswith('OK'): raise ValueError('Missing test evidence')
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SFINX 음의 상관 FX 페어 점검</title><style>body{{font:16px/1.7 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;background:#fbfcfe;color:#203347}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #cbd5df;text-align:right}}th{{background:#edf2f6}}td:first-child,th:first-child{{text-align:left}}.box{{padding:18px;background:#fff1d6;border-left:5px solid #b27204}}h2{{margin-top:32px}}a{{color:#0066a3}}</style>
<h1>음의 상관 FX 페어: 헤지 방향 확장과 비용 검증</h1>
<p>호가 관측 {kst(r['at_ms']/1000)} · 기준시장 입력 {kst(p['source_time'])} · 실주문 0건</p>
<div class="box"><b>이번 조합은 운용 후보로 채택하지 않습니다.</b> 24·72시간, 양쪽 롱·양쪽 숏, 기본·불리한 비용의 8개 경우 모두 비용 차감 예상 손익이 음수였습니다. 위험 추정이 개선되는 것과 수익 기회가 생기는 것은 별도로 판단했습니다. 기존 펀딩 페어·앙상블의 고정된 시험은 유지합니다.</div>
<h2>어떤 범위를 추가했는가</h2><p>기존 탐색은 양의 헤지 비율을 가진 반대 방향 페어였습니다. 이번에는 EUR/USD와 USD/JPY처럼 음의 상관관계를 가진 두 계약으로 범위를 넓혔습니다. 대회 티커 USDEUR는 이름과 달리 EUR/USD를 나타내고, 두 계약 모두 USDPR로 결제됩니다. USDJPY 보유 수량을 실제 엔화 현금 잔고와 혼동하지 않았습니다.</p>
<p>최근 168시간 로그수익률에서 β = Cov(rA,rB)/Var(rB)는 <b>{d['beta']:.6f}</b>, 상관계수는 <b>{d['correlation']:.6f}</b>였습니다. rA−βrB를 구성하려면 β가 음수일 때 두 계약의 포지션 부호가 같습니다. 수량은 총 명목금액을 1:|β|로 나누고 각 오라클 가격으로 환산합니다. β는 진입 판단 시점까지의 자료만 사용했습니다.</p>
<table><tr><th>같은 총 명목 비중 기준</th><th>추정 시간당 표준편차</th></tr><tr><td>음의 β에 맞게 같은 방향 보유</td><td>{d['correct_hourly_sigma']*100:.5f}%</td></tr><tr><td>비중 크기는 같고 두 번째 종목 부호만 반대</td><td>{d['forced_opposite_hourly_sigma']*100:.5f}%</td></tr></table>
<p>두 번째 행은 진단용 가상 비교이며 기존 실거래가 이런 포지션을 취했다는 뜻은 아닙니다. 같은 방향 페어의 달러 순명목금액은 0이 아니므로 계좌 순노출 한도에도 포함했습니다. 이 계산은 공분산 헤지이며 공적분·평균회귀·수익성 증명이 아닙니다.</p>
<p>24시간 순환 블록을 500회 재표집했을 때 β의 2.5·50·97.5백분위는 {d['descriptive_beta_quantiles'][0]:.3f} / {d['descriptive_beta_quantiles'][1]:.3f} / {d['descriptive_beta_quantiles'][2]:.3f}였습니다. 7일의 짧은 과거 구간에서 나온 기술적 안정성 진단이며, 독립 표본 500개나 미래 보장 구간으로 해석하지 않습니다.</p>
<h2>실제 par 호가에서의 비용 결과</h2><p>최대 계획 명목금액은 12만 USDPR이나 두 종목의 현재 진입·청산 호가 수량을 함께 적용해 기본 조건은 약 4만, 불리한 조건은 약 2만 USDPR로 제한됐습니다. 양수 예상 펀딩은 절반만 반영했습니다. 오라클 쪽으로 가격 괴리가 절반 줄어드는 가정과 미래 청산 깊이는 검증되지 않았습니다.</p>
<table><tr><th>방향</th><th>보유 시간</th><th>비용</th><th>예상 가격손익</th><th>예상 펀딩</th><th>예상 수수료</th><th>예상 순손익</th><th>예상 순손익 bp</th></tr>'''
    for row in r['rows']:
        q = row['proposal']
        if not q: continue
        body += '<tr>' + ''.join('<td>'+html.escape(str(v))+'</td>' for v in [
            '양쪽 숏' if row['direction'] < 0 else '양쪽 롱', row['horizon_hours'], row['scenario'],
            f"{q['predicted_price_pnl']:+.2f}", f"{q['predicted_funding']:+.2f}", f"{q['predicted_fees']:.2f}",
            f"{q['predicted_net_pnl']:+.2f}", f"{q['predicted_net_bps']:+.3f}"]) + '</tr>'
    body += '''</table><p><b>이 표는 모의 체결이나 실현 수익률이 아닌 현재 스냅샷의 기대값입니다.</b> 불리한 비용 조건은 수량도 절반이므로 손실 달러 금액이 작다고 성능이 더 좋다는 뜻은 아닙니다. 같은 명목금액 기준의 bp를 함께 비교합니다. 가장 나은 양쪽 숏 72시간도 예상 펀딩이 호가·가격 비용과 수수료를 상쇄하지 못했습니다. 기회가 없으므로 새 미래 시험이나 주문을 추가하지 않았습니다.</p>
<h2>관련 최신 논문을 어떻게 적용했는가</h2>
<p><a href="https://arxiv.org/html/2606.04574v1">Lebiedź·Ślepaczuk (2026)</a>은 양의 β를 요구하는 전통적 롱·숏 페어와, 통계 신호 위에 PPO·LSTM 실행 정책을 올리는 구성을 다룹니다. 상위 100개 암호화폐 중 20개 페어를 선택하는 Binance 시간봉 환경이고 기본 체결 비용은 편도 5bp입니다. 따라서 이 논문을 근거로 음의 β의 같은 방향 보유가 수익성을 갖는다고 주장하지 않았습니다. 이번 확장은 계약의 실제 표시 단위와 수익률 공분산에 근거한 별도 가설입니다.</p>
<p><a href="https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1749337/full">2026년 DNN·LSTM 공적분 페어 연구</a>는 거래 비용·유동성·체결 제약을 명시적으로 모델링하지 않았다고 밝힙니다. 예측 성능이나 논문의 성과를 그대로 par 순손익으로 옮길 근거가 되지 않습니다. 이번 점검에서도 헤지 방향 개선과 양쪽 비용 차감 후 수익을 분리했습니다.</p>
<p>계약 표시 단위·결제·수수료·펀딩은 <a href="https://perpetual-alpha26.github.io/docs/">대회 공식 문서</a>와 <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding">Hyperliquid 펀딩 문서</a>를 대조했습니다. 현재 API 응답에는 호가·오라클·마크·공개 예상 펀딩과 수신 시각을 저장했습니다.</p>
<h2>구현 검증과 다음 판단</h2><p>6개 테스트가 통과했습니다. 양의 β에서는 기존 계산을 재현하고, 음의 β의 수량 부호, 두 종목의 펀딩 현금액, 양쪽 진입·청산 깊이, 역상관 합성 자료의 헤지, 움직이지 않는 자료의 거부를 확인했습니다. 기존 실거래 및 사전에 고정한 비교 시험 코드 434개 의존 파일은 변경하지 않았습니다.</p>
<p>이번 결과로 제외한 것은 현재 조건의 FX 펀딩 후보입니다. 기존 전략보다 PnL·Q가 높은 전략을 찾았다는 결론은 아닙니다. 실행 중인 EWY–SP500 72시간 기회 검증과 7일간 네 전략 비교가 다음 실현 성과 근거입니다.</p><ul>'''
    paths = [(out/'plan.json','현재 호가 평가 전에 고정한 규칙'),(out/'hedge_diagnostic.json','헤지·재표집 진단'),
        (out/'results.json','공개 API 원응답과 8개 비용 결과'),(HERE/'quote.py','부호를 보존하는 페어 계산'),
        (HERE/'verification.log','6개 테스트 결과'),(ROOT/'reports/ensemble_carry_20261017/report.html','실행 중인 별도 앙상블 비교 설계')]
    body += ''.join(f'<li><a href="{path}">{label}</a></li>' for path,label in paths) + '</ul></html>'
    destination = ROOT/'reports/signed_pairs_20261017'; destination.mkdir(exist_ok=True)
    (destination/'report.html').write_text(body)
    save(HERE/'work_state.json',dict(updated_at=datetime.now().timestamp(),
        previous_goal_turn='progress: ensemble code implemented, 16 tests passed, future comparison registered and scheduled',
        current_turn='progress: signed-beta FX support implemented and 6 tests passed; actual quote economics rejects all 8 cases',
        beta=d['beta'], correlation=d['correlation'], costs_evaluated=8, eligible_both_costs=r['eligible_in_both_costs'],
        best_base_expected_bps=-29.116245494153016, best_stress_expected_bps=-35.121121091682284,
        actual_pnl=None, actual_Q=None, existing_strategy_improvement_proven=False,
        new_forward_trial_started=False, live_strategy_changed=False, sealed_outcomes_read=False,
        unchanged_source_count=len(p['sources']), goal_complete=False,
        next_evidence='Previously registered EWY–SP500 probe release 2026-10-20 04:54 KST; 7-day four-arm comparison release 2026-10-24 21:00 KST. Do not change frozen rules using intermediate outcomes.',
        next_action='Verify live process health and evidence availability; do not repeat failed historical parameter searches or represent new software tests as alpha validation.',
        report=str((destination/'report.html').relative_to(ROOT))))
    print(destination/'report.html')


if __name__ == '__main__': main()
