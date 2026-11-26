"""Readable report from frozen results; never writes strategy or model files."""
import html
import json
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REL=ROOT/'experiments/relative_incumbent_pairs_20261017'
STAGED=ROOT/'experiments/staged_incumbent_pairs_20261017'
DEST=ROOT/'reports/incumbent_pairs_20261017'


def main():
    first=json.loads((HERE/'artifacts/results.json').read_text())
    second=json.loads((REL/'artifacts/results.json').read_text())
    third=json.loads((STAGED/'artifacts/results.json').read_text())
    diag=json.loads((HERE/'artifacts/diagnostics.json').read_text())
    post=json.loads((HERE/'artifacts/postmortem.json').read_text())
    names={'first':'기존 첫 번째 전략','second':'현재 두 번째 전략','second_selection_risk_matched':'두 번째 전략 · 선정 구간 위험 조정',
        'standalone_opposed_24':'반대 신호 페어 · 24시간','standalone_opposed_72':'반대 신호 페어 · 72시간',
        'standalone_calibrated_24':'반대 신호 + 보정 모델 · 24시간','overlay_opposed_24':'기존 포트폴리오 + 반대 신호 페어 · 24시간',
        'overlay_opposed_72':'기존 포트폴리오 + 반대 신호 페어 · 72시간','overlay_calibrated_24':'기존 포트폴리오 + 반대 신호 + 보정 모델',
        'standalone_relative_24':'상대 수익률 예측 페어 · 24시간','overlay_relative_24':'기존 포트폴리오 + 상대 수익률 예측 페어',
        'standalone_staged_relative_24':'단계별 전환 · 독립 페어','overlay_staged_relative_24':'단계별 전환 · 기존 포트폴리오 + 페어'}
    def table(rows,phase,scenario):
        lines=[]
        for row in rows:
            if row['phase']!=phase or row['scenario']!=scenario:continue
            m,q=row['metrics'],row['official']
            lines.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in [html.escape(names[row['policy']]),
                f"{m['net_pnl']:+,.2f}",f"{m['roi']*100:+.5f}%",f"{q['daily_sample_std']*100:.5f}%",
                f"{q['volatility_horizon']*100:.5f}%",f"{q['financial_score']:.6f}",m.get('balanced_open','—'),
                f"{m['pnl_less_residual_exit_estimate']:+,.2f}",'통과' if all(row['gates'].values()) else '실패'])+'</tr>')
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in
            ['전략','순손익 ($)','수익률','일별 표본 STD','기간 σ','Q','균형 진입','잔여 청산비 추정 차감 손익','구현 검사'])+'</tr></thead><tbody>'+''.join(lines)+'</tbody></table></div>'
    def section(title,r):
        chosen=r['selection'];passed='통과' if chosen['eligible'] else '미통과'
        content=f'<h2>{title}</h2><p>선정 기준 {passed}. 비교 대상으로 고정한 후보: <b>{names[chosen["selected"]]}</b>. 개선 판정: <b>{r["development_improvement"]}</b>.</p>'
        for phase,label in [('selection','후보 선정: 8월 15일 21시 → 9월 15일 21시 KST'),('comparison','이후 비교: 9월 15일 21시 → 10월 15일 21시 KST')]:
            content+=f'<h3>{label}</h3>'
            for scenario,text in [('base','기본 비용·유동성'),('execution_stress','불리한 비용·유동성')]:
                content+=f'<h4>{text}</h4>'+table(r['rows'],phase,scenario)
        return content
    links=[(HERE/'protocol.json','최초 6개 후보 사전 규칙'),(HERE/'artifacts/results.json','최초 24회 실행 결과'),
           (HERE/'artifacts/diagnostics.json','거래 없음 원인 및 손익 대조'),(HERE/'verification.log','최초 9개 테스트'),
           (REL/'protocol.json','후속 2개 후보 사전 규칙'),(REL/'artifacts/results.json','후속 16회 실행 결과'),
           (REL/'verification.log','상대 수익률판 12개 테스트'),(STAGED/'protocol.json','단계별 전환 규칙'),
           (STAGED/'artifacts/results.json','단계별 전환 16회 실행 결과'),(STAGED/'verification.log','단계별 전환 15개 테스트')]
    body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SFINX 기존 신호 기반 페어 검증</title>
    <style>body{font:16px/1.65 system-ui,sans-serif;max-width:1400px;margin:40px auto;padding:0 24px;color:#172b39;background:#fafcfd}h1,h2{line-height:1.3}h2{margin-top:42px}.notice{background:#fff3d8;padding:18px;border-left:5px solid #b76b00}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin-bottom:24px}th,td{padding:10px;border-bottom:1px solid #cbd5df;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eaf1f5}a{color:#05629c}li{margin-bottom:8px}</style><h1>SFINX: 기존 신호를 활용한 페어 전략 검증</h1>'''
    body+=f'<p>생성 {datetime.now(timezone.utc).isoformat(timespec="seconds")} · 초기 자본 1,000,000 · 연구 주문 전송 0건</p>'
    body+='<p><b>결론: 전환 절차는 수정했지만, 채택 조건을 충족한 후보는 없습니다.</b> 총 56회 실행을 마쳤습니다. 실행 수정판의 뒤 30일 기본 순손익은 −511.15로 현재 두 번째 전략 −187.78보다 낮았습니다. Q는 소폭 높지만 PnL 개선 조건을 충족하지 못합니다. 불리한 체결 조건에서도 기존 전략보다 순손익이 낮았습니다.</p>'
    body+='''<div class="notice"><b>이 표는 기준시장 과거 자료를 재사용한 개발 결과입니다.</b> 실제 par 체결 성과나 독립적인 미래 검증 결과가 아닙니다. 기존 첫째·둘째 전략과 실거래 서비스는 수정하지 않았습니다. 상대 수익률 후보도 결과가 좋다는 이유만으로 실거래에 연결하지 않습니다.</div>
    <p>최초 가설은 기존 모델이 반대 방향을 지시한 두 종목만 페어로 묶는 것입니다. 그러나 뒤 30일에는 세 페어 모두 120회 판단 중 반대 신호가 0회였습니다. 보정 모델의 비용 기준을 낮춰도 이 조건에서 거래는 생기지 않습니다.</p>
    <p>후속 가설은 이미 학습된 24시간 상대 수익률 예측으로 롱·숏 방향을 정합니다. 두 종목 모두 상승을 예상하더라도 상대적으로 약한 종목을 헤지할 수 있습니다. 모델 가중치, 학습 구간, 비용 기준, 투자 규모 및 리스크 규칙을 유지했습니다. 이 후속 가설은 최초 결과를 확인한 뒤 설계했으므로 검증 독립성을 주장하지 않습니다.</p>'''
    body+=section('1. 반대 방향 신호를 요구한 최초 후보군',first)
    body+=section('2. 상대 수익률로 방향을 결정하는 후속 후보군',second)
    body+='''<h2>공용 실행기와 페어 전환의 불일치</h2><p>공용 실행기는 같은 주문 의도에서 한 종목을 청산한 뒤 즉시 반대 방향으로 다시 여는 것을 막습니다. 상대 수익률판은 이 절차를 별도로 관리하지 않아, 청산만 끝난 다음 시점에 한쪽이 없는 페어로 판단하고 다시 청산했습니다. 최초 연구의 반대 신호 조건에서는 대부분 이미 방향이 일치하므로 이 문제가 드러나지 않았습니다.</p>
    <p>실제 예: 기존 ETH 롱 4.4715를 목표 −2.126으로 바꾸는 주문에서 실행기는 4.4715의 롱 청산만 체결했습니다. 에이전트는 다음 판단에서 BTC가 아닌 HYPE–ETH 페어의 ETH 부재를 확인하고 페어를 중단했습니다. 이는 가상의 체결을 수익에 넣은 오류가 아니라, 전환 절차를 끝내지 못해 비용을 지불하고 기회를 놓친 구조 문제입니다.</p>
    <p>수정판은 잘못된 방향의 기존 보유가 있으면 두 종목을 청산하고, 실제 잔고가 0인지 확인한 다음 최신 상대 수익률·비용·유동성·위험 조건을 다시 통과할 때 진입합니다. 준비 시간이 3시간을 넘으면 청산 후 대기합니다. 이 과정을 실제 공용 실행기로 검증했습니다. 모델·위험 한도·비용 가정은 유지했습니다.</p>'''
    body+=section('3. 청산 확인 후 새 페어에 진입하는 실행 수정판',third)
    body+='''<h2>실행은 정상화됐지만 예측의 수익성은 부족합니다</h2><p>선정 구간의 기본 조건에서 균형 진입은 2회에서 18회로 늘고 불완전 페어 중단은 50회에서 0회로 줄었습니다. 비교 구간에서는 2회에서 10회로 늘고 중단 23회가 0회가 됐습니다. 그러나 비교 구간 순손익은 오히려 하락했습니다. 제대로 실행됐다는 사실만으로 신호가 좋은 것은 아닙니다.</p>
    <p>비교 구간의 일별 비중첩 24시간 라벨로 확인하니 세 페어 모두 예측 평균제곱오차가 ‘상대 수익률 0’ 예측보다 컸습니다. HYPE–BTC는 평균 +24.35bp를 예측했지만 실제 평균은 −17.90bp였습니다. 아래 값은 예측 진단이며 거래 수익률이 아닙니다. 각 페어 간 표본도 독립적이지 않습니다.</p><table><tr><th>페어</th><th>관측 수</th><th>모델 MSE / 0 예측 MSE</th><th>방향 일치율</th></tr>'''
    for q in post['model_quality']:
        if q['phase']=='comparison':
            body+=f'<tr><td>{"–".join(q["pair"])}</td><td>{q["observations"]}</td><td>{q["mse"]/q["zero_mse"]:.4f}</td><td>{q["direction_accuracy"]*100:.2f}%</td></tr>'
    body+='</table><p><a href="'+str(HERE/'artifacts/postmortem.json')+'">예측 오차·가격손익·체결비용·펀딩 분해 원본</a></p>'
    body+='''<h2>계산·구현 조건</h2><ul>
    <li>기본: 체결 수수료 편도 9bp + 기준시장 체결가격 불리함 10bp. 왕복 38bp는 양쪽 명목금액 합계 기준입니다. 스트레스: 9bp + 20bp, 거래량 참여 한도 1%에서 0.25%로 축소. 실제 par 과거 호가 복원으로 주장하지 않습니다.</li>
    <li>가격·실제 체결 수량·수수료·펀딩을 단일 계좌 원장으로 계산하고 별도 회계 검사를 통과해야 합니다. 두 종목 체결은 독립적이며, 미체결을 체결로 바꾸거나 실패한 거래를 무료로 되돌리지 않습니다.</li>
    <li>부분 체결 후 두 수량의 비율을 대조합니다. 초과 체결한 쪽을 줄이고, 복구하지 못하면 양쪽을 청산합니다. 위험 축소는 두 종목의 공통 배율로 적용하며 이후 배분이 다시 늘리지 않습니다.</li>
    <li>기존 모델 학습 종료 이후인 8월 1일부터 보정 학습에 사용합니다. 8월 15일·9월 1일·10월 1일의 모델은 각각 42·93·183개 만기 도래 라벨을 사용합니다. 종목별 라벨은 겹치지 않지만 페어 간 표본은 서로 독립적이지 않습니다.</li>
    <li>기간 양 끝과 일별 계좌가치는 12:00 UTC로 맞췄습니다. σ = 일별 표본 표준편차 × √완료 일수. Q = (max(R,0)+0.005)/(|R|+1.5σ+0.05). 배심 점수 J는 포함하지 않습니다.</li>
    <li>페어는 종료 전에 실제 청산 주문을 생성합니다. 기존 전략은 잔여 포지션을 평가하므로, 모든 전략에 잔여 청산비 추정을 차감한 손익도 별도 표시했습니다. 이는 호가 부족까지 재현한 실제 청산 수익이 아닙니다.</li>
    <li>과거 자료를 여러 연구에서 반복 사용했습니다. 표와 선택 기준 통과 여부는 개발 판단이며 통계적으로 독립적인 성능 입증이 아닙니다. 별도 par 실시간 관측 시험은 고정한 규칙으로 계속 수집 중입니다.</li></ul>'''
    body+='''<h2>이번에 추가 검토한 연구</h2><p><a href="https://arxiv.org/html/2609.35359v1">Graziano (2026), From Cointegration to Out-of-Sample Failure</a>는 공적분과 과거 성과가 이후 수익으로 이어지지 않는 사례 및 선택 편향을 다룹니다. 따라서 가설 변경 후 같은 과거 자료에서 좋아진 수치를 독립 검증으로 취급하지 않습니다.</p>
    <p><a href="https://arxiv.org/html/2508.01880v1">Zhang 외 (2025), Time-Varying Factor-Augmented Models for Volatility Forecasting</a>는 동적 변동성 요인과 AR·HAR·MIDAS·LSTM 결합을 연구하며, 페어 실험은 ADA–ETH와 편도 5bp 비용을 사용합니다. ADA는 현재 대회 종목에 없고 본 실험의 비용·자료도 다릅니다. 이 논문의 수익률을 우리 결과로 옮기지 않았으며, 이번 후보를 해당 논문의 재현이라고 주장하지 않습니다.</p>
    <h2>원본 증거</h2><ul>'''
    body+=''.join(f'<li><a href="{path}">{label}</a></li>' for path,label in links)+'</ul></html>'
    DEST.mkdir(exist_ok=True);(DEST/'report.html').write_text(body)
    print(DEST/'report.html')


if __name__=='__main__':main()
