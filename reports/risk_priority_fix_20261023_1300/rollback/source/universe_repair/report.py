"""Reviewable scope fix, matched results and deployment decision."""
import html,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/universe_repair_20261022'
NAMES={'legacy_core_v1':'기존 운용판','all_market_scope_only':'범위만 10종목 확대',
       'all_market_signed_core_v1':'10종목·양방향 수정판'}


def main():
    read=lambda n:json.loads((OUT/n).read_text())
    ref=read('reference_results.json');pas=read('pas_results.json');decision=read('deployment_decision.json')
    assert len(ref)==18 and len(pas)==9
    assert all(r['risk_breach_bars']==0 for r in ref)
    assert all(r['risk_breach_events']==0 for r in pas)
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    reference=table(['기간','체결 조건','버전','순손익 USD','총자본 수익률','기간 변동성 σ','Q','체결 종목 수'],[
        [r['phase'],r['scenario'],NAMES[r['arm']],f"{r['net_pnl']:+,.2f}",f"{r['return_fraction']*100:+.6f}%",
         f"{r['volatility']*100:.6f}%",f"{r['Q']:.6f}",len(r['symbols'])] for r in ref])
    pas_table=table(['체결 조건','버전','순손익 USDP','청산비용 예비금 차감 후','Q','체결 종목 수'],[
        [r['scenario'],NAMES[r['arm']],f"{r['net_pnl']:+,.2f}",f"{r['pnl_after_liquidation_reserve']:+,.2f}",f"{r['Q']:.6f}",len(r['traded_symbols'])] for r in pas])
    packet=read('staged/signal.json');audit=packet['body']['universe_audit'];anchors=packet['body']['frame']['anchor_weights']
    current=table(['종목','고려 대상','핵심 신호','위험 감축','현재 원시 목표 / 우리 자본'],[
        [r['symbol'],r['considered'],r['core_decision'],r['risk_reduced'],f'{a*100:+.4f}%'] for r,a in zip(audit,anchors)])
    attribution=table(['종목','가격 손익(체결비용 포함)','수수료','펀딩','순손익'],[
        [r['symbol'],f"{r['price_pnl']:+.4f}",f"{r['fees']:.4f}",f"{r['funding']:+.4f}",f"{r['net_pnl']:+.4f}"]
        for r in read('pas_asset_attribution.json')['all_market_signed_core_v1']])
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    body=f'''<h1>종목 범위 수정과 동일 조건 비교</h1><p>{stamp}</p>
<p><strong>핵심 신호의 3종목 제한과 롱 중심 설정을 제거한 코드·실행 경로는 검증했다. 하지만 이 수정판은 비교 수익과 Q가 악화되어 실거래로 활성화하지 않았다. 현재 실행자는 기존 <code>legacy_core_v1</code>이다.</strong></p>
<h2>확인한 원인과 변경</h2>
<p>기존 주문 실행기는 10종목을 지원했지만 <code>candidate_handoff.producer.generate</code>는 <code>scope=legacy, short=False</code>로 핵심 EMA·채널·되돌림을 생성했다. 따라서 실행기의 지원 범위만으로 전체 종목 판단을 구현했다고 한 것은 잘못이었다.</p>
<p>새 <code>universe_repair/core.py</code>는 10종목에 같은 핵심 규칙을 적용하고 추세 진입·종료를 롱과 숏에 대칭 적용한다. 종목별 신호·목표·위험 감축을 기록한다. 별도의 기존 페어 전문가 BTC/ETH, BTC/HYPE는 유지했다. 이 두 페어의 구성 여부가 핵심 전략의 다른 종목 주문을 차단하지 않는다.</p>
<p>10/60일 EMA, 5/10/20/30일 채널, 7일 되돌림, 고정 Chronos·regime 보정, 기존 위험 한도와 수량 배율은 유지했다. 새 모델 훈련이나 기간별 최적 파라미터 선택은 하지 않았다. 10종목을 평가한다는 뜻이지 항상 10종목을 보유한다는 뜻은 아니다.</p>
<h2>검증</h2><p>57개 테스트 통과. 합성 상승·하락 입력에서 10종목 모두 실제 주문 송신 함수와 체결 장부까지 도달했다. 이 검증의 거래소는 모의 구현이며 실제 주문은 0건이다. 종목 순서 불변성, 미래 추가 시 과거 판단 불변성, 신호 없을 때 강제 매수 없음, 잘못된 배포 버전 거절, 기존 보유·팀원 수량 보존을 확인했다.</p>
<p>비교는 현재 운용판·범위만 확대·범위와 양방향 수정의 세 개를 결과 확인 전에 고정했다. 각 50,000 운용자본 + 비교용 50,000 현금, 같은 비용·펀딩·기간·위험 한도를 사용했다. 총자본 수익률 분모는 100,000이다. 모든 결과는 이미 검토했던 과거 구간을 사용한 개발 비교이며 독립 검증이 아니다.</p>
<h2>외부 기준시장 비교 18회</h2><p>validation: 8월 1일~9월 1일, comparison: 9월 1일~10월 15일, continuous: 8월 1일~10월 15일. 모두 UTC 12:00 경계. σ는 완료 일별 수익률의 표본 표준편차 × √n인 기간 변동성이다. Q는 대회 수식을 적용한 모의 값이다. 실제 PAS 또는 공식 순위의 성과가 아니다.</p>{reference}
<h2>PAS 10월 19~21일 비교 9회</h2><p>한국시간 10월 19일 21:00~21일 21:00. 실제 체결 테이프와 펀딩을 이용한 조건부 재현이다. base_1pct는 관측 거래량의 1%까지 체결된다고 가정한다. stress_025pct는 0.25%와 더 높은 비용, base_10pct는 10% 체결 가정이다. 과거 전체 호가·마크·오라클이 없어 실제 체결 확정이나 공식 수익률로 해석할 수 없다. 기존 페어 신호는 이 구간 모두 0이어서 핵심 전략 결과를 비교했다.</p>{pas_table}
<p>수정판은 8종목에서 체결됐지만, 확대 자체가 수익성을 높이지 못했다. 기준 조건에서 가격 손실이 수수료보다 컸고 ETH·HYPE 숏 등 여러 포지션이 손실을 냈다. 단순히 수수료 문제라고 설명할 수 없다.</p>{attribution}
<h2>현재 신호의 종목별 확인</h2><p>18:00 KST 완료 입력으로 생성한 수정판이다. 아래 비중은 실제 주문 또는 체결이 아니라 원시 목표다. 현재 PAS 가격, 보유 수량, 최종 위험 배분, 수량 단위와 호가에 따라 변경될 수 있다.</p>{current}
<h2>실거래 상태와 조회 중단</h2><p>수정판을 활성화하지 않았다. 공개 상태를 종목별로 반복 확인하던 점검에서 HTTP429가 발생했고 기존 실행자도 안전 중단됐다. 미확정 주문이 없음을 확인한 후 기존 체크포인트로 복구했다. 우리 BTC +0.01899, 팀원 ETH −0.4064 보유분을 유지했고, 이 검토에서 보낸 주문은 없다.</p>
<p>이후 점검 도구의 종목별 전체 이력·계좌 재조회 루프를 제거했다. 공개 사전 점검은 첫 주문 후보만 수행하며 나머지는 목표 계산으로 명시한다. 이번 실계좌 사전 점검 전체는 완료되지 않았다. 미사용 종목의 Cross 1x 설정·설정 후 증거금 확인도 서명 없는 점검으로 완료했다고 표시하지 않았다.</p>
<h2>파일</h2><ul><li><code>universe_repair/core.py</code>: 범위·양방향 계산과 종목별 근거</li><li><code>candidate_handoff/producer.py</code>: 명시적 버전 선택, 신호 검증, 누락 종목 거절</li><li><code>candidate_handoff/runner.py</code>: 실행자와 신호 버전 대조</li><li><code>candidate_handoff/portfolio.py</code>: 판단 원장에 버전·종목 근거 기록</li><li><code>universe_repair/study.py</code>: 동일 조건 비교</li><li><code>tests.txt</code>, <code>reference_results.json</code>, <code>pas_results.json</code>, <code>deployment_decision.json</code>: 원본 결과·배포 판단</li></ul>
<p><strong>수정 완료의 범위:</strong> 종목 제한 제거 코드와 모의 실행 검증. <strong>완료하지 않은 것:</strong> 수익성 개선, 새 수정판의 실거래 활성화, 모든 종목의 실계좌 사전 점검.</p>'''
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>종목 범위 수정 검증</title><style>body{font:16px/1.65 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172238}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{border:1px solid #d9e0ea;padding:7px;text-align:right}th{background:#eef3f9}td:first-child,td:nth-child(2){text-align:left}h1{font-size:28px}h2{margin-top:36px}code{background:#f1f4f8;padding:2px 4px}</style>'+body+'</html>')
    print(OUT/'report.html')


if __name__=='__main__':main()
