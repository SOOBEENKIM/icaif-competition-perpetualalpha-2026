"""Render reviewed evidence, including unsuccessful candidates and correction."""
import html
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path

from .research import HERE,ROOT,OUT,sha,save,protected

LABELS={
 'first_strategy':'첫 번째 전략', 'second_strategy':'현재 두 번째 전략',
 'second_75pct_cash':'현재 전략 75% + 현금 25%',
 'pair_ou':'BTC–ETH 평균회귀 페어', 'pair_kalman':'BTC–ETH 칼만 페어 — 보정 불가',
 'relative_momentum':'암호화폐 상대 모멘텀', 'channel_breakout':'다자산 장기 돌파',
 'blend_pair_ou':'기존 75% + 평균회귀 페어 25%',
 'blend_pair_kalman':'기존 75% + 칼만 25% — 현금으로 대체',
 'blend_relative_momentum':'기존 75% + 상대 모멘텀 25%',
 'blend_channel_breakout':'기존 75% + 장기 돌파 25%',
 'blend_equal_experts':'기존 75% + 네 후보 균등 혼합 25%'
}


def main():
    result=json.loads((OUT/'reviewed/results.json').read_text())
    initial=json.loads((OUT/'results.json').read_text())
    snapshot=json.loads((OUT/'public_market_snapshot.json').read_text())
    counts=protected()
    frozen=json.loads((OUT/'frozen.json').read_text())
    for path,h in {**frozen['sources'],**frozen['inputs']}.items():assert sha(ROOT/path)==h,path
    reviewed=json.loads((OUT/'reviewed/frozen.json').read_text())
    assert sha(HERE/'calibration_review.py')==reviewed['source_sha256']
    assert sha(OUT/'results.json')==reviewed['initial_result_sha256']
    assert all(all(r['gates'].values()) for r in result['rows'])
    def pct(x):return '—' if x is None else f'{x*100:+.6f}%'
    def nonnegative(x):return '—' if x is None else f'{x*100:.6f}%'
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    order=list(LABELS)
    def results_table(phase,scenario):
        rows=[]
        for name in order:
            r=next(r for r in result['rows'] if (r['phase'],r['scenario'],r['policy'])==(phase,scenario,name));z=r['metrics']
            c=next(c for c in result['comparisons'] if (c['phase'],c['scenario'],c['policy'])==(phase,scenario,name))
            rows.append([LABELS[name],pct(z['roi']),nonnegative(z['daily_std_sample']),nonnegative(z['max_drawdown_magnitude']),
                 f"{z['fees']:,.2f}",z['fills'],pct(c['return_above_75pct_cash']),pct(z['roi_after_estimated_liquidation'])])
        return table(['구성','순수익률','일별 표본 STD','최대낙폭','수수료 USD','체결 조각','75% 현금 대조군 대비 수익 차이','추정 종료청산비 차감 ROI'],rows)
    eligible=[k for k in order if k.startswith('blend_') and k!='blend_pair_kalman']
    wins=[]
    for name in eligible:
        checks=[c for c in result['comparisons'] if c['policy']==name]
        if all(all(c['checks'].values()) for c in checks):wins.append(name)
    assert not wins
    mechanism_rows=[
      ['평균회귀 페어','BTC–ETH 상대가격 복원','과거 60일 OLS 헤지비율 + AR(1) 반감기','6시간 판단, z=2 진입 / 0.5 종료 / 4 손절, 7일 최대 보유','정식 공적분 검정은 아님. 두 계약의 체결 시차와 관계 붕괴 위험'],
      ['칼만 페어','변하는 상대 적정가격에서 벗어난 정도','전방 필터의 사전 혁신값, 시간가변 계수','z=2 진입, 72시간 제한. 미래를 쓰는 smoothing 없음','7월 무거래라 위험 보정 불가. 처음의 최대 규모 선택은 폐기'],
      ['상대 모멘텀','같은 암호화폐 시장 안의 강약 차이','14일 수익률에서 최근 1일 제외한 순위','상위 매수·하위 매도, 달러 금액 중립, 최소 72시간','BTC·ETH·HYPE 세 종목뿐. 시장 베타 중립을 뜻하지 않음'],
      ['장기 돌파','상승·하락 추세의 지속','28일 가격 채널 진입 / 14일 반대 채널 종료','10개 종목, 변동성 역수 배분, 양방향','기존 추세와 겹치는 부분이 많아 독립 수익원인지 별도 확인']]
    decor=[]
    for name in ('pair_ou','relative_momentum','channel_breakout'):
        c=next(c for c in result['comparisons'] if (c['phase'],c['scenario'],c['policy'])==('all','base',name))['complement']
        z=next(r for r in result['rows'] if (r['phase'],r['scenario'],r['policy'])==('all','base',name))['metrics']
        decor.append([LABELS[name],pct(z['roi']),f"{c['daily_return_correlation']:.4f}",
                      c['incumbent_loss_days'],pct(c['mean_candidate_on_incumbent_loss_days']),c['positive_candidate_on_loss_days']])
    source_rows=[
      ['2025 Financial Innovation, Copula-based trading of cointegrated cryptocurrency Pairs',
       '20종 암호화폐, BTC 기준 자산 관계와 copula를 사용. 수수료를 포함한 실증 연구.',
       '상대가격이라는 수익원만 참고. 여기서는 copula·20종 우주를 재현하지 않았음.'],
      ['2026 Frontiers, Deep learning-based pairs trading: real-time forecasting of co-integrated cryptocurrency pairs',
       'Johansen 관계와 DNN/LSTM 동적 앙상블을 결합. 논문은 거래비용을 명시적으로 모델링하지 않음.',
       '다른 모형의 결합 후보를 찾는 참고 자료. 그 예측 성과를 우리 시장 순수익의 증거로 사용하지 않음.'],
      ['Han·Kang·Ryu, Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market',
       '거래비용·가격 경로·청산 위험을 고려하면 상대 모멘텀의 근거가 약해질 수 있음을 보고.',
       '상대 모멘텀을 확정된 우위가 아닌 검증 후보로 포함. 공개본의 2013–2023 표본과 우리 기간은 다름.']]
    ts=datetime.fromtimestamp(snapshot['metadata']['received_at'],timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    quotes=[]
    for r in snapshot['summary']:
        quotes.append([r['coin'],f"{r['spread_bps']:.3f}",f"{r['buy_discount_to_oracle_bps']:.3f}",
                       f"{r['sell_premium_to_oracle_bps']:.3f}",f"{r['funding']*10000:.5f}",
                       f"{r['best_bid_notional']:,.2f}",f"{r['best_ask_notional']:,.2f}"])
    body='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>다른 수익원과 전략 앙상블 비교</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:36px auto;padding:0 24px;color:#17313c;background:#fafcfb}h1,h2{line-height:1.3}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0 25px}th,td{padding:10px;border:1px solid #d8e3df;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e8f0ed}.callout{background:#e9f3ef;border-left:5px solid #397b63;padding:16px 22px}.caution{background:#fff4df;padding:16px 22px}a{color:#206856}code{background:#edf1ef;padding:2px 5px}</style>
<h1>다른 수익원과 전략 앙상블 비교</h1>
<p class="callout"><strong>결론: 네 후보와 다섯 고정 앙상블을 구현·비교했지만, 현재 전략을 교체할 성능 근거는 확보하지 못했습니다.</strong>
수익률과 일별 변동성의 개선을 함께 확인했고, 투자 비중 축소만으로 생기는 변동성 감소를 별도 대조군으로 구분했습니다.
현재 실거래 전략은 이 실험에서 변경하지 않았고 새 주문도 보내지 않았습니다.</p>
<p>2026-08-01 00:00 UTC → 2026-10-13 12:00 UTC, 73.5일. 초기자본 1,000,000.
아래는 Hyperliquid 기준시장 시간봉과 실제 기준시장 펀딩으로 계산한 <strong>과거 모의 결과</strong>이며 par 실거래 수익률이 아닙니다.
이미 여러 개발에 사용한 기간이므로 독립 검증이라고 부르지 않습니다.</p>
<h2>구현한 후보와 모델</h2>'''+table(['후보','수익 가설','모형','고정 규칙','한계'],mechanism_rows)+'''
<p>앙상블은 기존 전략 75% + 개별 후보 25% 네 가지, 기존 75% + 네 후보 균등 혼합 25% 한 가지입니다.
한 지갑의 목표 수량을 먼저 합산·상계한 뒤 공통 주문·위험 관리와 장부로 재실행했습니다. 각 전략의 수익률을 단순 평균한 결과가 아닙니다.
별도 대조군은 기존 전략 75% + 현금 25%입니다. 첫·두 번째 전략의 10종목 우주와 공통 위험 한도는 유지하며,
암호화폐 전용 후보의 혼합은 기존 매크로 노출도 75%로 낮춥니다. 이 배분 차이는 현금 대조군으로 비교합니다.</p>
<h2>전체 기간 기본 비용 결과</h2>'''+results_table('all','base')+'''
<p>표의 칼만 후보는 위험 보정 불가로 비활성화한 후의 결과입니다. 따라서 칼만 혼합은 현금 대조군과 체결 및 자산 경로가 정확히 같습니다.
이를 개선된 전략으로 보지 않습니다. 최초 기록에서의 칼만 수익은 아래 정정에 남겼습니다.</p>
<h2>전체 기간 비용 스트레스 결과</h2>'''+results_table('all','stress')+'''
<p>기본: 편도 수수료 9bp + 충격 10bp, 거래량 참여율 1%. 스트레스: 수수료 9bp + 충격 20bp,
참여율 0.25%, 여기에 시간당 총노출 1bp의 가상 불리한 펀딩을 추가합니다.
이 추가 펀딩은 실제 par 펀딩이 아닙니다. 종료청산 열은 마지막 보유 수량의 비용 추정치만 추가 차감하며,
그 시점에 실제로 전량 청산할 수 있음을 입증하지 않습니다. 대회 수익은 종료 시 oracle 평가와도 구별해야 합니다.</p>
<h2>기간을 나누어 본 결과</h2><h3>8월 · 기본 비용</h3>'''+results_table('august','base')+'''
<h3>9월 1일 ~ 10월 13일 12:00 UTC · 기본 비용</h3>'''+results_table('september_october','base')+'''
<h2>기존 전략과 실제로 다른 결과를 내는가</h2>'''+table(['단독 후보','순수익률','기존 일수익과 상관계수','기존 손실 일수','그 날 후보 평균 수익률','그 날 후보 양수 일수'],decor)+'''
<p>평균회귀 페어는 기존과 낮은 상관관계를 보였지만 전체 비용 차감 수익은 음수였습니다. 상관관계가 낮다는 것만으로
섞을 이유가 충분하지 않습니다. 상대 모멘텀과 장기 돌파도 전체 기간 순수익이 음수였고,
혼합은 단순 현금 대조군보다 수익을 높이지 못했습니다. 후보 자체가 반드시 단독으로 모든 지표를 이겨야 하는 것은 아니며,
최종 판단 대상은 합친 계좌의 수익·위험입니다.</p>
<h2>발견하고 바로잡은 규모 보정 문제</h2>
<p class="caution">최초 칼만 후보는 +0.201997%를 기록했으나, 이는 비교 기간의 한 차례 신호에서 나온 결과였습니다.
7월 보정 기간에는 거래가 없어 모든 규모의 관측 위험이 0이었고, 처음 선택기는 최대 후보 규모 20%를 허용했습니다.
이것은 동등 위험 보정의 증거가 아닙니다. 원본 결과를 보존하고, 무거래 경로로는 위험 규모를 정할 수 없도록 수정했습니다.
해당 후보는 비활성화하고 배분액을 현금으로 남겼으며, 영향받은 18개 비교를 재실행했습니다.
신호·진입 임계값을 사후 조정하지 않았습니다. 추가 회귀 테스트는 무거래 경로가 최대 규모를 선택하지 못하도록 확인합니다.</p>
<p>다른 후보의 규모는 7월 전체 계좌 경로에서 현재 전략의 평균 예상 위험 102% 및 평균 총노출 125% 이내인 사전 고정 격자에서 정했습니다.
수익률로 규모를 선택하지 않았습니다. 이는 상한 보정이며 실제 변동성이 정확히 같은 비교는 아닙니다.
전체 기간과 하위 기간·기본 및 스트레스 모두에서 수익 증가, 일별 STD·MDD 비악화를 충족한 혼합 후보는 없습니다.</p>
<h2>실제 par 시장에서 확인한 별도 후보: 오라클 괴리 + 펀딩</h2>
<p>공개 API 조회 시각: '''+html.escape(ts)+'''. 사전 예약된 독립 실험의 봉인 자료를 열지 않고 새 공개 스냅샷을 받았습니다.
아래는 한 시점의 호가이며 수익 예측이나 백테스트가 아닙니다.</p>'''+table(['시장','호가차 bp','최우선 매수 시 oracle 할인 bp','최우선 매도 시 oracle 프리미엄 bp','표시 funding ×10⁴','최우선 bid 명목액','최우선 ask 명목액'],quotes)+'''
<p>이 시점에는 BTC·ETH·HYPE에서 매수 호가가 oracle보다 낮고 표시 펀딩도 음수였습니다. 이 조합은
가격 격차의 복원과 펀딩을 함께 보는 후보를 조사할 이유가 되지만, 같은 기초자산의 반대 계약을 확보한 무위험 차익거래가 아닙니다.
oracle 자체는 매도할 수 있는 호가가 아니며 수렴 시점도 보장되지 않습니다.</p>
<p>실제 표시 수량으로 500·2,000·10,000 명목액을 호가에 따라 계산했습니다. 500 명목액에서
SP500 매수의 이상적인 oracle 복원 폭은 약 18.672bp였고, 진입·oracle 가격 청산의 수수료를 각각 계산하면 약 0.655bp였습니다.
청산 호가차·지연·가격 변화를 추가하면 이 수치가 그대로 남지 않습니다. 따라서 이 스냅샷으로 즉시 수익성 있다고 판정하지 않았습니다.</p>
<h2>논문에서 가져온 것과 가져오지 않은 것</h2>'''+table(['1차 자료','확인 내용','이번 구현과의 관계'],source_rows)+'''
<p><a href="https://link.springer.com/article/10.1186/s40854-024-00702-7">2025 페어 트레이딩 논문</a> ·
<a href="https://doi.org/10.3389/fams.2026.1749337">2026 딥러닝 페어·앙상블 논문</a> ·
<a href="https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf">모멘텀의 실제 비용·위험 분석</a> ·
<a href="https://perpetual-alpha26.github.io/about">대회 평가 규정</a></p>
<h2>다음 개발 판단</h2>
<p>현재 보류: 같은 가격 방향을 예측하는 모델 이름만 추가하는 앙상블, 한 번 이긴 칼만 거래의 확대,
모든 약한 후보를 무조건 섞는 방식. 다음 연구 우선순위는 <strong>par의 실행 가능한 호가 대비 oracle 괴리와
펀딩 지속성을 함께 추정하는 후보</strong>입니다. 호가 수량·지연 후 진입가·실제 청산가를 목표로 삼고,
기존 방향 전략과 합친 계좌의 순성과를 검증해야 합니다. 이미 봉인한 독립 시험의 종료 전 결과를 가져다 튜닝하지 않습니다.</p>
<p>현재 후보들은 공통 시장 한도에서 비교했으며 대회 공식 종합점수는 계산하지 않았습니다.
공식 점수는 팀 간 정규화와 심사위원 평가가 필요합니다. 이 보고서는 후보 연구 결과이며 실거래 교체 승인이나 성능 보장이 아닙니다.</p>
<h2>검증과 재현 자료</h2>
<p>신호 미래정보 차단·페어 비용 단위·상계·중복 판단 방지·무거래 규모 보정 등 11개 테스트 통과.
최초 비교 72회, 규모 보정 21회, 정정 비교 18회. 첫·두 번째 기준 전략 8개 경로에서 기존 체결과 자산 경로를 정확히 재현했습니다.
수정된 최종 72개 비교의 장부·정보 시각·위험 검사도 통과했습니다.</p>
<p>원본 구현과 결과는 변경하지 않고 정정 모듈 및 결과를 별도로 보존했습니다. 실거래 의존 파일 '''+str(counts['active_protected_sources'])+'''개와
예약 실험 의존 파일 '''+str(counts['reserved_dependencies'])+'''개의 해시가 일치합니다. 이 작업의 실제 주문 전송은 0건입니다.</p>
<p>원자료: <code>experiments/complementary_alpha_20261017/artifacts/reviewed/results.json</code>.
실험 설정: <code>protocol.json</code>. 정정 진입점: <code>calibration_review.py</code>.
이미 실행된 폴더는 덮어쓰기를 거부합니다. 재실행 시 새 결과 경로를 사용하고 원본의 동결 결과를 보존해야 합니다.</p></html>'''
    dest=ROOT/'reports/complementary_alpha_20261017';dest.mkdir(parents=True,exist_ok=True)
    (dest/'report.html').write_text(body)
    save(dest/'summary.json',dict(rows=result['rows'],comparisons=result['comparisons'],
        disqualified_experts=result['unavailable_experts'],jointly_superior_candidates=wins,
        tests_passed=11,baseline_paths_reproduced=8,orders_sent=0,
        protected_counts=counts,historical_independence=False,live_replacement_approved=False))
    save(dest/'manifest.json',{str(q.relative_to(ROOT)):sha(q) for q in [*HERE.glob('*.py'),HERE/'protocol.json',
        OUT/'frozen.json',OUT/'results.json',OUT/'reviewed/results.json',OUT/'reviewed/frozen.json',
        OUT/'public_market_snapshot.json',OUT/'snapshot_capacity.json',HERE/'tests.log',dest/'report.html',dest/'summary.json']})
    print(dest/'report.html')


if __name__=='__main__':main()
