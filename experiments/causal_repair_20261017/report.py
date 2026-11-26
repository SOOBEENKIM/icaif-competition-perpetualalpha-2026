"""Evidence report for the completed repairs, including rejected economic results."""
import html
import json
from pathlib import Path

import numpy as np

from .data import HERE,OUT,sha,save

ROOT=HERE.parents[1];DEST=ROOT/'reports/causal_repair_20261017'


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def pct(x):return f'{x*100:.6f}%'
def money(x):return f'{x:,.2f}'


def main():
    DEST.mkdir(exist_ok=True)
    results=json.loads((OUT/'comparison/results.json').read_text());accepted=json.loads((OUT/'acceptance/results.json').read_text())
    forecasts=json.loads((OUT/'forecast/manifest.json').read_text());receipts=json.loads((OUT/'data_receipts.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    rows={(r['phase'],r['policy'],r['scenario']):r for r in results['rows']}
    names={'first_strategy':'첫 번째 전략','second_strategy':'두 번째 전략 · 현재 운용',
           'rolling_ungated':'확장 자료·월별 학습 · 검증 없는 적용',
           'rolling_gated':'종목별 검증만 적용 · 중간안',
           'portfolio_admitted':'전체 운용 검증까지 적용 · 최종 수정안'}
    for r in accepted['rows']:rows[r['phase'],r['policy'],r['scenario']]=r
    source_comparison=[]
    with np.load(ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz') as z:tt=z['market_times'];cc=z['close'][:,:3]
    for j,s in enumerate(['BTC','ETH','HYPE']):
        with np.load(OUT/f'external/{s}.npz') as z:et=z['times'];ec=z['close']
        common,a,b=np.intersect1d(tt,et,return_indices=True);basis=(ec[b]/cc[a,j]-1)*10000
        source_comparison.append({'symbol':s,'overlap_hours':len(common),'median_basis_bps':float(np.median(basis)),
            'p95_absolute_basis_bps':float(np.quantile(abs(basis),.95)),
            'hourly_return_correlation':float(np.corrcoef(np.diff(np.log(ec[b])),np.diff(np.log(cc[a,j])))[0,1])})
    save(DEST/'summary.json',{'verification':verification,'datasets':receipts['datasets'],'venue_comparison':source_comparison,
        'fits_completed':sum(f['available'] for f in forecasts['fits']),'recorded_economic_runs':len(results['rows'])+len(accepted['rows'])+2,
        'new_alpha_months_admitted':accepted['new_alpha_months_admitted'],
        'historical_independence':False,'live_replacement_approved':False,'decision':'Keep existing second live strategy. Corrected research controls reject the new alpha; no alpha improvement demonstrated.'})
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>원인 점검 후 수정·검증 결과 · 2026-10-17</title>
<style>body{font:16px/1.75 system-ui,sans-serif;max-width:1140px;margin:32px auto;padding:0 22px;color:#192638;background:#f5f7fa}section{background:white;padding:24px;margin:20px 0;border:1px solid #dfe5ed;border-radius:12px}h1,h2{line-height:1.4}h2{font-size:21px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce3eb;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f8}.scroll{overflow-x:auto}a{color:#185cab}.lead{border-left:5px solid #327854}.note{color:#536173}code{background:#eef2f6;padding:2px 5px}</style>
<h1>수정 구현과 비교 결과</h1>
<p>2026-10-17 · 운용 중인 전략 보존 · 실제 주문 0</p>
<section class="lead"><p><strong>데이터·시간 분할·위험 보정·종목별 검증·전체 운용 검증을 구현하고 실행했습니다.</strong> 최종 수정안은 검증에 실패한 새 모델을 채택하지 않고 기존 두 번째 전략을 유지합니다. 잘못된 모델 적용을 차단하는 수정은 확인됐지만, 새 예측 모델이 기존 전략보다 돈을 더 번다는 결과는 나오지 않았습니다.</p><p>최종 수정안의 수익률이 기존판과 같은 이유는 새 모델 채택이 0개월이기 때문입니다. 이 결과를 새 모델의 예측력 향상으로 보고하지 않습니다.</p></section>
<section><h2>무엇을 고쳤는가</h2>'''
    text+=table(['문제','구현한 수정','검증'],[
        ['짧은 한 구간 학습','외부 과거자료 확장, 종목별 이전 12개월 학습, 월별 재학습','19개 모델 생성; 모든 학습 정답이 적용 시점 전에 완성됨'],
        ['전체 평균이 특정 종목 실패를 가림','각 종목의 이전 60일 Hyperliquid 예측을 별도 검사','겹치지 않는 72h 구간 최소 16개; BTC/ETH/HYPE 개별 통과 여부'],
        ['빈 계좌 위험 보정','수량·현금·비용·펀딩을 이어 가는 순차 재생으로 위험 계수 선택','수익률로 계수를 고르지 않음; 목표 위험 이하를 충족하지 않으면 후보 사용 안 함'],
        ['학습 구간 보정이 이후에도 맞는다고 가정','매 판단에서 후보 목표의 예상 위험을 기준 전략 목표 이하로 제한','위험 상한 복원 금지, 다른 종목의 주문 보존, 강제 감축 우선 테스트'],
        ['예측 점수만 통과하면 실제 운용에 사용','이전 60일 전체 비용 차감 운용 결과도 비교','수익 개선·STD/MDD 비악화·최소 30개 일별 수익률 필요'],
        ['새 모델 도입과 함께 BRENT 등 제거','기존 10종목 유지; 검증된 암호화폐 부분에만 최대 25% 모델 기여','원유 등 다른 종목의 주문 불변 테스트'],
        ['수치 저장 중 numpy bool 오류','원자적 JSON 저장, 수치·배열 직렬화 수정','실패 증거 보존; 48개 재실행 결과가 필드별로 정확히 동일']])
    text+='''<p>새 모델은 XGBoost 72시간 예측입니다. 연속된 과거로부터 월마다 다시 학습하며, 미래 정답이 경계 밖으로 넘어가는 표본은 제외합니다. 조정 계수·모델 기여율을 결과에 맞춰 탐색하지 않았습니다. 기존 논문의 일부 방법을 적용한 변형 연구이며 원 논문의 전체 실험을 재현한 것은 아닙니다.</p></section>
<section><h2>데이터와 외부 거래소 차이</h2><p><a href="https://github.com/binance/binance-public-data">Binance 공식 공개자료</a>의 USD-M 선물 시간봉을 체크섬과 함께 수집했습니다. 없는 상장 이전 자료를 만들거나 누락 봉을 보간하지 않았습니다.</p>'''
    text+=table(['종목','수집 시간봉 수','최초 봉 종료 UTC','마지막 봉 종료 UTC'],[
        [s,d['bars'],str(np.datetime64(d['start'],'s')),str(np.datetime64(d['end'],'s'))] for s,d in receipts['datasets'].items()])
    text+=table(['종목','기준시장과 겹친 시간','가격 차이 절대값 95백분위(bp)','시간 수익률 상관계수'],[
        [x['symbol'],x['overlap_hours'],f"{x['p95_absolute_basis_bps']:.3f}",f"{x['hourly_return_correlation']:.6f}"] for x in source_comparison])
    text+='''<p>높은 가격 상관관계가 예측력 이전을 보장하지는 않습니다. 외부 자료로 학습한 모델은 Hyperliquid 기준시장에 대해 과거 시점별 예측을 만들고, 그 결과로 종목별 적격성을 검사합니다. HYPE는 상장 기간이 짧아 2026년 4·5월 모델을 만들지 않았습니다.</p>
<p>기준시장 자료는 mainnet reference이며 par 과거 호가가 아닙니다. 이번 재생은 다음 시간 종가에 비용과 거래량 제약을 적용합니다. 실제 대회 호가 체결 성능은 별도 검증 대상입니다.</p></section>
<section><h2>위험 보정 오류만 분리해서 수정</h2><p>동일한 과거 Ridge 예측을 그대로 두고 보정만 바꿨습니다. 기존 계수 55.4168 대신 7월 보유 경로 위험 기준을 통과한 계수 200을 선택했습니다. 이 진단의 7월 예측은 학습 내 예측이며, 새 후보의 위험 선택은 학습 외 예측을 사용합니다.</p>'''
    text+=table(['8월 진단','수익률','일별 STD','최대낙폭'],[
        [name,pct(rows['august',key,'base']['metrics']['roi']),pct(rows['august',key,'base']['metrics']['daily_std_sample']),pct(rows['august',key,'base']['metrics']['max_drawdown_magnitude'])]
        for key,name in [('old_flat_ridge','기존 빈 계좌 보정'),('repaired_path_ridge','실제 보유 경로 보정')]])
    text+='''<p>손실 감소에는 투자 규모 감소가 기여합니다. 위험 보정 수정이 알파 개선을 뜻하지 않습니다. 이후 구간의 위험을 고정 계수만으로 맞출 수 없어 최종 제어기에 판단 시점의 위험 상한도 추가했습니다. 위험을 목표 이하로 맞췄으며, 정확히 같은 실현 위험으로 맞췄다고 주장하지 않습니다.</p></section>
<section><h2>동일 조건의 성과 비교</h2><p>초기자본 1,000,000. UTC 2026-08-01~10-13 12:00의 73.5일 기준시장 재생입니다. 일별 STD는 완결된 UTC 일별 수익률의 표본 표준편차입니다. 끝의 보유분은 종가 평가이며 강제 청산 비용은 차감하지 않습니다.</p>'''
    text+=table(['기본 조건','수익률','일별 STD','최대낙폭','수수료','체결 수'],[
        [names[k],pct(z['roi']),pct(z['daily_std_sample']),pct(z['max_drawdown_magnitude']),money(z['fees']),z['fills']]
        for k in names for z in [rows['all',k,'base']['metrics']]])
    text+=table(['스트레스 조건','수익률','일별 STD','최대낙폭'],[
        [names[k],pct(z['roi']),pct(z['daily_std_sample']),pct(z['max_drawdown_magnitude'])]
        for k in names for z in [rows['all',k,'stress']['metrics']]])
    text+='''<p>모델을 추가한 중간안은 변동성은 줄였지만 수익이 더 낮았습니다. 따라서 교체하지 않았습니다. 최종 수정안의 기존판 동일 성과는 <strong>새 모델을 거절하고 기존판으로 돌아가는 동작</strong>의 검증입니다. 스트레스에는 매시간 총보유금액의 1bp 추가 펀딩 가정이 포함되며 실제 par 펀딩 수치가 아닙니다.</p></section>
<section><h2>왜 새 모델 사용을 막았는가</h2>'''
    text+=table(['적용 월','BTC 종목 검사','ETH 종목 검사','HYPE 종목 검사'],[
        [month,*['통과' if next(g for g in forecasts['gates'] if g['asset']==s and str(np.datetime64(g['cutoff'],'s'))[:7]==month)['enabled'] else '실패/표본 부족' for s in ['BTC','ETH','HYPE']]] for month in ['2026-08','2026-09','2026-10']])
    text+='''<p>8월에는 BTC·ETH가 종목 검사를 통과했습니다. 하지만 8월 진입 전에 볼 수 있었던 이전 60일 전체 운용 결과는 새 후보 −0.323543%, 기존판 −0.164387%였습니다. 이 후보를 그대로 적용하면 8월 결과도 기존판보다 낮았습니다. 따라서 전체 운용 적격성 검사에서 거절하게 고쳤습니다. 9·10월에는 종목 검사부터 통과하지 못했습니다.</p>
<p>전체 운용 검사 추가는 중간안 결과를 보고 설계한 후속 수정입니다. 각 월 판단에는 그 이전 자료만 사용하지만, 연구자가 이미 과거를 본 사실은 없어지지 않습니다. 독립된 미래 성과 검증으로 포장하지 않습니다.</p></section>
<section><h2>검증 결과와 현재 운용</h2>
<p>14개 회귀 테스트 통과: 미래 자료 변경에 따른 과거 특징·적격성 불변, 학습 정답 경계, 종목 분리, 위험 상한, 다른 종목 보존, 중복 판단 방지, 강제 감축, 비용 차감 성과 검사 및 결과 저장을 포함합니다. 기존 두 전략의 재생도 원래 저장 결과와 정확히 일치합니다.</p>
<p>실제 거래에 연결된 두 번째 전략과 별도로 봉인한 실험의 코드는 변경하지 않았고, 해당 실험의 중간 성과를 열람하지 않았습니다. 새 알파의 실제 주문은 0입니다. 새 예측을 활성화할 근거가 없으므로 기존 두 번째 전략을 유지합니다.</p>
<p>다음 교체 후보는 이 수정된 검증 경로에서 비용 차감 후 우위를 입증해야 합니다. 자료량 증가나 모델 학습 완료만으로 교체하지 않습니다.</p>'''
    text+=f'<p><a href="{HERE}/protocol.json">사전 설정</a> · <a href="{OUT}/comparison/results.json">중간안 전체 결과</a> · <a href="{OUT}/acceptance/results.json">최종 적격성 결과</a> · <a href="{HERE}/all_tests.log">테스트 기록</a> · <a href="{OUT}/verification.json">보존·재현 검증</a> · <a href="{HERE}/policy.py">제어기 코드</a> · <a href="{HERE}/forecast.py">학습·예측 코드</a></p></section></html>'
    (DEST/'report.html').write_text(text)
    save(DEST/'manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [*HERE.glob('*.py'),HERE/'protocol.json',OUT/'verification.json',DEST/'summary.json',DEST/'report.html']})
    print(DEST/'report.html')


if __name__=='__main__':main()
