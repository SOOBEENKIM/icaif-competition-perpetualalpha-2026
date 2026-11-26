import html,json
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha,ts
from experiments.complementary_alpha_20261017.research import protected
from experiments.netted_shadow_pairs_20261017.report import document,table,export_csv

HERE=Path(__file__).resolve().parent


def grid(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    out=ROOT/'reports/long_history_pairs_20261017';out.mkdir(exist_ok=True)
    r=json.loads((HERE/'artifacts_v2/results.json').read_text());models=json.loads((HERE/'artifacts_v2/models.json').read_text())
    audit=json.loads((HERE/'artifacts_v2/source_forecast_audit.json').read_text());signals=json.loads((HERE/'artifacts_v2/signal_audits.json').read_text())
    repair=json.loads((ROOT/'experiments/event_value_pairs_20261017/par_repair_artifacts/results.json').read_text())
    for folder in ('artifacts','artifacts_v2'):
        frozen=json.loads((HERE/folder/'frozen.json').read_text())
        for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    assert models==json.loads((HERE/'artifacts/models.json').read_text())
    guards=protected();prediction_rows=[];prediction_records=[]
    for kind in ['scalar','ridge']:
        for phase,a,b in [('validation','2026-08-01T12:00:00','2026-09-01T12:00:00'),('comparison','2026-09-01T12:00:00','2026-10-15T12:00:00')]:
            eligible=[x for x in signals['reference_'+kind] if ts(a)<=x['time']<ts(b) and x['method']!='unavailable']
            p=np.array([x['ratio']*x['full_gap']*10000 for x in eligible]);maximum=float(p.max()) if len(p) else None
            prediction_records.append(dict(kind=kind,phase=phase,sources=len(p),positive=int(sum(p>0)),maximum_price_bps=maximum,above_base_price_cost_and_buffer=int(sum(p>43))))
            prediction_rows.append([kind,phase,len(p),int(sum(p>0)),f'{maximum:.4f}' if maximum is not None else 'N/A',int(sum(p>43))])
    comparison=[]
    for phase in ['validation','comparison']:
        for arm in ['first_strategy','second_strategy','first_crypto_pairs','first_long_scalar','first_long_ridge']:
            for scenario in ['base','stress']:
                x=next(a for a in r['rows'] if (a['venue'],a['phase'],a['arm'],a['scenario'])==('reference',phase,arm,scenario));m=x['metrics']
                comparison.append([phase,arm,scenario,f"{m['net_pnl']:+,.2f}",f"{m['financial_score']:.8f}",m['completed_pairs']])
    forecasts=[]
    for pair,value in audit.items():
        for kind,m in value['metrics'].items():forecasts.append([pair,kind,m.get('events',len(value['rows'])),f"{m['rmse']*10000:.4f}",f"{m['bias']*10000:+.4f}"])
    body='''<h1>장기 자료로 학습한 페어별 예상 가격 이동 검증</h1>
<p class="note"><strong>장기 자료로 학습한 두 후보 모두 기존 두 전략의 PnL·Q를 개선하지 못했다.</strong> 88개 계좌 비교에서 새 페어 거래가 발생하지 않아 해당 기존 전략과 같은 성과였다. 예측한 가격 이동은 기본 비용을 충당할 만큼 크지 않았다. 실거래 전략과 설정은 변경하지 않았다.</p>
<h2>학습과 비교 자료를 분리한 방법</h2><p>BTC/ETH는 2023년 1월~2026년 2월 Binance USD-M 시간봉 27,720개, BTC/HYPE는 두 종목이 함께 관측되는 6,590개 시간봉을 사용했다. 같은 페어에서 보유기간이 겹치지 않는 완료 사건은 각각 111개와 17개였다. 학습은 2026년 3월 1일보다 먼저 완성된 목표만 사용했고 Hyperliquid 평가 기간의 결과는 모델 학습에 넣지 않았다. 거래소 가격을 하나로 이어 붙이지 않았다.</p>
<p>각 사건의 목표는 기존 고정 헤지·고정 평균·손절·시간 종료 규칙에 따른 가격 이동이다. 신호 가격에서 종료 판단 다음 시간봉 가격까지의 고정 수량별 현금 변화를 사용하고, 현지 표준편차로 정규화했다. 이는 가격 예측 목표이며 실제 체결 또는 비용 차감 수익이 아니다. 수수료와 펀딩은 실제 계좌 재현 단계에서 한 번만 계산한다.</p>
<h2>구현한 두 후보와 대조군</h2><p><strong>long_scalar:</strong> 페어별로 평균 회복 가정 대비 실제 가격 이동의 배율을 추정하고 24개 가상 관측에 해당하는 0 방향 보정을 적용했다. 서로 다른 페어의 결과를 합치지 않았다.</p>
<p><strong>long_ridge:</strong> 원래 신호의 Z 점수, 당시 헤지 비율로 계산한 1시간·24시간 회복 움직임, 로그 헤지 비율, 동일 통계량 기준 회복 여부를 입력으로 한 페어별 릿지 회귀다. 중앙값·사분위 범위로 정규화하고 사전에 고정한 정규화 강도와 값 범위를 사용했다. 48개 이상이면 회귀를 학습하며 BTC/HYPE의 17개 사례에는 선언한 단순 배율을 사용했다. 새 비율은 0~1로 제한하고 원래 평균·손절 기준은 바꾸지 않았다.</p>
<p>원래 첫째·둘째·페어 단독·혼합 두 개를 보존했다. 여기에 BTC/ETH와 BTC/HYPE만 남긴 <code>crypto_pairs</code> 대조군을 추가했다. 주식·지수 페어를 뺀 효과와 학습한 예상 이익의 효과를 구분하기 위해서다.</p>
<h2>왜 추가 진입이 생기지 않았는가</h2><p>BTC/ETH의 단순 배율은 음의 원추정치를 0으로 제한한 값이며, BTC/HYPE 배율은 약 0.0193이었다. 릿지의 평가 구간 최대 가격 이동 예측은 약 7.69bp였다. 기본 조건의 왕복 수수료·불리한 가격 이동 가정은 총 38bp이고 추가 진입 여유는 5bp다. 이는 실제 모든 주문의 확정 비용을 뜻하지 않으며, 이번 비교에 고정한 비용 가정이다. 실제 계좌 재현에서 펀딩과 현재 가격을 함께 계산한 뒤에도 새 페어가 열리지 않았다.</p>'''+grid(['예측','구간','지원 페어 신호','양의 예측','최대 가격 이동 bp','43bp 초과'],prediction_rows)+'''
<h2>기준시장 성과</h2><p>초기자본 1,000,000, validation: 8/01 12:00–9/01 12:00 UTC, comparison: 9/01 12:00–10/15 12:00 UTC. 같은 기간·자본·위험 한도를 사용했다. base 편도 수수료 9bp·불리한 가격 이동 10bp, stress 12bp·20bp 및 더 작은 체결 참여율과 펀딩 수령액 50%다. 아래 결과는 실제 par 계좌 수익이 아니다.</p>'''+grid(['기간','전략','비용','순손익','Q','완료 페어'],comparison)+'''
<p>기존 전략과 같은 성과가 나온 것은 개선이 아니다. 8월의 최소 완료 페어 수, 원래 두 전략 및 같은 암호화폐 집합의 대조군보다 높은 PnL·Q 조건을 충족하지 못했다. 선정 대표는 진단용 first_long_ridge이며 채택 자격은 false다. 이후 평가 결과로 대표나 배율을 바꾸지 않았다.</p>
<h2>가격 예측 오차와 거래 성과의 차이</h2><p>추가로 2025년 이전 자료만 학습해 이후 Binance 사건의 가격 예측을 확인했다. BTC/ETH 41건에서 회귀 예측의 RMSE는 가격 이동을 0으로 예측한 경우와 사실상 같았다. 평균으로 완전히 회복한다는 예측보다 오차가 작다는 사실만으로 비용을 넘는 알파가 생긴 것은 아니다. BTC/HYPE는 2025년 이전 자료가 없어 같은 시점의 모델 예측을 만들지 않았다.</p>'''+grid(['페어','예측 기준','사건 수','가격 이동 RMSE bp','예측 편향 bp'],forecasts)+'''
<h2>발견하고 수정한 자료 연결 오류</h2><p>기준시장 페어 배열은 BTC/ETH, EWY/EWJ, HYPE/EWJ, SP500/EWJ, BTC/HYPE 순서였고, par 배열은 BTC/HYPE, BTC/ETH 순서였다. 초기 연결기는 배열의 같은 열을 같은 페어로 취급했다. 새 코드의 양의 회복 금액 검사에서 이 오류를 발견해 계좌 재현 전에 중단했다. 실패한 실행과 학습 결과는 보존했다.</p>
<p>v2는 종목 조합으로 원래 페어를 찾고, 각 시장의 상태를 생성했던 원본 가격 자료로 입력을 계산한다. 페어 배열 순서를 바꿔도 결과가 같고, 없는 종목 조합은 명시적으로 실패하도록 검사했다. 수정 전후 학습 계수는 정확히 같다.</p>
<p>앞선 사건 보정 실험의 par 연결도 같은 문제를 가졌기에 36개 계좌를 재계산했다. 올바른 연결 후에도 그 구간의 페어 체결은 없어서 기존 계좌값·체결·PnL·Q가 모두 동일했다. 이전 par 신호 귀속 자료는 <code>event_value_pairs_20261017/par_repair_artifacts</code>로 교체하고, 오류 기록을 남겼다. 이전 기준시장 평가 구간의 원본 가격도 정확히 일치함을 확인했다.</p>
<h2>검증 범위와 다음 방향</h2><p>학습·예측 검사 7개 및 페어 식별·순서 불변 검사 4개, 총 11개 통과. 기존 다섯 대조군은 모든 시장·기간·비용 조건에서 체결과 계좌 이력을 정확히 재현했다. 이번 결과 88개와 이전 par 정정 재현 36개를 보존했다. 보호된 실거래 소스 373개와 예약 의존 파일 345개는 변경하지 않았다.</p>
<p>par 전체 기간은 여전히 캔들·펀딩에 따른 조건부 체결 모형이다. 완전한 과거 호가로 확인한 수익이 아니며, 신호가 없는 기존 관측 호가에 가상의 체결을 추가하지 않았다. Hyperliquid 과거 평가 자료는 반복 사용한 개발 자료로 독립적인 미래 검증이 아니다.</p>
<p>3σ 사건만 남기면 수년의 자료도 학습 사례가 적고, 이 후보의 보정은 예상 이익을 0에 가깝게 만들었다. 다음에는 사건이 발생한 시점만 학습하는 방식을 벗어나, 장기 자료에서 일정 간격의 미래 페어 가격 이동을 직접 학습하고 동일한 비용·체결 구조에서 비교한다. 과거의 짧은 자료로 했던 고정기간 예측과는 학습 자료·페어별 추정의 차이를 명시하고, 예측 오차뿐 아니라 실제 추가 거래의 비용 차감 손익으로 판단한다.</p>'''
    for venue in ['reference','par']:body+='<h2>'+('기준시장 전체 결과' if venue=='reference' else 'par 캔들 모형의 조건부 전체 결과')+'</h2>'+table([x for x in r['rows'] if x['venue']==venue])
    (out/'report.html').write_text(document('장기 자료 페어 예측 비교',body));export_csv(out/'comparison.csv',r['rows'])
    summary=dict(selection=r['selection'],actual_replays=88,prior_par_repair_replays=36,tests_passed=11,existing_controls_exact=True,
        corrected_prior_par_account_results_unchanged=repair['all_old_metrics_fills_snapshots_unchanged'],prediction_summary=prediction_records,
        model_coefficients_unchanged_by_mapping_repair=True,maximum_ownership_error=max(x['metrics']['maximum_ownership_error'] for x in r['rows']),
        protected=guards,orders_sent=0,automatic_live_replacement=False,historical_independence=False)
    save(out/'summary.json',summary)
    save(HERE/'work_state.json',dict(status='progress_longer_history_models_and_identity_mapping_verified_no_alpha',previous_goal_turn='progress',
        report=str((out/'report.html').relative_to(ROOT)),authoritative_results='artifacts_v2',actual_replays=88,prior_par_repair_replays=36,tests_passed=11,
        orders_sent=0,live_strategy_changed=False,
        findings=['Per-pair models trained on Binance price events matured before March2026, without Hyperliquid labels or venue splicing.',
            'BTC/ETH 111 events support ridge; BTC/HYPE 17 use declared scalar fallback.',
            'Sparse-event predictions remain below the fixed cost threshold: maximum 7.685bp versus 38bp round-trip base cost plus 5bp buffer.',
            'Both learned candidates produce no additional pair fills and equal native baselines; no improvement.',
            'Source-audit RMSE is nearly the same as zero-return forecast; lower error than full convergence is not alpha.',
            'Pair-column identity bug caught before new comparisons; v2 resolves by asset IDs and correct source snapshots.',
            'Previous event-calibration par mapping repaired; all 36 account outputs remain identical.'],
        next='Use older history for dense, source-anchored fixed-horizon paired price forecasts, fitted separately per pair with chronological calibration. Do not optimize only on sparse 3-sigma events. Audit prior fixed-horizon families, preserve all outcomes and exact shared-execution controls, evaluate additional trades and costs rather than RMSE alone.',
        remaining=['No candidate improves PnL/Q under required comparisons','Actual full-period par book profitability unverified','No independent prospective improvement evidence']))
    print(json.dumps(summary,ensure_ascii=False));print(out/'report.html')


if __name__=='__main__':main()
