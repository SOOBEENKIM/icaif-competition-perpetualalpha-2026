"""Auditable results; never converts a research winner into a deployment."""
import gzip,html,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,INPUT,read,save,sha
from .scaled_study import verify
from .candidate import snapshot

OUT=ROOT/'reports/pair_value_learning_20261023'
BUNDLE=ROOT/'deliverables/sfinx_coherent_pairs_candidate_20261023'


def table(rows,columns):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label,_ in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(key)))+'</td>' for key,_,fmt in columns)+'</tr>' for r in rows)+'</tbody></table>'


def main():
    verify()
    from .blend_study import verify as vb
    from .pas_blend import verify as vp
    vb();vp()
    def rows(folder):return [read(p) for p in sorted((folder/'rows').glob('*.json'))]
    incumbent=[r for r in read(ROOT/'reports/auto_pair_selection_20261023/reference_results.json') if r['arm']=='incumbent_fixed_two']
    adaptive=[r for r in rows(ROOT/'reports/pair_model_research_20261023/wave9') if r['arm']=='adaptive_hedge_72']
    scaled=[r for r in rows(OUT/'scaled_coherence') if r['arm']=='coherent_scaled_72']
    compare=incumbent+adaptive+scaled
    families=['','fixed_maturity_v2','net_value','sma_calibration','coherence','scaled_coherence','trend_blend']
    allreference=[]
    for family in families:
        for r in rows(OUT/family):allreference.append(dict(family=family or 'hierarchical_learning',**r))
    pas=[]
    for family in ['pas_v2','pas_coherence','pas84','pas_blend']:
        for r in rows(OUT/family):pas.append(dict(family=family,**r))
    assert len(allreference)==126 and len(pas)==54
    assert all(r['timing_pass'] and r['accounting_error']<1e-6 for r in allreference+pas)
    rejected=[r for r in allreference+pas if r['risk_breach_bars']!=0]
    assert all(r['risk_breach_bars']==0 for r in scaled)
    save(OUT/'reference_comparison.json',compare);save(OUT/'all_reference_results.json',allreference);save(OUT/'all_pas_results.json',pas)
    diagnostic=read(OUT/'forecast_diagnostics.json')
    BUNDLE.mkdir(parents=True,exist_ok=True)
    raw=dict(np.load(INPUT));t=1792713600;n=int(np.searchsorted(raw['market_times'],t))+1
    prefix={k:(v[:n] if v.ndim and len(v)==len(raw['market_times']) else v) for k,v in raw.items()}
    example=snapshot(prefix,t,t+10);example['receipt_is_historical_fixture']=True;example['not_current_trading_signal']=True
    save(BUNDLE/'historical_snapshot_example.json',example)
    selected=OUT/'scaled_coherence/coherent_scaled_72.npz';shutil.copy2(selected,BUNDLE/selected.name)
    paths=[ROOT/'pair_value_learning/candidate.py',ROOT/'pair_value_learning/scaled_coherence.py',ROOT/'pair_model_research/candidate.py',ROOT/'pair_model_research/adaptive_hedge.py',ROOT/'auto_pair_selection/selection.py',ROOT/'pair_regime_research/replay.py',ROOT/'pair_value_learning/test_candidate.py']
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate='coherent_scaled_72',status='reference_development_improvement_not_live_approved',
       source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},model_sha256=sha(selected),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),
       required_workspace=str(ROOT),standalone_distribution=False,entrypoint='pair_value_learning.candidate.snapshot',
       evaluation='2026-08-01T12:00:00Z/2026-10-15T12:00:00Z, reference market, already reused development',
       parameters=dict(universe_pairs=45,horizon_hours=72,fit_window_hours=1440,fit_time='00:00 UTC daily',sleeve_capital=50000,idle_reserve=50000,pair_gross_cap=6000),
       reference_results=scaled,independent_test=False,peer_superiority_proven=False,PAS_improvement_proven=False,
       orders_sent=0,live_connected=False,live_code_changed=False,
       check_command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest pair_value_learning.test_candidate pair_value_learning.test_scaled_coherence',
       promotion_requirements='Do not plug into old24h/fixedtwo executor. Explicit72h/all45 coordinator, venueprice/funding/legliquidity, ownership and paid lifecycle adapter would be required. This bundle does not send orders.')
    save(BUNDLE/'manifest.json',manifest)
    save(OUT/'verification.json',dict(reference_runs=126,PAS_runs=54,unit_tests=23,all_completed_runs_accounting_timing_pass=True,
         nominated_candidate_all_risk_checks_pass=True,rejected_risk_breach_runs=rejected,
         baseline_full_path_parity=read(OUT/'fixed_maturity_v2/parity.json'),funding_settlement_validation=read(OUT/'pas_v2/validation.json'),
         adapter_exact_parity_source_count=5,active_source_and_configuration_hashes_unchanged=True,orders_sent=0))
    num=lambda x:f'{x:,.2f}' if x is not None else '—'
    pct=lambda x:f'{x*100:.4f}%' if x is not None else '—'
    q=lambda x:f'{x:.6f}' if x is not None else '—'
    cols=[('arm','전략',str),('scenario','비용 조건',str),('net_pnl','순손익 USDP',num),('return_fraction','계좌 수익률',pct),('volatility','기간 변동성 σ',pct),('Q','Q',q),('max_drawdown','최대낙폭',pct),('completed_pairs','완료 페어',str),('risk_breach_bars','위험 초과 시점',str)]
    phases=[('phase','구간',str)]+cols
    whole=[r for r in compare if r['phase']=='continuous']
    pascols=[('family','실험',str),('arm','전략',str),('scenario','조건',str),('net_pnl','순손익 USDP',num),('completed_pairs','완료 페어',str),('fills','체결 조각',str)]
    forecastrows=[dict(arm=a,**r) for a,r in diagnostic['metrics'].items()]
    forecastcols=[('arm','예측',str),('n','중복 포함 사례',str),('MAE_bps','MAE bp',num),('direction_accuracy','방향 적중률',pct),('source_risk_weighted_MSE','위험 표준화 MSE',q),('selected_mean_price_return_minus_38bp','선별 신호 평균 가격손익−38bp',num)]
    body='''<h1>페어 예측·비용·체결 검증 결과</h1>
<p>작성: 2026-10-23. 기준시장 백테스트 개선 후보를 확보했지만, PAS에서의 성과 개선과 타 팀 대비 우위는 입증하지 못했다. 실거래 전략·주문·포지션은 변경하지 않았다.</p>
<h2>실제로 개선된 결과</h2><p>주 후보 <b>coherent_scaled_72</b>: 45개 관계의 72시간 상대수익 예측을 각 관계의 과거 변동성으로 표준화하고, 동일한 10개 종목 수익률로 설명할 수 있도록 함께 보정한다. 기존 SMA 헤지·AR(1) 평균회귀 예측이 입력이며, 새 대규모 사전학습 모델을 사용한 것은 아니다.</p>
<p>2026-08-01 12:00 UTC → 10-15 12:00 UTC. 우리 몫 50,000 + 유휴 50,000 USDP, 페어 명목 한도 6,000, 같은 방향성 기본 전략·위험 한도·수수료·펀딩·체결 지연. 자본 한도를 높여 만든 차이가 아니다. 기존 운용판은 고정 두 페어 모듈, 이전 연구판은 all45 adaptive72다.</p>'''+table(whole,cols)+'''
<p>신규 후보의 기본 조건 순손익 1,484.62는 가격 평가손익을 포함한다. 종료 잔여분 청산 비용 추정까지 차감하면 1,450.25 USDP다. 불리한 조건은 각각 821.39 / 764.66 USDP다. 전체 기본 조건에서 완료 페어는 14건뿐이다. 기간 σ는 일별 수익률 표본표준편차×√75이며, 일별 표준편차 자체나 연율 변동성이 아니다.</p>
<p>기본 조건: 편도 수수료9bp+영향10bp, 거래량 참여1%. 불리한 조건:12bp+20bp,참여0.25%,양의 펀딩 수취50%. 입·출구 모두 비용을 반영하고 독립 체결·미체결·한쪽 체결 후 감축을 모사한다. 비용이 커지면 선택·체결 경로도 달라져 최종 손익이 반드시 단조 감소하지 않는다.</p>
<h2>불리한 결과까지 포함한 같은 조건 비교</h2>'''+table(compare,phases)+'''
<p>신규 후보는 이전 all45 연구판의 전체 스트레스 결과(1,579.19 / Q0.255156)보다 낮다. 후반 비교 구간 스트레스에서는 −49.17 USDP로 기존 운용판의 +56.64보다 낮다. 따라서 모든 구간 우위가 아니다. 세 구간은 각각 포지션·현금을 초기화한 별도 재생이며, 연속 구간은 중간 보유분을 이어가므로 두 분할 손익의 합과 다를 수 있다.</p>
<h2>예측 자체의 개선과 남은 한계</h2>'''+table(forecastrows,forecastcols)+'''
<p>동일한 1,729개 시간 원점, 42,258개 사용 가능한 페어 예측. 72시간 정답이 평가 종료 전에 완성된 사례만 사용했다. 중첩된 정답이므로 42,258건의 독립 거래 표본이 아니다. 마지막 열은 각 예측 부호로 투자한 가격수익에서38bp만 차감한 진단치로, 펀딩·실제 체결·포트폴리오 손익과 다르다. 여전히 음수다. “비용을 넘는 예측력이 해결됐다”고 결론낼 수 없다.</p>
<p>source β를 고정하면 상대수익 y=W r이며, W의 페어 행은 (r_A−βr_B)/(1+β)다. D=diag(과거 변동성×√72), A=D⁻¹W, P=A A⁺로 두고 μ*=D P D⁻¹μ를 계산한다. 모든 동시 유효 페어에서 정답이 같은 가격 벡터에서 계산되면 ||D⁻¹(μ−y)||²−||D⁻¹(μ*−y)||²=||(I−P)D⁻¹μ||²≥0이다. 실제 자료와 합성 자료에서 확인했다. 이것은 선택된 일부 거래의 적중률·순손익·통계적 우위를 보장하지 않는다. 원래 관계 적격성 필터도 공적분 입증이나 미래 회귀 보장이 아니다.</p>
<h2>실제 PAS 가격 자료의 별도 검증</h2><p>2026-10-19 12:00 UTC → 10-23 00:00 UTC(한국시간 19일21시→23일09시),84시간. 완료된 PAS 시간봉·실제 펀딩 지급 시각·거래량으로 페어 모듈만 비교했다. 모든 후보에서 방향성 기본 전략은 제외했으므로 계좌 전체 성과가 아니다. 과거 L2·대기열·오라클 전체 시계열이 없어 정확한 체결 재현이 아닌 조건부 재생이다. 펀딩 평가가격은 지급 시점 기준시장 가격을 대용했다.</p>'''+table([r for r in pas if r['family'] in ['pas84','pas_blend']],pascols)+'''
<p>새 보정 후보도 기존 all45와 같은 HYPE/BRENT 진입을 선택해 기본 조건−6.85,스트레스−2.15 USDP였다. 완료 페어1건. 기준시장 개선이 PAS에서는 재현되지 않았다. 추세만 사용하는 대안은 기본+3.04,스트레스+0.46이지만, 기준시장 전체에서는+247.84/+170.22로 기존 운용판보다 낮았다. 좋은 구간만 골라 승자로 만들지 않았다.</p>
<p>관측된 한쪽 체결로 HYPE 3.87 중2.19를 다시 감축했고, HYPE/BRENT 청산에 여러 시간봉이 필요했다. 이는 이 재생 가정 안에서 확인한 실행 손실이며 실거래에서 같은 주문이 있었다는 뜻이 아니다. 오래 보유한 페어도 비용을 넘는 방향 예측을 하지 못했다. 단순 종목 확장이나 자금 확대만으로 해결되지 않는다.</p>
<p><a href="https://perpetual-alpha26.github.io/leaderboard">현재 공식 페이지</a>는 순위·점수가12:00 UTC 일별 표본을 사용하고 장중 PnL 차트와 다를 수 있다고 명시한다. pas84 원본행의 장중 Q는 가상 계산치이며 공식 순위 점수가 아니다. 완료일 기준 별도 재계산은 pas84/score_audit.json에 남겼다. 본문의 PAS 표에는 혼동을 피하려 Q를 표시하지 않았다. 기준시장 평가 종료는 모두12:00 UTC이므로 위 Q 비교는 이 구분에 영향받지 않는다.</p>
<h2>실험·실패·검증 기록</h2><p>이번 작업에서 완료한 기준시장 비교126건, PAS 조건부 비교54건. 서로 독립인 연구180건이 아니라 여러 후보×기간×비용 조건이다. 기존에 살펴본 과거 구간을 반복 사용했으므로 개발 성과다. 각 변경은 실행 전 코드·프로토콜·자료 해시를 고정했지만, 이것만으로 후보 선택의 과적합이 제거되지는 않는다. 독립 성과 검증이나 동료 팀보다 높은 등수라는 주장에 사용할 수 없다.</p>
<p>계층 릿지 수익 예측, 보유기간 고정, 가격-only 진입 필터와 펀딩 가치 분리, 과거 정답을 이용한 AR 크기 보정, 일반/위험 표준화 일관성 보정, 고정50:50추세 혼합을 비교했다. 학습 정답은 예측 시점 전에 만기가 끝난 것만 사용했다. 실패한 후보도 아래 원장에 보존했다. 초기 보유기간 실험의 감축 후 장부 앵커 오류, 한 시간에 두 펀딩 지급이 들어온 경우의 처리 오류는 연구 엔진에서 수정하고 별도 버전으로 다시 실행했다.</p>
<p>23개 테스트 통과. 백테스트와 후보 인터페이스가5개 실제 과거 입력 접두구간에서 동일한 예측·헤지·진입 방향을 출력했다. 미래 입력과 오래된 수신 시점은 거부한다. 회계 오차·시점 순서는 완료한 모든 비교에서 통과했다. 주 후보의6개 기준시장 조건과 모든 PAS 재생은 위험 한도를 통과했다. 탈락한 coherent_hierarchical_72는4개 조건에서8월19일20:00 UTC 일별 예상변동성 한도를 한 시점씩 초과했으며, 손실 성과와 함께 탈락 사유로 보존했다. 펀딩은 지급 건별 수취 할인 후 합산한다. 기존 경로 전체 이벤트·체결·잔고 동일성 검사도 통과했다.</p>
<p><a href="https://arxiv.org/html/2606.00060v1">2026년 비용을 반영한 워크포워드 예측 연구</a>는 예측 개선과 거래 순이익 사이의 차이, 비용 기반 거래 필터의 필요성을 참고하는 데 사용했다. 해당 논문의 모델·수익률을 재현했다는 의미는 아니다. 위 선형대수 보정의 수익성은 우리 자료에서 따로 검증한 것이다.</p>
<h2>검토 가능한 코드·산출물</h2><ul>
<li>후보 진입점: pair_value_learning/candidate.py → snapshot(data, source, received_at). 네트워크·키·주문 전송 없음.</li>
<li>핵심 보정: pair_value_learning/scaled_coherence.py. 부모 모형: pair_model_research/adaptive_hedge.py.</li>
<li>실험: pair_value_learning/scaled_study.py, pas_extension.py, blend_study.py, pas_blend.py.</li>
<li>후보 묶음: deliverables/sfinx_coherent_pairs_candidate_20261023/manifest.json. 모델 상태·재현 명령·소스/자료 해시·과거 입력 예제 포함. 현재 작업공간 의존 묶음이며 독립 배포본이 아니다.</li>
<li>전체 결과: all_reference_results.json, all_pas_results.json. 개별 실행 원본은 각 하위 폴더의 reference/ 또는 runs/에 압축 저장.</li>
<li>검증: tests.log, verification.json, forecast_diagnostics.json. 기존 실전 소스·설정 해시는 그대로다.</li></ul>
<p><b>판정:</b> 기존 운용판 대비 기준시장 전체 구간의 PnL·Q 개선을 확인한 연구 후보는 확보했다. 비용을 넘는 방향·크기 예측의 일반적 해결, PAS 개선, 타 팀 대비 우위는 미확인이다. 현재 실전 실행자에 연결하지 않았다.</p>
<details><summary>기준시장 전체 실험 원장126행</summary>'''+table(allreference,[('family','실험군',str)]+phases)+'''</details>
<details><summary>PAS 전체 실험 원장54행</summary>'''+table(pas,pascols)+'''</details>'''
    style='body{font:16px/1.65 system-ui,sans-serif;max-width:1400px;margin:40px auto;padding:0 24px;color:#203040}table{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0}th,td{padding:8px;border:1px solid #dce3eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef3f8}h1,h2{color:#153b59}details{overflow:auto;margin:24px 0}a{color:#176aa5}'
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 예측 개선 검증</title><style>'+style+'</style><body>'+body+'</body></html>')
    verify();print('Report and candidate bundle written;126 reference /54 PAS runs;live hashes unchanged')


if __name__=='__main__':main()
