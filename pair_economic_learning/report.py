"""Report every completed experiment and nominate only its demonstrated scope."""
import gzip,html,json,shutil,importlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from pair_value_learning.report import table
from .candidate import research_contract,snapshot

OUT=ROOT/'reports/pair_economic_learning_20261023'
BUNDLE=ROOT/'deliverables/sfinx_coherent_pairs_only_20261023'
WAVES=['wave1','wave2','wave3','wave4','wave5','wave6','wave7b','wave8','wave9','wave10']
MODULES=['study','meta_study','execution_study','reliability_study','combined_study','portfolio_study','fast_regime_study_v2','ranking_study','realized_study','value_study','pas_full_comparison']


def verify():
    for name in MODULES:importlib.import_module('pair_economic_learning.'+name).verify()
    for wave in WAVES:
        path=OUT/wave/'pas/freeze.json'
        if path.exists():
            for relative,digest in read(path).items():assert sha(ROOT/relative)==digest,relative


def collect(wave,sub='rows'):
    return [dict(wave=wave,**read(p)) for p in sorted((OUT/wave/sub).glob('*.json'))]


def main():
    verify()
    references=[r for w in WAVES for r in collect(w)]
    pas=[r for w in WAVES for r in collect(w,'pas/rows')]
    full=collect('pas_full48')
    assert len(references)==144 and len(pas)==132 and len(full)==9
    assert all(r['timing_pass'] and r['accounting_error']<1e-6 and r['risk_breach_bars']==0 for r in references+pas+full)
    assert read(OUT/'wave7/INVALIDATED.json')['valid_for_selection'] is False
    old=[r for r in read(ROOT/'reports/auto_pair_selection_20261023/reference_results.json') if r['arm']=='incumbent_fixed_two']
    previous=[read(p) for p in (ROOT/'reports/pair_value_learning_20261023/scaled_coherence/rows').glob('coherent_scaled_72_*.json')]
    candidate=[r for r in references if r['arm']=='unbounded_pairs_72']
    assert len(candidate)==len(previous)==len(old)==6
    matched=[]
    for r in candidate:
        o=next(x for x in old if (x['phase'],x['scenario'])==(r['phase'],r['scenario']))
        assert r['net_pnl']>o['net_pnl'] and r['Q']>o['Q']
        matched.append(dict(phase=r['phase'],scenario=r['scenario'],old_pnl=o['net_pnl'],new_pnl=r['net_pnl'],pnl_delta=r['net_pnl']-o['net_pnl'],old_Q=o['Q'],new_Q=r['Q']))
        trace=OUT/'wave3/reference'/f"unbounded_pairs_72_{r['phase']}_{r['scenario']}.json.gz"
        with gzip.open(trace,'rt') as fp:tr=json.load(fp)
        assert not tr['owner_fills']['base'] and all(not any(x['base_qty']) for x in tr['snapshots'])
        assert abs(tr['metrics']['net_pnl']-tr['metrics']['pair_owned_pnl'])<1e-6
    save(OUT/'reference_comparison.json',old+previous+candidate)
    save(OUT/'matched_improvement.json',matched)
    save(OUT/'all_reference_results.json',references);save(OUT/'all_pas_results.json',pas)
    save(OUT/'pas_full_strategy_results.json',full)
    BUNDLE.mkdir(parents=True,exist_ok=True)
    selected=OUT/'wave3/unbounded_pairs_72.npz';shutil.copy2(selected,BUNDLE/'forecast_state.npz')
    contract=research_contract();save(BUNDLE/'research_contract.json',contract)
    raw=dict(np.load(INPUT));source=1792713600;n=int(np.searchsorted(raw['market_times'],source))+1
    prefix={k:(v[:n] if v.ndim and len(v)==len(raw['market_times']) else v) for k,v in raw.items()}
    example=snapshot(prefix,source,source+10);example.update(historical_fixture=True,current_trading_signal=False)
    save(BUNDLE/'historical_snapshot.json',example)
    sources=[ROOT/'pair_economic_learning/candidate.py',ROOT/'pair_economic_learning/reference.py',ROOT/'pair_economic_learning/test_final_candidate.py',ROOT/'pair_value_learning/candidate.py',ROOT/'pair_value_learning/scaled_coherence.py',ROOT/'pair_model_research/candidate.py',ROOT/'pair_model_research/adaptive_hedge.py',ROOT/'pair_regime_research/replay.py',ROOT/'competition_live/config.json']
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate=contract['candidate'],cache_arm='unbounded_pairs_72',
        status='reference_development_improvement_only',standalone_distribution=False,workspace=str(ROOT),
        entrypoint='pair_economic_learning.candidate.snapshot',execution_contract='pair_economic_learning.candidate.research_contract',
        reference_passed_conditions=6,reference_results=candidate,comparison=matched,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),
        artifact_sha256={p.name:sha(p) for p in [BUNDLE/'forecast_state.npz',BUNDLE/'research_contract.json',BUNDLE/'historical_snapshot.json']},
        retrain_required=False,learned_execution_models_selected=False,
        selection='Post-comparison nomination of wave3 secondary ablation among24 development variants; not independent validation.',
        PAS84h_profitable=False,PAS48h_Q_improved=False,peer_superiority_proven=False,live_promoted=False,orders_sent=0,
        live_adapter_note='No live coordinator or ownership handoff is installed. The old24h fixedtwo runner must not be pointed at this72h45pair artifact without explicit integration.',
        reproduce='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pair_economic_learning.execution_study run',
        check='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest pair_economic_learning.test_final_candidate')
    save(BUNDLE/'manifest.json',manifest)
    audit=dict(valid_reference_runs=len(references),valid_PAS_module_runs=len(pas),valid_full_strategy_PAS_runs=len(full),
        invalidated_prototype_runs=12,invalidated_reason=read(OUT/'wave7/INVALIDATED.json'),
        all_valid_runs_accounting_timing_risk_pass=True,reference_all_six_PnL_Q_improved=True,
        active_source_and_config_hashes_unchanged=True,orders_sent=0,
        training=read(OUT/'wave9/label_audit.json'),
        baseline_parity=read(OUT/'wave8/parity.json'),single_cohort_parity=read(OUT/'wave6/parity.json'),
        tests=read(OUT/'tests.json'))
    save(OUT/'verification.json',audit)
    number=lambda x:f'{x:,.2f}' if x is not None else '—'
    pct=lambda x:f'{x*100:.4f}%' if x is not None else '—'
    score=lambda x:f'{x:.6f}' if x is not None else '—'
    cols=[('arm','후보',str),('phase','구간',str),('scenario','비용',str),('net_pnl','순손익',number),('terminal_pnl','청산비용 추정 차감',number),('return_fraction','계좌 수익률',pct),('volatility','기간 σ',pct),('Q','Q',score),('max_drawdown','최대낙폭',pct),('completed_pairs','완료 페어',str)]
    comparisons=[dict(r,arm={'unbounded_pairs_72':'새 페어 전용','coherent_scaled_72':'직전 연구 후보','incumbent_fixed_two':'현재 운용 구조 재생'}[r['arm']]) for r in old+previous+candidate]
    pc=[('arm','후보',str),('anchor','가격 기준',str),('scenario','비용',str),('net_pnl','순손익',number),('terminal_pnl','청산비용 추정 차감',number),('Q','완료일 기준 Q',score),('completed_pairs','완료 페어',str)]
    bestpas=[r for r in pas if r['arm']=='unbounded_pairs_72']
    body='''<h1>45개 페어 전략: 구성 개선과 실행 손익 학습 검증</h1>
<p><b>검증된 개선:</b> 기존과 같은 자본·기간·비용·위험 한도에서 <b>페어 전용 후보</b>가 6개 기준시장 비교 모두 순손익과 Q를 높였다. 75일 기본 조건 +1,595.52 USDP / Q0.265933, 스트레스 +1,098.89 / Q0.230751. 현재 운용 구조의 같은 재생은 +529.07 / Q0.151217, +299.83 / Q0.123253이다.</p>
<p><b>검증 범위:</b> 반복 사용한 과거 자료의 개발 성과다. PAS84시간 모의 재생은 여전히 손실이며, 타 팀보다 좋은 실전 성과나 모든 문제의 해결을 입증하지 않는다. 실전 프로그램·주문·포지션은 변경하지 않았다.</p>
<h2>무엇을 바꿨는가</h2>
<p>10개 종목의45개 조합을 매일 평가한다. 과거60일로 대칭 헤지 비율과 AR(1) 평균회귀 관계를 추정하고,72시간 상대수익을 예측한다. 각 관계의 과거 위험으로 표준화한 뒤, 동일한10개 종목 수익률로 설명되는 예측으로 일관성을 보정한다. 관계·자료가 부적격이면 건너뛰고, 적격한 관계도 예상수익이 실제 비용·펀딩·위험 제약을 넘을 때만 양방향 거래 후보가 된다.</p>
<p>이번에 선택한 변경은 이 예측에 별도로 섞었던 방향성 매매를 제거하는 것이다. 직전 연구 후보와 예측값·헤지·진입가격 한도·자본 한도는 같고, 방향성 기본 포지션과 페어 포지션이 같은 계좌에서 영향을 주는 경로를 없앴다. 예측 모델 자체가 추가로 좋아졌다는 주장은 아니다. 같은 예측에 복잡한 학습 모듈을 붙인 대안들도 비교했지만 선택하지 않았다.</p>
<p>페어는 한 번에 하나만 보유하며72시간 원래 만기를 갖는다. 경제성이 유지되고 위험 조건을 만족하는 경우 비용을 반영해 연장·헤지 조정할 수 있으며, 최초 보유 후168시간 한도를 둔다. 한쪽만 체결되면 한 차례 보완 또는 유료 감축·청산을 수행한다. 수익을 보장하는 무위험 차익거래가 아니다.</p>
<h2>같은 조건의 결과</h2>
<p>2026-08-01 12:00 UTC~10-15 12:00 UTC,75일. 우리 몫50,000 + 유휴50,000 USDP. 페어 명목 한도6,000 유지. 수익률·Q는 팀 전체100,000을 분모로 계산했다. σ는 일별 수익률 표본표준편차×√75이며 일별 표준편차 자체가 아니다. Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), J=0이면 점수0.8Q.</p>'''+table([r for r in comparisons if r['phase']=='continuous'],cols)+'''
<p>기본 조건은 편도 수수료9bp+가격 영향10bp, 거래량 참여1%. 스트레스는12bp+20bp,참여0.25%,양의 펀딩 수취50%. 진입과 청산 모두 비용을 부담한다. 비용 조건이 달라지면 진입·체결 경로가 달라지므로 손익의 단조 관계를 가정하지 않는다. 종료 보유분의 비용 차감은 추정치이며 실제 청산 완료로 세지 않았다.</p>
<p>새 후보의 기본 조건 평균 명목 투자액은5,682.41 USDP, 기존은13,978.95다. 더 크게 투자해서 만든 개선이 아니다. 새 후보는BTC·ETH·HYPE·EWY·BRENT·GOLD에 실제 모의 체결이 있었고,45개 모두 평가하지만10개 전부 보유하도록 강제하지 않는다. 전체 완료 페어14건, 숏 신규 체결 조각17개. 기본 σ는 기존0.8508%→0.8562%로 조금 높아졌고 최대낙폭은0.6850%→0.3441%로 감소했다. 스트레스 σ는0.7930%→0.5535%다.</p>
<h2>앞·뒤 구간도 함께 공개</h2>'''+table(comparisons,cols)+'''
<p>validation=8월1일~9월1일,comparison=9월1일~10월15일. 각각 자본·포지션을 초기화한 별도 재생이다. continuous는 보유분을 이어가므로 분할 손익 합과 다르다. 세 구간은 이미 반복 사용했으며 명칭validation/comparison을 미사용 검증으로 해석하면 안 된다. 새 후보는wave3의 사전 정의된 보조 후보를 전체 비교 후 선택한 것이며, 최초 주 후보였던10bp 추가 진입 제한안은 채택하지 않았다.</p>
<h2>PAS에서 무엇까지 확인됐는가</h2>
<p>10월19일12UTC~23일00UTC의84시간. 실제 PAS 완료 시간봉·거래량·펀딩 지급 자료로 동일한 페어 모듈을 재생했다. 과거 L2·주문 대기열과 오라클 전체 기록은 없어서 정확한 실거래 체결 재현은 아니다. 펀딩 평가가격은 기준시장 가격을 대용했다. 아래는 계좌 실적이 아닌 조건부 모의 결과다.</p>'''+table(bestpas,pc)+'''
<p>주 후보의 기본−6.85 / 스트레스−2.15 USDP, 완료 페어1건. 기준시장 예측 개선을 PAS 수익 개선으로 바꿔 말하지 않는다.48시간 평가만 보고 종료 시점을 선택해서 이 손실을 숨길 수 없다.</p>
<p>전체 전략과의 비교는 기존 기본 전략의 동결 자료가10월21일14UTC까지만 있어 마지막 공통 일별 표본인21일12UTC에서 끝냈다. 모든 계좌를19일12UTC에50,000으로 새로 시작했다. PAS 가격 이력의3h/24h 조건과 자료 신선도를 사용했으며 출시 전 가격을 만들어 넣지 않았다.72시간 페어의 청산이 끝나지 않은48시간 결과다.</p>'''+table(full,[c for c in cols if c[0]!='phase'])+'''
<p>이48시간에는 순손익이 개선됐지만 Q는 낮아졌다. 기본 조건 기존−2.99 / Q0.099888, 페어 전용+1.02 / Q0.099690. 이것을 종합 점수 개선으로 주장하지 않는다. 해당 기간에 후보로 실제 주문했다는 뜻도 아니다.</p>
<h2>실행 비용을 학습하도록 바꾼 실험</h2>
<p>단순 미래 가격 오차뿐 아니라 실제 모의 실행 순손익을 학습하도록 구현했다.147개 일별 원점에서3,490개 페어·비용 조건별 에피소드를 독립적으로 재생했다. 한쪽 체결558건, 불균형 복구343건의 비용을 포함했다. 96시간 종료 후 잔여 포지션이19건 있어 청산 비용 추정치를 별도로 차감했다.72시간 가격 목표가 끝나도96시간 실행 정답이 끝나기 전에는 학습에 쓰지 않는다.</p>
<p>과거90일, 매일 재학습,30일 감쇠, 자산 이름을 넣지 않은19개 특징, 깊이2/64라운드 고정모형을 사용했다. 실행 손익 순위 학습과 절대 손익 회귀·양수 진입 필터를 비교했고, 불리한 비용의 최소 손익을 목표로 하는 대안도 공개했다. 개선되지 않은 모델을 최종 후보에 억지로 넣지 않았다. 학습은 고립된72시간 거래이고 포트폴리오는 연장·자본 경쟁이 있으므로 이 차이도 남아 있다.</p>
<p>미래 실행 정답을10억으로 바꿔도 이전 일자의 모델 바이트와 예측이 동일한 검사를 통과했다. 입력 특징의 미래 가격 변경 불변성, source 고정 헤지 정답, 원점별 가중치, 실제 입력 접두구간5개와 후보 출력 일치, 기존 실행 경로 전체 동일성도 검사했다. 이런 검사는 직접 누출을 막지만 반복 실험에 따른 모델 선택 과적합을 제거하지 않는다.</p>
<h2>전체 실험과 오류 기록</h2>
<p>유효 기준시장144회, PAS 모듈132회, 전체 전략 PAS9회. 독립된285개 실험이 아니라 후보×기간×비용 조건 재생이다. 모든 완료 유효 실행은 회계·시점 순서·위험 한도 검사를 통과했다. 초기wave7의 미래 시점 모델 적격성으로 학습 사례를 거르던 오류는12개 결과를 무효화하고wave7b로 수정·재실행했다. 무효 결과는 승자 선정에 쓰지 않았고 원본을 남겼다.</p>
<p>비교한 것은 비선형 수익 예측, 온라인 평균회귀/추세 혼합, 비용 조건부 진입, 진입가격 추가 제한, 방향성 모듈 제거, 관계 신뢰도 감쇠, 단기 방향 확인,3개 동시 페어 분산, 짧은 만기 정답을 이용한 국면 선택, 기회 순위 학습, 실제 실행 손익 순위·회귀다. 자본·레버리지를 높이거나 특정 자산 이름·PAS 결과를 학습 정답에 넣어 수익을 만든 실험은 아니다.</p>
<h2>산출물과 재현</h2>
<ul><li>후보: deliverables/sfinx_coherent_pairs_only_20261023/manifest.json — 코드·자료 해시, 고정 예측 상태, 명시적 페어 전용 설정, 과거 입력 예제.</li>
<li>인터페이스: pair_economic_learning/candidate.py. 입력 시점 검증을 포함하며 네트워크·서명·주문 전송 없음. 별도 라이브 실행자를 설치한 것이 아니다.</li>
<li>전체 비교: all_reference_results.json, all_pas_results.json, pas_full_strategy_results.json, matched_improvement.json.</li>
<li>실행 학습: realized_labels.py, realized_ranker.py, execution_value.py. 원본 모델·프로토콜·실패 결과 모두 각wave 폴더에 보존.</li>
<li>검증: tests.log, tests.json, verification.json. 기존 실전 소스와 설정 해시는 동일하다.</li></ul>
<p><b>판정:</b> 같은 조건의 기준시장 PnL·Q가 개선된 후보를 구현·재현했다. PAS 실전 성과 개선 또는 타 팀 우위가 검증된 최종판으로 승인하지 않았다. 사용자가 요청한 “남은 문제 없이 남들보다 잘 되는 전략”이라고 보장할 근거는 없다.</p>
<details><summary>기준시장 전체144행</summary>'''+table(references,[('wave','실험군',str)]+cols)+'''</details>
<details><summary>PAS 전체132행</summary>'''+table(pas,[('wave','실험군',str)]+pc)+'''</details>'''
    style='body{font:16px/1.65 system-ui,sans-serif;max-width:1450px;margin:40px auto;padding:0 24px;color:#24364b}h1,h2{color:#193d58}table{border-collapse:collapse;width:100%;font-size:12px;margin:20px 0}th,td{border:1px solid #d9e2eb;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef4f8}details{overflow:auto;margin:24px 0}'
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 전용 개선 결과</title><style>'+style+'</style><body>'+body+'</body></html>')
    shutil.copy2(OUT/'report.html',BUNDLE/'report.html')
    verify();print('Report written:144 reference,132 PAS module,9 full PAS,live unchanged',flush=True)


if __name__=='__main__':main()
