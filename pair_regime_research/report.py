"""Publish both frozen experiments including failures; no outcome selection."""
import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from .study import ROOT, OUT, INPUT, read, save, sha, verify
from .diagnostics import OLD, AUDIT, csvsave
from pair_regime_adaptive.study import OUT as SECOND, verify as verify_second
from pair_model_research.adaptive_hedge import fit_pair
from pair_regime_adaptive.model import fit
from auto_pair_selection.selection import IDS, PAIRS

NAMES = {
    'incumbent_fixed_two':'현 운용 구조의 고정 2페어 기준판',
    'adaptive_hedge_72':'기존 45페어 적응형 헤지 후보',
    'ols_72':'가격 OLS 헤지 + 공통 관계 무효 처리',
    'relationship_72':'가격 OLS + 관계 안정성 검사',
    'persistence_72':'기존 헤지 + 추세 지속 경고',
    'stable_regime_72':'수정 1: 관계 안정성 + 추세 경고',
    'regime_choice_72':'수정 2: 수익률 헤지 + 회귀/추세 선택',
    'regime_ensemble_72':'수정 2 앙상블 비교군',
}


def number(x, digits=4):
    return '—' if x is None else f'{x:,.{digits}f}'


def table(headers, rows):
    esc=lambda x:html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def main():
    verify(); verify_second()
    rows=[r for r in read(ROOT/'reports/auto_pair_selection_20261023/reference_results.json') if r['arm']=='incumbent_fixed_two']
    rows += [read(p) for p in sorted((OLD/'rows').glob('adaptive_hedge_72_*.json'))]
    new=[read(p) for folder in [OUT,SECOND] for p in sorted((folder/'rows').glob('*.json'))]
    rows += new
    assert len(new)==36 and len(rows)==48
    assert all(r['timing_pass'] and r['accounting_error']<1e-6 and r['risk_breach_bars']==0 for r in new)
    save(OUT/'all_reference_results.json',rows)
    keys=['arm','phase','scenario','net_pnl','return_fraction','volatility','Q','score','pair_pnl','base_pnl','fees','funding','execution_cost','completed_pairs','fills','risk_breach_bars','accounting_error']
    csvsave(OUT/'all_reference_results.csv',[{k:r[k] for k in keys} for r in rows])
    d=dict(np.load(INPUT));logs=np.log(d['close']);active=(d['volume']>0)&(d['trades']>0)
    source=int(d['market_times'][-1]//86400*86400);i=int(np.searchsorted(d['market_times'],source));facts=[]
    for pair in [('ETH','BRENT'),('EWJ','BRENT'),('BTC','ETH'),('BTC','HYPE')]:
        k=PAIRS.index(pair);old=fit_pair(logs,active,i,IDS[k]);model,audit=fit(logs,active,i,IDS[k])
        stats=audit['return_statistics']
        facts.append(dict(pair='/'.join(pair),source=source,
                          old_level_correlation=old['level_correlation'] if old else None,
                          old_positive_beta=old['beta'] if old else None,
                          return_correlation=stats['correlation'],return_ols_beta=stats['beta'],
                          relationship_allowed=audit['eligible'],reasons=audit['reasons']))
    save(OUT/'confirmed_relationship_defects.json',facts)
    diagnostic=read(OUT/'pas_diagnostics.json');counts=read(OUT/'gate_counts.json');counts2=read(SECOND/'counts.json')
    formations=read(OUT/'formation_audit.json');reason_counts=Counter(reason for day in formations for row in day['pairs'] for reason in row.get('reasons',[]))
    v2=read(SECOND/'formation_regime_audit.json');last=v2[-1];eligible=sum(r['formation']['eligible'] for r in last['pairs'])
    tests=[(OUT/'tests.log').read_text(),(SECOND/'tests.log').read_text()]
    assert 'Ran 13 tests' in tests[0] and 'Ran 7 tests' in tests[1] and all(t.rstrip().endswith('OK') for t in tests)
    assert read(OUT/'parity.json')['exact_full_result_match']
    assert all(read(OUT/'input_prefix_parity.json').values())
    verdict=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        confirmed='The previous all45 research module admitted weak and unstable relationships as mean-reverting hedges. This is a demonstrated model-design weakness, not proof of the sole cause of live account PnL.',
        repair_implemented=True,unit_tests_passed=20,matched_reference_runs=36,
        old_engine_exact_full_reproduction=True,historical_input_prefix_unchanged=True,accounting_timing_risk_checks=True,
        pnl_improvement_established=False,PAS_profitable_prediction_established=False,
        promotion_recommended=False,live_changed=False,orders_sent=0,
        primary_candidates=['stable_regime_72','regime_choice_72'],
        shared_limit='Relationship filtering and risk-reducing hedges do not supply a calibrated, profitable 72h price forecast. Both frozen primary repairs underperformed old candidates.',
        data_reused=True,independent_holdout=False)
    save(OUT/'verdict.json',verdict)
    b='<h1>45개 페어: 관계 안정성·추세 지속 구분 수정과 검증</h1>'
    b+='<p>2026-10-23 연구. 입력은 2026-03-18 14:00 UTC부터 2026-10-23 08:00 UTC까지의 완료된 기준시장 시간봉으로 고정했습니다. PAS 평가는 직전 감사에서 보관한 실제 PAS 완료 시간봉만 사용합니다. 새 실거래 주문은 0건입니다.</p>'
    b+='<p class="note"><b>결론: 구조적 약점은 재현했고 두 수정안을 구현했습니다. 그러나 성능 개선은 입증하지 못했습니다.</b> 관계 검사만 강화하면 수익성이 해결된다는 종전 설명은 과했습니다. 두 수정안은 기존 후보보다 PnL·Q가 낮았고, 실제 PAS 예측 진단에서도 비용을 넘는 성과를 보이지 못했습니다. 실거래 교체 후보로 승인하지 않습니다.</p>'
    b+='<h2>무엇이 확실하고 무엇이 아닌가</h2><p>확인한 대상은 <b>아직 연결하지 않은 45페어 연구 후보</b>입니다. 현재 실거래의 전체 손익을 이 결함 탓으로 단정하지 않습니다. 현재 계좌에는 팀원 주문도 있으며, 계좌 합산 손익과 우리 전략 장부의 손익은 다릅니다. 다른 팀의 수익도 방향 예측, 펀딩, 투자 규모, 체결 비용으로 나누어 봐야 합니다. 직전 공개 지갑 감사에서는 모두 롱인 수익 계정도, 가격 손실을 펀딩 수익이 상쇄한 계정도 확인됐습니다. 따라서 “남들은 모두 롱·숏을 잘 맞춘다”는 주장은 관측 자료로 입증되지 않습니다. <a href="../market_candidate_audit_20261023/report.html">시각이 명시된 전체 시장·지갑 감사</a>.</p>'
    b+='<p>기존 후보는 양수 가격 공분산과 두 가격의 표준편차 비율로 양수 헤지 비중을 만든 뒤 잔차 AR 계수가 1보다 작으면 회귀를 기대했습니다. 약한 가격 상관이나 불안정한 관계를 배제하는 근거가 충분하지 않았습니다. 10월 23일 00:00 UTC 직전 60일 자료에서 다음을 재현했습니다.</p>'
    b+=table(['페어','가격 수준 상관','기존 양수 β','시간 수익률 상관','수익률 OLS β','새 관계 허용','배제 이유'],[[r['pair'],number(r['old_level_correlation'],6),number(r['old_positive_beta'],6),number(r['return_correlation'],6),number(r['return_ols_beta'],6),r['relationship_allowed'],', '.join(r['reasons'])] for r in facts])
    b+='<p>ETH/BRENT와 EWJ/BRENT의 음수 수익률 공분산은 이 모듈의 양수 β·반대 방향 두 다리 헤지가 단기 변동 위험을 줄이는 헤지라는 설명과 맞지 않습니다. 두 종목 이름을 금지한 것이 아니라 같은 수식으로 전 45개를 평가해 탈락시켰습니다. 다만 이것만으로 해당 페어의 모든 거래가 손실이라고 증명되는 것은 아닙니다.</p>'
    b+='<h2>구현한 두 실험</h2><p><b>수정 1:</b> 60일 가격 OLS 헤지, 20일씩 세 구간의 상관·β 안정성, 잔차 평형 이동, Engle–Granger 검정과 하루 45개 Holm 보정으로 관계를 검사했습니다. 6·24·72시간 괴리가 계속 확대되고 예상 회귀 속도를 넘으면 역방향 진입을 막습니다. 관계/추세 위반이 연속한 두 개의 새 시간봉에서 확인될 때 기존 체결 엔진에 유료 청산을 요청합니다. 같은 신호를 반복 읽은 횟수는 세지 않습니다.</p>'
    b+='<p>조건은 결과를 실행하기 전에 고정했습니다: 가격 상관 0.3, β의 세 구간 최대/최소 비율 2, 평형 이동 2σ, Holm 0.05, AR 반감기 6–336시간 등. 이 임계값을 보편적으로 옳다고 주장하지 않습니다. 검정은 I(1) 등 전제에 의존하고, 하루의 Holm 보정은 여러 날 반복 검사 전체의 오류율을 통제하지 않습니다. 합성 공적분 자료에서는 통과하는 페어가 존재했지만, 이번 실제 자료에서 모든 조건을 함께 만족한 진입 신호는 0개였습니다.</p>'
    b+='<p><b>수정 2:</b> 장기 공적분 주장과 별도로, 시간별 수익률의 최소분산 양수 헤지 β를 추정했습니다. 추정 분산 감소 5% 이상, 세 구간 양수 β, 최대/최소 비율 3 이내라는 조건을 적용합니다. 그 뒤 72시간 회귀 예측과 72시간 상대 추세 지속 예측을 함께 생성해 보관합니다. 선택 시점보다 정답이 엄격히 먼저 확정된 과거 예측만 비교합니다. 72시간 간격의 최소 10개 예측, 38bp 비용 대용치 차감 후 양의 성과, 0변화 예측보다 낮은 MAE를 요구합니다. 기준을 충족하는 모델이 없으면 새 진입을 보류합니다.</p>'
    b+='<p>모델 선택에는 당시 이미 생성된 예측과 그 예측 이후 확정된 기준시장 정답만 씁니다. 과거 예측을 나중의 계수로 다시 작성하지 않았습니다. 일일 선택은 직전 60일의 정답 확정 자료와 30일 반감기 가중치를 사용합니다. 두 모델이 모두 통과할 때의 앙상블도 별도로 비교했지만, 실제 입력에서는 두 모델이 함께 통과한 경우가 0건이어서 두 결과가 완전히 같았습니다. 앙상블 효과를 주장할 수 없습니다.</p>'
    b+='<p>기존 보유 포지션의 평가와 청산용 예측은 새 진입 불허와 분리했습니다. 수정 2에서는 단지 과거 모델 우위가 사라졌다는 이유만으로 즉시 유료 청산하지 않습니다. 구조적인 헤지 관계가 무효화되면 연속 두 신호 확인 규칙을 적용합니다. 오래된 입력은 신규 연장 허용 근거로 쓰지 않습니다. 누락을 두 번째 독립 관측으로 세지 않습니다.</p>'
    b+='<h2>동일 조건에서의 포트폴리오 결과</h2><p><b>2026-08-01 12:00 UTC–10-15 12:00 UTC, 기준시장 과거자료 개발 백테스트.</b> 우리 운용분 50,000 + 유휴 50,000으로 계좌 100,000, 페어 총액 최대 6,000, 동시에 한 페어, 동일한 방향 전략·공유 위험 한도·체결 엔진을 적용했습니다. 기본 조건은 편도 수수료 9bp·충격 10bp·거래량 참여율 1%; 불리한 조건은 12bp·20bp·0.25%·받는 펀딩 절반입니다. 표의 R·Q는 100,000 계좌 기준이며, 공식 PAS 실거래 수익률이나 공식 순위가 아닙니다.</p>'
    full=[]
    for arm in NAMES:
        for scenario in ['base','stress']:
            r=next(x for x in rows if x['arm']==arm and x['phase']=='continuous' and x['scenario']==scenario)
            full.append([NAMES[arm],scenario,number(r['net_pnl'],2),number(r['return_fraction']*100,4),number(r['volatility']*100,4),number(r['Q'],6),number(r['score'],6),r['completed_pairs'],number(r['pair_pnl'],2),number(r['base_pnl'],2)])
    b+=table(['구현','조건','순손익 USDP','계좌 R %','기간 σ %','Q','0.8Q (J=0)','완료 페어','페어 장부 손익','방향 장부 손익'],full)
    b+='<p>Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), Score=0.8Q+0.2J입니다. σ는 평가 일별 수익률 표본 표준편차에 √n을 곱한 값입니다. 모든 후보에 같은 계산을 적용했습니다. 수익률이 더 높아졌다고 주장할 수 있는 수정판은 없습니다.</p>'
    b+='<p>수정 1의 완전한 관계 검사는 모든 페어 진입을 막았고, 남은 방향 전략 손익이 −116.57이었습니다. 추세 경고만 추가한 비교군은 기존보다 조기 청산과 비용이 늘었습니다. 수정 2의 페어 장부는 기본 조건에서 +3.43에 그쳤고 전체 포트폴리오는 −101.37입니다. 헤지와 거래 선택 변경은 공유 위험 배분을 통해 방향 전략의 수량에도 영향을 주므로, 표의 장부 차이를 독립적인 인과 효과로 단순 합산해서는 안 됩니다.</p>'
    b+='<p>8월 구간, 9월 1일 이후 구간, 전 기간 연속 구간 각각에 기본/불리한 체결 조건을 적용했습니다. 새 6개 비교군×6조건=36회와 이전 기준판 12회를 모두 보존했습니다. 수정 2의 앙상블은 중복 결과이며 독립적인 추가 증거가 아닙니다. “OLS만 변경” 비교군도 공통 관계 무효 처리의 영향을 받으므로 순수 β만의 인과 실험이라고 부르지 않습니다.</p>'
    b+='<details><summary>전 48개 결과 보기</summary>'+table(['구현','기간','조건','순손익','Q','σ %','완료 페어'],[[NAMES[r['arm']],r['phase'],r['scenario'],number(r['net_pnl'],2),number(r['Q'],6),number(r['volatility']*100,4),r['completed_pairs']] for r in rows])+'</details>'
    b+='<h2>실제 PAS 완료 가격으로 별도 확인</h2><p>본대회 시작 이후 원점에서 72시간 뒤 가격까지 관측된 예측만 대조했습니다. 원점은 10월 19일 13:00 UTC부터 20일 07:00 UTC까지입니다. 취득 시각 이전에 끝난 PAS 1시간봉만 사용하고 거래 없는 봉·미완료 봉은 제외했습니다. 기존 197건의 진단은 이전 감사 수치와 정확히 일치합니다.</p>'
    b+=table(['구현','허용 예측 수','페어 수','방향 정확도 %','MAE bp','0변화 MAE bp','평균 방향 수익 bp','18bp 수수료만 차감 bp'],[[NAMES[r['arm']],r['permitted']['observations'],r['permitted']['pairs'],number(r['permitted']['accuracy']*100 if r['permitted']['accuracy'] is not None else None,2),number(r['permitted']['mae_bps'],2),number(r['permitted']['zero_mae_bps'],2),number(r['permitted']['directed_gross_bps'],2),number(r['permitted']['fee_only_bps'],2)] for r in diagnostic['results']])
    b+='<p class="note">이는 <b>실제 체결을 재현한 포트폴리오 백테스트가 아니라 가격 예측 진단</b>입니다. 72시간 정답이 크게 겹치며, 허용된 표본과 헤지 비중도 모델마다 다릅니다. 수정 2의 6건은 모두 BTC/EWJ 같은 페어의 겹치는 원점이고 방향이 모두 틀렸습니다. 0건인 후보에 정확도나 수익을 부여하지 않았습니다. 18bp 차감치에는 호가, 미체결, 펀딩, 자본 배분, 유료 헤지·청산이 없으므로 실제 순이익으로 제시하지 않습니다.</p>'
    b+='<h2>검증한 범위와 남은 한계</h2><ul><li>회귀·추세·미성숙 정답 차단·합성 안정 관계·미래 입력 불변성·보유 원장·신호 중복 처리 단위 검사 20개 통과.</li><li>기존 후보에 새 관계 플래그를 넣지 않으면 수정 엔진은 전 기간의 모든 체결·이벤트·회계·위험 결과를 정확히 재현했습니다.</li><li>36회 모두 시간순서 검사, 회계 대조 오차 1e−6 미만, 위험 한도 위반 0을 확인했습니다.</li><li>각 실험의 코드·파라미터·입력 해시는 결과 실행 전에 고정했습니다. 실운용 코드·설정의 기존 해시는 변경되지 않았습니다.</li><li>이전에 여러 번 사용한 기간입니다. 시점상 미래 정답을 보지 않는 코드라는 것과 연구 전체가 과적합되지 않았다는 것은 다릅니다. 독립 테스트 통과나 미래 수익 보장은 주장하지 않습니다.</li></ul>'
    b+=f'<p>최신 고정 입력의 수정 2 형성 시점에서 45개 중 관계 조건을 통과한 페어는 {eligible}개였습니다. 과거 예측 우위를 통과한 전문가가 있는 페어는 EWY/EWJ 1개였고, 08:00 UTC의 실제 신규 진입 임계값을 넘은 페어는 0개였습니다. 종목 평가를 두 페어로 제한한 결과가 아닙니다.</p>'
    b+='<p><b>현재 정확한 한계는 “관계를 가려내는 검사 부족”에 더해 “살아남은 관계에서 방향·크기를 비용 이상으로 예측하는 능력 부족”입니다.</b> 현재 검사는 약한 관계와 지속 괴리를 구분해 기록하고 거래를 차단할 수 있지만, 그 뒤의 가격 예측은 아직 이익 우위를 보이지 못했습니다. 필터를 더 붙이거나 임계값을 결과에 맞춰 느슨하게 만드는 것으로 해결됐다고 주장하지 않습니다.</p>'
    b+='<h2>코드·자료·재현</h2><ul><li><a href="../../pair_regime_research/model.py">수정 1 관계·추세 모델</a> / <a href="../../pair_regime_adaptive/model.py">수정 2 회귀·추세 선택 모델</a> / <a href="../../pair_regime_research/lifecycle.py">보유·연장·청산 규칙</a></li><li><a href="protocol.json">실험 1 사전 조건</a> / <a href="../pair_regime_adaptive_20261023/protocol.json">실험 2 사전 조건</a> / <a href="all_reference_results.csv">모든 결과 CSV</a> / <a href="pas_forecasts.csv">PAS 예측·정답 원장</a></li><li><a href="tests.log">13개 검사</a> / <a href="../pair_regime_adaptive_20261023/tests.log">7개 검사</a> / <a href="parity.json">기존 엔진 정확 재현</a> / <a href="verdict.json">최종 판정</a></li><li><a href="formation_audit.json">45개 관계의 일별 통과/탈락 원장</a> / <a href="../pair_regime_adaptive_20261023/formation_regime_audit.json">수익률 헤지·전문가 선택 원장</a> / <a href="current_pair_permissions.csv">고정 입력 마지막 시점별 허용 상태</a></li></ul>'
    b+='<p>재현: 프로젝트 루트에서 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pair_regime_research.study verify 및 pair_regime_adaptive.study verify. build/run은 고정 입력으로 실행되며 기존 결과를 덮어쓰는 파라미터 탐색을 하지 않습니다. 최초 freeze 명령은 이미 고정된 실험에 다시 사용할 수 없습니다. 통계 패키지는 <code>.tools/pair_stats_20261023</code>에 분리되어 있고 실거래 가상환경의 패키지는 바꾸지 않았습니다.</p>'
    b+='<p>통계 방법 근거: <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html">statsmodels Engle–Granger 공식 문서</a>, <a href="https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html">다중 검정 공식 문서</a>. 이 방법을 사용했다는 사실을 전략 수익성의 증거로 대신하지 않습니다.</p>'
    css='body{font:15px/1.7 system-ui,sans-serif;max-width:1450px;margin:32px;color:#173044}h1,h2{line-height:1.35;margin-top:32px}p{max-width:1200px}.note{background:#fff2d4;border-left:4px solid #bd840c;padding:16px}.scroll{overflow:auto;margin:16px 0}table{border-collapse:collapse;white-space:nowrap}th,td{padding:8px;border:1px solid #d4dfe7;text-align:right}th{background:#edf4f8}th:first-child,td:first-child{text-align:left}a{color:#1668ac}summary{cursor:pointer}code{background:#edf4f8}'
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 관계·추세 구분 검증</title><style>'+css+'</style>'+b+'</html>')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT/'pair_regime_research/diagnostics.py',__import__('pathlib').Path(__file__),
        OUT/'report.html',OUT/'verdict.json',OUT/'pas_diagnostics.json',OUT/'parity.json',
        OUT/'all_reference_results.csv',OUT/'confirmed_relationship_defects.json',OUT/'input_prefix_parity.json']}
    save(OUT/'final_artifact_manifest.json',manifest)
    verify();verify_second();print(json.dumps(verdict,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
