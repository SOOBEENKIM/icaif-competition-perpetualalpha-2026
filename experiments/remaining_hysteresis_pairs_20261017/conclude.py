"""Audit the complete holding-model experiment chain and retain the incumbent."""
import csv,html,json,time
from pathlib import Path
from .research import ROOT,HERE,OUT,V1,GATE,BUDGET,readrun,save,sha,verify,protected


def main():
    spec=json.loads((HERE/'protocol.json').read_text())
    v1=json.loads((V1/'results.json').read_text())
    v2=json.loads((BUDGET/'results.json').read_text())
    v3=json.loads((OUT/'results.json').read_text())
    base=json.loads((GATE/'results.json').read_text())
    for directory in [V1,BUDGET,OUT]:verify(json.loads((directory/'frozen.json').read_text())['sources'])
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    control=spec['control'];arms=[control,'first_dense_hourlyhedge_remaining_renewal_24',
         'first_dense_hourlyhedge_remaining_remaining_24','first_dense_hourlyhedge_remaining_budget_24',spec['candidate']]
    collections={arms[0]:(GATE,base),arms[1]:(V1,v1),arms[2]:(V1,v1),arms[3]:(BUDGET,v2),arms[4]:(OUT,v3)}
    lookup={}
    for arm,(_,data) in collections.items():
        for r in data['rows']:
            if r['arm']==arm:lookup[r['venue'],r['phase'],arm,r['scenario']]=r['metrics']
    reference=[];exact_later=[];exact_par=[];paths=[]
    for venue,phase in v3['windows']:
        for sc in ['base','stress']:
            baseline=readrun(GATE/'runs'/f'{venue}-{phase}-{control}-{sc}.json.gz')
            paths.append(GATE/'runs'/f'{venue}-{phase}-{control}-{sc}.json.gz')
            for arm in arms[1:]:
                directory=collections[arm][0];path=directory/'runs'/f'{venue}-{phase}-{arm}-{sc}.json.gz'
                run=readrun(path);paths.append(path);m=run['metrics'];b=baseline['metrics']
                delta={k:m[k]-b[k] for k in ['net_pnl','price_move_pnl','execution_cost','fees','funding']}
                assert abs(delta['net_pnl']-(delta['price_move_pnl']-delta['execution_cost']-delta['fees']+delta['funding']))<1e-6
                if venue=='reference':
                    reference.append(dict(phase=phase,scenario=sc,arm=arm,metrics=m,control=b,
                                          differences=delta,q_difference=m['financial_score']-b['financial_score']))
                    if phase=='comparison' and arm in arms[3:]:
                        for field in ['fills','snapshots','metrics','owner_fills','daily','funding_events']:
                            assert run[field]==baseline[field],(arm,sc,field)
                        exact_later.append(dict(arm=arm,scenario=sc,fields='financial snapshots, fills, ledgers, daily returns; decisions/limits differ'))
                else:
                    for field in ['events','fills','snapshots','metrics','owner_fills','daily','funding_events']:
                        assert run[field]==baseline[field],(venue,phase,arm,sc,field)
                    exact_par.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc))
    assert not v2['selection']['promote'] and not v3['selection']['promote']
    next_action=(
        'Retain decisiongate and cash_direct as improved reference controls. Stop this tested two-leg '
        'remaining-mean exit family: budget and post-fill hysteresis fixes recover later financial paths '
        'but do not add alpha, and August still underperforms. Before another model, inspect the existing '
        'pair-universe/sector selection experiments and distinguish already-tested configurations; focus '
        'a new bounded candidate on a demonstrable entry opportunity and actual par transfer rather than '
        'more exit-buffer searches on the same few winning trades. Freeze selection before later replay; '
        'no live replacement or reserved raw outcomes.'
    )
    conclusion=dict(at_unix=time.time(),classification='progress_holding_models_and_execution_fixes_evaluated',
        retained_improved_candidate=control,observed_reference_improvement_preserved=True,
        additional_pnl_improvement=False,reference_comparisons=reference,
        exact_later_financial_paths=exact_later,exact_unchanged_par_paths=exact_par,
        model_fits=sum(p['models'] for p in v1['preparation']),new_accounts=48,exact_control_replays=60,
        tests_by_experiment={'remaining_value':10,'remaining_budget':15,'remaining_hysteresis':4},
        distinct_tests=19,selections={'remaining_value':v1['selection'],'remaining_budget':v2['selection'],'remaining_hysteresis':v3['selection']},
        mechanisms=[
          'The simple renewal veto can reject a positive-preview opportunity because the original model allowed wider execution limits. Costed limit intersection fixes that without widening any old limit.',
          'The added model reused the positive pre-adjustment buffer at arrival. Separating the paid-holding threshold (-2bp) prevents immediate unwinds of inventory with positive incremental expected value. The old buffer was not a duplicated monetary fee.',
          'Those generic execution changes restore both later-period financial paths exactly, but the two-leg holding forecast still cuts August price-path gains. They are not an improvement over the retained candidate.',
          'All par paths are unchanged: no new holding/renewal opportunity or profitable completed par lifecycle was demonstrated.'
        ],
        next_action=next_action,protected=protected(),prospective_sources=len(prospective),
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,
        goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    dest=ROOT/'reports/remaining_hysteresis_pairs_20261017';dest.mkdir(exist_ok=False)
    labels={arms[0]:'기존 개선판 decisiongate',arms[1]:'보유 연장 거부',arms[2]:'조기 청산 추가',
            arms[3]:'두 모델의 가격 한도 교집합',arms[4]:'체결 후 보유 기준 분리'}
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>보유·청산 모델과 실행 개선의 최종 비교</title>
<style>body{font:15px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;line-height:1.65;color:#182b38}table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px}td,th{border:1px solid #ccd4db;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;background:#f4f6f8;padding:14px}.note{background:#eef4f8;padding:16px}</style>
<h1>보유·청산 모델과 실행 개선의 최종 비교</h1>
<p class="note">기존 개선판 decisiongate를 유지한다. 새 보유 모델 두 후보의 성과 저하를 확인한 뒤, 가격 한도 교집합과 체결 후 보유 기준 분리까지 순차 검증했다. 마지막 후보는 9월 이후 기존과 같은 체결·손익을 회복했지만 8월 성과가 낮아 추가 성능 개선으로 채택하지 않는다. 앞서 확인한 기존 1·2번 대비 decisiongate 개선은 그대로 보존한다.</p>
<p>초기자본 1M, 외부 기준시장 과거 자료의 비용 차감 계좌 시뮬레이션. 기간은 8월 1일~9월 1일 및 9월 1일~10월 15일이며 경계는 모두 12UTC다. 같은 과거를 반복 사용한 개발 결과이고 독립 미래 검증이 아니다. 각 후보의 8월 판정을 기록·고정한 뒤 이후를 평가했으며, 후속 수정은 별도 프로토콜·코드·결과로 남겼다. 실제 주문·전략 교체는 없다.</p>
<h2>무엇을 바꿨나</h2><p>1~24시간 후 두 종목 가격을 별도로 예측하고 실제 보유 수량, 현재 가격, 남은 만기, 추가 체결 비용·수수료·펀딩으로 연장과 청산을 검토했다. 원래 진입 신호와 수량·위험 한도·부분 체결 처리는 유지했다. 새 예측을 얹은 단순 연장 거부판과 조기 청산판이 먼저 실패했다.</p>
<p>이어 현재 예상 체결가격에서는 이익이 남지만 기존의 넓은 지정가 한도 때문에 거절되는 경우를 수정했다. 기존 한도와 새 모델의 예산을 교집합으로 좁혀 두 모델의 요구를 모두 만족하게 했다. 그다음 체결 전 +5bp의 신규 연장 기준과 체결 후 -2bp의 보유 종료 기준을 분리했다. -2bp는 앞선 보유 후보에 이미 정한 값이며 새로 탐색하지 않았다. 기존 손절·위험·불균형 청산은 우선한다.</p>
<p>이는 진입·보유 상태와 비용을 구분해야 한다는 <a href="https://arxiv.org/html/2502.04284v1">2025년 연구</a>, 두 종목의 비례 비용을 구분하는 <a href="https://arxiv.org/html/2604.02035v1">2026년 연구 Remark 2.2</a>의 관점과 연결된다. 논문의 HJB/RL 알고리즘이나 성과를 재현한 것은 아니며 이번 후보의 성과는 아래 별도 평가로 판단한다.</p>'''
    csv_rows=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            name='8월' if phase=='validation' else '9월 1일~10월 15일'
            cost='기본 비용' if sc=='base' else '불리한 체결 조건'
            page+=f'<h2>{name} · {cost}</h2><table><tr><th>후보</th><th>순손익</th><th>수익률</th><th>기간 σ</th><th>Q</th><th>최대낙폭</th><th>완료 페어</th></tr>'
            for arm in arms:
                m=lookup['reference',phase,arm,sc]
                page+=f"<tr><td>{labels[arm]}</td><td>{m['net_pnl']:+,.2f}</td><td>{100*m['roi']:+.6f}%</td><td>{100*m['volatility_horizon']:.6f}%</td><td>{m['financial_score']:.8f}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{m['completed_pairs']}</td></tr>"
                csv_rows.append(dict(phase=phase,scenario=sc,arm=arm,**{k:m[k] for k in ['net_pnl','roi','financial_score','volatility_horizon','max_drawdown','completed_pairs']}))
            page+='</table>'
    page+='''<h2>관측한 실패 원인과 수정 효과</h2>
<p>연장 거부판은 이후 기본 조건에서 가격 손익이 +316.53 늘었지만 체결 비용·수수료가 +467.24 늘어 계좌 순손익이 -151.59 감소했다. 가격 한도 교집합을 적용하자 이후 기간의 체결·잔고·손익은 기존 개선판과 정확히 같아졌다. 주문 가격 한도와 의사결정 로그는 다르므로 전체 주문 기록이 같다고 주장하지 않는다.</p>
<p>8월 24일에는 조정 체결 후 새 모델의 예상 보유 이익이 +17.32(기본)/+16.76(불리한 조건)였는데 신규 연장 기준 약 +59.86보다 작다는 이유로 바로 청산했다. 이 기준을 체결 후의 -2bp 보유 기준과 분리하자 해당 청산은 사라졌다. 원래 기준은 추가 안전 여유였으며 수수료를 장부에 두 번 넣은 버그는 아니다.</p>
<p>그 수정 후에도 8월 순손익은 기존 대비 기본 -361.19, 불리한 조건 -355.74 낮다. 비용 처리만 바꾸어 새 보유 모델의 예측력을 만들 수는 없었다. 이전 후보의 45개 보유 판단에서 가격 변화 예측 오차는 96.98bp로 변화 없음 기준 94.90bp보다 컸다. 이는 거래에서 선택된 겹치는 관측의 진단으로, 독립적인 일반 예측력 검정은 아니다.</p>
<h2>par와 검증 범위</h2><p>네 후보의 32개 par 조건 결과는 원래 개선판의 체결·계좌 경로와 정확히 같다. 이 자료에서는 신규 보유 검토나 연장 주문이 생기지 않았다. par의 새로운 수익 기회나 완료 페어 수익성을 입증한 실험으로 해석하지 않는다. 현재 candle 기반 체결 가정은 실제 L2의 모든 체결 가능성을 보장하지 않는다.</p>
<p>새 보유 모델의 일별·페어별·기간별 적합 12,668개를 만들고 이후 수정에서는 그대로 재사용했다. 신규 후보 계좌 48개, 원래 계좌 재현 60회를 완료했다. 각 실험 검사 10개·15개·4개(중복을 제외하면 19개)를 통과했다. 정수 입력 때문에 가격 소수가 잘리던 예산 함수 문제는 백테스트 동결 전에 검사로 발견·수정했고 실패 기록과 회귀 검사를 보존했다.</p>
<p>실제 운용 보호 소스 373개, 예약 의존 소스 345개, 향후 검증 소스 1,045개의 해시를 확인했다. 예약된 미래 원시 데이터는 열지 않았다. 내부 승격 실패는 공식 제출 금지 조건이 아니다.</p>
<h2>개발 판단</h2><p>decisiongate와 cash_direct를 개선된 비교 기준으로 유지한다. 이번 두 종목 평균 예측에 기반한 보유·청산 계열은 추가 채택하지 않는다. 같은 수익 거래에 맞추어 청산 임계값을 더 탐색하지 않고, 이미 시험한 페어 선정·섹터 실험을 확인한 뒤 새로운 진입 기회와 실제 par 전이 여부를 검증할 후보를 분리한다.</p>
<p><a href="../decision_gate_pairs_20261017/report.html">유지한 개선판 결과</a> · <a href="../decision_gate_robustness_20261017/report.html">위험 규모와 수익 집중도 검증</a> · <a href="../remaining_value_pairs_20261017/report.html">첫 보유 모델과 진단</a></p></html>'''
    (dest/'report.html').write_text(page)
    with (dest/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    cross=ROOT/'experiments/pairs_development_status_20261017/comparisons/remaining_holding_review.json';assert not cross.exists();save(cross,conclusion)
    save(HERE/'work_state.json',dict(status=conclusion['classification'],report=str((dest/'report.html').relative_to(ROOT)),
        retained_improved_candidate=control,additional_pnl_improvement=False,new_accounts=48,exact_controls=60,
        next_action=next_action,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
        historical_independence=False,goal_complete=False,no_progress_streak=0))
    budget_conclusion=dict(classification='progress_budget_intersection_recovers_later_paths',
        report=str((dest/'report.html').relative_to(ROOT)),selection=v2['selection'],
        later_financial_paths_equal_incumbent=True,august_underperforms_incumbent=True,
        tests_passed=15,new_accounts=12,exact_controls=24,orders_sent=0,live_strategy_changed=False,
        next_action='Hysteresis variant evaluated separately; retain decisiongate after full comparison.',goal_complete=False,no_progress_streak=0)
    save(BUDGET/'conclusion.json',budget_conclusion)
    save(BUDGET.parent/'work_state.json',budget_conclusion)
    files=[Path(__file__),OUT/'results.json',OUT/'conclusion.json',BUDGET/'results.json',BUDGET/'conclusion.json',
           V1/'results.json',V1/'diagnostics.json',V1/'conclusion.json',cross,dest/'report.html',dest/'results.csv',*paths]
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DONE holding-model family audited; retain decisiongate; no new PnL improvement',flush=True)


if __name__=='__main__':main()
