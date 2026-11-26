"""Record observed backtest improvement separately from strict selection flags."""
import gzip,json,time
from pathlib import Path
from .research import ROOT,HERE,OUT,PARENT,MATCHED,ARM,path_for,verify,save,sha


def main():
    result=json.loads((OUT/'results.json').read_text());diagnostics=json.loads((OUT/'diagnostics.json').read_text())
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in result['rows']}
    reference=[];exact_stress=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            new=lookup['reference',phase,ARM,sc];old=lookup['reference',phase,MATCHED,sc]
            natives={k:lookup['reference',phase,k,sc] for k in ['first_strategy','second_strategy']}
            assert all(new['net_pnl']>v['net_pnl'] and new['financial_score']>v['financial_score'] for v in natives.values())
            assert new['net_pnl']>=old['net_pnl'] and new['financial_score']>=old['financial_score']
            assert new['pnl_after_liquidation_reserve']>=old['pnl_after_liquidation_reserve']
            reference.append(dict(phase=phase,scenario=sc,candidate=new,retained=old,natives=natives,
                pnl_difference=new['net_pnl']-old['net_pnl'],q_difference=new['financial_score']-old['financial_score']))
            if sc=='stress':
                with gzip.open(path_for('reference',phase,sc),'rt') as f:old_run=json.load(f)
                with gzip.open(OUT/'runs'/f'reference-{phase}-{ARM}-{sc}.json.gz','rt') as f:new_run=json.load(f)
                for field in ['events','fills','metrics','snapshots']:assert old_run[field]==new_run[field],(phase,field)
                exact_stress.append(phase)
    verify(json.loads((OUT/'frozen.json').read_text())['sources'])
    verify(json.loads((OUT/'books/frozen.json').read_text())['sources'])
    verify(json.loads((OUT/'diagnostics_frozen.json').read_text())['sources'])
    verify(json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'])
    original=(ROOT/'experiments/conditional_completion_pairs_20261017/portfolio.py').read_text()
    expected=original.replace('from experiments.dense_history_pairs_20261017.policy import proposals','from .entry import proposals').replace('from .completion import completion_plan, after_completion','from experiments.conditional_completion_pairs_20261017.completion import completion_plan, after_completion')
    assert expected==(HERE/'portfolio.py').read_text()
    conclusion=dict(at_unix=time.time(),previous_goal_turn_classification='progress_shared_factor_candidate_completed',
        classification='progress_decision_gate_backtest_improvement_observed',
        improved_candidate=ARM,retained_comparison=MATCHED,reference_results=reference,
        observed_reference_pnl_q_improvement=True,reference_base_improved_in_both_periods=True,
        reference_stress_account_paths_identical=exact_stress,liquidation_reserve_result_preserved=True,
        predeclared_selection=result['selection'],
        selection_interpretation='The frozen strict rule demanded PnL/Q strictly above cash_direct in BOTH cost scenarios. Stress ties, so that record remains false. This does not negate observed base-case improvement or improvement over both native first/second in all four reference conditions. No criterion or earlier selection has been edited; the candidate is retained as a development result, not independent evidence or an automatically approved live replacement.',
        risk_limits='Same initial capital, risk ceilings and order-sizing formula; realized exposure differs. Comparison-period sigma improves 0.955481% to 0.941440%, while MDD worsens 1.183937% to 1.232809%. August sigma and MDD also increase. Q still improves in base cases.',
        mechanism='August gate rejected the August 19 short-pair entry and permitted earlier August 20 long entries. Later it delayed the October 5 16UTC entry into subsequent attempts, reducing combined episode losses; it did not eliminate all losses. Entry timing/held path, not universal fee savings, explains the observed improvement.',
        proxy_limit='The fixed-24h source proxy performs worse in the later period despite full-account improvement. No claim that the training proxy by itself has stable predictive value, or that changing holding rules will preserve this gain.',
        par_limit='Extended par base loss improves from -229.841769 to -186.308684 and the flat-start day from -50.382018 to -6.808615 by removing the overlay; those match native first/second. Older par comparison base is 0.040319 worse than cash_direct. No completed par pair or independent quote-level profitable lifecycle demonstrated.',
        new_account_replays=12,exact_control_replays=12,tests_passed=10,model_fits=526,
        maximum_solver_gap=diagnostics['solver']['gap_max'],candidate_book_observations=136,exact_control_book_observations=136,
        next_action='Retain cash_direct and decisiongate as explicit controls. Quantify the decision-gate gain at matched realized risk and its concentration by episode before a new model. Study holding/renewal separately: source 24h decision proxy and actual renewed trade payoff differ; do not repurpose the entry score as a mean return or reuse entry sunk costs for a holding decision.',
        orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    report=ROOT/'reports/decision_gate_pairs_20261017/report.html';page=report.read_text()
    page+='''<h2>결과: 기준시장 백테스트 추가 개선 확인</h2>
<p>기본 조건에서 8월 순손익은 기존 cash_direct +14238.3038에서 +22028.9398로, Sep1–Oct15는 +3422.1612에서 +3852.7636으로 높아졌다. Q는 각각 0.253469→0.295158, 0.124304→0.130237이다. 불리한 조건의 두 기간 계좌 경로·체결·손익은 기존 개선판과 완전히 같다. 초기 자본·한도·주문 산식은 동일하고 실제 노출은 달라질 수 있다. 남은 포지션의 추정 정리 비용을 별도로 차감해도 기본 조건의 개선은 유지된다.</p>
<p>기록된 strict 선정 조건은 두 비용 조건 모두에서 엄격한 초과를 요구했으므로 불리한 조건의 동률 때문에 false이다. 이 기록을 사후 수정하지 않는다. 이는 기본 조건의 성과 개선이나 기존 첫째·둘째를 네 reference 조건 모두에서 앞선 사실을 부정하지 않으며 대회 제출 금지 조건도 아니다. 이후 비교할 개발 후보로 보존한다.</p>
<p>Sep1–Oct15 기간 변동성은 0.955481%→0.941440%로 낮아졌지만 최대낙폭은 1.183937%→1.232809%로 높아졌다. 8월에는 기간 변동성·최대낙폭이 증가한다. 모든 위험 지표가 개선됐다고 주장하지 않는다.</p>
<h2>어떤 거래가 달라졌나</h2><p>8월 19일의 손실 숏 진입을 걸러내고 8월 20일 롱 진입 시점이 앞당겨졌다. 이후 비교 기간에는 10월 5일 16UTC의 진입이 빠지고 17UTC·21UTC 시도로 바뀌었다. 이 구간의 합산 손실이 줄었지만 손실 자체가 사라진 것은 아니다. 이후 기간 페어 가격 손익은 +4557.7068→+5083.0849, 거래 비용과 수수료는 오히려 증가했고 최종 페어 순손익은 +2978.0238→+3407.7921이다.</p>
<h2>par에서 확인한 범위</h2><p>확장 par 기본 조건 손익은 -229.841769→-186.308684, 마지막 하루는 -50.382018→-6.808615로 개선됐다. 필터가 페어 진입을 막으면서 기존 첫째·둘째와 같은 성과가 된 것이다. 완료 페어는 0이며 새로운 par 수익 기회가 입증된 것은 아니다. 이전 par 비교 기간은 기존 개선판보다 0.040319 낮고 불리한 비용 조건은 같다.</p>
<p>기존 호가 관측 68방향/비용조건에서는 필터가 모두 거절했고, 기존 가격 예측을 유지하므로 낙관적 비용 차감 기대값도 변하지 않았다. 원천 24시간 보조 수익의 이후 구간 평균은 -0.7181bp에서 -1.7931bp로 악화됐으므로, 보조 학습 지표를 계좌 수익이나 안정적 예측력으로 오인하지 않는다. 실제 보유 연장과 계좌 경로가 중요하다.</p>
<p>검사 10개, 모델 학습 526개, 신규 계좌 12개 및 원래 계좌 12개 정확 재현을 완료했다. 학습 해의 최대 primal-dual gap은 9.9982e-7 미만이다. 미래 검증 자료는 열지 않았고 실거래 주문·교체도 하지 않았다.</p>'''
    report.write_text(page)
    dest=ROOT/'experiments/pairs_development_status_20261017/comparisons';dest.mkdir(exist_ok=True)
    save(dest/'decision_gate.json',conclusion)
    save(HERE/'work_state.json',dict(status=conclusion['classification'],report=str(report.relative_to(ROOT)),
        improved_candidate=ARM,observed_reference_pnl_q_improvement=True,comparison_control=MATCHED,
        new_account_replays=12,exact_controls=12,tests_passed=10,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
        historical_independence=False,goal_complete=False,no_progress_streak=0,next_action=conclusion['next_action']))
    paths=[Path(__file__),OUT/'conclusion.json',report,dest/'decision_gate.json']
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print('DONE: additional reference backtest improvement; same stress paths; par limits retained',flush=True)


if __name__=='__main__':main()
