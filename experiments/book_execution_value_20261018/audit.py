"""Conclude execution-only gains without mislabeling them as strategy profits."""
import csv,html,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,save,sha,verify,protected
from .model import booster


def main():
    labels=json.loads((OUT/'labels.json').read_text());result=json.loads((OUT/'results.json').read_text())
    strategy=json.loads((OUT/'strategy_v2/results.json').read_text());spec=json.loads((HERE/'protocol.json').read_text())
    models=[json.loads((OUT/name).read_text()) for name in ['initial_model.json','later_model.json']]
    sources={}
    for name in ['evaluation_frozen.json','strategy/frozen.json','strategy_v2/frozen.json']:
        frozen=json.loads((OUT/name).read_text());verify(frozen['sources']);sources.update(frozen['sources'])
    rows={r['sample_id']:r for r in labels['rows']}
    for model in models:
        assert all(rows[i]['available_at']<model['fit_time'] and rows[i]['tape'] in model['tapes'] for i in model['training_ids'])
    assert result['selection']['frozen_before_later'] and result['selection']['execution_model_selected']
    assert len(strategy['trials'])==0
    old=json.loads((OUT/'strategy/results.json').read_text());assert len(old['trials'])==0 and old['exact_old_plans']==strategy['exact_old_plans']==272
    for name,n in [('test_results.txt',6),('model_test_results.txt',5)]:
        text=(HERE/name).read_text();assert f'Ran {n} tests' in text and text.rstrip().endswith('OK')
    rows_by_tape=[]
    for tape in spec['tapes']:
        subset=[r for r in labels['rows'] if r['tape']==tape]
        rows_by_tape.append(dict(tape=tape,rows=len(subset),unique_sources=len({r['source'] for r in subset}),
            outcomes=dict(Counter(r['outcome'] for r in subset)),states=dict(Counter(r['state'] for r in subset))))
    importance=[]
    for model in models:
        for kind in ['classifier','costs','fraction']:
            gains=booster(model[kind]).get_score(importance_type='gain')
            importance.append(dict(fit_time=model['fit_time'],kind=kind,
                gain_top=[dict(feature=labels['feature_names'][int(k[1:])],gain=v) for k,v in sorted(gains.items(),key=lambda kv:kv[1],reverse=True)[:8]],
                interpretation='Training split gain, not causal attribution or an independently validated feature selection.'))
    reasons=Counter(r['plan']['reason'] for r in strategy['source_plans'])
    calibration_improved=all(result[phase]['brier']<result[phase]['constant_brier'] and result[phase]['cost_rmse_bps']<result[phase]['constant_cost_rmse_bps'] for phase in ['validation','later'])
    diagnostic=dict(tape_groups=rows_by_tape,importance=importance,source_plan_reasons=dict(reasons),
        source_plans=len(strategy['source_plans']),chronological_availability_verified=True,
        calibration_and_cost_rmse_improved_both_periods=calibration_improved,
        retained_gate_rejections=sum(s['counts'].get('decision_gate_rejected',0) for s in strategy['summaries'] if s['reference_variant']=='gate'),
        unresolved_outcomes_never_assigned_cash_zero=True)
    save(OUT/'diagnostics.json',diagnostic)
    next_action='Retain the improved reference decisiongate and the new execution model as a research component. Execution prediction is no longer the only bottleneck: unchanged crypto forecasts have no cost-qualified quote plans in these tapes. Before another trading candidate, examine a pre-period, outcome-independent pair-universe rule (for example a correlation graph) on the existing six-pair forecasts; do not manually remove ETH/HYPE based on observed losses. Freeze the rule before August selection and later replay; keep all results labeled reused-history development and preserve reserved forward blindness.'
    conclusion=dict(at_unix=time.time(),classification='actual_book_execution_predictions_improved_no_additional_trading_pnl_verified',
        previous_goal_turn='progress_hurdle_comparison_and_volume_depth_mismatch_audit_completed',
        retained_improved_candidate='first_dense_hourlyhedge_decisiongate_24',retained_reference_improvement_preserved=True,
        simulated_execution_episodes=len(labels['rows']),unique_source_times=labels['unique_source_times'],
        tests_passed=11,exact_old_source_plans=272,model_fits=6,
        validation=result['validation'],later=result['later'],execution_model_selected=result['selection']['execution_model_selected'],
        added_strategy_entries=len(strategy['trials']),source_plans=len(strategy['source_plans']),source_plan_reasons=dict(reasons),
        full_horizon_pnl=None,financial_score=None,financial_improvement_claimed=False,
        next_action=next_action,protected=protected(),orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
        historical_independence=False,goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    dest=ROOT/'reports/book_execution_value_20261018';dest.mkdir(exist_ok=False)
    with (dest/'metrics.csv').open('w') as f:
        columns=['phase',*result['validation'].keys()];writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for phase in ['validation','later']:writer.writerow(dict(phase=phase,**result[phase]))
    page='''<!doctype html><meta charset="utf-8"><title>실제 호가 기반 체결·정리 비용 예측</title>
<style>body{font:15px system-ui;margin:32px;line-height:1.65}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px;text-align:right}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px}</style>
<h1>실제 호가 기반 체결·정리 비용 예측</h1>
<p><b>실제 수신 호가에서 체결 상태·보유 수량·추가 비용을 예측하는 모델은 단순 과거 평균보다 개선됐다. 기존 예측과 연결한 주문 선택에서는 비용을 넘는 신호가 없어 추가 PnL 개선은 확인하지 못했다.</b> 기존에 성과가 개선된 기준시장 후보는 유지한다.</p>
<h2>어떻게 달라졌는가</h2><p>캔들 거래량으로 가상의 체결 가능량을 정하는 대신, 허용된 과거 호가 5개 기록의 실제 잔량과 수신 시각을 사용했다. 앞 2개로 학습하고 3번째에서 모델 채택 기준을 저장했다. 이후 설정을 바꾸지 않고 앞 3개로 다시 학습해 뒤 2개를 평가했다. 과거에 이미 연구한 자료이므로 새 독립 미래 검증이 아니다.</p>
<p>각 시점에서 120k 상한과 동일 위험 한도 아래 실제 호가 잔량의 25/50/100% 크기, 현재 체결 예상가 대비 0/10bp 가격 제한을 비교했다. 양쪽 주문은 별도로 도착한다. 한쪽만 체결되면 같은 제한가로 한 번 보완하고, 실패하면 비용을 내고 정리한다. 관찰은 최대 60초이며, 확인되지 않은 주문·미정리 잔량은 불확실 상태로 보존했다. 펀딩 경계를 넘는 관측되지 않은 지급을 만들지 않았다.</p>
<p>균형 진입 성공은 장기 거래 완료가 아니다. 성공 시에는 실제 보유 비율과 원래 호가 대비 추가 진입 비용만 학습한다. 실패 후 정리된 경우에만 확인된 현금 손실을 비용으로 사용한다. 우연한 불균형 포지션 이익을 비용 감소로 보상하지 않는, 비음수 실행 위험 모형이다. 미확인 현금 결과를 0으로 만들지 않았지만 비용 회귀가 정리 완료 사례에 조건부라는 한계는 남는다.</p>'''
    page+=f"<p>가상 사례 {len(labels['rows']):,}개는 서로 다른 시각 {labels['unique_source_times']}개의 방향·수량·지정가·실행 조건 변형이다. 서로 독립적인 {len(labels['rows']):,}건의 실거래가 아니다. 학습 가중치는 같은 시각의 변형들이 합계 1이 되도록 했다.</p>"
    page+='<h2>분리한 기간의 예측 결과</h2><p>작을수록 좋은 오차 지표다. 비교 대상은 학습 구간의 상태 비율·평균 수량·평균 비용이며, 뒤 기간의 평균을 사용하지 않았다. Brier 점수는 정확도 비율이 아니다.</p><table><tr><th>기간</th><th>상태 오차: 평균</th><th>새 모델</th><th>비용 RMSE: 평균(bp)</th><th>새 모델(bp)</th><th>수량 RMSE: 평균</th><th>새 모델</th></tr>'
    for phase in ['validation','later']:
        m=result[phase];page+=f"<tr><td>{phase}</td><td>{m['constant_brier']:.6f}</td><td>{m['brier']:.6f}</td><td>{m['constant_cost_rmse_bps']:.4f}</td><td>{m['cost_rmse_bps']:.4f}</td><td>{m['constant_held_fraction_rmse']:.6f}</td><td>{m['held_fraction_rmse']:.6f}</td></tr>"
    page+='</table><p>뒤 평가에는 서로 다른 129개 시각의 11,740개 변형이 포함된다. 비용 평가에서 미정리·미확인 현금 496개는 제외했고 이를 손실 0으로 처리하지 않았다. 수량 오차는 균형 진입이 관측된 경우의 조건부 수량 비율 오차다. 기록은 몇 시간에 불과하므로 다른 시장 국면으로의 일반화는 확인되지 않았다.</p>'
    page+=f'''<h2>전략과 연결한 결과</h2><p>기존 cash_direct 예측과 decisiongate 예측을 모두 그대로 유지했다. 최대 수량, 수량만 최적화, 과거 평균 실행 위험, 새 실행 위험 모델의 네 가지 선택 방법을 비교했다. 원래 가격 예측·헤지·순이익 문턱을 바꾸지 않았다. 이전 호가 판단 {strategy['exact_old_plans']}개를 그대로 재현했다.</p>
<p>decisiongate는 검사한 시각에서 신규 진입 방향을 허용하지 않았다. 방향 필터가 없는 cash_direct도 세 가지 수량의 {len(strategy['source_plans']):,}개 판단 중 비용을 넘는 주문을 만들지 못했다. 따라서 새 체결 모델 때문에 거래가 늘거나 PnL이 개선됐다고 주장할 수 없다. 아래는 서로 겹치는 시각·양 방향·수량·기본/불리 조건의 판단 개수다.</p><pre>{html.escape(json.dumps(dict(reasons),ensure_ascii=False,indent=2))}</pre>
<p>최대 수량 후보가 작은 수량 후보와 우연히 같은 주문이 되더라도 비교 대상에서 사라지지 않도록 후보 중복 처리도 보완했다. 앞선 방향 필터 전용 실행과 두 예측을 모두 비교한 실행에서 실제로 선택된 주문은 모두 0개였다. 이 짧고 단절된 호가 기록에는 전체 전략 수익률이나 공식 Q를 붙이지 않는다.</p>'''
    page+='''<h2>다음 판단</h2><p>실행 예측 모델은 개선된 연구 구성요소로 보관하되, 실행기만 계속 바꿔 원래 없던 가격 예측 이익을 만들려 하지 않는다. 기존 두 암호화폐 페어 외의 후보를 검토할 때는 평가 전 기간의 자료로 정하는 객관적인 페어 구성 규칙부터 고정한다. 이미 본 손실 때문에 특정 페어만 수동 제거한 뒤 독립 검증이라고 주장하지 않는다. 기존 개선판의 기준시장 +25,868.79 / Q 0.30858은 유지되며, 새 모형의 실거래 성과와는 구분한다.</p>
<p>관련 검사 11개와 기존 호가 판단 대조 272개를 통과했다. 실거래 주문·전략 교체·보류된 미래 자료 열람은 없다.</p>'''
    (dest/'report.html').write_text(page)
    files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'strategy_comparison_protocol.json',HERE/'test_results.txt',HERE/'model_test_results.txt',
        OUT/'initial_model.json',OUT/'later_model.json',OUT/'validation.json',OUT/'later.json',OUT/'selection.json',OUT/'results.json',
        OUT/'diagnostics.json',OUT/'conclusion.json',OUT/'strategy/results.json',OUT/'strategy_v2/results.json',dest/'report.html',dest/'metrics.csv']
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources=sources,prospective_sources=len(prospective)))
    save(HERE/'work_state.json',dict(status='progress_actual_book_execution_learning_and_strategy_connection_completed',
        report=str((dest/'report.html').relative_to(ROOT)),execution_prediction_improved=True,additional_pnl_verified=False,
        next_action=next_action,goal_complete=False,no_progress_streak=0,orders_sent=0,live_strategy_changed=False))
    print('DONE',dest/'report.html','sources',len(sources),'execution improvement',calibration_improved,'strategy entries',len(strategy['trials']),flush=True)


if __name__=='__main__':main()
