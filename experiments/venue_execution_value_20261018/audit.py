"""Outcome diagnostics and final immutable record of the historical experiment."""
import csv
import gzip
import html
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,save,sha,ts,verify,protected
from .model import predict


def main():
    spec=json.loads((HERE/'protocol.json').read_text())
    labels=json.loads((OUT/'labels.json').read_text())
    results=json.loads((OUT/'results.json').read_text())
    models=json.loads((OUT/'value_models.json').read_text())
    books=json.loads((OUT/'books/results.json').read_text())
    repaired=json.loads((OUT/'book_reconciliation/results.json').read_text())
    sources={}
    for name in ['evaluation_frozen.json','evaluation_resume_v3_frozen.json','books/frozen.json','book_reconciliation/frozen.json']:
        frozen=json.loads((OUT/name).read_text());verify(frozen['sources']);sources.update(frozen['sources'])
    for name,count in [('test_results.txt',8),('control_v2_test_results.txt',4),('book_test_results.txt',4)]:
        text=(HERE/name).read_text();assert f'Ran {count} tests' in text and text.rstrip().endswith('OK')
    byid={r['sample_id']:r for r in labels['rows']}
    for model in models:
        for identity in model['training_ids']:
            row=byid[identity]
            assert row['status'] in ('closed','unfilled') and row['outcome_time']<model['fit_time']
            assert row['source']>=model['fit_time']-spec['training_window_days']*86400
    outcome_groups=[]
    for h in spec['horizons_hours']:
        for k,pair in enumerate(spec['pairs']):
            rows=[r for r in labels['rows'] if r['horizon']==h and r['pair_index']==k]
            mature=[r for r in rows if r['status'] in ('closed','unfilled')]
            closed=[r for r in rows if r['status']=='closed']
            outcome_groups.append(dict(horizon=h,pair=pair,counts=dict(Counter(r['status'] for r in rows)),
                distinct_sources=len({r['source'] for r in mature}),balanced_closed=sum(r['balanced'] for r in closed),
                aborted_closed=sum(r['aborted'] for r in closed),positive_closed=sum(r['net_pnl']>0 for r in closed),
                median_intended_gross=float(np.median([r['intended_gross'] for r in mature])) if mature else None,
                mean_net_bps=float(np.mean([r['net_pnl']/r['intended_gross']*10000 for r in mature])) if mature else None))
    predictions=[];prediction_rows=[]
    for phase,start,end in spec['windows']:
        start,end=ts(start),ts(end)
        for h in spec['horizons_hours']:
            rows=[r for r in labels['rows'] if r['horizon']==h and r['status'] in ('closed','unfilled')
                  and start<=r['source']<end and r['outcome_time']<end]
            values=[]
            for row in rows:
                past=[m for m in models if m['horizon']==h and m['fit_time']<=row['source']]
                if not past:continue
                model=max(past,key=lambda m:m['fit_time'])
                estimate=predict(model,np.asarray(row['features']),row['source'])*10000
                actual=row['net_pnl']/row['intended_gross']*10000
                values.append((estimate,actual))
                prediction_rows.append(dict(phase=phase,sample_id=row['sample_id'],prediction_bps=estimate,actual_bps=actual,fit_time=model['fit_time']))
            a=np.asarray(values)
            predictions.append(dict(phase=phase,horizon=h,overlapping_mature_rows=len(values),
                rmse_bps=float(np.sqrt(np.mean((a[:,0]-a[:,1])**2))) if len(a) else None,
                zero_rmse_bps=float(np.sqrt(np.mean(a[:,1]**2))) if len(a) else None,
                above_5bp=int(np.sum(a[:,0]>5)) if len(a) else 0,
                realized_mean_above_5bp=float(np.mean(a[a[:,0]>5,1])) if len(a) and np.any(a[:,0]>5) else None))
    accounts=[]
    for row in results['rows']:
        m=row['metrics'];file=OUT/'runs'/f"{row['phase']}-{row['arm']}-{row['scenario']}.json.gz"
        with gzip.open(file,'rt') as f:run=json.load(f)
        difference=m['net_pnl']-(m['price_move_pnl']-m['execution_cost']-m['fees']+m['funding'])
        assert abs(difference)<1e-5
        accounts.append(dict(phase=row['phase'],arm=row['arm'],scenario=row['scenario'],
            accounting_error=difference,events=dict(Counter(e['event'] for e in run['events'])),
            pair_owned_pnl=m['pair_owned_pnl'],base_owned_pnl=m['base_owned_pnl'],
            partial_exposure_minutes=m['unbalanced_minutes'],residual_gross=m['residual_gross']))
    unique_value_books={(t['time'],tuple(t['plan']['pair']),t['plan']['direction']) for t in books['trials'] if t['kind']=='value'}
    summary=dict(outcome_groups=outcome_groups,prediction_diagnostics=predictions,prediction_rows=prediction_rows,
        accounting_checks=accounts,strict_training_outcomes_checked=True,unique_value_book_opportunities=len(unique_value_books),
        prediction_diagnostics_condition_on_observed_closure=True,
        censoring_limitation='Long-unresolved entries are excluded from mature cash labels; this is informative censoring, not proof those positions have zero cost.',
        selection=results['selection'],orders_sent=0)
    save(OUT/'diagnostics.json',summary)
    retained=json.loads((ROOT/'experiments/lifecycle_stability_pairs_20261018/artifacts/conclusion.json').read_text())
    conclusion=dict(at_unix=time.time(),classification='par_execution_value_labels_corrected_no_additional_pnl_q_gain',
        previous_goal_turn='progress_identified_terminal_control_schema_mismatch',
        retained_improved_candidate='first_dense_hourlyhedge_decisiongate_24',
        retained_reference_improvement_preserved=True,new_venue_candidate_selected=results['selection']['selected'],
        label_counts=labels['counts'],virtual_label_replays=labels['virtual_account_replays'],
        label_closed_balanced=sum(g['balanced_closed'] for g in outcome_groups),
        label_closed_aborted=sum(g['aborted_closed'] for g in outcome_groups),
        value_fits=len(models),new_account_replays=42,economic_path_controls=results['exact_controls'],tests_passed=16,
        old_book_entry_probes=len(books['trials']),old_book_entry_paths_reproduced=repaired['exact_prior_entries'],
        old_book_reconciliation_statuses=repaired['statuses'],unique_value_book_opportunities=len(unique_value_books),
        old_book_full_horizon_exits=0,selection_not_official_submission_gate=True,
        next_action='Retain the improved reference decisiongate. Audit source-time liquidity persistence, lot/hedge geometry and informative censoring before a nonlinear or hurdle execution-value candidate. Freeze any new learner before evaluating later accounts; do not widen orders or count duplicate directional/horizon paths as independent evidence.',
        protected=protected(),orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
        historical_independence=False,goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    dest=ROOT/'reports/venue_execution_value_20261018';dest.mkdir(exist_ok=False)
    keys=['phase','arm','scenario','net_pnl','roi','financial_score','volatility_horizon','max_drawdown','completed_pairs','aborted_pairs','unbalanced_minutes']
    with (dest/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in results['rows']:writer.writerow({k:row.get(k,row['metrics'].get(k)) for k in keys})
    page='''<!doctype html><meta charset="utf-8"><title>par 체결 후 순손익 학습 비교</title>
<style>body{font:15px system-ui;margin:32px;line-height:1.65}table{border-collapse:collapse;font-size:13px}th,td{padding:6px;border:1px solid #bbb;text-align:right}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px}</style>
<h1>par 체결 후 순손익 학습 비교</h1>
<p><b>결론: 새 6시간·24시간 모델은 기존 계좌 대비 PnL과 Q의 추가 개선을 확인하지 못했다. 기존에 개선된 기준시장 decisiongate 후보는 유지한다.</b> 과거 자료를 반복 사용한 개발 평가이며 실거래나 독립적인 미래 검증이 아니다.</p>
<p>이전 가격 목표는 양 종목의 다음 캔들과 만기 캔들에 거래가 없어도 마지막 가격을 사용했다. 이번에는 같은 과거 par 자료에서 진입·한쪽 체결·한 번의 보완·유료 정리·실제 펀딩을 모의 실행하고, 실제로 포지션이 0이 된 뒤의 순손익을 학습했다. 전량 미체결은 관측된 0, 끝까지 미정리된 건은 검열 자료로 구분했다. 6시간·24시간 고정 보유, 원래 지정가 유지, 위험 한도 유지다.</p>
<p>입력은 당시 알려진 가격 괴리·수익률·거래량·펀딩·헤지 특징, 방향 상호작용과 주문 규모다. 7일 학습창, 3일 가중 반감기, 고정된 5회 학습 시각의 릿지 모델이다. 비용 차감 후 목표이므로 기본 수수료와 펀딩을 이중 차감하지 않는다. 불리한 조건에서는 추가 왕복 비용 26bp만 반영한다. 규칙과 자료를 먼저 고정했고, 10월 8~12일에서 선정을 저장한 뒤 12~16일, 12~17일을 비교했다. 마지막 두 구간은 겹친다. 시각은 UTC 12:00 경계다.</p>
<p>호가 대신 캔들 거래량을 사용하는 체결 모형은 실제 과거 주문장 전체를 재현하지 못한다. 종료 시 남은 포지션은 평가손익이며 무료 청산을 가정하지 않았다. 양 방향·보유기간의 경로는 중복되므로 독립 표본으로 세지 않는다.</p>
<h2>자료와 검증</h2>'''
    page+=f"<p>가상 계좌 경로 {labels['virtual_account_replays']}회. 청산 완료 689건 중 정상 보유 후 종료 75건, 진입 실패 후 정리 614건. 전량 미체결 302건, 미정리 37건. 학습에서 제외된 미정리 경로의 비용을 0으로 취급하지 않았지만, 완료된 경로만 학습하는 선택 편향 가능성은 남아 있다. 실제 포트폴리오 비교 42개, 기존 경제적 경로 대조 16개, 관련 단위검사 16개를 통과했다.</p>"
    page+='''<p>초기 대조 중단은 기존 주문 로그에 없던 entry_limit_blocked 필드 때문이었다. 원래 실행기도 가격 한도를 넘으면 체결 가능 수량을 0으로 만들었다. 추가 필드만 별도 집계하고 모든 공통 주문 필드·체결·계좌·판단·성과의 일치를 확인했다. 가격 제한에 걸렸다고 기록된 건은 두 기록 모두 체결량과 수량 한도가 0임을 검사했다. 학습 결과·모수·선정 규칙은 바꾸지 않았다.</p><h2>전체 계좌 비교</h2><table><tr>'''
    page+=''.join(f'<th>{html.escape(k)}</th>' for k in keys)+'</tr>'
    for row in results['rows']:
        values=[row.get(k,row['metrics'].get(k)) for k in keys]
        page+='<tr>'+''.join(f'<td>{v:.8f}</td>' if isinstance(v,float) else f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>'
    page+='</table><h2>순손익 예측 진단</h2><p>해당 구간에서 결과가 관측된 겹치는 가상 경로의 진단이다. 실제 선택된 포트폴리오 손익과 구분하며, 전체 미정리 경로의 예측 정확도를 입증하지 않는다.</p><pre>'+html.escape(json.dumps(predictions,ensure_ascii=False,indent=2))+'</pre>'
    page+=f'''<h2>실제 과거 호가 점검</h2><p>허용된 과거 호가 5개 구간에서 독립적인 진입 점검 12회를 했다. 같은 진입을 재현한 뒤 보완과 유료 정리를 관찰한 결과, 양쪽 진입 2회, 진입 실패 후 유료 정리 6회, 전량 미체결 4회였다. 새 순손익 모델의 6시간·24시간 설정은 <b>동일한 시장 기회 {len(unique_value_books)}개</b>를 선택했고 모두 정리 손실이 발생했다. 설정별 결과를 서로 다른 기회로 세면 안 된다. 두 정상 진입은 가격 목표 비교 모델의 결과다. 어느 경우에도 6시간·24시간 만기 청산은 관측되지 않았으므로 이 호가 점검의 수익률이나 Q를 만들지 않았다.</p>
<p>호가 점검은 실제 수신 지연·한정된 잔량·독립된 양쪽 도착을 사용한다. 한쪽 체결 시 원래 지정가로 한 번만 보완하고 실패하면 실제 반대 호가로 정리한다. 캔들 모델의 예측값을 호가 비용에 재보정한 성과라는 주장은 아니다.</p>
<h2>유지되는 기존 개선 성과</h2><p>별도 기준시장 8월 1일~10월 15일 연속 평가에서 기존 첫째 +13,490.84 / Q 0.23534, 둘째 +11,858.42 / Q 0.22507, 유지 후보 +25,868.79 / Q 0.30858이다. 이 표 위의 짧은 par 결과와 서로 다른 기간·시장이다. 새 실험의 실패가 기존 개선을 취소하지 않는다. 표본 수와 내부 후보 선정 조건은 공식 제출 금지 요건이 아니다.</p>
<h2>다음 검토</h2><p>진입 실패 후 정리 비용이 많은 경로를 대상으로, 당시 알 수 있던 거래 활동의 지속성·두 종목의 수량 단위와 헤지 비율·미정리 표본 편향을 먼저 진단한다. 이후 비선형 모델 또는 체결 성공과 조건부 손익을 나누는 모델을 고정해 비교한다. 실거래 주문·전략 교체는 하지 않았다.</p>'''
    (dest/'report.html').write_text(page)
    files=[*HERE.glob('*.py'),HERE/'test_results.txt',HERE/'control_v2_test_results.txt',HERE/'book_test_results.txt',
           OUT/'diagnostics.json',OUT/'conclusion.json',OUT/'results.json',OUT/'selection.json',OUT/'books/results.json',
           OUT/'book_reconciliation/results.json',*sorted((OUT/'runs').glob('*.json.gz')),dest/'report.html',dest/'results.csv']
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources=sources,prospective_sources=len(prospective)))
    save(HERE/'work_state.json',dict(status='progress_execution_value_comparison_and_old_book_repair_completed',
        report=str((dest/'report.html').relative_to(ROOT)),new_account_replays=42,economic_controls=16,tests_passed=16,
        next_action=conclusion['next_action'],orders_sent=0,live_strategy_changed=False,goal_complete=False,no_progress_streak=0))
    print('DONE',dest/'report.html','sources',len(sources),'diagnostics',predictions,flush=True)


if __name__=='__main__':main()
