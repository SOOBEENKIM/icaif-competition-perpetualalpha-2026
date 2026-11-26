"""Calibrate predictions, audit selected actions and retain the prior improvement."""
import csv,gzip,html,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,PARENT,save,sha,verify,protected
from .model import predict
from experiments.par_direct_forecast_20261017.data import ts


def main():
    spec=json.loads((HERE/'protocol.json').read_text());data=json.loads((OUT/'labels.json').read_text())
    models=json.loads((OUT/'models.json').read_text());results=json.loads((OUT/'results.json').read_text())
    books=json.loads((OUT/'books/results.json').read_text());repairs=json.loads((OUT/'book_reconciliation/results.json').read_text())
    capacity=json.loads((OUT/'capacity/results.json').read_text());sources={}
    for name in ['evaluation_frozen.json','dataset_frozen.json','books/frozen.json','book_reconciliation/frozen.json','capacity/frozen.json']:
        d=json.loads((OUT/name).read_text());verify(d['sources']);sources.update(d['sources'])
    tests=(HERE/'test_results.txt').read_text();assert 'Ran 7 tests' in tests and tests.rstrip().endswith('OK')
    ids={r['sample_id']:r for r in data['rows']}
    for model in models:
        assert all(ids[i]['initial_time']<model['fit_time'] for i in model['initial_ids'])
        assert all(ids[i]['status'] in ('closed','unfilled') and ids[i]['outcome_time']<model['fit_time'] for i in model['cash_ids'])
    diagnostics=[];predictions=[]
    for phase,start,end in spec['windows']:
        start,end=ts(start),ts(end)
        for h in spec['horizons_hours']:
            probabilities=[];priors=[];truth=[];cash=[]
            for row in data['rows']:
                if row['horizon']!=h or not start<=row['source']<end or row['initial_time']>=end:continue
                model=max((m for m in models if m['horizon']==h and m['fit_time']<=row['source']),key=lambda m:m['fit_time'],default=None)
                if model is None:continue
                value,parts=predict(model,np.asarray(row['features']),row['source'],'hurdle')
                direct,_=predict(model,np.asarray(row['features']),row['source'],'tree')
                probabilities.append(parts['initial_probabilities']);priors.append(model['prior']);truth.append(np.eye(3)[row['initial_class']])
                realized=row['net_pnl']/row['intended_gross'] if row['status'] in ('closed','unfilled') and row['outcome_time']<end else None
                if realized is not None:cash.append((value*10000,direct*10000,realized*10000))
                predictions.append(dict(phase=phase,horizon=h,sample_id=row['sample_id'],fit_time=model['fit_time'],
                    initial_class=row['initial_class'],probabilities=parts['initial_probabilities'],hurdle_value=value,tree_value=direct,observed_cash_return=realized))
            a,b,y=map(np.asarray,[probabilities,priors,truth]);c=np.asarray(cash)
            diagnostics.append(dict(phase=phase,horizon=h,overlapping_initial_rows=len(a),overlapping_cash_rows=len(c),
                multiclass_brier=float(np.mean(np.sum((a-y)**2,axis=1))),constant_prior_brier=float(np.mean(np.sum((b-y)**2,axis=1))),
                balanced_brier=float(np.mean((a[:,2]-y[:,2])**2)),balanced_prior_brier=float(np.mean((b[:,2]-y[:,2])**2)),
                hurdle_cash_rmse_bps=float(np.sqrt(np.mean((c[:,0]-c[:,2])**2))),tree_cash_rmse_bps=float(np.sqrt(np.mean((c[:,1]-c[:,2])**2))),
                zero_cash_rmse_bps=float(np.sqrt(np.mean(c[:,2]**2))),
                hurdle_positive_count=int(np.sum(c[:,0]>5)),hurdle_positive_realized_mean_bps=float(np.mean(c[c[:,0]>5,2])) if np.any(c[:,0]>5) else None))
    paths=[]
    for row in results['new_rows']:
        m=row['metrics'];path=OUT/'runs'/f"{row['phase']}-{row['arm']}-{row['scenario']}.json.gz"
        with gzip.open(path,'rt') as f:run=json.load(f)
        assert abs(m['net_pnl']-(m['price_move_pnl']-m['execution_cost']-m['fees']+m['funding']))<1e-5
        intents=[e for e in run['events'] if e['event']=='pair_intent'];components=[]
        for e in intents:
            comp=e.get('execution_value_components',{})
            if comp:
                contribution=np.array(comp['initial_probabilities'])*comp['conditional_net_returns']
                components.append(dict(time=e['time'],pair=e['pair'],initial_unbalanced_contribution=float(contribution[1]),
                    initial_balanced_contribution=float(contribution[2]),expected_net_bps=e['expected_net_bps']))
        paths.append(dict(phase=row['phase'],arm=row['arm'],scenario=row['scenario'],events=dict(Counter(e['event'] for e in run['events'])),
            components=components,pair_owned_pnl=m['pair_owned_pnl'],base_owned_pnl=m['base_owned_pnl']))
    mismatch=[r for r in capacity['rows'] if not r['source_both_traded'] and r['initial_state']==2]
    unique_mismatch={(r['time'],tuple(r['pair']),r['direction']) for r in mismatch}
    save(OUT/'diagnostics.json',dict(calibration=diagnostics,predictions=predictions,selected_action_paths=paths,
        strict_initial_and_cash_availability=True,capacity_mismatch_examples=mismatch,unique_balanced_capacity_mismatches=len(unique_mismatch)))
    conclusion=dict(at_unix=time.time(),classification='execution_state_model_completed_no_robust_pnl_q_gain_proxy_mismatch_verified',
        previous_goal_turn='progress_execution_value_comparison_and_old_book_repair_completed',
        retained_improved_candidate='first_dense_hourlyhedge_decisiongate_24',retained_reference_improvement_preserved=True,
        selected=results['selection']['selected'],new_account_replays=24,exact_parent_accounts=42,tests_passed=7,
        reused_episode_paths=1028,new_episode_replays=0,model_bundles=len(models),
        initial_class_counts=data['initial_classes'],cash_statuses=data['cash_statuses'],
        independent_book_probes=len(books['trials']),book_repair_statuses=repairs['statuses'],
        capacity_probe_counts=capacity['counts'],unique_balanced_capacity_mismatches=len(unique_mismatch),
        capacity_audit_is_not_profit_evidence=True,
        next_action='Stop treating volume-capped candle labels as calibrated exchange fill probabilities. Retain the improved reference alpha model and implement a matched source-book-depth sizing/paired IOC adapter with current quote economics and paid repair, compared against the candle-volume gate on the same allowed books. Do not infer long-horizon PnL from short tapes or unblind reserved forward outcomes.',
        protected=protected(),historical_independence=False,reserved_raw_read=False,orders_sent=0,live_strategy_changed=False,goal_complete=False,no_progress_streak=0)
    save(OUT/'conclusion.json',conclusion)
    dest=ROOT/'reports/venue_hurdle_value_20261018';dest.mkdir(exist_ok=False)
    keys=['phase','arm','scenario','net_pnl','financial_score','roi','volatility_horizon','max_drawdown','completed_pairs','aborted_pairs','unbalanced_minutes']
    with (dest/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in results['rows']:w.writerow({k:r.get(k,r['metrics'].get(k)) for k in keys})
    page='''<!doctype html><meta charset="utf-8"><title>초기 체결과 조건부 순손익 분리 실험</title>
<style>body{font:15px system-ui;margin:32px;line-height:1.65}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:7px;text-align:right}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px}</style>
<h1>초기 체결과 조건부 순손익 분리 실험</h1>
<p><b>새 혼합 모델은 앞 구간에서 작은 개선을 보였지만 뒤 구간에서 기존 계좌를 넘지 못했다. 기존 개선된 기준시장 decisiongate 후보는 유지한다.</b> 이 실험에서 더 중요한 발견은, 캔들 거래량 제한이 실제 호가의 체결 가능 잔량과 다르다는 점을 같은 과거 시점의 호가로 확인한 것이다. 이 발견 자체는 수익 개선 증거가 아니다.</p>
<h2>구현과 평가</h2><p>기존 1,028개 가상 진입 경로를 다시 실행하지 않고, 최초 IOC 도착 시점의 미체결·불균형 체결·균형 체결을 분리했다. 아직 청산되지 않은 경로도 관측된 초기 체결 상태만 학습에 쓴다. 이후 실제 정리된 순손익만 조건부 손익 학습에 사용하며, 미정리 손익을 0으로 만들지 않는다. 조건부 손익 학습에 남는 검열 편향은 해결됐다고 주장하지 않는다.</p>
<p>초기 상태는 고정 깊이 2의 부스팅 분류기, 조건부 손익은 규제된 릿지 모델이다. 샘플이 12건 미만인 조건부 모델은 0으로 축소한 평균을 쓴다. 같은 입력의 순손익 부스팅 회귀도 비교했다. 이전 특징에 과거 거래 활동 지속성·두 종목의 동시 활동 전이·주문 최소 단위·목표 수량을 추가했다. 창 길이·학습일·규제·횟수를 결과 전에 고정했고 주문 크기·지정가·청산 규칙은 유지했다. 미래 거래량과 미래 결과를 바꾸어도 과거 특징과 모델이 변하지 않는 검사를 포함했다.</p>
<p>10월 8~12일 12:00 UTC에서 먼저 선정을 저장한 뒤 12~16일, 12~17일을 평가했다. 뒤 두 구간은 겹치며 모든 구간은 재사용된 개발 자료다. 새 계좌 24개, 기존 계좌의 체결·잔고·주문·판단·성과 경로 대조 42개, 관련 검사 7개를 완료했다. 전량 미체결 상태의 조건부 손익은 정확히 0이다. 예측값은 기본 비용 차감 후 값이며 불리한 조건에서는 기존과 같은 추가 비용만 차감했다.</p>
<p><a href="https://arxiv.org/html/2512.05734v1">KANFormer(2025)</a>는 주문 대기열·호가·주문 행동을 이용하고 검열을 고려해 체결 시간을 예측한다. 여기서는 체결 상태와 손익을 구분한다는 문제 설정을 참고했다. 필요한 주문 대기열 자료가 없으므로 논문 모델이나 수익을 재현한 결과가 아니다. <a href="https://arxiv.org/abs/2403.02572">상태 의존 주문 흐름 연구</a> 역시 실제 대기열 모형과 이 캔들 기반 대용 모형을 동일시할 근거가 되지 않는다.</p>
<h2>기본 조건 계좌 결과</h2><p>초기자본 1M. 평가손익을 포함한다. 전체 기본/불리 조건과 이전 모델은 results.csv에 있다.</p><table><tr>'''
    display=['phase','arm','net_pnl','financial_score','completed_pairs','aborted_pairs','unbalanced_minutes']
    page+=''.join(f'<th>{k}</th>' for k in display)+'</tr>'
    for r in results['rows']:
        if r['scenario']!='base' or (r['arm']!='first_strategy' and '_hurdle_' not in r['arm']):continue
        page+='<tr>'+''.join(f'<td>{v:.8f}</td>' if isinstance(v,float) else f'<td>{html.escape(str(v))}</td>' for v in [r.get(k,r['metrics'].get(k)) for k in display])+'</tr>'
    page+='</table><h2>체결 확률과 손익 정확도</h2><p>다중 클래스 Brier는 낮을수록 좋다. 고정 확률 비교는 해당 학습 시점의 과거 비율이다. 뒤 4일에서는 소폭 개선됐지만 하루를 더한 겹치는 구간에서는 오히려 나빠졌다. 양쪽 균형 체결 후 현금 손익을 학습할 표본은 학습 시점별 3~14개뿐이다. 가상 경로는 양 방향·보유기간·시점이 겹치며 독립 거래 수가 아니다.</p><pre>'+html.escape(json.dumps(diagnostics,ensure_ascii=False,indent=2))+'</pre>'
    page+='''<h2>실제 호가와 캔들 거래량 제한의 차이</h2><p>추가 진단은 수익 신호를 사용하지 않고, 같은 120k 상한과 원래 위험 한도에서 독립적인 가상 주문을 만들었다. 당시 호가로 지정가를 고정하고, 1초/3초 뒤 수신 호가의 실제 잔량에 각각 체결했다. 앞 캔들 거래량이 0인 경우에도 양쪽이 체결되는 사례가 있었다. 캔들 모형의 5분 지연과 호가 모형의 지연은 다르므로 이를 동일 실행 조건의 수익 비교로 해석하면 안 된다. 서로 겹치는 주문을 하나의 포트폴리오로 합산하지 않는다.</p>'''
    page+=f"<p>진단 {capacity['counts']['probes']}회 중 앞 캔들에서 두 종목 모두 거래되지 않은 경우 {capacity['counts']['source_inactive']}회, 그 중 양쪽 균형 체결 {len(mismatch)}회였다. 기본/불리 조건을 중복 제거하면 서로 다른 시점·페어·방향은 <b>{len(unique_mismatch)}개</b>다. 그 외 대부분은 불균형 체결이어서 체결 가능성만으로 안전한 페어 진입이라고 볼 수도 없다.</p>"
    page+='<table><tr><th>UTC 시각</th><th>페어</th><th>조건</th><th>캔들 기반 가능 금액</th><th>호가 모의 체결 금액</th></tr>'
    for r in mismatch:
        stamp=time.strftime('%Y-%m-%d %H:%M',time.gmtime(r['time']))
        page+=f"<tr><td>{stamp}</td><td>{'/'.join(r['pair'])}</td><td>{r['scenario']}</td><td>{r['candle_volume_gross_capacity']:.2f}</td><td>{r['received_filled_gross']:.2f}</td></tr>"
    page+='</table><p>위 금액은 수익이 아니라 독립적인 모의 진입 명목금액이다. 이 주문들의 장기 청산 수익은 확인하지 않았다. 이번 모델이 만든 실제 신호의 호가 점검은 별도 3회이며, 진입 실패 후 유료 정리·양쪽 진입 후 만기 관측 부족·전량 미체결이 각각 1회였다. 여기에도 전체 수익률이나 Q를 붙이지 않는다.</p>'
    page+='''<h2>다음 구현 방향</h2><p>캔들 거래량 상한에서 생성한 라벨을 실제 거래소 체결 확률로 해석하거나, 이 가정 위에서 분류기만 계속 변경하지 않는다. 이미 개선된 기준시장 예측은 유지하고, 실제 호가 잔량으로 양쪽 목표 수량을 맞추는 실행 계층을 별도 비교한다. 현재 가격에서 예상 손익과 비용을 다시 계산하고 원래 지정가·위험 한도·유료 정리를 지킨다. 짧은 호가 기록으로 장기 수익을 만들거나 보류된 미래 자료를 미리 보지 않는다.</p>
<p>기존 별도 기준시장 8월 1일~10월 15일 개선 후보의 +25,868.79 / Q 0.30858은 유지된다. 그 결과를 이 짧은 par 결과나 실제 거래 수익과 섞지 않는다. 실거래 주문·교체는 없다.</p>'''
    (dest/'report.html').write_text(page)
    files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',OUT/'models.json',OUT/'labels.json',OUT/'results.json',
           OUT/'diagnostics.json',OUT/'conclusion.json',OUT/'selection.json',OUT/'books/results.json',OUT/'book_reconciliation/results.json',
           OUT/'capacity/results.json',*sorted((OUT/'runs').glob('*.json.gz')),dest/'report.html',dest/'results.csv']
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'conclusion_frozen.json',dict(at_unix=time.time(),sources=sources,prospective_sources=len(prospective)))
    save(HERE/'work_state.json',dict(status='progress_hurdle_comparison_and_volume_depth_mismatch_audit_completed',
        report=str((dest/'report.html').relative_to(ROOT)),new_accounts=24,exact_controls=42,tests_passed=7,
        next_action=conclusion['next_action'],goal_complete=False,no_progress_streak=0,orders_sent=0,live_strategy_changed=False))
    print('DONE',dest/'report.html','sources',len(sources),'unique source-volume mismatches',len(unique_mismatch),flush=True)


if __name__=='__main__':main()
