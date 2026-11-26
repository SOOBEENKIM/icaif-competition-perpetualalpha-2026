"""Post-comparison prediction/episode attribution; never reselect a candidate."""
import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha
from experiments.dense_history_pairs_20261017.research import load
from .research import HERE,OUT,PARENT,MATCHED,DIRECT,verify


def main():
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
    d,_,_=inputs(cfg);forecasts=load(OUT/'reference_issued_forecasts.npz')
    original=load(DIRECT/'reference_issued_forecasts.npz');pred=forecasts['predictions'];beta=forecasts['beta']
    actual_legs=forecasts['target']*forecasts['unit']
    actual=(actual_legs[...,0]-beta*actual_legs[...,1])/(1+beta)
    joint=(pred[...,0]-beta*pred[...,1])/(1+beta)
    direct=original['prediction_direct'];times=forecasts['times']
    valid=forecasts['fresh']&forecasts['terminal_fresh']&np.isfinite(actual)&np.isfinite(joint)&np.isfinite(direct)
    diagnostics=[];episodes=[]
    arms={'direct_cash':MATCHED,'joint':'first_dense_hourlyhedge_joint_24','common':'first_dense_hourlyhedge_common_24'}
    for phase in ['validation','comparison']:
        with gzip.open(PARENT/'runs'/f'reference-{phase}-{MATCHED}-base.json.gz','rt') as f:control=json.load(f)
        mask=valid&((times>=control['start'])&(forecasts['maturity']<=control['end'])&(times%86400==43200))[:,None]
        for kind,value in [('zero',np.zeros_like(joint)),('direct',direct),('joint',joint)]:
            a=actual[mask];v=value[mask]
            diagnostics.append(dict(phase=phase,model=kind,nonoverlap_pair_days=len(a),
                mean_prediction_bps=float(v.mean()*10000),mean_actual_bps=float(a.mean()*10000),
                rmse_bps=float(np.sqrt(np.mean((v-a)**2))*10000),
                directional_accuracy=float(np.mean(v*a>0)) if kind!='zero' else None,
                note='Pair-days overlap across assets; 24h-spaced forecast sources do not make different pairs statistically independent.'))
        for kind,arm in arms.items():
            for sc in ['base','stress']:
                path=(PARENT if kind=='direct_cash' else OUT)/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz'
                with gzip.open(path,'rt') as f:run=json.load(f)
                audit=episode_attribution(run,d,cfg['scenarios'][sc])
                for episode in audit['episodes']:
                    intent=next(e for e in run['events'] if e['event']=='pair_intent' and e['time']==episode['time'])
                    episode['expected_price_bps']=intent['expected_price_bps']
                    episode['expected_funding_bps']=intent['expected_funding_bps']
                    episode['predicted_leg_returns']=(np.array(intent['terminal_prices'])/intent['source_prices']-1).tolist()
                    episode['direction']=intent['direction']
                    episode['exit_reasons']=[e['reason'] for e in run['events'] if episode['time']<=e['time']<=episode['end'] and e['event']=='exit_requested']
                episodes.append(dict(phase=phase,model=kind,scenario=sc,base_owned_pnl=run['metrics']['base_owned_pnl'],**audit))
    books=json.loads((OUT/'books/results.json').read_text())
    result=dict(at_unix=time.time(),prediction_checks=diagnostics,episode_attribution=episodes,book_summary=books['summary'],
        conclusion='Neither candidate qualifies. Joint worsens later reference base cost; common leaves account outcomes unchanged. All par candidate fills and economic account results match the conditional-cash parent, with no completed pairs. Joint adds six unfilled order records in each of the overlapping par comparison and extended windows. No candidate passes current-book economics on the allowed old observations.',
        next_action='Do not add model capacity or loosen costs based on August. Audit the failure of the source-time price-spread signal and training target relative to later pair price movements; explicitly separate forecast error from execution costs. Preserve current live and reserved independent evaluation.',
        historical_independence=False,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,goal_complete=False)
    save(OUT/'diagnostics.json',result)
    for row in diagnostics:print(row,flush=True)
    for row in episodes:
        if row['phase']=='comparison' and row['model']=='joint' and row['scenario']=='base':
            print('LATER_JOINT_EPISODES',json.dumps(row,ensure_ascii=False),flush=True)
    state_path=HERE/'work_state.json';s=json.loads(state_path.read_text());s.update(status='progress_joint_leg_rejected_after_accounts_books_and_attribution',
        diagnostics=str((OUT/'diagnostics.json').relative_to(ROOT)),book_exact_controls=books['exact_control_bound_observations'],
        next_action=result['next_action'],performance_improvement_established=False,previous_turn_classification='progress: capacity and expected-return bound audit completed')
    save(state_path,s)
    report=ROOT/'reports/joint_leg_pairs_20261017/report.html';text=report.read_text()
    extra='''<section id="diagnostics"><h2>결과 판정과 실제 호가 점검</h2>
<p><strong>두 후보 모두 교체 기준을 통과하지 못했다.</strong> joint의 8월 기본 비용 PnL은 +19,580.62 USDPR로 부모 후보 +14,238.30보다 높았지만, 완료 페어가 1개뿐이고 불리한 비용에서는 부모 후보보다 낮았다. 9월 1일–10월 15일 기본 비용 PnL은 −37.71 USDPR로 부모 후보 +3,422.16 및 기존 첫째 +459.53, 둘째 +572.88보다 낮았다. common의 모든 계좌 성과는 부모 후보와 같았다.</p>
<p>par에서도 두 후보 모두 부모 후보의 체결·계좌 성과를 개선하지 못했고 완료 페어는 0개다. 확장 기간 Oct8–Oct17 기본 비용 PnL −229.84 USDPR, 단독 하루 Oct16–Oct17 −50.38 USDPR이며 실제 지갑 수익이 아닌 모의 계좌 결과다. 단독 하루는 σ/Q를 계산할 표본이 부족하다.</p>
<p>기존 공개 호가의 동일한 17개 시각·34개 종목쌍 시각을 양방향으로 대조했다. 두 비용 조건 각각 68개 관측에서 두 후보 모두 비용을 넘는 진입 계획이 없었다. 기존 direct의 낙관적 최우선 호가 상한 136개를 정확히 재현했다. 이 기록은 24시간 진입·청산 경로가 아니므로 실제 수익률이나 Q를 부여하지 않는다.</p>
<p>두 출력 모델은 계산의 표현력을 높였으나 수익 예측 개선으로 이어지지 않았다. 8월 성과만으로 채택하거나 비용·진입 조건을 완화하지 않는다. 추가 회계 분해 및 24시간 간격 예측 오차는 diagnostics.json에 저장했다. 미헤지 시간은 캔들·거래량 모의 체결에 따른 값이며 실제 거래소의 청산 소요시간이 아니다.</p></section>'''
    assert 'id="diagnostics"' not in text
    report.write_text(text.replace('<h2>8월 후보 선정</h2>',extra+'<h2>8월 후보 선정</h2>',1))
    verify(json.loads((OUT/'frozen.json').read_text())['sources'])
    verify(json.loads((OUT/'books/frozen.json').read_text())['sources'])
    reserved=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(reserved)
    print('DONE frozen sources intact; reserved dependencies',len(reserved),flush=True)


if __name__=='__main__':main()
