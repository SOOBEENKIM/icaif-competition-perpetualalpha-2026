"""Cost-sensitive entry gate on unchanged cash_direct forecasts and execution."""
import copy,csv,gzip,html,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.adaptive_residual_pairs_20261017.model import frozen_predictions
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.model import design,labels,map_state
from experiments.dense_history_pairs_20261017.research import load
from experiments.rolling_feature_pairs_20261017.model import rolling as direct_rolling
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as PAR_INPUT,save,sha
from experiments.conditional_completion_pairs_20261017.research import verify
from .portfolio import replay
from .model import rolling,map_gate

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
PARENT=ROOT/'experiments/conditional_completion_pairs_20261017/artifacts'
DIRECT=ROOT/'experiments/rolling_feature_pairs_20261017/artifacts'
EXT=ROOT/'experiments/historical_par_extension_20261017/artifacts_v3'
MATCHED='first_dense_hourlyhedge_cash_direct_24';ARM='first_dense_hourlyhedge_decisiongate_24'


def path_for(venue,phase,sc):return PARENT/'runs'/f'{venue}-{phase}-{MATCHED}-{sc}.json.gz'


def render(result):
    dest=ROOT/'reports/decision_gate_pairs_20261017';dest.mkdir(exist_ok=True)
    keys=['venue','phase','arm','scenario','net_pnl','roi','financial_score','volatility_horizon','max_drawdown','completed_pairs','unbalanced_minutes']
    with (dest/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in result['rows']:w.writerow({k:r.get(k,r['metrics'].get(k)) for k in keys})
    page='''<!doctype html><meta charset="utf-8"><title>비용을 학습하는 페어 진입 필터</title><style>body{font:15px system-ui;margin:32px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:7px;text-align:right}td:first-child{text-align:left}pre{background:#f5f5f5;padding:16px;white-space:pre-wrap}</style>
<h1>비용을 학습하는 페어 진입 필터</h1><p>기존 개선판 cash_direct의 예상 가격·헤지·주문 수량·지정가·위험 한도·펀딩·보유 연장·정리 규칙을 유지한다. 새 모델은 같은 20개 특징으로 롱/숏/대기 판단의 SPO+ 손실을 학습하며, 신규 진입의 방향만 허용하거나 거절한다. 점수를 수익률이나 미래 가격으로 대입하지 않는다.</p>
<p>학습용 보조 문제는 판단 시점부터 24시간의 고정 헤지 가격 수익률에서 왕복 38bp를 차감하는 세 행동 문제다. 이 보조 문제에는 실제 펀딩·지연·수량 제한·정리·보유 연장이 없으며 계좌 성과가 아니다. 실제 백테스트는 기존 엔진으로 이 조건들을 별도 반영한다. 모델은 일별 갱신, 과거 90일, 6시간 표본, 30일 가중 반감기, 최소 200개의 엄격히 만기가 지난 표본을 사용한다. 표본은 서로 겹친다.</p>
<p>원래 가격 예측과 통제 계좌 12개를 재현하고 새 계좌 12개를 비교한다. 초기 1M, 동일 기본/불리한 실행 조건. 8월에서 먼저 판정하고 이후를 평가하되 모든 과거 기간은 이미 사용한 개발 자료임을 명시한다. 완료 페어 수는 증거량이지 공식 제출 요건이 아니다. 실제 par 호가와 캔들 거래량은 구분한다. 실거래 변경은 없다.</p>
<p><a href="https://arxiv.org/abs/1710.08005">SPO+ 원 논문</a>의 손실을 세 행동·고정 비용 문제에 독립적으로 유도했다. <a href="https://arxiv.org/html/2609.33472v1">2026년 9월 연구</a>가 설명하듯 의사결정의 일치가 평균 수익률 예측의 일치를 보장하지 않으므로, 이 모델은 기존 가격 예측을 대체하지 않는다. 논문 실험 성과를 재현했다는 주장이 아니다.</p>'''
    page+='<h2>8월 사전 판정</h2><pre>'+html.escape(json.dumps(result['selection'],ensure_ascii=False,indent=2))+'</pre>'
    for venue,phase in result['windows']:
        page+=f'<h2>{venue} / {phase}</h2><table><tr><th>전략</th><th>비용</th><th>순손익</th><th>수익률</th><th>Q</th><th>기간 변동성</th><th>MDD</th><th>완료 페어</th><th>미헤지 분</th></tr>'
        for r in result['rows']:
            if (r['venue'],r['phase'])!=(venue,phase):continue
            m=r['metrics'];q='—' if m['financial_score'] is None else f"{m['financial_score']:.8f}"
            v='—' if m['volatility_horizon'] is None else f"{100*m['volatility_horizon']:.6f}%"
            page+=f"<tr><td>{html.escape(r['arm'])}</td><td>{r['scenario']}</td><td>{m['net_pnl']:+.6f}</td><td>{100*m['roi']:+.6f}%</td><td>{q}</td><td>{v}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{m['completed_pairs']}</td><td>{m['unbalanced_minutes']:.0f}</td></tr>"
        page+='</table>'
    (dest/'report.html').write_text(page);return str((dest/'report.html').relative_to(ROOT))


def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text());tests=(HERE/'test_results.txt').read_text()
    assert 'Ran 10 tests' in tests and tests.rstrip().endswith('OK')
    parent=json.loads((PARENT/'results.json').read_text());windows=parent['windows']
    rows=[r for r in parent['rows'] if r['arm'] in ['first_strategy','second_strategy',MATCHED]];assert len(rows)==36
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    refs={'reference':ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
          'par_extended':ROOT/'practice_live/updates/20261017T120009814481Z/inputs.npz'}
    expert=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/model_protocol.json').read_text());models={};model_paths=[]
    for pair in expert['pairs']:
        path=ROOT/expert['model_directory']/('_'.join(pair)+'_24.json');models[tuple(pair),24]=json.loads(path.read_text());model_paths.append(path)
    sources=json.loads((PARENT/'frozen.json').read_text())['sources'].copy()
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',PARENT/'results.json',PARENT/'frozen.json',*refs.values(),*model_paths,
           ROOT/'experiments/shared_factor_pairs_20261017/artifacts/conclusion.json',ROOT/'experiments/pairs_development_status_20261017/status.json']
    paths+=[path_for(v,phase,sc) for v,phase in windows for sc in ['base','stress']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn='progress_shared_factor_comparison_completed',historical_independence=False))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':load(PAR_INPUT/'inputs.npz'),'par_extended':load(EXT/'inputs.npz')}
    controls={v:load(DIRECT/f'{v}_direct_24_state.npz') for v in ['reference','par']};controls['par_extended']=load(EXT/'direct_state.npz')
    states={};preparation=[]
    for venue,path in refs.items():
        ref=load(path);pairs=cfgs[venue]['dense_pairs']
        f=design(ref['market_times'],ref['close'],ref['volume'],ref['trades'],p['symbols'],pairs);raw=frozen_predictions(f,models,pairs,24)
        y,maturity=labels(ref['market_times'],ref['close'],f,24);terminal=np.zeros_like(f['fresh']);terminal[:-24]=f['fresh'][24:]
        old_spec={**spec,'candidates':['direct'],'diagnostic_variants':[]}
        old,_,_=direct_rolling(ref['market_times'],f,y,maturity,raw,terminal,old_spec)
        actions,scores,available,stamps,records=rolling(ref['market_times'],f,y,maturity,raw,terminal,spec)
        np.savez_compressed(OUT/f'{venue}_issued_forecasts.npz',times=ref['market_times'],target=y,maturity=maturity,
            unit=f['vol']*np.sqrt(24),prediction_direct=old['direct'],economic_action=actions,economic_available=available,
            decision_score=scores,fit_stamps=stamps,fresh=f['fresh'],terminal_fresh=terminal)
        save(OUT/f'{venue}_fits.json',records)
        for v in (['reference'] if venue=='reference' else ['par','par_extended']):
            exact=map_state(data[v]['times'],ref,f,old['direct'],f['pairs'],source_hours=1,lifetime_hours=1)
            for key in controls[v]:np.testing.assert_array_equal(exact[key],controls[v][key],err_msg=f'{v}-{key}')
            state=map_gate(data[v]['times'],ref['market_times'],actions,available,exact)
            for key in controls[v]:np.testing.assert_array_equal(state[key],controls[v][key])
            np.savez_compressed(OUT/f'{v}_gate_state.npz',**state);states[v]=state
        preparation.append(dict(venue=venue,models=len(records),last_source=int(ref['market_times'][-1]),
            strict_maturity=all(r['last_maturity']<r['fit_time'] for r in records),
            max_solver_gap=max(r['solver']['gap'] for r in records),max_solver_iterations=max(r['solver']['iterations'] for r in records)))
        print('PREPARED',preparation[-1],flush=True)
    save(OUT/'preparation.json',preparation)
    verified=[]
    for venue,phase in windows:
        for sc in ['base','stress']:
            with gzip.open(path_for(venue,phase,sc),'rt') as f:old=json.load(f)
            cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
            run=replay(data[venue],controls[venue],None,cfg,p,MATCHED,sc,old['start'],old['end'])
            for key in ['events','fills','metrics','snapshots']:assert run[key]==old[key],(venue,phase,sc,key)
            verified.append(dict(venue=venue,phase=phase,scenario=sc))
        print('EXACT',venue,phase,flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    def record(venue,phase,sc):
        cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
        with gzip.open(path_for(venue,phase,sc),'rt') as f:old=json.load(f)
        run=replay(data[venue],states[venue],None,cfg,p,ARM,sc,old['start'],old['end'])
        with gzip.open(OUT/'runs'/f'{venue}-{phase}-{ARM}-{sc}.json.gz','wt') as f:json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=run['metrics'];lookup[venue,phase,ARM,sc]=m
        rows.append(dict(venue=venue,phase=phase,arm=ARM,scenario=sc,metrics=m,pnl_advantage_matched=m['net_pnl']-lookup[venue,phase,MATCHED,sc]['net_pnl']))
        print(venue,phase,sc,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],'pairs',m['completed_pairs'],flush=True)
    for sc in ['base','stress']:record('reference','validation',sc)
    checks=[]
    for sc in ['base','stress']:
        m=lookup['reference','validation',ARM,sc]
        checks.append(dict(scenario=sc,positive_pnl=bool(m['net_pnl']>0),no_risk_breach=bool(m['risk_breach_bars']==0),
            above_native_pnl_q=bool(all(m['net_pnl']>lookup['reference','validation',b,sc]['net_pnl'] and m['financial_score']>lookup['reference','validation',b,sc]['financial_score'] for b in ['first_strategy','second_strategy'])),
            above_matched_pnl_q=bool(m['net_pnl']>lookup['reference','validation',MATCHED,sc]['net_pnl'] and m['financial_score']>lookup['reference','validation',MATCHED,sc]['financial_score']),
            completed_pairs=int(m['completed_pairs']),internal_sample_flag=bool(m['completed_pairs']>=3)))
    selection=dict(candidate=ARM,checks=checks,historical_improvement_over_natives=all(r['above_native_pnl_q'] for r in checks),
        historical_improvement_over_retained=all(r['above_matched_pnl_q'] for r in checks),
        selected_for_further_research=all(r['positive_pnl'] and r['no_risk_breach'] and r['above_native_pnl_q'] and r['above_matched_pnl_q'] for r in checks),
        sample_flag_is_not_official_requirement=True,frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);decision_hash=sha(OUT/'selection.json')
    for venue,phase in windows[1:]:
        for sc in ['base','stress']:record(venue,phase,sc)
    assert sha(OUT/'selection.json')==decision_hash;verify(sources);verify(prospective)
    result=dict(rows=rows,windows=windows,selection=selection,new_account_replays=12,exact_control_replays=verified,tests_passed=10,
        protected=protected(),orders_sent=0,live_strategy_changed=False,historical_independence=False,reserved_raw_read=False)
    save(OUT/'results.json',result);report=render(json.loads((OUT/'results.json').read_text()))
    save(HERE/'work_state.json',dict(status='progress_decision_gate_accounts_completed',report=report,selection=selection,
        new_account_replays=12,exact_controls=12,tests_passed=10,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,
        historical_independence=False,goal_complete=False,no_progress_streak=0))
    print('DONE',report,flush=True)


if __name__=='__main__':main()
