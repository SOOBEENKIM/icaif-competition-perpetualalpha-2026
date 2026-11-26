"""Daily raw-return target ablation; all trading logic remains the frozen parent."""
import copy,csv,gzip,html,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.adaptive_residual_pairs_20261017.model import frozen_predictions
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.model import design,labels,map_state
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as PAR_INPUT,save,sha
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.conditional_completion_pairs_20261017.portfolio import replay
from .model import rolling

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
PARENT=ROOT/'experiments/conditional_completion_pairs_20261017/artifacts'
DIRECT=ROOT/'experiments/rolling_feature_pairs_20261017/artifacts'
EXT=ROOT/'experiments/historical_par_extension_20261017/artifacts_v3'
MATCHED='first_dense_hourlyhedge_cash_direct_24';ARM='first_dense_hourlyhedge_raw_24'


def path_for(venue,phase,sc):return PARENT/'runs'/f'{venue}-{phase}-{MATCHED}-{sc}.json.gz'


def render(result):
    dest=ROOT/'reports/raw_return_pairs_20261017';dest.mkdir(exist_ok=True)
    keys=['venue','phase','arm','scenario','net_pnl','roi','financial_score','volatility_horizon','max_drawdown','completed_pairs','unbalanced_minutes']
    with (dest/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in result['rows']:w.writerow({k:r.get(k,r['metrics'].get(k)) for k in keys})
    page='''<!doctype html><meta charset="utf-8"><title>페어 수익률 정규화 비교</title><style>body{font:15px system-ui;margin:32px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:7px;text-align:right}td:first-child{text-align:left}pre{background:#f4f5f6;padding:16px;white-space:pre-wrap}</style>
<h1>페어 수익률을 직접 예측하는 비교</h1><p>기존 일별 갱신 릿지는 페어의 24시간 수익률을 해당 시점의 변동성으로 나누어 학습하고, 예측 시 현재 변동성을 다시 곱한다. 새 raw 후보는 같은 자료에서 변동성으로 나누지 않은 수익률을 학습하고 직접 출력한다. 입력 특징·학습 표본·학습 시각·가중치·릿지 규제·목표값 이상치 제한·예측 상한·헤지 비율은 유지한다.</p>
<p>이는 학습 목표와 손실의 상대 가중치에 대한 가설이다. 기존 정규화가 구현 오류라는 뜻이 아니며, 과거 다른 par 모델도 원래 수익률을 예측한 바 있다. 이번에는 장기간 기준시장 자료를 사용하는 동일한 일별 종목쌍 회귀에서 이 차이만 분리한다.</p>
<p>모형 검증 8개, 원래 정규화 예측 배열 및 부모 계좌 12개 정확 재현, 새 후보 계좌 12개. 초기 1M, 동일 수수료·지연·양쪽 독립 체결·펀딩·청산·위험 한도. 기존 조건부 헤지 완성 및 유료 보유 연장 구조를 그대로 사용한다. 실제 주문이나 실거래 교체는 없다.</p>
<p>기준시장 8월에서 먼저 판정하고 이후 기준시장 Sep1–Oct15, par Oct1–Oct8 및 Oct8–Oct16, 확장 par Oct8–Oct17 및 하루 Oct16–Oct17을 비교한다. 모두 12UTC 경계다. 과거 자료를 재사용한 개발 비교이며 독립 검증이나 실제 지갑 성과가 아니다. 캔들 거래량은 실제 호가 잔량과 구분하며 미헤지 시간은 모의 체결 결과다. 하루 단독 Q/변동성은 표본 부족으로 공란이다.</p>'''
    page+='<h2>8월 판정</h2><pre>'+html.escape(json.dumps(result['selection'],ensure_ascii=False,indent=2))+'</pre>'
    for venue,phase in result['windows']:
        page+=f'<h2>{venue} / {phase}</h2><table><tr><th>전략</th><th>비용</th><th>PnL USDPR</th><th>수익률</th><th>Q</th><th>기간 변동성</th><th>MDD</th><th>완료 페어</th><th>미헤지 분</th></tr>'
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
    assert 'Ran 8 tests' in tests and tests.rstrip().endswith('OK')
    parent=json.loads((PARENT/'results.json').read_text());windows=parent['windows']
    rows=[r for r in parent['rows'] if r['arm'] in ['first_strategy','second_strategy',MATCHED]];assert len(rows)==36
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    refs={'reference':ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
          'par_extended':ROOT/'practice_live/updates/20261017T120009814481Z/inputs.npz'}
    expert_spec=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/model_protocol.json').read_text());models={}
    model_paths=[]
    for pair in expert_spec['pairs']:
        path=ROOT/expert_spec['model_directory']/('_'.join(pair)+'_24.json')
        models[tuple(pair),24]=json.loads(path.read_text());model_paths.append(path)
    sources=json.loads((PARENT/'frozen.json').read_text())['sources'].copy()
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',PARENT/'results.json',PARENT/'frozen.json',*refs.values(),*model_paths,
           ROOT/spec['prior_artifact'],ROOT/'experiments/joint_leg_pairs_20261017/artifacts/conclusion.json']
    paths += [path_for(v,phase,sc) for v,phase in windows for sc in ['base','stress']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths});verify(sources)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn='progress_training_target_weight_audit_completed',historical_independence=False))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':load(PAR_INPUT/'inputs.npz'),'par_extended':load(EXT/'inputs.npz')}
    controls={v:load(DIRECT/f'{v}_direct_24_state.npz') for v in ['reference','par']};controls['par_extended']=load(EXT/'direct_state.npz')
    states={};preparation=[]
    for venue,path in refs.items():
        ref=load(path);pairs=cfgs[venue]['dense_pairs']
        f=design(ref['market_times'],ref['close'],ref['volume'],ref['trades'],p['symbols'],pairs)
        raw=frozen_predictions(f,models,pairs,24)
        y,maturity=labels(ref['market_times'],ref['close'],f,24);terminal=np.zeros_like(f['fresh']);terminal[:-24]=f['fresh'][24:]
        cash,control,stamps,records=rolling(ref['market_times'],f,y,maturity,raw,terminal,spec)
        np.savez_compressed(OUT/f'{venue}_issued_forecasts.npz',times=ref['market_times'],target=y,maturity=maturity,
                            unit=f['vol']*np.sqrt(24),prediction_raw=cash,prediction_direct=control,fit_stamps=stamps,fresh=f['fresh'],terminal_fresh=terminal)
        save(OUT/f'{venue}_fits.json',records)
        for v in (['reference'] if venue=='reference' else ['par','par_extended']):
            exact=map_state(data[v]['times'],ref,f,control,f['pairs'],source_hours=1,lifetime_hours=1)
            for key in controls[v]:np.testing.assert_array_equal(exact[key],controls[v][key],err_msg=f'{v}-{key}')
            state=map_state(data[v]['times'],ref,f,cash,f['pairs'],source_hours=1,lifetime_hours=1)
            np.testing.assert_array_equal(state['directions'],controls[v]['directions'])
            np.savez_compressed(OUT/f'{v}_raw_state.npz',**state);states[v]=state
        preparation.append(dict(venue=venue,models=len(records),last_source=int(ref['market_times'][-1]),strict_maturity=all(r['last_maturity']<r['fit_time'] for r in records)))
        print('PREPARED',venue,len(records),'fits; original states exact',flush=True)
    save(OUT/'preparation.json',preparation)
    verified=[]
    for venue,phase in windows:
        for sc in ['base','stress']:
            with gzip.open(path_for(venue,phase,sc),'rt') as f:old=json.load(f)
            cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
            run=replay(data[venue],controls[venue],None,cfg,p,MATCHED,sc,old['start'],old['end'])
            for key in ['events','fills','metrics','snapshots']:assert run[key]==old[key],(venue,phase,sc,key)
            verified.append(dict(venue=venue,phase=phase,scenario=sc))
        print('EXACT',venue,phase,'two matched accounts',flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    def record(venue,phase,sc):
        cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
        with gzip.open(path_for(venue,phase,sc),'rt') as f:old=json.load(f)
        run=replay(data[venue],states[venue],None,cfg,p,ARM,sc,old['start'],old['end'])
        with gzip.open(OUT/'runs'/f'{venue}-{phase}-{ARM}-{sc}.json.gz','wt') as f:json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=run['metrics'];lookup[venue,phase,ARM,sc]=m
        rows.append(dict(venue=venue,phase=phase,arm=ARM,scenario=sc,metrics=m,pnl_advantage_matched=m['net_pnl']-lookup[venue,phase,MATCHED,sc]['net_pnl']))
        print(venue,phase,sc,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],'pairs',m['completed_pairs'],'unbalanced',m['unbalanced_minutes'],flush=True)
    for sc in ['base','stress']:record('reference','validation',sc)
    checks=[];margins=[]
    for sc in ['base','stress']:
        m=lookup['reference','validation',ARM,sc]
        checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
        for baseline in ['first_strategy','second_strategy',MATCHED]:
            b=lookup['reference','validation',baseline,sc];checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
            if baseline!=MATCHED:margins.append(m['net_pnl']-b['net_pnl'])
    selection=dict(candidate=ARM,eligible=all(checks),minimum_native_pnl_advantage=min(margins),frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for venue,phase in windows[1:]:
        for sc in ['base','stress']:record(venue,phase,sc)
    assert sha(OUT/'selection.json')==selection_hash;verify(sources)
    result=dict(rows=rows,windows=windows,selection=selection,new_account_replays=12,exact_control_replays=verified,tests_passed=8,
                protected=protected(),orders_sent=0,live_strategy_changed=False,historical_independence=False,reserved_raw_read=False)
    save(OUT/'results.json',result);report=render(result)
    save(HERE/'work_state.json',dict(status='progress_raw_target_accounts_completed',report=report,selection=selection,
        new_account_replays=12,exact_controls=12,tests_passed=8,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,goal_complete=False))
    print('DONE',report,flush=True)


if __name__=='__main__':main()
