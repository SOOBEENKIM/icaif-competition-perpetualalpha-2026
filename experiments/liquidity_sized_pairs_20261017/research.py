"""Frozen initial liquidity sizing comparison; original execution unchanged."""
import copy,csv,gzip,html,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as PAR_INPUT,save,sha
from experiments.dense_history_pairs_20261017.research import load
from experiments.conditional_completion_pairs_20261017.research import old_path,verify,DIRECT,EXT,CONTROL
from .portfolio import replay
from .policy import capacity_volume

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
PARENT=ROOT/'experiments/conditional_completion_pairs_20261017/artifacts'
CASH='first_dense_hourlyhedge_cash_direct_24'

def path_for(venue,phase,arm,scenario):
    if arm==CONTROL:return old_path(venue,phase,scenario)
    assert arm==CASH
    return PARENT/'runs'/f'{venue}-{phase}-{arm}-{scenario}.json.gz'

def render(result):
    report=ROOT/'reports/liquidity_sized_pairs_20261017';report.mkdir(exist_ok=True)
    columns=['venue','phase','arm','scenario','net_pnl','roi','financial_score','volatility_horizon','max_drawdown','mean_gross','maximum_gross','fees','funding','completed_pairs','aborted_pairs','unbalanced_minutes','residual_gross','pnl_after_liquidation_reserve']
    with (report/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in result['rows']:writer.writerow({k:row[k] if k in row else row['metrics'].get(k) for k in columns})
    tables=[]
    for venue,phase in result['windows']:
        rows=[]
        for row in result['rows']:
            if (row['venue'],row['phase'])!=(venue,phase):continue
            m=row['metrics'];q='—' if m['financial_score'] is None else f"{m['financial_score']:.8f}"
            vol='—' if m['volatility_horizon'] is None else f"{100*m['volatility_horizon']:.6f}%"
            rows.append(f"<tr><td>{html.escape(row['arm'])}</td><td>{row['scenario']}</td><td>{m['net_pnl']:+.6f}</td><td>{q}</td><td>{vol}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{100*m['mean_gross']:.4f}%</td><td>{m['completed_pairs']}</td><td>{m['aborted_pairs']}</td><td>{m['unbalanced_minutes']:.0f}</td><td>{m['residual_gross']:.3f}</td></tr>")
        tables.append(f'<h2>{venue} / {phase}</h2><table><tr><th>전략</th><th>비용</th><th>PnL USDPR</th><th>Q</th><th>기간 변동성</th><th>MDD</th><th>평균 총노출</th><th>완료 페어</th><th>진입 실패</th><th>미헤지 분</th><th>잔여 명목액</th></tr>'+''.join(rows)+'</table>')
    page='''<!doctype html><meta charset="utf-8"><title>페어 초기 유동성 크기 산정</title><style>body{font:15px system-ui;margin:32px;line-height:1.65}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:7px;text-align:right}td:first-child{text-align:left}pre{padding:16px;background:#f4f5f6;overflow:auto}</style>
<h1>초기 페어 주문량: 현재 거래량과 과거 12개 관측의 중앙값</h1>
<p>median: 기존 회귀 페어의 초기 주문량에만 유동성 제한을 추가한다. median_cash: 같은 제한과 직전 조건부 헤지 완성을 함께 적용한다. 기존 예측값·위험 한도·수수료·보유/청산 규칙을 유지하고, 실제 모의 체결에는 원래 거래량과 가격을 사용한다. 매수·매도 수량, 최소 주문액 및 헤지 비율을 함께 다시 산정한다.</p>
<p>설정은 현재 완료 관측과 과거 11개를 포함한 12개 거래량의 중앙값이다. 거래 없는 관측의 0을 포함하고, 관측 누락이 있으면 신규 진입을 보류한다. 기준시장은 시간봉이므로 12시간, par는 5분봉이므로 1시간이다. 미래 호가 잔량을 예측했다고 주장하지 않는다. 자체 실행 가설이며 논문 재현이 아니다.</p>
<p>첫째·둘째·기존 회귀 페어·조건부 헤지 후보를 대조한다. 초기 자본은 모두 1M USDPR·빈 포지션이다. 실제 자본 투입이 줄어든 효과를 확인하도록 평균 노출과 잔여 포지션을 함께 표시한다. 적게 거래해 손실이 감소한 것을 새로운 수익 기회 검증으로 취급하지 않는다.</p>
<p>새 비교 24개, 기존 경로 정확 대조 24개, 테스트 10개. 8월 선택을 이후 비교 전에 고정했다. 모든 기간은 재사용한 개발 자료이며 실제 지갑 결과나 독립 검증이 아니다. par 캔들/거래량 기반 모의 체결이며 역사적 호가를 복원하지 않았다. 실제 주문 0, 실거래 변경 0.</p>
<p>기준시장 validation: Aug1–Sep1, comparison: Sep1–Oct15. par validation: Oct1–Oct8, comparison: Oct8–Oct16. par_extended extended: Oct8–Oct17, appended_day_flat_start: Oct16–Oct17. 모두 경계는 12UTC(한국 21시). 하루만인 경우 σ/Q는 표본 부족으로 공란이다. 잔여 포지션은 평가손익에 포함하며 가상으로 청산하지 않는다.</p>'''
    page+='<h2>8월 선택</h2><pre>'+html.escape(json.dumps(result['selection'],ensure_ascii=False,indent=2))+'</pre>'+''.join(tables)
    (report/'report.html').write_text(page)
    return str((report/'report.html').relative_to(ROOT))

def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text());tests=(HERE/'test_results.txt').read_text()
    assert 'Ran 10 tests' in tests and tests.rstrip().endswith('OK')
    parent=json.loads((PARENT/'results.json').read_text());windows=parent['windows']
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    rows=[r for r in parent['rows'] if r['arm'] in ['first_strategy','second_strategy',CONTROL,CASH]]
    assert len(rows)==48
    sources=json.loads((PARENT/'frozen.json').read_text())['sources'].copy()
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',PARENT/'results.json',PARENT/'frozen.json']
    paths += [ROOT/x['path'] for x in spec['prior_audit']]
    paths += [path_for(v,phase,arm,sc) for v,phase in windows for arm in [CONTROL,CASH] for sc in ['base','stress']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths});verify(sources)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn='progress_24_conditional_completion_replays',historical_independence=False))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':load(PAR_INPUT/'inputs.npz'),'par_extended':load(EXT/'inputs.npz')}
    states={v:load(DIRECT/f'{v}_direct_24_state.npz') for v in ['reference','par']};states['par_extended']=load(EXT/'direct_state.npz')
    coverage=[]
    for venue,d in data.items():
        volume=capacity_volume(d['par_volume'],d['par_trades'],spec['window_bars'])
        # Independent prefix construction protects the estimator's causal boundary.
        cut=len(d['times'])-30
        np.testing.assert_array_equal(volume[:cut],capacity_volume(d['par_volume'][:cut],d['par_trades'][:cut],spec['window_bars']))
        np.savez_compressed(OUT/f'{venue}_capacity.npz',times=d['times'],capacity_volume=volume)
        coverage.append(dict(venue=venue,rows=len(volume),bar_seconds=int(d['times'][1]-d['times'][0]),
            unavailable=int((~np.isfinite(volume)).sum()),zero=int((volume==0).sum()),current_volume_reduced=int((volume<d['par_volume']).sum())))
    save(OUT/'capacity_audit.json',coverage)
    verified=[]
    for venue,phase in windows:
        for arm in [CONTROL,CASH]:
            for scenario in cfgs[venue]['scenarios']:
                cfg=copy.deepcopy(cfgs[venue])
                if arm==CASH:cfg['conditional_completion']={'risk_aversion':0.}
                with gzip.open(path_for(venue,phase,arm,scenario),'rt') as f:old=json.load(f)
                run=replay(data[venue],states[venue],None,cfg,p,arm,scenario,old['start'],old['end'])
                for key in ['fills','events','metrics','snapshots']:assert run[key]==old[key],(venue,phase,arm,scenario,key)
                verified.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario))
        print('EXACT',venue,phase,'4 control accounts',flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    def record(venue,phase,kind,scenario):
        cfg=copy.deepcopy(cfgs[venue]);cfg['entry_capacity']={'window_bars':spec['window_bars']}
        if spec['variants'][kind]['conditional_completion']:cfg['conditional_completion']={'risk_aversion':0.}
        matched=CASH if spec['variants'][kind]['conditional_completion'] else CONTROL
        with gzip.open(path_for(venue,phase,matched,scenario),'rt') as f:old=json.load(f)
        arm=f'first_dense_hourlyhedge_{kind}_direct_24'
        run=replay(data[venue],states[venue],None,cfg,p,arm,scenario,old['start'],old['end'])
        with gzip.open(OUT/'runs'/f'{venue}-{phase}-{arm}-{scenario}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=run['metrics'];b=lookup[venue,phase,matched,scenario]
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=m,matched=matched,
            pnl_advantage_matched=m['net_pnl']-b['net_pnl'],q_advantage_matched=m['financial_score']-b['financial_score'] if m['financial_score'] is not None else None))
        lookup[venue,phase,arm,scenario]=m
        print(venue,phase,kind,scenario,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],'completed',m['completed_pairs'],'unbalanced',m['unbalanced_minutes'],flush=True)
    for kind in spec['variants']:
        for sc in cfgs['reference']['scenarios']:record('reference','validation',kind,sc)
    rankings=[]
    for kind in spec['variants']:
        arm=f'first_dense_hourlyhedge_{kind}_direct_24';checks=[];matched_checks=[];margins=[]
        matched=CASH if spec['variants'][kind]['conditional_completion'] else CONTROL
        for sc in cfgs['reference']['scenarios']:
            m=lookup['reference','validation',arm,sc]
            checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for native in ['first_strategy','second_strategy']:
                b=lookup['reference','validation',native,sc]
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']]);margins.append(m['net_pnl']-b['net_pnl'])
            b=lookup['reference','validation',matched,sc]
            matched_checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
        rankings.append(dict(arm=arm,eligible=all(checks) and all(matched_checks),native_eligible=all(checks),matched_improvement=all(matched_checks),minimum_native_pnl_advantage=min(margins)))
    rankings.sort(key=lambda r:(-r['eligible'],-r['minimum_native_pnl_advantage'],r['arm']))
    selection=dict(selected=rankings[0]['arm'],eligible=rankings[0]['eligible'],ranking=rankings,frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for venue,phase in windows[1:]:
        for kind in spec['variants']:
            for sc in cfgs[venue]['scenarios']:record(venue,phase,kind,sc)
    assert sha(OUT/'selection.json')==selection_hash;verify(sources)
    result=dict(rows=rows,windows=windows,selection=selection,new_account_replays=24,exact_control_replays=verified,capacity_coverage=coverage,
        protected=protected(),reserved_sources_unchanged=1045,tests_passed=10,orders_sent=0,live_strategy_changed=False,historical_independence=False)
    save(OUT/'results.json',result);report=render(result)
    save(HERE/'work_state.json',dict(status='progress_initial_liquidity_capacity_24_replays_done',report=report,new_account_replays=24,
        exact_control_replays=len(verified),tests_passed=10,selection=selection,orders_sent=0,live_strategy_changed=False,
        historical_independence=False,reserved_raw_read=False,goal_complete=False))
    print('DONE',report,flush=True)

if __name__=='__main__':main()
