"""Frozen conditional completion comparison, identical original entry models."""
import copy,csv,gzip,html,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as PAR_INPUT,save,sha,ts
from experiments.dense_history_pairs_20261017.research import load
from .portfolio import replay

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
DIRECT=ROOT/'experiments/rolling_feature_pairs_20261017/artifacts'
EXT=ROOT/'experiments/historical_par_extension_20261017/artifacts_v3'
CONTROL='first_dense_hourlyhedge_direct_24'

def verify(sources):
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path

def old_path(venue,phase,scenario):
    if venue=='par_extended':return EXT/'runs'/f'{phase}-{CONTROL}-{scenario}.json.gz'
    return DIRECT/'runs'/f'{venue}-{phase}-{CONTROL}-{scenario}.json.gz'

def render(results):
    report=ROOT/'reports/conditional_completion_pairs_20261017';report.mkdir(exist_ok=True)
    rows=results['rows'];keys=['venue','phase','arm','scenario','net_pnl','roi','financial_score','volatility_horizon','max_drawdown','fees','funding','completed_pairs','aborted_pairs','unbalanced_minutes','residual_gross']
    with (report/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:row[k] if k in row else row['metrics'].get(k) for k in keys})
    parts=[]
    for venue,phase in results['windows']:
        cells=[]
        for row in rows:
            if (row['venue'],row['phase'])!=(venue,phase):continue
            m=row['metrics'];q='—' if m['financial_score'] is None else f"{m['financial_score']:.8f}"
            vol='—' if m['volatility_horizon'] is None else f"{100*m['volatility_horizon']:.6f}%"
            cells.append(f"<tr><td>{html.escape(row['arm'])}</td><td>{row['scenario']}</td><td>{m['net_pnl']:+.6f}</td><td>{100*m['roi']:+.6f}%</td><td>{q}</td><td>{vol}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{m['completed_pairs']}</td><td>{m['aborted_pairs']}</td><td>{m['unbalanced_minutes']:.0f}</td></tr>")
        parts.append(f'<h2>{venue} / {phase}</h2><table><tr><th>전략</th><th>비용</th><th>PnL USDPR</th><th>수익률</th><th>Q</th><th>기간 변동성</th><th>MDD</th><th>완료 페어</th><th>진입 실패</th><th>미헤지 분</th></tr>'+''.join(cells)+'</table>')
    page='''<!doctype html><meta charset="utf-8"><title>페어의 조건부 헤지 완성 비교</title><style>body{font:15px system-ui;margin:32px;line-height:1.6}td,th{border:1px solid #bbb;padding:7px;text-align:right}table{border-collapse:collapse;font-size:13px}td:first-child{text-align:left}pre{background:#f5f5f5;padding:16px}</style>
<h1>한쪽 체결 이후: 헤지 완성과 청산의 남은 가치 비교</h1>
<p>첫째 전략 + 기존 일별 갱신 회귀 페어를 유지하고, 최초 진입의 한쪽만 체결된 경우를 변경했다. cash는 남은 현금가치, risk는 같은 현금가치와 전체 계좌 공분산 위험(γ=64)을 함께 비교한다. 이미 낸 수수료와 손익은 장부에 남으며 의사결정에서 이중 차감하지 않는다. 현재 즉시 청산 예상값 대비 남은 이점 안에서만 미체결 종목의 지정가를 조정한다.</p>
<p>초기 진입·목표 크기·헤지 비율·예측값·시그널 만기·위험 상한·수수료·유동성 제약은 같다. 헤지 재시도는 한 번이며 실제 체결 실패 시 기존 유료 청산 절차를 따른다. 청산 대안의 현재 가격 예상은 의사결정용 근사이며 전량 즉시 체결을 보장하지 않는다. 회계에서는 실제 모의 체결을 기다린다.</p>
<p>비교 원리: <a href="https://arxiv.org/html/2601.00738v1">2026 CEX–DEX 실행 실패 후 의사결정 연구</a>의 실패 후 대안 구분과 <a href="https://arxiv.org/html/2502.04284v1">2025 거래비용·상태 기반 의사결정 연구</a>를 참고했다. 자체 한 단계 근사이며 논문의 최적 제어·학습 모델 재현이 아니다. 무제한 유동성이나 원자적 동시 체결을 가정하지 않았다.</p>
<p>모든 평가 기간은 재사용 과거 자료다. 기준시장과 실제 par 캔들 기반 모의 성과를 구분하며, 과거 호가의 24시간 진입·청산 검증이나 실제 지갑 수익이 아니다. 새 후보 24개 계좌, 기존 경로 정확 대조 12개, 테스트 9개. August 후보 선택을 나머지 비교 전에 고정했다. 실제 주문과 자동 교체는 없다.</p>
<p>기간: 기준시장 validation Aug1 12UTC–Sep1 12UTC, comparison Sep1–Oct15 12UTC. par validation Oct1–Oct8 12UTC, comparison Oct8–Oct16 12UTC. par_extended extended Oct8–Oct17 12UTC, appended_day_flat_start Oct16–Oct17 12UTC. 계좌마다 초기 1M와 빈 포지션이며 하루 단독 σ/Q는 표본 부족으로 공란이다. par_extended는 실제 정산 시각 반영 어댑터를 사용하고 기존 구간은 그대로 보존했다. 잔여 포지션은 평가액이며 청산을 가정하지 않는다.</p>'''
    page+='<h2>8월 선택 결과</h2><pre>'+html.escape(json.dumps(results['selection'],ensure_ascii=False,indent=2))+'</pre>'+''.join(parts)
    (report/'report.html').write_text(page)
    return str((report/'report.html').relative_to(ROOT))

def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    assert 'Ran 9 tests' in (HERE/'test_results.txt').read_text() and (HERE/'test_results.txt').read_text().rstrip().endswith('OK')
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    old_results=json.loads((ROOT/'experiments/calibrated_daily_tree_pairs_20261017/artifacts/results.json').read_text())
    arms=['first_strategy','second_strategy',CONTROL]
    rows=[row for row in old_results['rows'] if row['arm'] in arms]
    ext_results=json.loads((EXT/'results.json').read_text())
    rows += [dict(venue='par_extended',phase=row['window'],arm=row['arm'],scenario=row['scenario'],metrics=row['metrics']) for row in ext_results['rows'] if row['arm'] in arms]
    assert len(rows)==36
    windows=[('reference','validation'),('reference','comparison'),('par','validation'),('par','comparison'),
             ('par_extended','extended'),('par_extended','appended_day_flat_start')]
    reserved=ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json'
    sources=json.loads((EXT/'frozen.json').read_text())['sources'].copy()
    sources.update(json.loads(reserved.read_text())['sources'])
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',EXT/'inputs.npz',EXT/'direct_state.npz',EXT/'results.json',reserved]
    paths += [ROOT/x['path'] for x in spec['prior_audit']]
    paths += [DIRECT/f'{v}_direct_24_state.npz' for v in ['reference','par']]
    paths += [old_path(venue,phase,scenario) for venue,phase in windows for scenario in ['base','stress']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths});verify(sources)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn='progress_extra_par_day_one_leg_failure',historical_independence=False))
    rd,_,p=inputs(cfgs['reference'])
    data={'reference':rd,'par':load(PAR_INPUT/'inputs.npz'),'par_extended':load(EXT/'inputs.npz')}
    states={v:load(DIRECT/f'{v}_direct_24_state.npz') for v in ['reference','par']}
    states['par_extended']=load(EXT/'direct_state.npz');verified=[]
    for venue,phase in windows:
        for scenario in cfgs[venue]['scenarios']:
            with gzip.open(old_path(venue,phase,scenario),'rt') as f:old=json.load(f)
            run=replay(data[venue],states[venue],None,cfgs[venue],p,CONTROL,scenario,old['start'],old['end'])
            for key in ['fills','events','metrics','snapshots']:assert run[key]==old[key],(venue,phase,scenario,key)
            verified.append(dict(venue=venue,phase=phase,scenario=scenario));print('EXACT',venue,phase,scenario,flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    def record(venue,phase,kind,scenario):
        cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':spec['candidates'][kind]}
        with gzip.open(old_path(venue,phase,scenario),'rt') as f:old=json.load(f)
        arm=f'first_dense_hourlyhedge_{kind}_direct_24'
        run=replay(data[venue],states[venue],None,cfg,p,arm,scenario,old['start'],old['end'])
        with gzip.open(OUT/'runs'/f'{venue}-{phase}-{arm}-{scenario}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=run['metrics'];baseline=lookup[venue,phase,CONTROL,scenario]
        row=dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=m,
                 pnl_advantage_direct=m['net_pnl']-baseline['net_pnl'],
                 q_advantage_direct=m['financial_score']-baseline['financial_score'] if m['financial_score'] is not None else None)
        rows.append(row);lookup[venue,phase,arm,scenario]=m
        print(venue,phase,kind,scenario,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],'completed',m['completed_pairs'],'unbalanced',m['unbalanced_minutes'],flush=True)
    for kind in spec['candidates']:
        for sc in cfgs['reference']['scenarios']:record('reference','validation',kind,sc)
    rankings=[]
    for kind in spec['candidates']:
        arm=f'first_dense_hourlyhedge_{kind}_direct_24';checks=[];matched=[];margins=[]
        for sc in cfgs['reference']['scenarios']:
            m=lookup['reference','validation',arm,sc]
            checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for native in ['first_strategy','second_strategy']:
                b=lookup['reference','validation',native,sc]
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']]);margins.append(m['net_pnl']-b['net_pnl'])
            b=lookup['reference','validation',CONTROL,sc]
            matched.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
        rankings.append(dict(arm=arm,eligible=all(checks) and all(matched),native_eligible=all(checks),matched_improvement=all(matched),minimum_native_pnl_advantage=min(margins)))
    rankings.sort(key=lambda r:(-r['eligible'],-r['minimum_native_pnl_advantage'],r['arm']))
    selection=dict(selected=rankings[0]['arm'],eligible=rankings[0]['eligible'],ranking=rankings,frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for venue,phase in windows[1:]:
        for kind in spec['candidates']:
            for sc in cfgs[venue]['scenarios']:record(venue,phase,kind,sc)
    assert sha(OUT/'selection.json')==selection_hash;verify(sources)
    result=dict(rows=rows,windows=windows,selection=selection,new_account_replays=24,exact_control_replays=verified,
        protected=protected(),reserved_sources_unchanged=1045,tests_passed=9,orders_sent=0,live_strategy_changed=False,historical_independence=False)
    save(OUT/'results.json',result);report=render(result)
    save(HERE/'work_state.json',dict(status='progress_conditional_completion_24_replays_done',report=report,new_account_replays=24,
        exact_control_replays=len(verified),tests_passed=9,selection=selection,orders_sent=0,live_strategy_changed=False,
        historical_independence=False,reserved_raw_read=False,goal_complete=False))
    print('DONE',report,flush=True)

if __name__=='__main__':main()
