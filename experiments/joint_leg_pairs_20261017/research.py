"""Frozen two-leg forecast experiment, historical accounts and exact controls."""
import copy, csv, gzip, html, json, time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.model import design
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as PAR_INPUT, save, sha
from experiments.conditional_completion_pairs_20261017.research import verify
from .model import targets_and_units, rolling, map_forecasts
from .portfolio import replay

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
PARENT=ROOT/'experiments/conditional_completion_pairs_20261017/artifacts'
DIRECT=ROOT/'experiments/rolling_feature_pairs_20261017/artifacts'
EXT=ROOT/'experiments/historical_par_extension_20261017/artifacts_v3'
MATCHED='first_dense_hourlyhedge_cash_direct_24'


def path_for(venue,phase,scenario):
    return PARENT/'runs'/f'{venue}-{phase}-{MATCHED}-{scenario}.json.gz'


def render(result):
    dest=ROOT/'reports/joint_leg_pairs_20261017';dest.mkdir(exist_ok=True)
    keys=['venue','phase','arm','scenario','net_pnl','roi','financial_score','volatility_horizon','max_drawdown','completed_pairs','unbalanced_minutes']
    with (dest/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in result['rows']:w.writerow({k:r.get(k,r['metrics'].get(k)) for k in keys})
    page='''<!doctype html><meta charset="utf-8"><title>페어 개별 종목 예측 비교</title><style>body{font:15px system-ui;margin:32px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:7px;text-align:right}td:first-child{text-align:left}pre{background:#f4f5f6;padding:16px;white-space:pre-wrap}</style>
<h1>페어 예측을 두 종목의 목표가격으로 연결</h1>
<p>기존 단일 페어 수익률을 +μ/−μ로 두 종목에 배분하는 방식은 정확한 헤지의 합산 가격손익을 재현하지만, 개별 종목의 기대수익률을 유일하게 결정하지 못한다. 이는 합산 계산 오류라고 단정할 수 없으며, 한쪽 체결·반올림·헤지 변경 상황에서 확인할 모형 가정이다.</p>
<p>joint: 동일한 과거 특징에서 두 종목의 24시간 수익률을 각각 릿지 회귀로 예측한다. common: 기존 페어 예측을 그대로 두고 joint에서 예측한 공통 움직임만 추가한다. 두 후보 모두 목표가격을 초기 진입·한쪽 체결 후 판단·보유 연장·유료 헤지 변경까지 전달한다. 공통 후보는 정확한 헤지의 가격손익을 보존하지만 반올림과 미래 청산 비용 때문에 주문·순손익까지 반드시 같지는 않다.</p>
<p>학습: 최근 90일, 반감기 30일, 6시간 간격 자료, 매일 00UTC 재학습, 예측은 매시간. 두 종목 모두 거래가 있던 시작·종료 자료만 학습하며 종료 시점은 학습 시각보다 엄격히 이전이다. 종목별 과거 168시간 변동성으로 목표를 정규화한다. 새 후보 24개 계좌와 부모 경로 12개 정확 대조, 구현 검증 11개. 동일 초기 자본 1M, 비용·위험·체결·청산 조건. 모든 기간은 재사용 개발 자료이며 독립 검증이나 실제 지갑 수익이 아니다.</p>
<p>초기 주문량을 과거 거래량 중앙값으로 제한한 이전 후보는 사용하지 않는다. 다만 계좌 백테스트 자체의 캔들·거래량 체결 근사는 남아 있으며 실제 호가 결과와 구분해야 한다. 실거래 주문·자동 교체 없음.</p>
<p><a href="https://arxiv.org/html/2601.00738v1">2026 실행 실패 후 재판단 연구</a>와 <a href="https://arxiv.org/html/2502.04284v1">2025 비용·상태 기반 거래 연구</a>는 판단 구조의 참고자료다. 두 출력 회귀 및 공통 움직임 비교는 자체 가설이며 논문 전략 재현이나 논문 수익률 재현을 주장하지 않는다.</p>'''
    page+='<h2>8월 후보 선정</h2><pre>'+html.escape(json.dumps(result['selection'],ensure_ascii=False,indent=2))+'</pre>'
    for venue,phase in result['windows']:
        page+=f'<h2>{venue} / {phase}</h2><table><tr><th>전략</th><th>비용</th><th>PnL USDPR</th><th>수익률</th><th>Q</th><th>기간 변동성</th><th>MDD</th><th>완료 페어</th><th>미헤지 분</th></tr>'
        for r in result['rows']:
            if (r['venue'],r['phase'])!=(venue,phase):continue
            m=r['metrics'];q='—' if m['financial_score'] is None else f"{m['financial_score']:.8f}"
            v='—' if m['volatility_horizon'] is None else f"{100*m['volatility_horizon']:.6f}%"
            page+=f"<tr><td>{html.escape(r['arm'])}</td><td>{r['scenario']}</td><td>{m['net_pnl']:+.6f}</td><td>{m['roi']*100:+.6f}%</td><td>{q}</td><td>{v}</td><td>{m['max_drawdown']*100:.6f}%</td><td>{m['completed_pairs']}</td><td>{m['unbalanced_minutes']:.0f}</td></tr>"
        page+='</table>'
    (dest/'report.html').write_text(page)
    return str((dest/'report.html').relative_to(ROOT))


def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    tests=(HERE/'test_results.txt').read_text();assert 'Ran 11 tests' in tests and tests.rstrip().endswith('OK')
    parent=json.loads((PARENT/'results.json').read_text());windows=parent['windows']
    rows=[r for r in parent['rows'] if r['arm'] in ['first_strategy','second_strategy','first_dense_hourlyhedge_direct_24',MATCHED]]
    assert len(rows)==48
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    ref_paths={'reference':ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
               'par_extended':ROOT/'practice_live/updates/20261017T120009814481Z/inputs.npz'}
    sources=json.loads((PARENT/'frozen.json').read_text())['sources'].copy()
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',PARENT/'results.json',PARENT/'frozen.json',*ref_paths.values()]
    paths += [path_for(v,phase,sc) for v,phase in windows for sc in ['base','stress']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths});verify(sources)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn='progress_volume_capacity_rejected_and_best_touch_forecast_economics_audited',historical_independence=False))
    rd,_,p=inputs(cfgs['reference']);data={'reference':rd,'par':load(PAR_INPUT/'inputs.npz'),'par_extended':load(EXT/'inputs.npz')}
    controls={v:load(DIRECT/f'{v}_direct_24_state.npz') for v in ['reference','par']};controls['par_extended']=load(EXT/'direct_state.npz')
    refs={v:load(path) for v,path in ref_paths.items()};old_ref=load(ROOT/cfgs['par']['reference_run']/'inputs.npz')
    for key in ['market_times','close','volume','trades']:
        np.testing.assert_array_equal(refs['par_extended'][key][:len(old_ref[key])],old_ref[key])
    states={};preparation=[]
    for venue,ref in refs.items():
        f=design(ref['market_times'],ref['close'],ref['volume'],ref['trades'],p['symbols'],cfgs[venue]['dense_pairs'])
        y,maturity,unit,tf=targets_and_units(ref,f,spec['horizon'])
        forecasts,stamps,records=rolling(ref['market_times'],f,y,maturity,unit,tf,spec)
        np.savez_compressed(OUT/f'{venue}_issued_forecasts.npz',times=ref['market_times'],target=y,unit=unit,
                            predictions=forecasts,fit_stamps=stamps,maturity=maturity,terminal_fresh=tf,fresh=f['fresh'],beta=f['beta'])
        save(OUT/f'{venue}_fits.json',records)
        for v in (['reference'] if venue=='reference' else ['par','par_extended']):
            for kind in spec['candidates']:
                state=map_forecasts(data[v]['times'],ref['market_times'],forecasts,controls[v],kind)
                if kind=='common':np.testing.assert_array_equal(state['mu'],controls[v]['mu'])
                np.savez_compressed(OUT/f'{v}_{kind}_state.npz',**state);states[v,kind]=state
        preparation.append(dict(venue=venue,daily_pair_fits=len(records),features=f['names'],
                                latest_source=int(ref['market_times'][-1]),strict_maturity=all(r['last_maturity']<r['fit_time'] for r in records)))
        print('PREPARED',venue,len(records),'daily pair fits',flush=True)
    save(OUT/'preparation.json',preparation)
    verified=[]
    for venue,phase in windows:
        for sc in ['base','stress']:
            with gzip.open(path_for(venue,phase,sc),'rt') as f:old=json.load(f)
            cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
            run=replay(data[venue],controls[venue],None,cfg,p,MATCHED,sc,old['start'],old['end'])
            for key in ['fills','events','metrics','snapshots']:assert run[key]==old[key],(venue,phase,sc,key)
            verified.append(dict(venue=venue,phase=phase,scenario=sc))
        print('EXACT',venue,phase,'two parent accounts',flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows}
    def record(venue,phase,kind,sc):
        cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
        with gzip.open(path_for(venue,phase,sc),'rt') as f:old=json.load(f)
        arm=f'first_dense_hourlyhedge_{kind}_24'
        run=replay(data[venue],states[venue,kind],None,cfg,p,arm,sc,old['start'],old['end'])
        with gzip.open(OUT/'runs'/f'{venue}-{phase}-{arm}-{sc}.json.gz','wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=run['metrics'];b=lookup[venue,phase,MATCHED,sc]
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc,metrics=m,matched=MATCHED,pnl_advantage_matched=m['net_pnl']-b['net_pnl']))
        lookup[venue,phase,arm,sc]=m
        print(venue,phase,kind,sc,'PnL',round(m['net_pnl'],6),'Q',m['financial_score'],'pairs',m['completed_pairs'],'unbalanced',m['unbalanced_minutes'],flush=True)
    for kind in spec['candidates']:
        for sc in ['base','stress']:record('reference','validation',kind,sc)
    rankings=[]
    for kind in spec['candidates']:
        arm=f'first_dense_hourlyhedge_{kind}_24';checks=[];margins=[]
        for sc in ['base','stress']:
            m=lookup['reference','validation',arm,sc]
            checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for base in ['first_strategy','second_strategy',MATCHED]:
                b=lookup['reference','validation',base,sc]
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
                if base!=MATCHED:margins.append(m['net_pnl']-b['net_pnl'])
        rankings.append(dict(arm=arm,eligible=all(checks),minimum_native_pnl_advantage=min(margins)))
    rankings.sort(key=lambda r:(-r['eligible'],-r['minimum_native_pnl_advantage'],r['arm']))
    selection=dict(selected=rankings[0]['arm'],eligible=rankings[0]['eligible'],ranking=rankings,frozen_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    for venue,phase in windows[1:]:
        for kind in spec['candidates']:
            for sc in ['base','stress']:record(venue,phase,kind,sc)
    assert sha(OUT/'selection.json')==selection_hash;verify(sources)
    result=dict(rows=rows,windows=windows,selection=selection,new_account_replays=24,exact_control_replays=verified,
                tests_passed=11,protected=protected(),orders_sent=0,live_strategy_changed=False,historical_independence=False,reserved_raw_read=False)
    save(OUT/'results.json',result);report=render(result)
    save(HERE/'work_state.json',dict(status='progress_joint_leg_forecast_accounts_completed',report=report,selection=selection,
         new_account_replays=24,exact_controls=12,tests_passed=11,orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,goal_complete=False))
    print('DONE',report,flush=True)


if __name__=='__main__':main()
