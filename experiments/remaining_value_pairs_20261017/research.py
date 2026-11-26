"""Frozen remaining-inventory model comparison; research replay only."""
import copy
import csv
import gzip
import html
import json
import time
from pathlib import Path
import numpy as np

from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.complementary_alpha_20261017.research import protected
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.dense_history_pairs_20261017.model import design
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as PAR_INPUT, save, sha
from .model import prepare, map_remaining
from .portfolio import replay

HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'
PARENT=HERE.parent/'decision_gate_pairs_20261017/artifacts'
EXT=HERE.parent/'historical_par_extension_20261017/artifacts_v3'
CONTROL='first_dense_hourlyhedge_decisiongate_24'


def runpath(venue,phase,sc):
    return PARENT/'runs'/f'{venue}-{phase}-{CONTROL}-{sc}.json.gz'


def readrun(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def render(result):
    dest=ROOT/'reports/remaining_value_pairs_20261017';dest.mkdir(exist_ok=False)
    page='''<!doctype html><meta charset="utf-8"><title>페어 보유·연장 가치 모델 비교</title>
<style>body{font:15px system-ui;margin:32px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #bbb;text-align:right}td:first-child{text-align:left}pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}</style>
<h1>실제 보유 수량과 남은 만기를 반영한 보유·연장 모델</h1>
<p>기존 개선판 decisiongate의 진입·원래 예측·수량·위험 한도·부분 체결·수수료·펀딩·청산 실행은 유지했다. 새 모델은 두 종목의 1~24시간 후 수익률을 각각 예측하고 실제 보유 수량의 현금가치로 합산한다. 매일 과거 90일의 엄격히 만기 지난 자료를 사용하여 갱신하고, 원래 20개 페어 특징·릿지 회귀·6시간 표본·30일 가중 반감기를 유지한다. 진입 필터 점수를 미래 가격에 대입하지 않는다.</p>
<p>renewal 후보: 원래 허용된 연장만 추가 검토하여, 새 모델이 남은 이익으로 비용을 감당하지 못한다고 판단하면 거부한다. 헤지 변경이 필요할 때는 원래 지정가 한도에서의 최악 비용까지 차감하고 실제 체결 도착 후에도 고정된 예측을 다시 대조한다. remaining 후보: 같은 연장 검토에 더해 6시간마다 실제 보유 수량의 현재 승인 만기까지 가치를 계산하여, 지금 청산보다 2bp 넘게 불리하면 청산을 요청한다. 이미 낸 진입 수수료는 다시 청구하지 않는다. 청산 요청과 실제 완료는 구분한다.</p>
<p>예측이 없으면 원래 동작을 유지하고 그 횟수를 기록한다. 학습·평가는 모두 이미 재사용한 과거 자료다. 외부 기준시장 예측 가격을 미래 par 가격으로 연결하는 기존 가정은 남아 있다. 이 실험은 그 가격 괴리의 미래 수렴을 증명하지 않는다. 캔들 거래량 기반 체결과 실제 L2 호가는 다르다.</p>
<p><a href="https://arxiv.org/html/2604.02035v1">2026년 연구 Remark 2.2</a>는 페어의 비례 수수료를 정확히 처리하려면 두 자산의 가격을 따로 보아야 한다고 설명한다. 해당 연구의 OU/RL 알고리즘을 재현한 것이 아니라 그 회계 구분을 적용했다. <a href="https://arxiv.org/html/2502.04284v1">2025년 alpha decay 연구</a>는 보유 상태·미래 신호·비용을 함께 고려할 근거이며, 이번 회귀 기반 거부 규칙을 최적 Bellman 정책으로 주장하지 않는다.</p>'''
    page+='<h2>8월에서 먼저 기록한 후보 선택</h2><pre>'+html.escape(json.dumps(result['selection'],ensure_ascii=False,indent=2))+'</pre>'
    csv_rows=[]
    for venue,phase in result['windows']:
        page+=f'<h2>{venue} / {phase}</h2><table><tr><th>전략</th><th>조건</th><th>순손익</th><th>수익률</th><th>기간 σ</th><th>Q</th><th>최대낙폭</th><th>완료 페어</th></tr>'
        for row in result['rows']:
            if (row['venue'],row['phase'])!=(venue,phase):continue
            m=row['metrics'];q='—' if m['financial_score'] is None else f"{m['financial_score']:.8f}"
            v='—' if m['volatility_horizon'] is None else f"{100*m['volatility_horizon']:.6f}%"
            page+=f"<tr><td>{row['arm']}</td><td>{row['scenario']}</td><td>{m['net_pnl']:+.4f}</td><td>{100*m['roi']:+.6f}%</td><td>{v}</td><td>{q}</td><td>{100*m['max_drawdown']:.6f}%</td><td>{m['completed_pairs']}</td></tr>"
            csv_rows.append(dict(venue=venue,phase=phase,arm=row['arm'],scenario=row['scenario'],
                                **{k:m[k] for k in ['net_pnl','roi','financial_score','volatility_horizon','max_drawdown','completed_pairs']}))
        page+='</table>'
    page+='<h2>보유 판단 기록</h2><pre>'+html.escape(json.dumps(result['holding_decisions'],ensure_ascii=False,indent=2))+'</pre>'
    page+='<p>초기자본 1M, PnL은 기간 말 보유 평가손익을 포함한다. Q 계산은 현재 연구 기준식이며 J를 포함한 공식 확정 순위가 아니다. 실제 운용은 변경하지 않았고 예약된 미래 원시 자료는 열지 않았다.</p>'
    (dest/'report.html').write_text(page)
    with (dest/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    return str((dest/'report.html').relative_to(ROOT))


def main():
    OUT.mkdir(exist_ok=False);(OUT/'runs').mkdir()
    spec=json.loads((HERE/'protocol.json').read_text())
    tests=(HERE/'test_results.txt').read_text();assert 'Ran 10 tests' in tests and tests.rstrip().endswith('OK')
    parent=json.loads((PARENT/'results.json').read_text());windows=parent['windows'];rows=parent['rows'].copy()
    cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    refs={'reference':ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
          'par_extended':ROOT/'practice_live/updates/20261017T120009814481Z/inputs.npz'}
    sources=json.loads((PARENT/'frozen.json').read_text())['sources'].copy()
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',PARENT/'results.json',PARENT/'frozen.json',
           PARENT/'conclusion.json',*refs.values(),EXT/'inputs.npz',PAR_INPUT/'inputs.npz',
           HERE.parent/'joint_leg_pairs_20261017/model.py',HERE.parent/'decision_gate_robustness_20261017/artifacts/conclusion.json']
    paths += [runpath(v,ph,sc) for v,ph in windows for sc in ['base','stress']]
    paths += [PARENT/f'{v}_gate_state.npz' for v in ['reference','par','par_extended']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths});verify(sources)
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources'];verify(prospective)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn=spec['previous_goal_turn']))
    rd,_,p=inputs(cfgs['reference'])
    data={'reference':rd,'par':load(PAR_INPUT/'inputs.npz'),'par_extended':load(EXT/'inputs.npz')}
    controls={v:load(PARENT/f'{v}_gate_state.npz') for v in data};states={};preparation=[]
    for venue,path in refs.items():
        ref=load(path);f=design(ref['market_times'],ref['close'],ref['volume'],ref['trades'],p['symbols'],cfgs[venue]['dense_pairs'])
        prediction,stamps,records=prepare(ref,f,spec)
        np.savez_compressed(OUT/f'{venue}_issued_forecasts.npz',times=ref['market_times'],predictions=prediction,fit_stamps=stamps,horizons=spec['horizons'])
        save(OUT/f'{venue}_fits.json',records)
        for v in (['reference'] if venue=='reference' else ['par','par_extended']):
            state=map_remaining(data[v]['times'],ref['market_times'],prediction,stamps,controls[v],spec['horizons'])
            for key in controls[v]:np.testing.assert_array_equal(state[key],controls[v][key],err_msg=f'{v}-{key}')
            np.savez_compressed(OUT/f'{v}_remaining_state.npz',**state);states[v]=state
        row=dict(venue=venue,models=len(records),strict_maturity=all(r['last_maturity']<r['fit_time'] for r in records),
                 horizons=spec['horizons'],last_source=int(ref['market_times'][-1]))
        assert row['strict_maturity'];preparation.append(row);print('PREPARED',row,flush=True)
    save(OUT/'preparation.json',preparation)
    verified=[]
    for venue,phase in windows:
        for sc in ['base','stress']:
            old=readrun(runpath(venue,phase,sc));cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
            run=replay(data[venue],states[venue],None,cfg,p,CONTROL,sc,old['start'],old['end'])
            for key in ['events','fills','metrics','snapshots','owner_fills']:assert run[key]==old[key],(venue,phase,sc,key)
            verified.append(dict(venue=venue,phase=phase,scenario=sc));print('EXACT',venue,phase,sc,flush=True)
    lookup={(r['venue'],r['phase'],r['arm'],r['scenario']):r['metrics'] for r in rows};decisions=[]
    def record(venue,phase,kind,sc):
        arm=f'first_dense_hourlyhedge_remaining_{kind}_24'
        old=readrun(runpath(venue,phase,sc));cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
        cfg['remaining_value']={**spec['holding'],'mode':kind}
        run=replay(data[venue],states[venue],None,cfg,p,arm,sc,old['start'],old['end'])
        with gzip.open(OUT/'runs'/f'{venue}-{phase}-{arm}-{sc}.json.gz','wt') as f:json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        m=run['metrics'];lookup[venue,phase,arm,sc]=m
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=sc,metrics=m,
                         pnl_advantage_control=m['net_pnl']-old['metrics']['net_pnl']))
        events=run['events'];reviews=[e for e in events if e['event']=='remaining_value_check']
        decisions.append(dict(venue=venue,phase=phase,kind=kind,scenario=sc,
             holding_reviews=len(reviews),unavailable_holding=sum(not e['available'] for e in reviews),
             early_close_requests=sum(e['close'] for e in reviews),
             renewal_vetoes=sum(e.get('reason')=='remaining_model_renewal_veto' for e in events),
             unavailable_renewal=sum(e.get('remaining_model_available') is False for e in events),
             arrival_joint_unwinds=sum(e.get('reason')=='arrival_joint_remaining_cash_insufficient' for e in events)))
        print(venue,phase,kind,sc,'PnL',round(m['net_pnl'],5),'Q',m['financial_score'],decisions[-1],flush=True)
    for kind in spec['candidates']:
        for sc in ['base','stress']:record('reference','validation',kind,sc)
    ranking=[]
    for kind in spec['candidates']:
        arm=f'first_dense_hourlyhedge_remaining_{kind}_24';checks=[]
        for sc in ['base','stress']:
            m=lookup['reference','validation',arm,sc];b=lookup['reference','validation',CONTROL,sc]
            checks.append(dict(scenario=sc,pnl_difference=m['net_pnl']-b['net_pnl'],q_difference=m['financial_score']-b['financial_score'],
                               no_risk_breach=m['risk_breach_bars']==0,completed_pairs=m['completed_pairs']))
        eligible=all(c['pnl_difference']>=-1e-8 and c['q_difference']>=-1e-12 and c['no_risk_breach'] for c in checks)
        eligible &= any(c['pnl_difference']>1e-8 and c['q_difference']>1e-12 for c in checks)
        ranking.append(dict(kind=kind,arm=arm,checks=checks,promote=bool(eligible),
                            minimum_pnl_advantage=min(c['pnl_difference'] for c in checks),minimum_q_advantage=min(c['q_difference'] for c in checks)))
    ranking.sort(key=lambda r:(r['minimum_pnl_advantage'],r['minimum_q_advantage'],r['kind']),reverse=True)
    eligible=[r for r in ranking if r['promote']]
    selection=dict(selected=eligible[0]['arm'] if eligible else None,ranking=ranking,frozen_before_later_replays=True,
                   historical_independence=False,sample_count_is_not_official_submission_requirement=True)
    save(OUT/'selection.json',selection);selected_hash=sha(OUT/'selection.json')
    for venue,phase in windows[1:]:
        for kind in spec['candidates']:
            for sc in ['base','stress']:record(venue,phase,kind,sc)
    assert sha(OUT/'selection.json')==selected_hash;verify(sources);verify(prospective)
    result=dict(rows=rows,windows=windows,selection=selection,holding_decisions=decisions,preparation=preparation,
                exact_control_replays=verified,new_account_replays=24,tests_passed=10,
                protected=protected(),prospective_sources=len(prospective),orders_sent=0,
                live_strategy_changed=False,reserved_raw_read=False,historical_independence=False)
    save(OUT/'results.json',result);report=render(json.loads((OUT/'results.json').read_text()))
    save(HERE/'work_state.json',dict(status='progress_remaining_value_candidates_completed',report=report,
         selection=selection,new_account_replays=24,exact_controls=12,tests_passed=10,
         orders_sent=0,live_strategy_changed=False,reserved_raw_read=False,historical_independence=False,
         goal_complete=False,no_progress_streak=0))
    print('DONE',report,flush=True)


if __name__=='__main__':main()
