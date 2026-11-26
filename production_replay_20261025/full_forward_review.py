"""All prespecified primaries, with verified funding feedback and exit costs.

An evaluator audit fixed before outcomes are opened. This neither tunes a
candidate nor promotes a diagnostic arm or modifies a live checkpoint.
"""
import copy,gzip,html,json,time,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .protocol import ROOT,OUT,save,sha

DEST=OUT/'complete_forward_review'
END=1792915500;WARM=1792897500
EXPERIMENTS=[
    dict(name='entry_lifetime',primary='fixed_entry_1h',controls=['incumbent'],start=WARM,replay_start=WARM),
    dict(name='capital_usage',primary='size_only_2',controls=['incumbent'],start=1792900800,replay_start=1792900800),
    dict(name='schedule_recovery',primary='fresh_block_catchup',controls=['incumbent'],start=1792904400,replay_start=WARM),
    dict(name='decision_side_quote',primary='frozen_side_quote',controls=['incumbent','fixed_entry_1h'],start=1792908000,replay_start=WARM),
    dict(name='quote_history',primary='quote_inputs',controls=['incumbent','frozen_side_quote'],start=1792908000,replay_start=WARM),
]
CONTRACTS=['forward_contract.json','capital_forward/protocol.json','catchup/protocol.json',
           'quote_anchor/protocol.json','quote_history/protocol.json','forward_review/protocol.json']


def freeze():
    assert time.time()<END and not (DEST/'protocol.json').exists()
    sources=[Path(__file__),ROOT/'production_replay_20261025/cash_replay.py',
             ROOT/'production_replay_20261025/book_replay.py',ROOT/'production_replay_20261025/liquidation_audit.py']
    contracts={p:json.loads((OUT/p).read_text()) for p in CONTRACTS}
    for c in contracts.values():
        for p in [*c.get('hashes',{}),*c.get('candidate_hashes',{})]:sources.append(ROOT/p)
    save(DEST/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),experiments=EXPERIMENTS,end=END,
        mechanism='Rerun unchanged registered strategies using verified public settlement cash per unit inside the ledger at each settlement time. All quantity/risk feedback recalculated. No future settlement can enter an earlier state or feature. Original oracle-approximation results remain visible.',
        acceptance='Each primary: positive netPnL and positive fully-liquidated endPnL, and greater than ALL prespecified controls after end liquidation, in both base and stress. Complete funding/exit depth, no accounting/risk/unknown/timing failures. Paired advantage must exceed combined settlement rounding error. Missing evidence never becomes zero cost.',
        interpretation='Multiple overlapping prospective comparisons. Not independent replications, not a long-horizon efficacy proof or official daily Q. Do not substitute diagnostic arms. Any selection among primaries is disclosed selection from this family, not a fresh independent validation of the selected winner.',
        contracts={p:sha(OUT/p) for p in CONTRACTS},hashes={str(p.relative_to(ROOT)):sha(p) for p in set(sources)},
        seed_sha=sha(OUT/'forward_seed.pkl'),orders_sent=0))


def check():
    c=json.loads((DEST/'protocol.json').read_text())
    for p,h in c['hashes'].items():assert sha(ROOT/p)==h,p
    for p,h in c['contracts'].items():assert sha(OUT/p)==h,p
    assert sha(OUT/'forward_seed.pkl')==c['seed_sha']
    return c


def prepare(seed,e,arm):
    p=copy.deepcopy(seed);direct='incumbent'
    if e['name'] in ['entry_lifetime','capital_usage']:direct=arm
    elif e['name']=='schedule_recovery' and arm!='incumbent':
        from .catchup_study import ActivatedCatchup
        p.__class__=ActivatedCatchup;p.research_activation=e['start']
    elif e['name'] in ['decision_side_quote','quote_history'] and arm!='incumbent':
        if arm=='quote_inputs':
            from .quote_history import Activated
            p.__class__=Activated;p.activation=e['start'];p.quote_mode=arm
        else:
            from .quote_study import Activated
            p.__class__=Activated;p.activation=e['start'];p.branch=arm
    return p,direct


def decision(rows,experiment):
    checks=[]
    for scenario in ['base','stress']:
        arms={r['arm']:r for r in rows if r['scenario']==scenario}
        primary=arms[experiment['primary']]
        controls=[arms[a] for a in experiment['controls']]
        valid=all(r.get('valid',False) for r in [primary,*controls])
        positive=valid and primary['net_pnl']>0 and primary['net_after_terminal_exit']>0
        comparisons=[]
        for control in controls:
            advantage=primary['net_after_terminal_exit']-control['net_after_terminal_exit'] if valid else None
            bound=primary['cash_rounding_bound']+control['cash_rounding_bound']
            comparisons.append(dict(control=control['arm'],advantage_after_exit=advantage,rounding_bound=bound,
                                    passed=bool(valid and advantage>bound)))
        checks.append(dict(scenario=scenario,valid=valid,positive=bool(positive),comparisons=comparisons,
                           passed=bool(positive and all(c['passed'] for c in comparisons))))
    return checks


def run():
    check()
    while time.time()<END+1260:
        save(DEST/'status.json',dict(at=time.time(),state='waiting_for_fixed_end_and_public_settlements',orders_sent=0))
        time.sleep(min(30,max(1,END+1260-time.time())))
    check();assert not (DEST/'started.json').exists();save(DEST/'started.json',dict(at=time.time()))
    from competition_live.profile import load_profile
    from competition_live.runtime import private_load
    from .book_replay import load_books
    from .cash_replay import run_cash
    from .liquidation_audit import compare_at_common_end,value_at
    profile=load_profile();streams,capture=load_books((OUT/'capture').glob('*.jsonl.gz'),END)
    save(DEST/'capture_manifest.json',capture)
    units=json.loads((OUT/'settlement_check/cash_units.json').read_text())
    funding=[]
    for symbol in profile['trading']['symbols']:
        rows=json.loads((OUT/'forward_evaluation'/f'{symbol}_funding.json').read_text())['rows']
        funding.extend(dict(r,coin='pas:'+symbol) for r in rows if WARM*1000<r['time']<=END*1000)
    assert funding,'No settled history'
    source=sorted([json.loads(p.read_text()) for p in (OUT/'forward_signals').glob('*.json')],key=lambda r:r['received_at'])
    seed=private_load(OUT/'forward_seed.pkl')['portfolio'];groups=[];cache={}
    for e in EXPERIMENTS:
        rows=[]
        for scenario,sc in dict(base=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=1.),
                                stress=dict(fee_bps=12.,depth_fraction=.5,latency_ms=3000,receipt_multiplier=.5)).items():
            results={};prefix=None
            for arm in [*e['controls'],e['primary']]:
                state,direct=prepare(seed,e,arm)
                key=(scenario,e['replay_start'],e['start'] if arm!='incumbent' else None,arm,direct)
                if key not in cache:
                    cache[key]=run_cash(streams,source,profile,sc,e['replay_start'],END,state,direct,funding,cash_units=units)
                r=copy.deepcopy(cache[key]);r['arm']=arm;results[arm]=r
                before=[s for s in r['snapshots'] if s['time']<e['start']]
                if prefix is None:prefix=before
                else:assert before==prefix,'Inventories changed before registered activation'
                folder=DEST/e['name'];folder.mkdir(exist_ok=True)
                with gzip.open(folder/f'{arm}_{scenario}.json.gz','wt') as fp:json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
            compared=compare_at_common_end(streams,profile,sc,results,e['start'],END)
            for row in compared:
                row['scenario']=scenario;r=results[row['arm']]
                cash=[f for f in r['funding'] if e['start']<f['time']/1000<=END]
                row['cash_rounding_bound']=sum(f['maximum_rounding_error'] for f in cash)
                row['funding']=sum(float(f['delta']['usdc']) for f in cash)
                orders=[o for o in r['orders'] if e['start']<=o['arrival']<=END and o['filled']]
                row['fill_count']=len(orders)
                row['fees']=sum(o['filled']*o['price']*sc['fee_bps']/10000 for o in orders)
                if row.get('net_pnl') is not None:
                    row['price_pnl']=row['net_pnl']+row['fees']-row['funding']
                    row['return_on_50000_pct']=row['net_pnl']/50000*100
                    samples=[s for s in r['snapshots'] if e['start']<=s['time']<=END]
                    equities=np.array([s['equity'] for s in samples]);peaks=np.maximum.accumulate(equities)
                    row['observed_max_drawdown_pct']=float(np.max(1-equities/peaks)*100)
                    row['mean_notional']=float(np.mean([s['notional'] for s in samples]))
                rows.append(row)
        checks=decision(rows,e)
        groups.append(dict(**e,rows=rows,checks=checks,passes_short_paper_gate=all(c['passed'] for c in checks)))
        save(DEST/'results.json',dict(at=time.time(),experiments=groups,conditional_paper_only=True,
            selection_family_size=len(EXPERIMENTS),official_Q=None,official_daily_volatility=None,orders_sent=0))
        save(DEST/'status.json',dict(at=time.time(),state='evaluating',completed=len(groups),orders_sent=0))
    render(groups)
    save(DEST/'status.json',dict(at=time.time(),state='complete',orders_sent=0))


def render(groups):
    lines=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>PAS 실제 호가·정산 대조</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%;margin-bottom:30px}td,th{border:1px solid #ddd;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}</style>',
        '<h1>PAS 실제 호가·정산 대조</h1>',
        '<p>후보를 고정한 뒤 수집한 PAS 호가와 실제 발표 신호로 계산했습니다. 공개 계좌의 펀딩 정산액으로 매 시점의 잔고와 이후 판단을 재실행했습니다. 모두 조건부 모의 체결이며, 실제 주문이나 운용 전략 변경은 없습니다.</p>',
        '<p>자기 자본 50,000 USDP, 동일 보유분, 동일 위험 한도에서 비교했습니다. 팀원 손익은 제외합니다. 각 실험의 시작 시각은 아래에 표시하며 종료는 2026-10-25 17:05 KST입니다. 서로 겹치는 다섯 비교이며, 다섯 번의 독립 검증이 아닙니다.</p>']
    def fmt(x):return '확인 불가' if x is None else f'{x:,.6f}'
    for g in groups:
        at=datetime.fromtimestamp(g['start'],timezone.utc).isoformat()
        lines.append('<h2>'+html.escape(g['name'])+'</h2><p>시작 UTC '+at+' · 지정 후보 '+g['primary']+' · 짧은 모의 기준 '+('통과' if g['passes_short_paper_gate'] else '미통과')+'</p>')
        lines.append('<table><tr><th>후보</th><th>조건</th><th>순손익 USDP</th><th>종료 청산 후 USDP</th><th>가격 손익</th><th>수수료</th><th>펀딩</th><th>체결 수</th><th>유효성</th></tr>')
        for r in g['rows']:
            lines.append('<tr><td>'+r['arm']+'</td><td>'+r['scenario']+'</td>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k in ['net_pnl','net_after_terminal_exit','price_pnl','fees','funding'])+'<td>'+str(r['fill_count'])+'</td><td>'+('확인' if r['valid'] else html.escape(r.get('reason','자료·실행 검사 실패')) )+'</td></tr>')
        lines.append('</table>')
    lines.append('<p>총 미사용 구간은 몇 시간에 불과합니다. 공식 일별 변동성·Q나 장기 수익성을 추정하지 않습니다. 규모 후보는 더 큰 노출의 효과를 포함하며, 예측력이 향상됐다는 증거로 해석하지 않습니다. 통과 후보를 선택하면 이 자료는 그 선택에 사용된 자료가 됩니다.</p><p><a href="results.json">전체 수치·실패·위험 기록</a> · <a href="protocol.json">결과 열람 전 고정한 기준</a></p></html>')
    (DEST/'report.html').write_text('\n'.join(lines))


if __name__=='__main__':globals()[sys.argv[1]]()
