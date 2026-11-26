"""Post-cutoff evaluation of the frozen second experiment; no live orders."""
import copy
import gzip
import html
import json
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from .replication import ROOT, OUT, PREVIOUS, START, END, check
from .protocol import save, sha


def freeze():
    check()
    if time.time() >= START or (OUT/'evaluation_contract.json').exists():
        raise ValueError('Evaluator must be registered before the fixed start')
    wallets=json.loads((PREVIOUS/'reserved_eight_hours/protocol.json').read_text())['wallets_for_funding_cash_only']
    save(OUT/'evaluation_contract.json',dict(at=time.time(),source_sha=sha(__file__),
        parent_sha=sha(OUT/'protocol.json'),wallets_for_settlement_cash_only=wallets,
        computation='Original lot-exact/cash-exact/data-aware replay and unchanged qualification functions. Reconnect/receipt checks and actual-ledger audit included. All three arms, both phases and both costs.',
        funding_start=START-21600,first_open_not_before=END+65,
        orders_sent=0,live_changed=False))


def validate():
    c=check();e=json.loads((OUT/'evaluation_contract.json').read_text())
    if sha(__file__)!=e['source_sha'] or sha(OUT/'protocol.json')!=e['parent_sha']:
        raise ValueError('Registered evaluator or contract changed')
    return c,e


def collect(c,e):
    if time.time()<END+65:raise ValueError('Do not open evaluation outcomes early')
    from .retrospective import request
    from .settlement_audit import cash_units
    from .exact_settlement import read_records
    funding=[]
    for symbol in c['profile']['trading']['symbols']:
        rows=request(dict(type='fundingHistory',coin='pas:'+symbol,
            startTime=e['funding_start']*1000,endTime=END*1000),OUT/'funding_schedule'/f'{symbol}.json')
        if len(rows)>=500:raise ValueError('Potentially truncated funding schedule')
        funding.extend(dict(r,coin='pas:'+symbol) for r in rows)
    for wallet in e['wallets_for_settlement_cash_only']:
        rows=request(dict(type='userFunding',user=wallet,startTime=e['funding_start']*1000,
            endTime=END*1000+999),OUT/'funding_cash'/f'{wallet}.json')
        if len(rows)>=500:raise ValueError('Potentially truncated settlement records')
    records=read_records(OUT/'funding_cash')
    units,rejected=cash_units(records)
    save(OUT/'cash_units.json',units);save(OUT/'rejected_cash_units.json',rejected)
    save(OUT/'funding_schedule.json',funding)
    return funding,records,units


def read_ledger():
    from competition_live.runtime import private_load
    source=ROOT/'.runtime/pas_budget_lifecycle/coordinator.pkl'
    for _ in range(10):
        before=sha(source);p=private_load(source)['portfolio'];after=sha(source)
        if before==after:break
        time.sleep(1)
    else:raise ValueError('Actual checkpoint changed during read')
    a=p.account.physical
    reconstructed=a.initial-sum(f['signed_qty']*f['price']+f['fee'] for f in a.fills)+sum(f['signed_amount'] for f in a.funding_events)
    if abs(reconstructed-a.shadow_cash)>1e-6 or p.fatal or p.account.unknown or p.account.pending_orders:
        raise ValueError('Actual owner ledger not reconciled')
    save(OUT/'actual_owner_ledger.json',dict(checkpoint_sha=after,initial=a.initial,
        fills=a.fills,funding_events=a.funding_events,shadow_cash=a.shadow_cash,
        accounting_error=abs(reconstructed-a.shadow_cash),orders_sent=0))
    return a


def annotate(row,r,scenario):
    row['scenario']=scenario
    events=[f for f in r['funding'] if START<f['time']/1000<=END]
    row['cash_rounding_bound']=sum(f['maximum_rounding_error'] for f in events)
    row['funding']=sum(float(f['delta']['usdc']) for f in events)
    orders=[o for o in r['orders'] if START<=o['arrival']<=END and o['filled']]
    row['fill_count']=len(orders)
    row['fees']=sum(f['fee'] for f in r['fills'])
    if row.get('net_pnl') is not None:
        row['price_pnl']=row['net_pnl']+row['fees']-row['funding']
        row['return_on_50000_pct']=row['net_pnl']/50000*100
        equities=np.array([s['equity'] for s in r['snapshots']])
        row['observed_max_drawdown_pct']=float(np.max(1-equities/np.maximum.accumulate(equities))*100)
        row['mean_notional']=float(np.mean([s['notional'] for s in r['snapshots']]))
    return row


def evaluate(c,funding,records,units):
    from competition_live.runtime import private_load
    from . import book_replay,cash_replay,data_aware_book_replay,lot_sweep,liquidation_audit
    from . import qualification_review,venue_feasibility
    from .full_forward_review import prepare,decision
    from .exact_settlement import run_exact
    from .delayed_forward_review import delayed_submit
    from . import seal_capture
    # Seal only received events at/before the registered endpoint. Original
    # historical feature archive is separate and ends before this test starts.
    with patch.object(seal_capture,'OUT',OUT),patch.object(seal_capture,'END',END):
        seal_capture.seal()
    files=[PREVIOUS/'sealed_forward_capture/received_through_fixed_end.jsonl.gz',
           OUT/'sealed_forward_capture/received_through_fixed_end.jsonl.gz']
    streams,manifest=book_replay.load_books(files,END)
    save(OUT/'capture_manifest.json',manifest)
    source=sorted([json.loads(p.read_text()) for p in (OUT/'forward_signals').glob('*.json')],key=lambda r:r['received_at'])
    source=[s for s in source if s['received_at']<=END]
    seed_info=json.loads((OUT/'seed_manifest.json').read_text())
    if seed_info['at']>=START or seed_info['seed_sha']!=sha(OUT/'forward_seed.pkl'):
        raise ValueError('Initial condition not preserved')
    seed=private_load(OUT/'forward_seed.pkl')['portfolio']
    if seed.pair is not None:raise ValueError('Active pair renewal is outside this frozen trial')
    ledger=read_ledger();profile=c['profile']
    experiment=dict(name='quote_history',primary=c['primary'],controls=c['controls'],start=START,replay_start=START)
    groups={};capacity={};fidelity={}
    for phase in c['phases_seconds']:
        label=f'realistic_phase{phase}';rows=[];folder=OUT/label
        for scenario,sc in c['scenarios'].items():
            results={}
            for arm in [*c['controls'],c['primary']]:
                validate()
                p,direct=prepare(seed,experiment,arm)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(book_replay,'run_book',data_aware_book_replay.run_book_data_aware))
                    stack.enter_context(patch.object(data_aware_book_replay,'PHASE_SECONDS',phase))
                    stack.enter_context(patch.object(cash_replay.CashVenue,'submit',delayed_submit))
                    stack.enter_context(patch.object(book_replay,'sweep',lot_sweep.sweep))
                    r=run_exact(streams,source,profile,sc,START,END,p,direct,funding,
                                cash_units=units,records=records)
                r['arm']=arm;results[arm]=r;folder.mkdir(exist_ok=True)
                with gzip.open(folder/f'{arm}_{scenario}.json.gz','wt') as fp:
                    json.dump(r,fp,allow_nan=False,default=lambda x:x.tolist())
                with patch.object(venue_feasibility,'END',END):
                    capacity[(label,arm,scenario)]=venue_feasibility.assess(r,streams,profile,sc,seed.account.reserved.qty)
                if arm=='incumbent' and scenario=='base':
                    with patch.object(qualification_review,'END',END):
                        fidelity[label]=qualification_review.actual_comparison(ledger,streams,profile,experiment,r)
                save(OUT/'status.json',dict(at=time.time(),state='evaluating',phase=phase,scenario=scenario,arm=arm,orders_sent=0))
            with patch.object(liquidation_audit,'sweep',lot_sweep.sweep):
                compared=liquidation_audit.compare_at_common_end(streams,profile,sc,results,START,END)
            rows.extend(annotate(row,results[row['arm']],scenario) for row in compared)
        groups[label]=dict(**experiment,rows=rows,checks=decision(rows,experiment))
        save(folder/'results.json',groups[label])
    verdict=qualification_review.qualification(groups,capacity,fidelity)
    result=dict(at=time.time(),start=START,end=END,primary=c['primary'],comparisons=groups,
        actual_incumbent_comparisons=fidelity,verdict=verdict,
        capacity=[dict(timing=k[0],arm=k[1],scenario=k[2],**v) for k,v in capacity.items()],
        selection_from_previous_family=True,prior_family_size=5,
        conditional_paper=True,official_Q=None,official_volatility=None,
        orders_sent=0,live_changed=False)
    save(OUT/'results.json',result);render(result)
    save(OUT/'status.json',dict(at=time.time(),state='complete',verdict=verdict,orders_sent=0))


def render(result):
    def fmt(x):return '확인 불가' if x is None else f'{x:,.5f}'
    lines=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>PAS 호가 이력 후보 추가 검증</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd}</style>',
        '<h1>호가 이력 후보: 별도 미사용 기간 검증</h1>',
        '<p>2026-10-25 18:00~10-26 00:05 KST. 자기 자본 50,000 USDP, 같은 초기 보유분과 위험 한도. 단위 USDP. 기존 실거래는 변경하지 않았습니다.</p>',
        '<p>앞선 다섯 후보 비교에서 선택한 quote_inputs를 사전에 고정했습니다. 아래는 실제 수신 호가와 발표 신호를 사용한 조건부 모의 체결이며 실제 수익이 아닙니다. 비용·지연·청산·실제 기존판 재현 오차를 모두 비교합니다.</p>',
        '<p>고정한 단기 조건: '+('통과' if result['verdict']['short_conditional_trial_eligible'] else '미통과')+'</p>',
        '<table><tr><th>실행 시점</th><th>비용</th><th>전략</th><th>순손익</th><th>종료 청산 후</th><th>수수료</th><th>펀딩</th><th>체결 수</th><th>자료·회계</th></tr>']
    for label,g in result['comparisons'].items():
        for r in g['rows']:
            lines.append('<tr><td>'+html.escape(label)+'</td><td>'+r['scenario']+'</td><td>'+r['arm']+'</td>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k in ['net_pnl','net_after_terminal_exit','fees','funding'])+'<td>'+str(r['fill_count'])+'</td><td>'+str(r['valid'])+'</td></tr>')
    lines+=['</table><ul>']+['<li>'+html.escape(s)+'</li>' for s in result['verdict']['reasons']]+['</ul>',
        '<p>단기 조건 통과와 장기 우월성은 구분합니다. 공식 일별 변동성·Q, 과적합 부재를 이 기간만으로 주장하지 않습니다.</p>',
        '<p><a href="results.json">전체 결과</a> · <a href="protocol.json">시작 전 고정 조건</a> · <a href="evaluation_contract.json">평가 코드 고정</a></p></html>']
    (OUT/'report.html').write_text('\n'.join(lines))


def run():
    c,e=validate()
    if (OUT/'first_open.json').exists():raise FileExistsError('Keep the first complete evaluation; do not overwrite')
    while time.time()<END+65:
        save(OUT/'status.json',dict(at=time.time(),state='collecting_unopened_interval',start=START,end=END,orders_sent=0))
        time.sleep(min(30,max(1,END+65-time.time())))
    validate()
    save(OUT/'first_open.json',dict(at=time.time(),start=START,end=END,orders_sent=0))
    try:
        funding,records,units=collect(c,e)
        evaluate(c,funding,records,units)
    except Exception as error:
        save(OUT/'failure.json',dict(at=time.time(),kind=type(error).__name__,error=str(error),orders_sent=0))
        save(OUT/'status.json',dict(at=time.time(),state='failed',error=str(error),orders_sent=0))
        raise


if __name__=='__main__':
    globals()[sys.argv[1]]()
