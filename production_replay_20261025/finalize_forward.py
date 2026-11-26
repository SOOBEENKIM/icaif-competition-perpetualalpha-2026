"""Assemble the fixed future experiments after their registered endings.

No strategy tuning or live installation. Keeps failed/diagnostic arms visible.
"""
import gzip
import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from .protocol import ROOT, OUT, save, sha

DEST = OUT/'forward_review'
END = 1792915500
EXPERIMENTS = [
    dict(name='entry_lifetime', directory='forward_evaluation', primary='fixed_entry_1h', start=1792897500),
    dict(name='capital_usage', directory='capital_forward', primary='size_only_2', start=1792900800),
    dict(name='schedule_recovery', directory='catchup/future', primary='fresh_block_catchup', start=1792904400),
]


def freeze():
    assert time.time() < END
    assert not (DEST/'protocol.json').exists()
    files = [Path(__file__), ROOT/'production_replay_20261025/liquidation_audit.py',
             ROOT/'production_replay_20261025/settlement_audit.py']
    save(DEST/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), end=END,
        experiments=EXPERIMENTS,
        evaluation='Same exact start/end valuation; reconstruct arrival-time fills and cash from the common initial ledger. Deduct executable observed-depth liquidation including fees. Missing exit depth, missing settlement or accounting/risk failures invalidate promotion. Keep original approximate-funding run; public-settlement revaluation is fixed-path sensitivity, not hidden policy rerun.',
        decision='For each preregistered primary, positive own net PnL and greater than control after terminal exit, in both base and stress; all audit checks pass. Diagnostic hourly candidate cannot become a primary after results. Passing is short conditional paper evidence only, not proof of long-term alpha or automatic deployment.',
        statistical_limit='Multiple prespecified overlapping comparisons, not independent replications. No p-value or official daily-volatility/Q claim from these hours. Do not imply absence of overfitting.',
        hashes={str(p.relative_to(ROOT)):sha(p) for p in files}, orders_sent=0))


def check():
    p=json.loads((DEST/'protocol.json').read_text())
    for rel,digest in p['hashes'].items():
        if sha(ROOT/rel)!=digest:
            raise ValueError('Pre-outcome audit implementation changed: '+rel)
    return p


def run():
    check()
    while time.time() < END+900:
        save(DEST/'status.json',dict(at=time.time(),state='waiting_for_original_experiments',orders_sent=0))
        time.sleep(min(30,max(1,END+900-time.time())))
    check()
    from competition_live.profile import load_profile
    from competition_live.runtime import private_load
    from .book_replay import load_books
    from .liquidation_audit import compare_at_common_end
    from .settlement_audit import path_cash_sensitivity
    profile=load_profile();seed=private_load(OUT/'forward_seed.pkl')['portfolio']
    streams,capture=load_books((OUT/'capture').glob('*.jsonl.gz'),END)
    save(DEST/'capture_manifest.json',capture)
    cashpath=OUT/'settlement_check/cash_units.json'
    units=json.loads(cashpath.read_text()) if cashpath.exists() else []
    settlements=[]
    for s in profile['trading']['symbols']:
        path=OUT/'forward_evaluation'/f'{s}_funding.json'
        if path.exists():
            settlements.extend(dict(r,coin='pas:'+s) for r in json.loads(path.read_text())['rows'])
    groups=[]
    for experiment in EXPERIMENTS:
        rows=[]
        for scenario,sc in dict(base=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=1.),
                                stress=dict(fee_bps=12.,depth_fraction=.5,latency_ms=3000,receipt_multiplier=.5)).items():
            results={}
            for arm in ['incumbent',experiment['primary']]:
                path=OUT/experiment['directory']/f'{arm}_{scenario}.json.gz'
                if path.exists():
                    with gzip.open(path,'rt') as fp:results[arm]=json.load(fp)
                else:rows.append(dict(arm=arm,scenario=scenario,valid=False,reason='original_result_missing'))
            for row in compare_at_common_end(streams,profile,sc,results,experiment['start'],END):
                row['scenario']=scenario
                r=results[row['arm']]
                cash=path_cash_sensitivity(r,seed.account.physical.fills,units,profile['trading']['symbols'],
                                          experiment['start'],END,sc['receipt_multiplier'],settlements)
                row['settlement_sensitivity']=cash
                if not cashpath.exists() or not settlements or not cash['complete']:
                    row['valid']=False;row['settlement_verification_missing']=True
                if row.get('net_pnl') is not None and cash['complete']:
                    original_cash=sum(float(f['delta']['usdc']) for f in r['funding']
                                      if experiment['start'] < f['time']/1000 <= END)
                    row['fixed_path_funding_difference']=cash['funding']-original_cash
                rows.append(row)
        decisions=[]
        for scenario in ['base','stress']:
            group={r['arm']:r for r in rows if r['scenario']==scenario}
            c=group.get('incumbent',{});n=group.get(experiment['primary'],{})
            passed=bool(c.get('valid') and n.get('valid') and n.get('net_pnl',-1)>0
                        and n.get('net_after_terminal_exit',-1)>0
                        and n['net_after_terminal_exit']>c['net_after_terminal_exit'])
            # Funding revaluation cannot silently reverse the paired advantage.
            if passed:
                advantage=n['net_after_terminal_exit']-c['net_after_terminal_exit']
                sensitivity=n.get('fixed_path_funding_difference',0)-c.get('fixed_path_funding_difference',0)
                rounding=sum(x['settlement_sensitivity']['maximum_rounding_error'] for x in [c,n])
                passed=advantage+sensitivity>rounding
            decisions.append(dict(scenario=scenario,passed=passed))
        groups.append(dict(**experiment,rows=rows,checks=decisions,
                           passes_short_paper_gate=all(d['passed'] for d in decisions),
                           live_replacement_authorized_by_this_report=False))
    save(DEST/'results.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),experiments=groups,
         conditional_paper_only=True,official_volatility=None,official_Q=None,actual_orders_sent=0))
    render(groups)
    save(DEST/'status.json',dict(at=time.time(),state='complete',orders_sent=0))


def render(groups):
    lines=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>PAS 고정 후보 비교</title>',
           '<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}small{color:#555}</style>',
           '<h1>PAS 고정 후보 비교</h1><p>실제 수신한 호가·신호로 계산한 조건부 모의 결과입니다. 실제 주문은 0건이며 현재 운용 전략을 변경하지 않았습니다. 동일 보유분·자본·평가 시각에서 비용과 청산 가능성을 대조했습니다.</p>',
           '<p>짧은 기간이므로 공식 일별 변동성과 Q는 계산하지 않습니다. 겹치는 기간의 여러 후보 비교를 서로 독립된 검증으로 취급하지 않습니다.</p>']
    def fmt(v):return '확인 불가' if v is None else f'{v:,.6f}'
    for g in groups:
        lines.append('<h2>'+html.escape(g['name'])+'</h2><p>짧은 모의 비교 기준: '+('통과' if g['passes_short_paper_gate'] else '미통과')+'</p>')
        lines.append('<table><tr><th>후보</th><th>조건</th><th>순손익 USDP</th><th>종료 청산 반영</th><th>자료·실행 검사</th></tr>')
        for r in g['rows']:
            lines.append('<tr><td>'+html.escape(r['arm'])+'</td><td>'+r['scenario']+'</td><td>'+fmt(r.get('net_pnl'))+'</td><td>'+fmt(r.get('net_after_terminal_exit'))+'</td><td>'+('통과' if r['valid'] else html.escape(r.get('reason','불완전 자료 또는 검사 실패')) )+'</td></tr>')
        lines.append('</table>')
    lines.append('<p>펀딩 대조는 <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding">공식 결제 방식</a>에 따라 공개 계좌의 실제 지급액을 사용합니다. 고정 거래 경로에 대한 금액 민감도이며, 결제금 변경에 따른 정책 재실행은 아닙니다.</p>')
    lines.append('<p>이 보고서는 자동 교체 승인이 아닙니다. 통과하더라도 긴 기간의 성과 보장이나 실제 체결 보장으로 해석할 수 없습니다. 전체 상세 결과는 <a href="results.json">JSON</a>에 남깁니다.</p></html>')
    (DEST/'report.html').write_text('\n'.join(lines))


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
