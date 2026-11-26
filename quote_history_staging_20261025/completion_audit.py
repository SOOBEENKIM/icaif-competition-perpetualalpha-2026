"""Finish the existing frozen trial with accounting checks, never new trials.

Only reads completed research outputs. No signer, live checkpoint, service
control, network, model fitting, candidate selection or order submission.
"""
import gzip
import html
import json
import math
import time

from production_replay_20261025.replication import OUT, START, END
from production_replay_20261025.replication_evaluate import validate
from production_replay_20261025.protocol import save, sha


def close(a, b, label, tolerance=1e-6):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > tolerance:
        raise ValueError(f'{label}: {a!r} != {b!r}')


def path_accounting(r, row, scenario, symbols):
    """Rebuild cash and inventory directly from fills and settled funding."""
    first, last = r['snapshots'][0], r['snapshots'][-1]
    if first['time'] != START or last['time'] != END:
        raise ValueError('Missing common endpoint snapshot')
    qty = list(first['qty'])
    cash = first['equity'] - sum(q*m for q, m in zip(qty, first['marks']))
    orders = [o for o in r['orders'] if o['filled']]
    if len(orders) != len(r['fills']):
        raise ValueError('Fill/order count mismatch')
    fees = 0.
    for f, o in zip(r['fills'], orders):
        if not START < o['arrival'] <= END:
            raise ValueError('Fill outside fixed evaluation interval')
        j = f['asset']
        if symbols[j] != o['symbol']:
            raise ValueError('Fill/order asset mismatch')
        close(f['signed_qty'], o['side']*o['filled'], 'fill quantity')
        close(f['price'], o['price'], 'fill price')
        close(f['fee'], abs(f['signed_qty'])*f['price']*scenario['fee_bps']/10000, 'fee')
        cash -= f['signed_qty']*f['price'] + f['fee']
        qty[j] += f['signed_qty']
        fees += f['fee']
    funding = sum(float(f['delta']['usdc']) for f in r['funding']
                  if START < f['time']/1000 <= END)
    cash += funding
    for a, b in zip(qty, last['qty']):
        close(a, b, 'ending inventory')
    end_equity = cash + sum(q*m for q, m in zip(qty, last['marks']))
    close(end_equity, last['equity'], 'ending equity')
    pnl = end_equity-first['equity']
    close(pnl, row['net_pnl'], 'net PnL')
    close(fees, row['fees'], 'total fees')
    close(funding, row['funding'], 'total funding')
    close(pnl+fees-funding, row['price_pnl'], 'price PnL')
    close(pnl/50000*100, row['return_on_50000_pct'], 'return denominator')
    if row.get('net_after_terminal_exit') is not None:
        close(pnl+row['terminal_exit']['priced_partial_adjustment'],
              row['net_after_terminal_exit'], 'terminal liquidation PnL')
    return dict(accounting_pass=True, initial_equity=first['equity'],
                initial_qty=first['qty'], net_pnl=pnl, fills=len(orders))


def audit():
    if time.time() < END+65:
        raise ValueError('Outcome interval is still closed')
    contract = json.loads((OUT/'completion_audit_contract.json').read_text())
    if contract['source_sha'] != sha(__file__) or contract['at'] >= END:
        raise ValueError('Completion audit changed after registration')
    c, e = validate()
    main = json.loads((OUT/'results.json').read_text())
    temporal = json.loads((OUT/'temporal_audit.json').read_text())
    opened = json.loads((OUT/'first_open.json').read_text())
    seed = json.loads((OUT/'seed_manifest.json').read_text())
    if opened['at'] < e['first_open_not_before']:
        raise ValueError('Outcomes opened before registered cutoff')
    if seed['at'] >= START or seed['seed_sha'] != sha(OUT/'forward_seed.pkl'):
        raise ValueError('Initial condition changed')
    if (main['start'], main['end']) != (START, END):
        raise ValueError('Evaluation interval changed')
    if main['orders_sent'] or main['live_changed']:
        raise ValueError('Research output reports a live mutation')
    if not temporal['complete'] or temporal['original_result_sha'] != sha(OUT/'results.json'):
        raise ValueError('Timestamp audit incomplete or results changed')
    checks = []
    for phase in c['phases_seconds']:
        label = f'realistic_phase{phase}'
        group = main['comparisons'][label]
        for row in group['rows']:
            p = OUT/label/f"{row['arm']}_{row['scenario']}.json.gz"
            with gzip.open(p, 'rt') as f:
                r = json.load(f)
            checks.append(dict(path=str(p.relative_to(OUT)), sha256=sha(p),
                **path_accounting(r, row, c['scenarios'][row['scenario']], c['profile']['trading']['symbols'])))
    if len(checks) != 12:
        raise ValueError('Expected all twelve prespecified paths')
    for x in checks[1:]:
        close(x['initial_equity'], checks[0]['initial_equity'], 'shared starting equity')
        if x['initial_qty'] != checks[0]['initial_qty']:
            raise ValueError('Arms started with different inventory')
    actual = json.loads((OUT/'actual_owner_ledger.json').read_text())
    actual_cash = actual['initial']-sum(f['signed_qty']*f['price']+f['fee'] for f in actual['fills'])
    actual_cash += sum(f['signed_amount'] for f in actual['funding_events'])
    close(actual_cash, actual['shadow_cash'], 'actual owner ledger cash')
    result = dict(at=time.time(), complete=True, accounting_pass=True, paths=checks,
        result_sha=sha(OUT/'results.json'), temporal_sha=sha(OUT/'temporal_audit.json'),
        original_qualification=main['verdict']['short_conditional_trial_eligible'],
        final_qualification=temporal['eligible'], next_experiment_started=False,
        automatic_live_replacement=False, orders_sent=0, live_changed=False)
    save(OUT/'completion_audit.json', result)
    render(main, result)
    return result


def render(main, audit_result):
    rows = []
    for phase, g in main['comparisons'].items():
        for scenario in ('base', 'stress'):
            arms = {r['arm']:r for r in g['rows'] if r['scenario'] == scenario}
            a, b = arms['incumbent'], arms[main['primary']]
            values = [a['net_pnl'], b['net_pnl'], b['net_pnl']-a['net_pnl'],
                      a['net_after_terminal_exit'], b['net_after_terminal_exit']]
            rows.append('<tr><td>'+html.escape(phase)+'</td><td>'+scenario+'</td>'+''.join(
                '<td>'+('미확인' if v is None else f'{v:,.6f}')+'</td>' for v in values)+'</tr>')
    verdict = '통과' if audit_result['final_qualification'] else '미통과'
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>PAS 호가 이력 검증 종료 결과</title>
<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}</style>
<h1>기존판과 호가 이력 검증판 비교</h1>
<p>2026-10-25 18:00 ~ 10-26 00:05 한국시간. 본인 자본 50,000 USDP, 같은 초기 보유분·신호·위험 한도. 단위 USDP.</p>
<p>수신한 PAS 호가와 발표 신호를 이용한 조건부 모의 체결 결과입니다. 실제 후보의 거래 손익이 아닙니다. 미사용 구간을 사전에 고정했지만 약 6시간의 한 구간이므로 장기 우월성이나 과적합 부재를 증명하지 않습니다.</p>
<table><tr><th>실행 시점</th><th>비용</th><th>기존 순손익</th><th>검증판 순손익</th><th>차이</th><th>기존 청산 후</th><th>검증판 청산 후</th></tr>'''+''.join(rows)+'''</table>
<p>12개 경로의 현금·수량·수수료·펀딩·손익 계산 대조 완료. 사전 교체 검토 기준: '''+verdict+'''.</p>
<p>상대적인 손익 개선 여부와 사전 검토 기준 통과 여부는 구분합니다. 공식 일별 변동성·Q·순위는 이 단기 비교에서 계산하지 않습니다.</p>
<p>이번 연구는 종료하며 다른 전략이나 실험을 추가하지 않습니다. 실제 운용 전략은 바꾸지 않았습니다.</p>
<p><a href="report.html">세부 결과와 미통과 사유</a> · <a href="results.json">원본 결과</a> · <a href="completion_audit.json">회계 대조</a> · <a href="temporal_audit.json">정산 시각 대조</a></p></html>'''
    (OUT/'completion_report.html').write_text(document)


def main():
    if (OUT/'completion_audit.json').exists():
        raise FileExistsError('Preserve the first completion audit')
    while time.time() < END+65:
        time.sleep(min(30, max(1, END+65-time.time())))
    while True:
        if (OUT/'failure.json').exists():
            raise ValueError('Original evaluation failed; no replacement experiment will run')
        try:
            status = json.loads((OUT/'status.json').read_text())
            temporal = json.loads((OUT/'temporal_audit.json').read_text())
            if status.get('state') == 'complete' and temporal.get('complete'):
                break
            if temporal.get('complete') is False:
                raise ValueError('Original timestamp audit did not complete')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.time() > END+3600:
            raise TimeoutError('Original evaluation did not finish within its audit deadline')
        time.sleep(15)
    return audit()


if __name__ == '__main__':
    try:
        print(json.dumps(main(), ensure_ascii=False))
    except Exception as error:
        save(OUT/'completion_audit_failure.json', dict(at=time.time(), complete=False,
            kind=type(error).__name__, error=str(error), next_experiment_started=False,
            orders_sent=0, live_changed=False))
        raise
