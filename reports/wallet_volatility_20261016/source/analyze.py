"""Daily volatility from public, whole-account equity snapshots.

This is a fixed, retrospective analysis, not trading code. It intentionally
rejects external cash-flow intervals instead of treating deposits as returns.
"""
import bisect
import csv
import datetime as dt
from decimal import Decimal as D
import hashlib
import html
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PRIOR = PROJECT / 'reports/wallet_leaderboard_20261016'
USER = '0x125778065a578d598d88bddaa3fdd635280d0e47'
DAY = 86_400_000
START = 1791849600000  # 2026-10-13 00:00 UTC / 09:00 KST
END = START + 3 * DAY
KST = dt.timezone(dt.timedelta(hours=9))


def stamp(t):
    return dt.datetime.fromtimestamp(t / 1000, KST).isoformat(timespec='seconds')


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def external_events(ledger, lo, hi):
    result = []
    for event in ledger:
        if not lo < event['time'] <= hi:
            continue
        delta = event['delta']
        # Same wallet moving collateral between spot and par is internal.
        internal = (delta.get('type') == 'send'
                    and delta.get('user', '').lower() == delta.get('destination', '').lower())
        if not internal:
            result.append(event)
    return result


def sampled_equity(portfolio):
    result = {}
    series = dict(portfolio)
    for name in ('allTime', 'month', 'week', 'day'):
        for time, equity in series[name]['accountValueHistory']:
            value = D(equity)
            if time in result and abs(result[time] - value) > D('.00001'):
                raise ValueError(f'Conflicting equity at {time}')
            result[time] = value
    # Do not merge pnlHistory: its baseline can differ between API windows.
    return sorted(result), result


def exposure_days(fills):
    fills = [x for x in fills if x['coin'].startswith('par:')]
    result = []
    for index in range(3):
        lo, hi = START + index * DAY, START + (index + 1) * DAY
        position = {}
        for fill in fills:
            if fill['time'] >= lo:
                continue
            coin = fill['coin']
            position[coin] = position.get(coin, D(0)) + D(fill['sz']) * (1 if fill['side'] == 'B' else -1)
        trades = sum(lo <= x['time'] < hi for x in fills)
        result.append({'has_exposure_or_fill': any(x != 0 for x in position.values()) or trades > 0,
                       'fills': trades})
    return result


def make_row(wallet, exchange):
    address = wallet['address']
    raw = json.loads((ROOT / 'raw' / f'{address}.json').read_text())
    previous = json.loads((PRIOR / 'raw/wallets' / f'{address}.json').read_text())
    assert previous['ledger']['pagination_complete']
    assert previous['history_end_ms'] >= END
    times, points = sampled_equity(raw['response'])
    selected = []
    for index in range(4):
        boundary = START + index * DAY
        j = bisect.bisect_right(times, boundary) - 1
        if j < 0:
            raise ValueError(f'{address}: missing daily boundary')
        time = times[j]
        lag = boundary - time
        if lag > 3 * 3_600_000 or points[time] <= 0:
            raise ValueError(f'{address}: stale or unfunded boundary')
        selected.append((time, points[time], boundary, lag))
    flows = external_events(previous['ledger']['rows'], selected[0][0], selected[-1][0])
    if flows:
        raise ValueError(f'{address}: external cash flows require a different return method')
    exposure = exposure_days(previous['fills']['rows'])
    returns, daily = [], []
    for i in range(3):
        t0, e0, b0, l0 = selected[i]
        t1, e1, b1, l1 = selected[i + 1]
        r = (e1 - e0) / e0
        returns.append(float(r))
        daily.append({
            'wallet_id': wallet['wallet_id'], 'address': address,
            'nominal_start_kst': stamp(b0), 'nominal_end_kst': stamp(b1),
            'observed_start_kst': stamp(t0), 'observed_end_kst': stamp(t1),
            'start_lag_minutes': l0 / 60000, 'end_lag_minutes': l1 / 60000,
            'start_equity': str(e0), 'end_equity': str(e1),
            'external_cash_flow': '0', 'equity_change': str(e1 - e0),
            'return_percent': float(r * 100),
            'has_par_exposure_or_fill': exposure[i]['has_exposure_or_fill'],
            'par_fill_count': exposure[i]['fills'],
        })
    sample_sd = statistics.stdev(returns) * 100
    population_sd = statistics.pstdev(returns) * 100
    assert abs(sample_sd - float(np.std(returns, ddof=1) * 100)) < 1e-12
    assert abs(population_sd - float(np.std(returns, ddof=0) * 100)) < 1e-12
    chained = float(np.prod(1 + np.array(returns)) - 1)
    total_return = float(selected[-1][1] / selected[0][1] - 1)
    assert abs(chained - total_return) < 1e-12
    source = exchange.get(address)
    performance = dict(source['windowPerformances']) if source else {}
    all_time = performance.get('allTime', {})
    active_returns = [r for r, x in zip(returns, exposure) if x['has_exposure_or_fill']]
    row = {
        'volatility_rank_42': None,
        'wallet_id': wallet['wallet_id'], 'address': address, 'is_user': address == USER,
        'single_grant_1m': wallet['single_grant_1m'],
        'daily_std_sample_percent': sample_sd,
        'daily_std_population_percent': population_sd,
        'relative_volatility_score_0_100': None,
        'observed_days': 3,
        'exposure_days': len(active_returns),
        'exposure_days_only_std_percent': statistics.stdev(active_returns) * 100 if len(active_returns) >= 2 else None,
        'fills_in_period': sum(x['fills'] for x in exposure),
        'first_par_fill_kst': wallet['first_fill_kst'],
        'last_par_fill_kst_at_prior_snapshot': wallet['last_fill_kst'],
        'common_period_return_percent': total_return * 100,
        'common_period_equity_change': str(selected[-1][1] - selected[0][1]),
        'max_boundary_lag_minutes': max(x[3] for x in selected) / 60000,
        'last_equity_observed_kst': stamp(selected[-1][0]),
        'exchange_all_time_pnl': all_time.get('pnl'),
        'exchange_all_time_roi_percent': float(D(all_time['roi']) * 100) if all_time else None,
        'exchange_all_time_volume': all_time.get('vlm'),
        'exchange_all_time_pnl_rank_within_42': None,
        'exchange_snapshot_account_value': source.get('accountValue') if source else None,
        'portfolio_received_at_utc': raw['received_at_utc'],
        'external_flow_count_in_observed_period': 0,
        'official_team_identity': 'unverified',
    }
    return row, daily


def render(rows, method):
    escape = html.escape
    user = next(x for x in rows if x['is_user'])
    def f(value, digits=6):
        return '—' if value is None else f'{float(value):,.{digits}f}'
    def table(subset):
        lines = []
        for row in subset:
            address = row['address']
            cells = [str(row['volatility_rank_42'] or '—'),
                     escape(row['wallet_id'] + (' · 내 지갑' if row['is_user'] else '')),
                     f'<code>{address}</code>',
                     f(row['daily_std_sample_percent'], 9),
                     f(row['relative_volatility_score_0_100'], 4),
                     f"{row['exposure_days']} / 3", str(row['fills_in_period']),
                     f(row['common_period_return_percent'], 6),
                     f(row['exchange_all_time_pnl'], 6),
                     f(row['exchange_all_time_roi_percent'], 6),
                     f(row['max_boundary_lag_minutes'], 1)]
            lines.append(f'<tr class="{"mine" if row["is_user"] else ""}">' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
        headers = ['변동성 순위', '지갑', '주소', '일별 표본 STD (%) ↓', '변동성 참고점수 / 100',
                   '노출일수', '기간 체결 수', '공통기간 수익률 (%)', '거래소 누적 PNL', '거래소 누적 ROI (%)', '최대 표본 지연 (분)']
        return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(lines) + '</tbody></table></div>'
    primary = sorted([x for x in rows if x['single_grant_1m']], key=lambda x: (x['daily_std_sample_percent'], x['wallet_id']))
    others = sorted([x for x in rows if not x['single_grant_1m']], key=lambda x: x['daily_std_sample_percent'])
    content = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>공개 계좌 이력 기반 변동성 비교 · 2026-10-16</title>
<style>body{{font:15px system-ui,sans-serif;margin:32px;color:#182337;background:#f7f9fc}}h1{{font-size:26px}}p,li{{line-height:1.7}}.card{{background:white;border:1px solid #dce3ef;padding:20px;border-radius:12px;margin:18px 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}th,td{{padding:11px;border-bottom:1px solid #dde3eb;text-align:right}}th{{background:#e9eff8;cursor:pointer}}td:nth-child(2),td:nth-child(3){{text-align:left}}.mine{{background:#def4ea;font-weight:700}}input{{font:inherit;padding:12px;width:min(650px,90%);border:1px solid #abb8cb;border-radius:8px}}a{{color:#165bd6}}small{{color:#485a73}}code{{font-size:12px}}</style>
<h1>일별 수익률 변동성 비교</h1>
<p>공통기간: <b>2026-10-13 09:00 → 2026-10-16 09:00 KST</b> · UTC 기준 완료된 3일 · 공개 표본으로 추정 · 공식 대회 순위 아님</p>
<div class="card"><b>내 지갑 {USER}</b><p>일별 표본 표준편차 <b>{f(user['daily_std_sample_percent'], 9)}%</b> · 변동성이 낮은 순서 <b>{user['volatility_rank_42']} / 42</b> · 비교군 내 변동성 참고점수 <b>{f(user['relative_volatility_score_0_100'], 4)} / 100</b><br>
거래 노출이 있었던 날: <b>{user['exposure_days']} / 3일</b>. 거래 이후 완료된 일별 수익률이 1개이므로, 거래한 날만의 표준편차는 계산하지 않음.<br>
마지막 실제 계좌 표본: <b>{user['last_equity_observed_kst']}</b>. 이후의 장중 손익은 이번 일별 STD에 포함하지 않음.</p></div>
<p><a href="leaderboard.csv">전체 지갑 CSV</a> · <a href="daily_returns.csv">날짜별 계좌가치·수익률 CSV</a> · <a href="analysis.json">계산 결과 JSON</a> · <a href="validation.json">검산 결과</a></p>
<div class="card"><b>계산·비교 기준</b><ul>
<li>일별 수익률 = (종료 계좌가치 − 시작 계좌가치) / 시작 계좌가치. 해당 표본 기간에 외부 입출금이 없음을 원장으로 확인. 같은 지갑의 spot→par 담보 이동은 외부 입출금이 아님.</li>
<li>일별 STD는 수익률 3개의 표본 표준편차(n−1). 연율화하지 않음. 모집단 표준편차(n)는 CSV에 별도 제공. 두 방식의 순위는 같음.</li>
<li>거래소 portfolio API의 <b>전체 계좌가치</b>를 사용하므로 대기자금과 미실현손익이 포함됨. par만의 과거 오라클 평가액을 재구성한 공식 대회 지표와는 다름.</li>
<li>각 UTC 자정 이하의 마지막 공개 표본을 사용. 지갑별 실제 관측 시각이 다르며 최대 지연은 {f(method['maximum_boundary_lag_minutes'], 1)}분. 누락 시각을 선형 보간하거나 미래값으로 채우지 않음.</li>
<li>42개 지갑은 앞서 100만 USDPR 단일 지급이 확인된 비교군. 모든 공식 참가팀 또는 Keyrock 제외가 보증된 명단이 아님. 나머지 18개도 별도 계산.</li>
<li>참고점수 = 100 × (비교군 최대 STD − 해당 STD) / (최대 STD − 최소 STD). 심사위원 점수와 합치지 않음. 적은 거래일·작은 노출 때문에 낮은 변동성이 나올 수 있음.</li>
<li>거래소 누적 PNL·ROI는 공개 리더보드 원본. 공통 3일 수익률과 기간·분모가 다름. 파일 Last-Modified: {escape(method['exchange_file_last_modified'])}; 조회: {escape(method['exchange_retrieved_at_utc'])}. 새로고침 시각이 곧 각 행의 평가 시각이라는 뜻은 아님.</li>
</ul></div>
<input id="filter" placeholder="지갑 주소 또는 W53 검색" aria-label="지갑 검색">
<h2>100만 USDPR 단일 지급 확인 지갑 42개</h2>{table(primary)}
<h2>기타 자금 규모·유입 지갑 18개</h2><p>주소만으로 시장봇 여부를 확정하지 않으며, 42개 비교군의 점수 범위에 포함하지 않았습니다.</p>{table(others)}
<p>출처: <a href="https://perpetual-alpha26.github.io/about">대회 평가 규정</a> · <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint">공개 API</a> · <a href="https://app.hyperliquid-testnet.xyz/leaderboard">거래소 리더보드</a></p>
<script>document.getElementById('filter').addEventListener('input',e=>{{let q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});document.querySelectorAll('th').forEach(h=>h.addEventListener('click',()=>{{let t=h.closest('table'),i=[...h.parentNode.children].indexOf(h),b=t.tBodies[0],up=h.dataset.up!=='1';h.dataset.up=up?'1':'0';[...b.rows].sort((a,b)=>{{let x=a.cells[i].textContent.replaceAll(',',''),y=b.cells[i].textContent.replaceAll(',',''),nx=Number(x),ny=Number(y);return(up?1:-1)*(Number.isFinite(nx)&&Number.isFinite(ny)?nx-ny:x.localeCompare(y));}}).forEach(r=>b.appendChild(r));}}));</script></html>'''
    (ROOT / 'report.html').write_text(content)


def main():
    prior = json.loads((PRIOR / 'analysis.json').read_text())
    stats = json.loads((ROOT / 'raw/exchange_leaderboard.json').read_text())
    exchange = {w['ethAddress'].lower(): w for w in stats['response']['leaderboardRows']}
    rows, daily = [], []
    for wallet in prior['wallets']:
        row, days = make_row(wallet, exchange)
        rows.append(row)
        daily.extend(days)
    primary = [r for r in rows if r['single_grant_1m']]
    assert len(rows) == 60 and len(primary) == 42 and len(daily) == 180
    sigmas = [r['daily_std_sample_percent'] for r in primary]
    lo, hi = min(sigmas), max(sigmas)
    for row in primary:
        row['volatility_rank_42'] = 1 + sum(x < row['daily_std_sample_percent'] for x in sigmas)
        row['relative_volatility_score_0_100'] = (hi - row['daily_std_sample_percent']) / (hi - lo) * 100
        pnl = D(row['exchange_all_time_pnl'])
        row['exchange_all_time_pnl_rank_within_42'] = 1 + sum(D(r['exchange_all_time_pnl']) > pnl for r in primary)
    rows.sort(key=lambda r: (not r['single_grant_1m'], r['daily_std_sample_percent'], r['wallet_id']))
    method = {
        'nominal_start_kst': stamp(START), 'nominal_end_kst': stamp(END),
        'daily_timezone': 'UTC, chosen explicitly; official day boundary not specified in the public rule reviewed',
        'equity': 'Whole-account accountValueHistory, including idle spot collateral; not perp-only equity',
        'sampling': 'Last available observation at or before each daily UTC boundary, at most 3 hours old',
        'maximum_boundary_lag_minutes': max(r['max_boundary_lag_minutes'] for r in rows),
        'cashflows': 'No external ledger events over any selected observed period; one same-wallet spot-to-par send excluded',
        'return': '(E_end - E_start) / E_start, only after the external-flow audit passed',
        'standard_deviation': 'Primary: sample ddof=1 of three daily returns; population ddof=0 also exported; neither annualized',
        'score': '100*(max_sigma - sigma)/(max_sigma - min_sigma), within the fixed 42-wallet comparison cohort only',
        'scope': '60 previously observed par-trading wallets, 42 single-1M grants; no verified team or Keyrock mapping',
        'official_score': False, 'historical_oracle_valuation': False,
        'exchange_retrieved_at_utc': stats['received_at_utc'],
        'exchange_file_last_modified': stats['headers'].get('Last-Modified'),
        'source_urls': ['https://perpetual-alpha26.github.io/about',
                        'https://api.hyperliquid-testnet.xyz/info', stats['source']],
    }
    validation = {
        'passed': True, 'wallets': len(rows), 'primary_wallets': len(primary),
        'daily_return_records': len(daily), 'missing_boundary_points': 0,
        'nonpositive_boundary_equities': 0, 'unsupported_cashflow_intervals': 0,
        'ledger_history_complete_through_cutoff': True,
        'numpy_and_statistics_standard_deviations_agree': True,
        'chained_daily_returns_reconcile_to_total_equity_change': True,
        'zero_volatility_primary_wallets': sum(s == 0 for s in sigmas),
        'no_external_flows_subtracted_twice': True,
        'pnl_window_rebasing_avoided_by_using_equity_history_only': True,
        'no_current_partial_day_in_standard_deviation': True,
        'notes': 'Public histories are sparse. Checks establish calculation consistency, not exact midnight or oracle valuations.',
    }
    save(ROOT / 'analysis.json', {'method': method, 'wallets': rows, 'daily_returns': daily})
    save(ROOT / 'validation.json', validation)
    write_csv(ROOT / 'leaderboard.csv', rows)
    write_csv(ROOT / 'daily_returns.csv', daily)
    render(rows, method)
    source_files = list((ROOT / 'raw').glob('*.json')) + list((ROOT / 'source').glob('*.py'))
    source_files += [PRIOR / 'raw/wallets' / f"{r['address']}.json" for r in rows]
    save(ROOT / 'source_checksums.json', {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    print(json.dumps({'validation': validation, 'user': next(r for r in rows if r['is_user'])}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
