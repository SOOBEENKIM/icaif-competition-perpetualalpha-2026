"""Reconstruct par-only net returns from public settlement and capital records."""
import argparse
from collections import defaultdict, Counter
from decimal import Decimal as D
import csv
import datetime as dt
import html
import itertools
import json
from pathlib import Path

Z = D(0)
USER = '0x125778065a578d598d88bddaa3fdd635280d0e47'
TEAM_FUNDER = '0xc1beb07c770638b956601f2c6e54f643a99687ba'
KST = dt.timezone(dt.timedelta(hours=9))

def dec(value):
    return D(str(value or 0))

def date(ms):
    return dt.datetime.fromtimestamp(ms / 1000, KST).strftime('%Y-%m-%d %H:%M:%S') if ms else ''

def short(address):
    return address[:8] + '…' + address[-6:]

def fmt(value, n=4):
    return '—' if value is None else f'{dec(value):,.{n}f}'

def ordered_fills(fills):
    """Follow exchange startPosition transitions, including reversed same-ms rows."""
    out, errors = [], []
    current = defaultdict(D)
    for _, group in itertools.groupby(sorted(fills, key=lambda f: f['time']), key=lambda f: f['time']):
        by_coin = defaultdict(list)
        for fill in group:
            by_coin[fill['coin']].append(fill)
        for coin, rows in by_coin.items():
            edges = defaultdict(list)
            for fill in rows:
                edges[dec(fill['startPosition'])].append(fill)
            stack, path = [(current[coin], None)], []
            while stack:
                node, _ = stack[-1]
                if edges[node]:
                    fill = edges[node].pop()
                    qty = dec(fill['sz']) * (1 if fill['side'] == 'B' else -1)
                    stack.append((node + qty, fill))
                else:
                    _, fill = stack.pop()
                    if fill is not None:
                        path.append(fill)
            path.reverse()
            if len(path) != len(rows):
                errors.append('incomplete position chain: ' + coin)
                path = rows
            for fill in path:
                if dec(fill['startPosition']) != current[coin]:
                    errors.append('position transition mismatch: ' + coin)
                current[coin] += dec(fill['sz']) * (1 if fill['side'] == 'B' else -1)
                out.append(fill)
    return out, sorted(set(errors))

def cashflows(wallet, end, spot_pairs):
    address = wallet['address']
    flows, costs, issues = [], [], []
    for row in wallet['ledger']['rows']:
        if row['time'] > end:
            continue
        delta = row['delta']
        if delta.get('token') != 'USDPR':
            continue
        sender = delta.get('user', '').lower()
        recipient = delta.get('destination', '').lower()
        amount = dec(delta.get('amount'))
        if delta['type'] == 'spotGenesis':
            flows.append((row['time'], amount, 'genesis'))
        elif sender == recipient == address:
            pass  # Same owner's spot/perp transfer is not an external deposit.
        elif recipient == address:
            flows.append((row['time'], amount, 'external_deposit'))
        elif sender == address:
            flows.append((row['time'], -amount, 'external_withdrawal'))
        else:
            issues.append('unclassified USDPR ledger: ' + delta['type'])
        if sender == address and delta.get('feeToken') == 'USDPR':
            costs.append((row['time'], dec(delta.get('fee'))))
    for fill in wallet['fills']['rows']:
        if fill['time'] <= end and fill['coin'] in spot_pairs:
            qty = dec(fill['sz']) * (1 if fill['side'] == 'B' else -1)
            if fill['feeToken'] == 'USDPR':
                qty -= dec(fill['fee'])
            flows.append((fill['time'], qty, 'USDPR_spot_conversion'))
    return sorted(flows), costs, issues

def reconstruct(fills, funding, costs, prices):
    positions, entries = defaultdict(D), defaultdict(D)
    symbols = defaultdict(lambda: {'closed': Z, 'fees': Z, 'funding': Z, 'notional': Z, 'price_cashflow': Z,
                                   'fills': 0, 'maker_fills': 0, 'maker_notional': Z})
    issues = []
    for fill in fills:
        coin = fill['coin']
        qty = dec(fill['sz']) * (1 if fill['side'] == 'B' else -1)
        px, before, closed = dec(fill['px']), positions[coin], dec(fill['closedPnl'])
        after = before + qty
        if not before:
            entries[coin] = px
        elif before * qty > 0:
            entries[coin] = (abs(before) * entries[coin] + abs(qty) * px) / abs(after)
        elif after and before * after > 0:
            # Infer venue settlement basis, preserving its rounding on partial exits.
            entries[coin] = px - closed / min(abs(before), abs(qty)) * (1 if before > 0 else -1)
        elif after:
            entries[coin] = px
        else:
            entries[coin] = Z
        positions[coin] = after
        symbol = symbols[coin]
        symbol['closed'] += closed
        symbol['price_cashflow'] -= qty * px
        if fill['feeToken'] == 'USDPR':
            symbol['fees'] += dec(fill['fee'])  # fee already includes builderFee.
        else:
            issues.append('non-USDPR fee: ' + coin)
        notional = abs(qty) * px
        symbol['notional'] += notional
        symbol['fills'] += 1
        if not fill['crossed']:
            symbol['maker_fills'] += 1
            symbol['maker_notional'] += notional
    for row in funding:
        symbols[row['delta']['coin']]['funding'] += dec(row['delta']['usdc'])
    for coin, symbol in symbols.items():
        qty = positions[coin]
        if qty and coin not in prices:
            issues.append('missing oracle: ' + coin)
        symbol['quantity'] = qty
        symbol['entry'] = entries[coin]
        symbol['unrealized'] = qty * (prices.get(coin, Z) - entries[coin]) if qty else Z
        # closedPnl uses rounded position bases and is not always exactly the
        # self-financing return. Reconcile with executed quantities and prices.
        symbol['price_pnl'] = symbol['price_cashflow'] + qty * prices.get(coin, Z)
        symbol['reported_pnl_rounding_adjustment'] = symbol['price_pnl'] - symbol['closed'] - symbol['unrealized']
        symbol['net'] = symbol['price_pnl'] - symbol['fees'] + symbol['funding']
    result = {key: sum((s[key] for s in symbols.values()), Z)
              for key in ('closed', 'fees', 'funding', 'unrealized', 'notional', 'maker_notional', 'reported_pnl_rounding_adjustment')}
    result['transfer_fees'] = sum((cost for _, cost in costs), Z)
    result['net'] = result['closed'] - result['fees'] + result['funding'] + result['unrealized'] + result['reported_pnl_rounding_adjustment'] - result['transfer_fees']
    return result, dict(symbols), issues

def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k,v in row.items()})

def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(v)+'</th>' for v in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    root, raw = args.report, args.report / 'raw'
    market = json.loads((raw / 'market_now.json').read_text())
    manifest = json.loads((raw / 'manifest.json').read_text())
    end = market['queried_at_ms']
    oracle = {u['name']: dec(c['oraclePx']) for u,c in zip(market['response'][0]['universe'], market['response'][1])}
    mark = {u['name']: dec(c['markPx']) for u,c in zip(market['response'][0]['universe'], market['response'][1])}
    wallets = [json.loads(p.read_text()) for p in sorted((raw/'wallets').glob('*.json'))]
    ids = json.loads((args.previous/'raw/wallet_ids.json').read_text())
    old_analysis = json.loads((args.previous/'analysis.json').read_text())
    old_rows = {r['address']: r for r in old_analysis.get('wallets', old_analysis.get('leaderboard', []))}
    next_id = max(int(v[1:]) for v in ids.values()) + 1
    for wallet in wallets:
        if wallet['par_fill_count'] and wallet['address'] not in ids:
            ids[wallet['address']] = f'W{next_id:02}'
            next_id += 1
    (raw/'wallet_ids.json').write_text(json.dumps(ids, indent=2))
    trades = defaultdict(lambda: defaultdict(list))
    for wallet in wallets:
        for fill in wallet['fills']['rows']:
            if fill['coin'].startswith('par:') and fill['time'] <= end:
                trades[(fill['coin'], fill['time'], fill['tid'])][fill['side']].append((wallet['address'], fill))
    mismatches, unpaired = [], []
    for key, sides in trades.items():
        if len(sides['B']) != 1 or len(sides['A']) != 1:
            unpaired.append(key)
        else:
            buy, sell = sides['B'][0][1], sides['A'][0][1]
            if (dec(buy['px']), dec(buy['sz'])) != (dec(sell['px']), dec(sell['sz'])):
                mismatches.append(key)
    rows, fills_out, funding_out, flows_out, positions_out, orders_out, symbol_out, inventory = [], [], [], [], [], [], [], []
    details = {}
    for wallet in wallets:
        address = wallet['address']
        wid = ids.get(address, 'USER' if address == USER else '')
        fills, issues = ordered_fills([f for f in wallet['fills']['rows'] if f['coin'].startswith('par:') and f['time'] <= end])
        flows, costs, flow_issues = cashflows(wallet, end, manifest['usdpr_spot_pairs'])
        issues += flow_issues
        funding = [f for f in wallet.get('funding', {}).get('rows', []) if f['time'] <= end and f['delta'].get('coin', '').startswith('par:')]
        metrics, symbols, accounting_issues = reconstruct(fills, funding, costs, oracle)
        issues += accounting_issues
        for field in ('fills', 'ledger', 'funding'):
            if not wallet.get(field, {}).get('pagination_complete'):
                issues.append(field + ': incomplete history')
        if wallet['fills'].get('retention_limit_possible'):
            issues.append('10,000 recent fill retention limit may omit history')
        firstfund = min((t for t,v,_ in flows if v > 0), default=None)
        if fills and (firstfund is None or fills[0]['time'] < firstfund):
            issues.append('capital before first fill not established')
        one_grant = len(flows) == 1 and flows[0][1] == D(1000000) and flows[0][2] == 'external_deposit'
        grant_confirmed = one_grant and any(r['delta'].get('user', '').lower() == TEAM_FUNDER and
                            r['delta'].get('destination', '').lower() == address and r['delta'].get('token') == 'USDPR'
                            and dec(r['delta'].get('amount')) == D(1000000) for r in wallet['ledger']['rows'])
        capital = sum((v for _,v,_ in flows), Z)
        denominator = sum((v * D(end-t) / D(end-firstfund) for t,v,_ in flows), Z) if firstfund is not None and end > firstfund else Z
        return_percent = metrics['net'] / denominator * 100 if denominator > 0 else None
        if denominator <= 0:
            issues.append('no positive invested-capital denominator')
        live = wallet.get('positions', {}).get('response', {})
        live_positions = [p['position'] for p in live.get('assetPositions', [])]
        orders = wallet.get('orders', {}).get('response', [])
        if not isinstance(orders, list):
            orders = []
        live_time = live.get('time')
        inv = {'wallet_id': wid, 'address': address, 'fills': len(fills), 'single_grant_1m': grant_confirmed,
               'net_capital_USDPR': str(capital), 'first_capital_kst': date(firstfund),
               'discovery_sources': wallet['discovery_sources'], 'issues': sorted(set(issues))}
        inventory.append(inv)
        for t,value,kind in flows:
            flows_out.append({'wallet_id':wid, 'address':address, 'time_ms':t,'time_kst':date(t),'amount_USDPR':str(value),'kind':kind})
        for p in live_positions:
            positions_out.append({'wallet_id':wid, 'address':address,'query_time_kst':date(live_time), **p})
        for order in orders:
            orders_out.append({'wallet_id':wid, 'address':address,'query_received_utc':wallet['orders']['received_at_utc'], **order})
        for f in funding:
            funding_out.append({'wallet_id':wid,'address':address,'time_kst':date(f['time']),'time_ms':f['time'],**f['delta']})
        for f in fills:
            key = f['coin'],f['time'],f['tid']
            peers = trades[key]['A' if f['side']=='B' else 'B']
            fills_out.append({'wallet_id':wid,'address':address,'time_kst':date(f['time']),**f,
                'notional_USDPR':str(dec(f['px'])*dec(f['sz'])),
                'fee_bps':str(dec(f['fee'])/(dec(f['px'])*dec(f['sz']))*10000),
                'counterparty':peers[0][0] if len(peers)==1 else ''})
        for coin, values in symbols.items():
            symbol_out.append({'wallet_id':wid,'address':address,'coin':coin,'oracle_at_cutoff':str(oracle.get(coin)),
                               **{k:str(v) if isinstance(v,D) else v for k,v in values.items()}})
        if not fills and address != USER:
            continue
        prior = old_rows.get(address, {})
        row = {'rank':None, 'wallet_id':wid, 'address':address,'single_grant_1m':grant_confirmed,
               'group':'100만 USDPR 지급 확인' if grant_confirmed else '기타 자금 규모·유입',
               'return_percent':float(return_percent) if return_percent is not None and not issues else None,
               'net_pnl_USDPR':str(metrics['net']), 'net_capital_USDPR':str(capital),
               'reconstructed_equity_USDPR':str(capital+metrics['net']),
               **{k+'_USDPR':str(metrics[k]) for k in ('closed','fees','funding','unrealized','transfer_fees','notional','reported_pnl_rounding_adjustment')},
               'fill_count':len(fills),'new_fill_count':len(fills)-prior.get('fill_count',prior.get('lifetime_par_fill_count',0)),
               'first_fill_kst':date(fills[0]['time']) if fills else None,'last_fill_kst':date(fills[-1]['time']) if fills else None,
               'maker_fill_percent':sum(not f['crossed'] for f in fills)/len(fills)*100 if fills else None,
               'maker_notional_percent':float(metrics['maker_notional']/metrics['notional']*100) if metrics['notional'] else None,
               'open_positions_at_cutoff':sum(s['quantity']!=0 for s in symbols.values()),
               'queried_position_count':len(live_positions),'queried_order_count':len(orders),
               'position_query_kst':date(live_time),
               'largest_markets':', '.join(c.replace('par:','') for c,s in sorted(symbols.items(),key=lambda cs:-cs[1]['notional'])[:3]),
               'first_capital_kst':date(firstfund),'cutoff_kst':date(end),
               'return_method':'net par PnL / 1,000,000 USDPR' if one_grant else 'Modified Dietz since first recorded USDPR funding',
               'previous_return_percent':prior.get('return_percent',prior.get('period_return_percent')),
               'previous_rank':prior.get('rank'),
               'mark_net_pnl_USDPR':str(reconstruct(fills,funding,costs,mark)[0]['net']),
               'mark_return_percent':float(reconstruct(fills,funding,costs,mark)[0]['net']/denominator*100) if denominator > 0 and not issues else None,
               'daily_return_std_percent':None, 'official_total_score':None,
               'history_issues':'; '.join(sorted(set(issues))), 'official_team_identity':'unverified'}
        rows.append(row)
        details[address] = {'symbols':symbols,'fills':fills[-15:][::-1], 'live_positions':live_positions, 'orders':orders,
                           'spot_snapshot':wallet.get('spot',{}), 'mark_valued_net_pnl_USDPR':str(reconstruct(fills,funding,costs,mark)[0]['net'])}
    eligible = sorted([r for r in rows if r['single_grant_1m'] and r['fill_count'] and r['return_percent'] is not None], key=lambda r:-r['return_percent'])
    for i,row in enumerate(eligible, 1):
        row['rank'] = i
    other = sorted([r for r in rows if r['fill_count'] and not r['single_grant_1m']],key=lambda r:-(r['return_percent'] if r['return_percent'] is not None else -1e100))
    rows.sort(key=lambda r:(r['rank'] is None,r['rank'] or 1000,-(r['return_percent'] or 0)))
    validation = {'snapshot_cutoff_kst':date(end),'wallets_collected':len(wallets),'trading_wallets':sum(bool(r['fill_count']) for r in rows),
        'one_million_grant_trading_wallets':sum(r['single_grant_1m'] and bool(r['fill_count']) for r in rows),
        'ranked_one_million_grant_wallets':len(eligible),'wallet_side_fills':len(fills_out),
        'unique_executions':len(trades),'unpaired_executions':len(unpaired),'price_quantity_mismatches':mismatches,
        'history_issue_wallets':[{k:r[k] for k in ('wallet_id','address','history_issues')} for r in rows if r['history_issues']],
        'unprocessed_wallets':manifest['unprocessed'],
        'scope':'observed public wallets; official team address mapping unavailable; practice period'}
    for name, data in [('leaderboard',rows),('fills',fills_out),('funding',funding_out),('capital_flows',flows_out),
                       ('positions',positions_out),('open_orders',orders_out),('by_market',symbol_out),('wallet_inventory',inventory)]:
        write_csv(root/(name+'.csv'),data)
    identity = {'keyrock_role_confirmed':True, 'keyrock_confirmed_addresses':[],
        'sources':['https://app.notion.com/p/3eb73946edee809d94ebc05f95828180',
                   'https://app.notion.com/p/3eb73946edee810b995de6f2fcfa2608',
                   'https://stats-data.hyperliquid.xyz/Testnet/leaderboard'],
        'findings':'Organizer email names Keyrock as market maker. Discord October 8 Q6/A6 discusses its launch. Neither supplied page provides Keyrock wallet addresses. Fresh public testnet leaderboard has no Keyrock name; observed trading addresses have no displayName.',
        'exclusion_policy':'Exactly one 1M USDPR team-distributor grant and no additional USDPR capital flows is the primary comparison filter. Other funding patterns, including large accounts and the public par deployer, are outside that ranking. This filter is evidence of funding conditions, not proof of wallet ownership or proof that every market maker has been removed.',
        'outside_primary_ranking':[{'wallet_id':r['wallet_id'],'address':r['address'],
             'net_capital_USDPR':r['net_capital_USDPR'],'reason':'different capital amount or funding pattern; Keyrock ownership unverified'} for r in other]}
    (root/'identity_and_exclusions.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2))
    method = {'cutoff_ms':end,'cutoff_kst':date(end),'oracle_prices':{c:str(p) for c,p in oracle.items()},
       'return':'Exact executed-price PnL = sum(signed quantity * (cutoff oracle - execution price)), minus actual USDPR fees plus funding cashflows minus USDPR transfer fees. The difference from reported closedPnl plus reconstructed open PnL is separately disclosed as a rounding adjustment. Initial and additional deposits are capital, not profit.',
       'period':'Cumulative since first recorded USDPR capital; each wallet has its own first trading date. This is not a matched-start or official competition ranking.',
       'capital_groups':'Primary ranking requires exactly one 1,000,000 USDPR external deposit from the observed team distributor and no additional USDPR capital flows. Other capital patterns use Modified Dietz and are displayed separately.',
       'equity':'USDPR capital plus par trading PnL reconstructed from public records; unrelated USDC markets and protocol/deployer revenue are excluded.',
       'timestamps':'All rankings use the same recorded oracle context and fill/funding cutoff. Live leverage, position state and resting orders were fetched later and retain separate query timestamps.',
       'limitations':'No official address-to-team or Keyrock address mapping; only observed wallets. Historical daily oracle equity and jury scores unavailable, so official total scores and risk ranks are not fabricated. PnL is reconstructed from public trade/funding records, not a claim of exact venue account-balance reconciliation.',
       'sources':['https://perpetual-alpha26.github.io/about','https://perpetual-alpha26.github.io/leaderboard','https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint']}
    (root/'analysis.json').write_text(json.dumps({'method':method,'validation':validation,'wallets':rows,'details':details},ensure_ascii=False,indent=2,default=str))
    (root/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    columns = ['순위','지갑','주소','순수익률 %','순손익 USDPR','체결 / 신규','주요 종목','미결제 포지션 / 주문']
    def cells(r):
        return [r['rank'] or '—',('내 지갑 · ' if r['address']==USER else '')+r['wallet_id'],r['address'],fmt(r['return_percent'],6),fmt(r['net_pnl_USDPR'],2),
                f"{r['fill_count']} / +{r['new_fill_count']}",r['largest_markets'],f"{r['queried_position_count']} / {r['queried_order_count']}"]
    body = f'<h1>공개 지갑 수익률 리더보드</h1><p class="lead">{date(end)} KST 관측 오라클 기준 · 연습 기간 · 실제 체결 기록 분석</p>'
    body += '<p>순손익은 실제 매수·매도 가격과 수량, 잔여 포지션의 오라클 가치, 수수료·펀딩·USDPR 송금비용으로 재구성합니다. 입금은 수익에서 제외합니다. API 표시 실현손익의 반올림 차이는 별도 보정 항목으로 공개합니다. 100만 USDPR를 한 번 지급받고 추가 자금 유입·유출이 없는 지갑끼리 순위를 비교합니다. 지갑별 거래 시작일과 운용 기간은 다릅니다.</p>'
    body += f'<p>발견 지갑 {len(wallets)}개 · 체결 지갑 {validation["trading_wallets"]}개 · 100만 지급 순위 {len(eligible)}개 · 시장 체결 {len(trades):,}건 / 지갑별 체결 {len(fills_out):,}건. 공식 팀 이름과 주소의 연결은 확인되지 않았습니다. 10월 19일 본선 전의 비공식 참고표입니다.</p>'
    body += '<p>수익률이 0.06%라면 100만 USDPR 기준 순손익은 600 USDPR입니다. 개별 포지션의 레버리지 ROE와 계좌 수익률은 다릅니다. 일별 위험 점수와 심사위원 점수가 없어 공식 종합 순위는 산출하지 않습니다.</p>'
    body += f'<p><strong>Keyrock 구분:</strong> 주최 측 메일에서 시장조성 업체임을 확인했지만, 제공된 공지·공개 리더보드에는 해당 지갑 주소가 없습니다. 100만 지급 조건이 다른 {len(other)}개 거래 지갑은 본 순위에서 제외했습니다. 이 분류만으로 Keyrock 지갑을 모두 식별·제외했다고 볼 수는 없습니다. <a href="identity_and_exclusions.json">주소별 제외 내역과 근거</a></p>'
    body += '<p>'+ ' · '.join(f'<a href="{name}.csv">{label} CSV</a>' for name,label in [('leaderboard','전체 수익률'),('fills','전체 체결·상대 지갑'),('by_market','종목별 손익'),('positions','현재 포지션'),('open_orders','현재 미체결'),('capital_flows','입출금'),('funding','펀딩')]) + '</p>'
    body += '<h2>100만 USDPR 지급 지갑 · 거래한 지갑 순위</h2>' + table(columns,[cells(r) for r in eligible])
    user_row = next((r for r in rows if r['address']==USER),None)
    if user_row:
        body += '<h2>사용자 지갑</h2>'+table(columns,[cells(user_row)])
        body += '<p>'+html.escape(f"오라클 기준 순손익 {fmt(user_row['net_pnl_USDPR'],4)} USDPR / {fmt(user_row['return_percent'],6)}%. 같은 시각 마크 가격 기준 순손익 {fmt(user_row['mark_net_pnl_USDPR'],4)} USDPR / {fmt(user_row['mark_return_percent'],6)}%. 거래소 화면은 마크 가격과 조회 시점에 따라 다를 수 있습니다. 일별 STD·공식 종합 점수는 미산출입니다.")+'</p>'
    body += '<h2>다른 자금 규모의 거래 지갑</h2><p>봇·시장 운영 계정도 포함될 수 있지만, 주소만으로 운영 주체를 단정하지 않습니다. 추가 입출금이 있는 경우 시간 가중 자본을 분모로 하는 Modified Dietz를 사용합니다. 위의 100만 지급 지갑 순위와 합산하지 않습니다.</p>' + table(columns+['순자금 USDPR','오류'],[cells(r)+[fmt(r['net_capital_USDPR'],2),r['history_issues']] for r in other])
    bad_grant = [r for r in rows if r['single_grant_1m'] and r['fill_count'] and r['return_percent'] is None]
    if bad_grant:
        body += '<h2>기록 검증 미통과 · 순위 제외</h2>'+table(columns+['오류'],[cells(r)+[r['history_issues']] for r in bad_grant])
    body += '<h2>지갑별 체결·포지션 상세</h2><input id="filter" placeholder="주소, W 번호 또는 종목으로 검색" oninput="document.querySelectorAll(\'details.wallet\').forEach(d=>d.hidden=!d.textContent.toLowerCase().includes(this.value.toLowerCase()))">'
    for row in rows:
        detail = details[row['address']]
        body += '<details class="wallet"><summary>'+html.escape(f"{row['wallet_id']} · {row['address']} · {fmt(row['return_percent'],6)}% · {fmt(row['net_pnl_USDPR'],2)} USDPR")+'</summary>'
        body += '<p>'+html.escape(f"첫 체결 {row['first_fill_kst']} / 마지막 체결 {row['last_fill_kst']} KST. 누적 {row['fill_count']}회, 메이커 체결 비중 {fmt(row['maker_fill_percent'],2)}%. 실현손익 {fmt(row['closed_USDPR'],4)}, 수수료 {fmt(row['fees_USDPR'],4)}, 펀딩 {fmt(row['funding_USDPR'],4)}, 미실현손익 {fmt(row['unrealized_USDPR'],4)} USDPR.")+'</p>'
        body += table(['종목','체결 수','거래대금','표시 실현손익','수수료','펀딩','미실현','반올림 보정','순손익','기준시각 포지션'],[[c]+[fmt(s[k],0 if k=='fills' else 4) for k in ('fills','notional','closed','fees','funding','unrealized','reported_pnl_rounding_adjustment','net','quantity')] for c,s in detail['symbols'].items()])
        body += '<p>최근 체결 최대 15건. 전체 기록과 거래 상대 주소는 전체 체결 CSV에 포함됩니다.</p>'
        body += table(['시각 KST','종목','방향','가격','수량','실현손익','수수료','체결'],[[date(f['time']),f['coin'],f['dir'],f['px'],f['sz'],f['closedPnl'],f['fee'],'테이커' if f['crossed'] else '메이커'] for f in detail['fills']])
        body += '<p>'+html.escape(f"포지션 조회 {row['position_query_kst']} KST. 레버리지·현재 포지션·주문은 순위 기준 시각보다 나중에 조회했습니다.")+'</p>'
        body += table(['종목','보유 수량','평균 진입가','레버리지','거래소 표시 미실현손익','청산가'],[[p['coin'],p['szi'],p['entryPx'],str(p['leverage']),p['unrealizedPnl'],str(p.get('liquidationPx'))] for p in detail['live_positions']])
        body += table(['종목','매수/매도','지정가','잔여수량','reduceOnly','주문번호'],[[o['coin'],o['side'],o['limitPx'],o['sz'],o.get('reduceOnly'),o['oid']] for o in detail['orders']])
        if row['history_issues']:
            body += '<p class="warning">'+html.escape(row['history_issues'])+'</p>'
        body += '</details>'
    body += '<h2>검증과 범위</h2><pre>'+html.escape(json.dumps(validation,ensure_ascii=False,indent=2))+'</pre>'
    coverage_path=root/'coverage_validation.json'
    if coverage_path.exists():
        coverage=json.loads(coverage_path.read_text())
        body += '<p>'+html.escape(f"시장 체결 수 교차 검증: 완료된 종목·시간 구간 {coverage['completed_coin_hours_checked']:,}개, 시장 체결 {coverage['completed_hour_market_executions']:,}건, 불일치 {len(coverage['candle_count_mismatches'])}개. 후속 체결을 반영한 현재 포지션 {coverage['position_wallets_checked']}지갑 검증 통과 여부: {coverage['passed']}.")+'</p>'
    if (root/'balance_reconciliation.json').exists():
        body += '<p>포지션이 없는 상위 지갑 W13·W26의 재구성 자산을 거래소 실제 USDPR 잔액과 대조했습니다. 차이는 각각 0, −0.000005 USDPR였습니다. <a href="balance_reconciliation.json">잔액 대조 원본</a></p>'
    body += '<p>공개 API의 최근 체결 보관 한도는 10,000건입니다. 전날 저장된 원본과 신규 기록을 합치고, 체결별 시작 포지션 연결 및 매수·매도 양측 가격·수량을 대조했습니다. 발견되지 않은 주소나 API 보관 범위 밖의 이력은 누락될 수 있습니다. 별도 검증 파일은 <a href="coverage_validation.json">coverage_validation.json</a>을 참조하세요.</p>'
    body += '<p><a href="https://perpetual-alpha26.github.io/about">공식 평가 기준</a> · <a href="https://perpetual-alpha26.github.io/leaderboard">공식 등록 팀 목록</a> · <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint">API 원본 정의</a></p>'
    css = 'body{max-width:1500px;margin:40px auto;padding:0 24px;font:15px/1.65 system-ui;color:#243047;background:#f7f9fc}h1{font-size:30px}.lead{font-size:19px}h2{margin-top:36px}.scroll{overflow:auto;background:white;border:1px solid #dee4ed;border-radius:10px}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{padding:9px 12px;border-bottom:1px solid #e8edf4;text-align:right}td:nth-child(2),td:nth-child(3){text-align:left}th{background:#edf2fa}details{background:white;margin:12px 0;padding:14px;border:1px solid #dce4ef;border-radius:10px}summary{cursor:pointer;font-weight:600}input{padding:12px;width:min(600px,90%);font:inherit}pre{white-space:pre-wrap;overflow:auto;background:#edf2fa;padding:18px}.warning{color:#b03a26}a{color:#1658b6}'
    (root/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 지갑 수익률 리더보드</title><style>'+css+'</style><body>'+body+'</body></html>')
    print(json.dumps({'validation':validation,'top':eligible[:10],'other':other,'user':user_row},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
