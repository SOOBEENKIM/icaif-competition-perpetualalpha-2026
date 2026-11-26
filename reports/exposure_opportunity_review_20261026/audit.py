"""Read-only diagnosis of the archived sizing overlay. Never imports a signer.

History is reused development data, not a new performance test. Quote sweeps
describe visible depth now, not future exit liquidity or guaranteed fills.
"""
import copy
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + '\n')


def blockers(base, old, protected, allowed, liquid, edge, symbols):
    rows = []
    for j, symbol in enumerate(symbols):
        reasons = []
        if protected[j]: reasons.append('risk_protected')
        if not allowed[j]: reasons.append('entry_filter')
        if not liquid[j]: reasons.append('reference_activity_missing')
        if old[j] != 0 and (old[j] * base[j] <= 0 or abs(base[j]) < abs(old[j]) - 1e-12):
            reasons.append('reduction_or_reversal_first')
        if base[j] == 0: reasons.append('no_base_target')
        if edge[j] <= .0043: reasons.append('edge_not_above_43bp')
        rows.append(dict(symbol=symbol, reasons=reasons, eligible=not reasons,
                         signed_edge_bp=float(edge[j] * 10000)))
    return rows


def history():
    from adaptive_leverage.controller import AdaptiveController
    from adaptive_leverage.study import live_manifest
    from competition_live.profile import load_profile
    from pas_performance_research.extension import OUT as EXT, START, END, SYMBOLS
    from pas_strategy_audit.data import load
    from pas_counterfactual_20261022.engine import simulate
    before = live_manifest()
    cfg, d = load(EXT)
    f = dict(np.load(EXT / 'candidate_inputs.npz'))
    audit_rows = []

    class Audited(AdaptiveController):
        def propose(self, at, prices, qty, equity, raw, sigma, volume, trades, protocol, **kwargs):
            action = super().propose(at, prices, qty, equity, raw, sigma, volume, trades, protocol, **kwargs)
            if action is not None:
                row = self.leverage_trace[-1]
                i = int(np.searchsorted(self.times, at))
                base = np.asarray(self.trace[-1]['target_qty'])
                reasons = blockers(base, qty, self.protected[i], self.trace[-1]['entry_filter_pass'],
                                   (volume > 0) & (trades > 0), row['signed_edge'], SYMBOLS)
                assert [r['eligible'] for r in reasons] == row['eligible']
                audit_rows.append(dict(time=int(at), multiplier=row['selected_target_multiplier'],
                                       utility_gain=row['forecast_utility_gain'], before=row['before'],
                                       after=row['after'], assets=reasons))
            return action

    controller = Audited(f['times'], f['forecast'], f['carry'], f['anchor'], f['protected'], 5.,
                         price_times=d['times'] / 1000, prices=d['close'],
                         entry_rule='pullback_any3', age=d['age_ms'] / 1000)
    tape, funding = [], []
    for s in SYMBOLS:
        for b in json.loads((EXT / 'raw' / f'pas_candles_{s}.json').read_text())['data']:
            at = int(b['t']) + 60000
            if float(b['v']) > 0 and b['n'] > 0 and START * 1000 <= at <= END * 1000:
                tape.append(dict(time=at, coin='pas:' + s, qty=float(b['v']),
                                 vwap=float(b['c']), buy_price=float(b['h']), sell_price=float(b['l'])))
        for r in json.loads((EXT / 'raw' / f'pas_funding_{s}.json').read_text())['data']:
            funding.append(dict(time=r['time'], coin='pas:' + s,
                                rate=float(r['fundingRate']), cash_per_unit=None))
    cfg.update(start_ms=START * 1000, cutoff_ms=END * 1000)
    sc = json.loads((EXT / 'protocol.json').read_text())['scenarios']['base_1pct']
    result = simulate(cfg, d, copy.deepcopy(json.loads((EXT / 'events.json').read_text())),
                      tape, funding, load_profile()['trading'], sc, controller=controller)
    archived = ROOT / 'reports/adaptive_leverage_20261026/pas/adaptive_leverage_continuous71h_base_1pct.json.gz'
    with gzip.open(archived, 'rt') as fp: original = json.load(fp)
    assert result == original, 'Instrumentation changed archived simulation'
    assert before == live_manifest(), 'Live source/config changed'
    counts = {}
    for row in audit_rows:
        for asset in row['assets']:
            for reason in asset['reasons']: counts[reason] = counts.get(reason, 0) + 1
    save('historical_audit.json', dict(start=START, end=END, rounds=len(audit_rows),
        asset_rounds=sum(len(r['assets']) for r in audit_rows), blocker_counts=counts,
        eligible_asset_rounds=sum(a['eligible'] for r in audit_rows for a in r['assets']),
        amplified_rounds=sum(r['multiplier'] > 1 for r in audit_rows),
        exact_archived_reproduction=True, independent_test=False, metrics=result['metrics'],
        daily_score_cutoff=original['daily'][-1]['time'], trace=audit_rows,
        archived_sha256=hashlib.sha256(archived.read_bytes()).hexdigest(), live_unchanged=True))
    print(json.dumps(dict(rounds=len(audit_rows), blockers=counts, exact_reproduction=True)), flush=True)


def sweep(levels, quantity):
    left, cash = quantity, 0.
    for level in levels:
        take = min(left, float(level['sz']))
        cash += take * float(level['px']); left -= take
        if left <= 1e-12: break
    return dict(covered=left <= 1e-12, filled_quantity=quantity-left,
                vwap=cash / (quantity-left) if quantity > left else None)


def current():
    from adaptive_leverage.policy import select_exposure
    from adaptive_leverage.study import live_manifest
    from candidate_handoff.public import PublicPAS
    from candidate_handoff.producer import validate
    from candidate_handoff.portfolio import target_from_frame
    from competition_live.profile import load_profile
    from joint_wallet.book import JointBook
    before = live_manifest(); profile = load_profile(); symbols = profile['trading']['symbols']
    runtime = ROOT / '.runtime/pas_budget_lifecycle'
    api = PublicPAS(profile); books = []
    try:
        for s in symbols:
            response = api.post(dict(type='l2Book', coin='pas:' + s))
            assert response['coin'] == 'pas:' + s
            books.append(dict(symbol=s, received_at=time.time(), data=response))
        meta, contexts = api.market()
        market_at = time.time()
        account = api.post(dict(type='clearinghouseState', dex='pas', user=profile['team_wallet']))
        received = time.time()
    finally: api.session.close()
    packet = json.loads((runtime / 'signal.json').read_text())
    frame = validate(packet, profile, received, expected_revision='all_market_signed_core_v1',
                     expected_policy='pullback_any3')['frame']
    ownership = json.loads((runtime / 'ownership.json').read_text())
    book = JointBook.restore(ownership)
    status = json.loads((runtime / 'status.json').read_text())
    assert status['pair_state'] is None, 'Separate base and active pair inventory before review'
    pos = {r['position']['coin']:r['position'] for r in account['assetPositions']}
    np.testing.assert_allclose([float(pos.get('pas:'+s,{}).get('szi',0)) for s in symbols],book.physical.qty,rtol=0,atol=1e-9)
    assert [m['name'] for m in meta['universe']] == ['pas:'+s for s in symbols]
    assert meta['collateralToken'] == profile['collateral_token']
    px = np.array([float(c['markPx']) for c in contexts]); book.reconcile(px)
    ours = book.owners['ours']; equity = ours.equity(px)
    action, trace = target_from_frame(frame, px, ours.qty, equity, profile['trading'])
    assert action is not None
    selection = select_exposure(base_qty=action['target_qty'],current_qty=ours.qty,prices=px,equity=equity,
        forecast=frame['price_forecast_168h'],funding=frame['reference_funding_forecast_168h'],
        hourly_covariance=frame['sigma'],protected=frame['protected'],entry_allowed=trace['entry_filter_pass'],
        liquid=(np.array(frame['volume'])>0)&(np.array(frame['trades'])>0),
        decimals=profile['trading']['execution']['decimals'],risk=profile['trading']['risk'])
    rows = blockers(np.array(action['target_qty']),ours.qty,frame['protected'],trace['entry_filter_pass'],
                    (np.array(frame['volume'])>0)&(np.array(frame['trades'])>0),selection['signed_edge'],symbols)
    assert [r['eligible'] for r in rows] == selection['eligible']
    fees = [f['fee']/abs(f['signed_qty']*f['price'])*10000 for f in ours.fills if f['signed_qty']]
    for j,(row,c,b) in enumerate(zip(rows,contexts,books)):
        bids,asks=b['data']['levels']; assert bids and asks
        mid=(float(bids[0]['px'])+float(asks[0]['px']))/2
        depth=[]
        for notional in [1000,5000]:
            q=notional/mid;buy=sweep(asks,q);sell=sweep(bids,q)
            depth.append(dict(notional=notional,buy=buy,sell=sell,
                immediate_roundtrip_bp=(buy['vwap']-sell['vwap'])/mid*10000+18 if buy['covered'] and sell['covered'] else None))
        row.update(direction='long' if action['target_qty'][j]>0 else 'short' if action['target_qty'][j]<0 else 'flat',
            own_notional=float(ours.qty[j]*px[j]),target_notional=float(action['target_qty'][j]*px[j]),
            reference_price_forecast_bp=frame['price_forecast_168h'][j]*10000,
            reference_funding_168h_bp=frame['reference_funding_forecast_168h'][j]*10000,
            pas_funding_hourly_bp=float(c['funding'])*10000,
            mark_oracle_basis_bp=(float(c['markPx'])/float(c['oraclePx'])-1)*10000,
            visible_book_depth=depth,book_received_at=b['received_at'],
            book_age_at_read_seconds=b['received_at']-b['data']['time']/1000)
    assert before == live_manifest()
    save('public_snapshot.json',dict(at=received,market_at=market_at,meta=meta,contexts=contexts,account=account,books=books))
    save('signal_snapshot.json',packet);save('ownership_snapshot.json',ownership)
    save('current_audit.json',dict(at=received,signal_source=frame['source_time'],own_equity=equity,
        own_gross_notional=float(abs(ours.qty)@px),risk=profile['trading']['risk'],
        leverage_policy=status['leverage_policy'],selection={k:v for k,v in selection.items() if k!='trials'},
        assets=rows,observed_fee_bp_range=[min(fees),max(fees)],observed_fee_bp_median=float(np.median(fees)),
        fee_model='9bp per side illustrative sweep, checked against owned fills; future exit liquidity unknown',
        scope='Unsigned hypothetical rebalance at captured marks; not the actual scheduled order',
        live_unchanged=True,orders_sent=0))
    print(json.dumps(dict(at=received,equity=equity,eligible=selection['eligible'],
        gross=float(abs(ours.qty)@px),fee_bp_median=float(np.median(fees)))),flush=True)


if __name__=='__main__':
    {'history':history,'current':current}[sys.argv[1]]()
