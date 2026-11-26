"""Current public PAS account + local ownership -> unsigned sizing comparison."""
import json
import time
from pathlib import Path
import numpy as np
import requests
from competition_live.profile import load_profile
from candidate_handoff.producer import validate
from candidate_handoff.portfolio import target_from_frame
from joint_wallet.book import JointBook
from .policy import select_exposure, select_margin
from .study import ROOT, OUT, live_manifest


def main():
    original = live_manifest()
    p = load_profile()
    runtime = ROOT / '.runtime/pas_budget_lifecycle'
    packet = json.loads((runtime / 'signal.json').read_text())
    book = JointBook.restore(json.loads((runtime / 'ownership.json').read_text()))
    status = json.loads((runtime / 'status.json').read_text())
    if status.get('pair_state') is not None:
        raise ValueError('Preview requires separate base/pair inventory when a pair is active')
    started = time.time()
    with requests.Session() as session:
        def read(payload):
            # Only two public info queries; never use /exchange or a signer.
            response = session.post(p['rest_url'] + '/info', json=payload, timeout=20)
            response.raise_for_status()
            return response.json()
        meta, contexts = read({'type': 'metaAndAssetCtxs', 'dex': 'pas'})
        account = read({'type': 'clearinghouseState', 'dex': 'pas', 'user': p['team_wallet']})
    now = time.time()
    if now - started > 15 or not -2 <= now - account['time'] / 1000 <= 15:
        raise ValueError('Public snapshot too stale')
    f = validate(packet, p, now, expected_revision='all_market_signed_core_v1',
                 expected_policy='pullback_any3')['frame']
    symbols = p['trading']['symbols']
    if ([m['name'] for m in meta['universe']] != ['pas:' + s for s in symbols]
            or meta['collateralToken'] != p['collateral_token']):
        raise ValueError('Venue identity differs')
    positions = {r['position']['coin']: r['position'] for r in account['assetPositions']}
    if set(positions) - {'pas:' + s for s in symbols}:
        raise ValueError('Unexpected physical market')
    actual = [float(positions.get('pas:' + s, {}).get('szi', 0.)) for s in symbols]
    np.testing.assert_allclose(actual, book.physical.qty, rtol=0, atol=1e-9)
    np.testing.assert_allclose(book.owners['ours'].qty, status['own_qty'], rtol=0, atol=1e-9)
    marks = np.array([float(c['markPx']) for c in contexts])
    book.reconcile(marks)
    ours, teammate = book.owners['ours'], book.owners['teammate']
    equity, other_equity = ours.equity(marks), teammate.equity(marks)
    action, trace = target_from_frame(f, marks, ours.qty, equity, p['trading'])
    if action is None:
        raise ValueError('No incumbent target for comparison')
    selection = select_exposure(base_qty=action['target_qty'], current_qty=ours.qty,
        prices=marks, equity=equity, forecast=f['price_forecast_168h'],
        funding=f['reference_funding_forecast_168h'], hourly_covariance=f['sigma'],
        protected=f['protected'], entry_allowed=trace['entry_filter_pass'],
        liquid=(np.array(f['volume']) > 0) & (np.array(f['trades']) > 0),
        decimals=p['trading']['execution']['decimals'], risk=p['trading']['risk'])
    leverage = []
    for s in symbols:
        position = positions.get('pas:' + s)
        if position is None:
            # A hypothetical new asset setting must be queried before live use.
            raise ValueError('Missing active leverage; request activeAssetData before preview')
        if position['leverage']['type'] != 'cross':
            raise ValueError('Isolated position requires a different margin model')
        leverage.append(position['leverage']['value'])
    maximum = [m['maxLeverage'] for m in meta['universe']]
    margin = select_margin(own_notional=np.array(selection['target_qty']) * marks,
        teammate_notional=teammate.qty * marks, own_equity=equity, teammate_equity=other_equity,
        current_leverage=leverage, exchange_max=maximum, leverage_ceiling=3)
    result = dict(at=now, signal_source=f['source_time'], signal_sha256=packet['sha256'],
        symbols=symbols, own_equity=equity, teammate_equity=other_equity,
        current_own_notional=float(abs(ours.qty) @ marks),
        current_own_actual_leverage=float(abs(ours.qty) @ marks / equity),
        current_exchange_leverage=leverage, exchange_max_leverage=maximum,
        baseline_target_qty=action['target_qty'], exposure=selection, margin=margin,
        orders_sent=0, setting_updates=0, live_changed=False, live_ready=False,
        scope='Hypothetical rebalance now, not the scheduled next order. Same incumbent risk limits.',
        forecast_scope='Frozen reference-market 168h forecast; not calibrated PAS expected profit.',
        margin_scope='3x research setting ceiling and 80% per-owner margin budget; not approved live limits.')
    OUT.mkdir(exist_ok=True)
    (OUT / 'public_snapshot.json').write_text(json.dumps(dict(at=now, meta=meta, contexts=contexts, account=account)))
    (OUT / 'preview_signal.json').write_text(json.dumps(packet))
    (OUT / 'current_preview.json').write_text(json.dumps(result, indent=2))
    assert original == live_manifest(), 'Running code/config changed during preview'
    print(json.dumps({k: result[k] for k in ('at', 'own_equity', 'current_own_notional',
                     'current_own_actual_leverage', 'current_exchange_leverage')}))
    print(json.dumps(dict(selected_target_multiplier=selection['selected_target_multiplier'],
                         margin=margin, live_changed=False)))


if __name__ == '__main__': main()
