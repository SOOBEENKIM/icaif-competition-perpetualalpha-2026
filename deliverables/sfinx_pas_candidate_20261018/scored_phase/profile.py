"""Explicit scored-phase boundaries; practice/research artifacts stay immutable."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'deliverables/sfinx_pairs_candidate_20261018_final'
CANDIDATE = 'first_dense_hourlyhedge_decisiongate_24'
START = int(datetime(2026, 10, 19, 12, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 11, 6, 12, tzinfo=timezone.utc).timestamp())
SYMBOLS = ('BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY')


def verify_bundle():
    manifest = json.loads((BUNDLE/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (BUNDLE/name).resolve()
        if not path.is_relative_to(BUNDLE) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Selected candidate bundle changed: '+name)
    return hashlib.sha256((BUNDLE/'manifest.json').read_bytes()).hexdigest()


def make_profile(snapshot):
    original = json.loads((BUNDLE/'config.json').read_text())
    if original['candidate'] != CANDIDATE:
        raise ValueError('Unexpected strategy selection')
    markets = {m['name']:m for m in snapshot['meta']['universe']}
    if set(markets) != {'pas:'+s for s in SYMBOLS}:
        raise ValueError('Unexpected PAS universe')
    token = snapshot['meta']['collateralToken']
    tokens = [t for t in snapshot['spot_meta']['tokens'] if t['index']==token]
    if len(tokens)!=1 or tokens[0]['name']!='USDP':
        raise ValueError('PAS collateral is not USDP')
    p = copy.deepcopy(original['trading'])
    factor = 100_000/p['initial']
    p['initial'] = 100_000.
    p['risk']['asset_notional'] *= factor
    p['execution']['decimals'] = [markets['pas:'+s]['szDecimals'] for s in SYMBOLS]
    configs = copy.deepcopy(original['configs'])
    for config in configs.values():
        config['pair_max_gross'] *= factor
    return dict(version=1, candidate=CANDIDATE, source_bundle=str(BUNDLE.relative_to(ROOT)),
        source_manifest_sha256=verify_bundle(), network='testnet', dex='pas', collateral='USDP',
        rest_url='https://api.hyperliquid-testnet.xyz', ws_url='wss://api.hyperliquid-testnet.xyz/ws',
        team_wallet='0x125778065a578d598D88bddaA3FdD635280d0E47',
        initial_capital=100_000., collateral_token=token,
        start_unix=START,end_unix=END,start_utc='2026-10-19T12:00:00Z',end_utc='2026-11-06T12:00:00Z',
        start_kst='2026-10-19T21:00:00+09:00',end_kst='2026-11-06T21:00:00+09:00',
        snapshot_hour_utc=12, paper_deadline_date='2026-11-06',paper_deadline_time_zone=None,
        paper_deadline_time=None, paper_deadline_note='Announcement specifies the date only; do not assume the trading close is the paper deadline.',
        venue_caps=dict(per_market=10_000_000.,total=50_000_000.,scope='All market participants, not per-team buying power.'),
        eligibility=dict(practice_fill_required=True,scored_phase_fill_required=True,minimum_daily_returns_for_score=2),
        scoring=dict(lambda_=1.5,c=.05,epsilon=.005,Q_weight=.8,jury_weight=.2,jury_default=0.),
        trading=p,configs=configs,pairs=original['pairs'],gate_spec=original['gate_spec'],
        execution=dict(maximum_single_order_notional=15_000.,end_liquidation_seconds=900,
            enable_orders=False,require_actual_funding=True,allow_automatic_transfer=False,
            allowed_prefix='pas:',blocked_prefixes=['par:','pa:'],minimum_order_value=10.),
        baseline_target_rule=original['baseline_target_rule'],
        training_data_note='Historical reference and PAR data are retained as training/research sources, never as PAS execution quotes.',
        migration_note='Scale absolute risk/sizing caps by capital ratio; retain statistical model, fractional limits and learned parameters. Read PAS lots and caps from API.')


def load_profile():
    p = json.loads((Path(__file__).parent/'config.json').read_text())
    if (p['dex'],p['collateral'],p['candidate'],p['initial_capital'],p['start_unix'],p['end_unix']) != ('pas','USDP',CANDIDATE,100_000.,START,END):
        raise ValueError('Scored profile does not match the announced phase')
    if p['rest_url'] != 'https://api.hyperliquid-testnet.xyz':
        raise ValueError('Testnet endpoint required')
    return p


if __name__ == '__main__':
    snapshot=json.loads((Path(__file__).parent/'artifacts/public_preflight/snapshot.json').read_text())
    target=Path(__file__).parent/'config.json'
    if target.exists(): raise FileExistsError('Do not silently replace the scored profile')
    target.write_text(json.dumps(make_profile(snapshot),indent=2,ensure_ascii=False)+'\n')
    print(target)
