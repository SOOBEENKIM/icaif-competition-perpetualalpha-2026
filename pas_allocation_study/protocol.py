import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/pas_allocation_study_20261021'
SYMBOLS = ['BTC','ETH','HYPE','SP500','EWY','EWJ','BRENT','GOLD','USDEUR','USDJPY']


def ts(s):
    return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native(x):
    if hasattr(x, 'tolist'): return x.tolist()
    raise TypeError(type(x).__name__)


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=native, allow_nan=False))


def freeze():
    path = OUT/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    pointer = json.loads((ROOT/'practice_live/latest.json').read_text())
    from competition_live.profile import verify_deployment
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        experiment='Partially pooled return model plus venue error correction and funding; one nominated policy, no search',
        symbols=SYMBOLS, pointer=pointer, input_hash=sha(ROOT/pointer['inputs']),
        target_hash=sha(ROOT/pointer['run']/'targets.npz'),
        train_start=ts('2026-04-01T00:00:00Z'), train_end=ts('2026-07-01T00:00:00Z'),
        calibration_start=ts('2026-07-01T00:00:00Z'), calibration_end=ts('2026-08-01T00:00:00Z'),
        evaluation_start=ts('2026-08-01T12:00:00Z'), evaluation_end=ts('2026-10-15T12:00:00Z'),
        pas_start=ts('2026-10-19T12:00:00Z'), pas_end=ts(pointer['completed_bar_utc']),
        horizons=[24], execution_delay_hours=1, label_sampling_hours=24,
        common_ridge=1., group_ridge=10., calibration='Single nonnegative attenuation in [0,1], fitted to July matured 24h labels; no PnL selection',
        target_clip_sigma=4., forecast_clip_sigma=1., daily_target_risk=.001,
        entry_cost=.0038, entry_buffer=.0005, decision_hours=6,
        funding_projection='Half of last 24 observed hourly mean, extended 24h. Receipts are not guaranteed.',
        basis_model='Past hourly log(PAS trade-close/reference-close), ridge AR(1) increment, min 48 consecutive pairs, max 96h. Slope shrunk to zero, stable coefficient only. No future oracle assumed.',
        basis_minimum_pairs=48, basis_ridge_prior_pairs=24, basis_lookback=96,
        team_capital=100000., user_capital=50000., reserve=50000.,
        reference_arms=['incumbent','size_only','directional','exposure_control'],
        pas_arms=['incumbent','size_only','directional','venue_aware','exposure_control'],
        pair_overlay='Exact incumbent pair models and execution kept in reference comparisons. PAS uses base-only conditional comparison; actual gate inactivity is audited separately and never assumed indefinitely.',
        exposure_control='July-only ratio of mean candidate target gross to incumbent target gross, then identical original risk projection. Actual later exposures may differ; report them.',
        nomination='venue_aware is nominated before outcomes; directional is its external-price component and ablation. No winner substitution.',
        success_rule={
            'functional':'No future-label/preprocessing leakage, prefix invariance, zero accounting errors or unintended risk breaches; real bidirectional and all-market support.',
            'reference':'Directional nominee must have positive net PnL and beat incumbent AND size-only in PnL and Q in August, September-Oct15 and continuous, base AND stress. Continuous PnL must also beat exposure control.',
            'uncertainty':'Paired 7-day moving-block bootstrap 95% lower bound of mean daily PnL advantage over incumbent >0 on later window. Report 3-day sensitivity too.',
            'pas':'Venue-aware must beat incumbent and size-only PnL and Q in both scenarios on same PAS interval, after costs. No selection or tuning using PAS outcomes.',
            'deployment':'Reference and PAS passes are necessary. Incomplete historical books/marks or shared-wallet ownership unresolved prevent claiming live readiness. No automatic live replacement.'},
        scenarios={'base':{'fee_bps':9.,'impact_bps':10.,'participation':.01,'receipt_multiplier':1.},
                   'stress':{'fee_bps':12.,'impact_bps':20.,'participation':.0025,'receipt_multiplier':.5}},
        historical_independence=False,
        provenance='All retained periods previously exposed. Chronological causal backtests are development evidence, not a fresh independent holdout or proof of no selection overfitting.',
        orders_sent=0, live_changed=False, deployment_hashes=verify_deployment()['files'])
    save(path,p)
    return p


if __name__=='__main__': print(json.dumps(freeze(),indent=2))
