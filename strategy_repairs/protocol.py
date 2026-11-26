import hashlib
import json
import time
from datetime import datetime, timezone
from competition_live.profile import ROOT, load_profile, verify_deployment, verify_bundle

OUT = ROOT / 'reports/isolated_repairs_20261020'
ARMS = {
    'incumbent': dict(scope='legacy', short=False, cross_exit=False, risk=False, original_size=False),
    'risk_only': dict(scope='legacy', short=False, cross_exit=False, risk=True, original_size=False),
    'short_only': dict(scope='legacy', short=True, cross_exit=False, risk=False, original_size=False),
    'short_and_exit': dict(scope='legacy', short=True, cross_exit=True, risk=False, original_size=False),
    'risk_short_exit': dict(scope='legacy', short=True, cross_exit=True, risk=True, original_size=False),
    'risk_short_exit_all': dict(scope='all', short=True, cross_exit=True, risk=True, original_size=False),
    'size_only': dict(scope='legacy', short=False, cross_exit=False, risk=False, original_size=True),
}


def freeze():
    OUT.mkdir(exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    pointer = json.loads((ROOT/'practice_live/latest.json').read_text())
    p = load_profile()
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), arms=ARMS,
        pointer=pointer, pas_start_ms=1792411200000, pas_cutoff_ms=int(time.time()//60)*60000,
        team_initial=100000., user_initial=50000., reserve=50000., risk=p['trading']['risk'],
        historical_independence=False, new_model_training=False,
        inherited_models='Retain frozen Chronos/regime forecasts and causally rolling pair/gate states. '
                         'Do not fit coefficients to PAS outcomes or search thresholds.',
        risk_contract='Preserve the existing combined regime/Chronos attenuation on the trend component: '
                      'size the ungated trend first, apply gate last, then add the unchanged reversion component. '
                      'Protect a resulting net exposure reduction against the 20% controller buffer and turnover limit. '
                      'Do not rescale surviving assets after a risk cut. Pair owner and aggregate risk caps unchanged.',
        short_contract='Mirror existing EMA/channel conditions. The short_only arm preserves old long exits; '
                       'short_and_exit additionally activates the separately declared EMA crossover exit. '
                       'Universe expansion is evaluated only in the separate all arm.',
        sizing_contract='Trace rather than assume a bug. size_only restores the already-existing original scale, '
                        'without forcing capital use. It is diagnostic and ineligible for strategy selection.',
        execution_contract='Same signal, quantities, sealed prices and account risk; fresh quote per needed asset '
                           'after confirmation delays. Unknown outcomes halt. No live orders.',
        selection='Evaluate all arms in August first; freeze nomination before September results. '
                  'Eligible strategy must beat incumbent PnL and team Q in BOTH base and stress, '
                  'with zero account/timing violations. Exclude size_only. Rank worst PnL advantage, '
                  'then name; if none passes, nominee is null. Show every later result regardless.',
        later='September 1–October 15 and continuous August–October 15; both scenarios. '
              'The history has already been viewed, so no independent-holdout claim.',
        pas='Separate Oct 19 start through fixed cutoff. Reference input signals only, actual past publication '
            'times; PAS future candles are evaluation/fill data only. Missing historical books/marks limit '
            'execution validity. No invented Q when only one completed daily return exists.',
        forward='Freeze source hash after functional tests. New public quote capture after that freeze '
                'validates execution causally, not long-term strategy profitability.',
        deployment_hashes=verify_deployment()['files'], bundle_hash=verify_bundle(),
        source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in (ROOT/pointer['inputs'],ROOT/pointer['run']/'targets.npz')},
        orders_sent=0)
    path.write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    print(json.dumps(freeze(),indent=2))
