"""Read immutable public history, reproduce incumbents, never open a signer."""
import json
import hashlib
from pathlib import Path
import numpy as np
from replay_lab.common_comparison.models import Market
from strategy.agent import TAIL
from experiments.performance_attribution_20261016.agent import BudgetMatchedAgent

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'artifacts'


def ts(x): return int(np.datetime64(x, 's').astype('int64'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()) + '\n')


def prepare():
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((HERE / 'protocol.json').read_text())
    run = ROOT / cfg['reference_run']
    verification = json.loads((run / 'verification.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    assert sha(run / 'inputs.npz') == manifest['inputs_sha256']
    assert sha(run / 'targets.npz') == verification['targets_sha256']
    with np.load(run / 'inputs.npz') as z: inp = {k: z[k].copy() for k in z.files}
    with np.load(run / 'targets.npz') as z: old = {k: z[k].copy() for k in z.files}
    p = json.loads((TAIL / 'baseline.json').read_text())['protocol']
    m = Market(*(inp[k] for k in ('market_times', 'close', 'volume', 'trades', 'funding', 'observed')), tuple(p['symbols']))
    assert np.array_equal(m.times, old['times'])
    agent = BudgetMatchedAgent()
    for i in np.flatnonzero((np.arange(len(m.times)) >= 1008) & (m.times % 21600 == 0)):
        context = np.log(m.close[i-167:i+1].T).astype('float32')
        agent.incumbent.agent.signal.cached[(int(m.times[i]), context.tobytes())] = old['chronos_forecast'][i].copy()
    print('Reproduce both incumbent target paths using verified cached forecasts', flush=True)
    agent.prepare_history(m, {k: v for k, v in inp.items() if k not in ('market_times', 'funding', 'observed')},
                          funding_source='mainnet_reference_settled')
    np.testing.assert_array_equal(agent.incumbent.weights, old['default'])
    with np.load(ROOT / 'experiments/risk_preserving_20261016/artifacts/targets.npz') as z:
        n = len(z['times'])
        np.testing.assert_array_equal(agent.incumbent.weights[:n], z['incumbent'])
        np.testing.assert_array_equal(agent.weights[:n], z['candidate'] * agent.multiplier)
        np.testing.assert_array_equal(agent.sigma[:n], z['sigma'])
    times = np.arange(ts('2026-10-01T00:00:00'), ts(cfg['end_utc'])+3600, 3600)
    ix = np.searchsorted(m.times, times)
    assert np.array_equal(m.times[ix], times)
    shape = (len(times), len(m.symbols))
    close, volume, trades, funding = [np.full(shape, np.nan) for _ in range(4)]
    # Preserve full funding history before the evaluation start for past-only forecasts.
    history_start = ts('2026-09-29T00:00:00')
    ft = np.arange(history_start, times[-1]+3600, 3600)
    rates = np.full((len(ft), len(m.symbols)), np.nan)
    sources = {str(run / f): sha(run / f) for f in ('inputs.npz', 'targets.npz')}
    coverage = []
    for j, s in enumerate(m.symbols):
        cp = ROOT / cfg['par_history'] / f'candles_{s}_1h.json'
        fp = cp.with_name(f'funding_{s}.json')
        sources[str(cp)] = sha(cp); sources[str(fp)] = sha(fp)
        cr = json.loads(cp.read_text())['response']
        fr = json.loads(fp.read_text())
        assert fr['pagination_complete']
        candles = {int(r['t']) // 1000 + 3600: r for r in cr}
        for i, t in enumerate(times):
            r = candles.get(int(t))
            if r is None: raise ValueError(f'Missing candle {s} {t}')
            assert int(r['T']) + 1 == int(t)*1000
            close[i, j], volume[i, j], trades[i, j] = float(r['c']), float(r['v']), float(r['n'])
        for r in fr['rows']:
            raw = int(r['time']); t = raw // 3600000 * 3600
            assert raw - t*1000 < 1000
            k = (t-history_start)//3600
            if 0 <= k < len(ft):
                v = float(r['fundingRate'])
                if np.isfinite(rates[k,j]): assert rates[k,j] == v
                rates[k,j] = v
        funding[:,j] = rates[np.searchsorted(ft,times),j]
        coverage.append(dict(symbol=s, hours=len(times), active_hours=int((volume[:,j]>0).sum()),
                             zero_volume_hours=int((volume[:,j]==0).sum()), funding_hours=int(np.isfinite(funding[:,j]).sum())))
    assert np.isfinite(close).all() and np.all(close>0) and np.isfinite(funding).all()
    np.savez_compressed(OUT / 'inputs.npz', times=times, close=close, volume=volume, trades=trades,
        funding=funding, funding_times=ft, funding_history=rates, oracle_proxy=m.close[ix],
        first=agent.incumbent.weights[ix], second=agent.weights[ix], retention=agent.retention[ix],
        sigma=agent.sigma[ix], ref_times=m.times, ref_close=m.close)
    save(OUT / 'data_manifest.json', dict(sources=sources, coverage=coverage,
        baseline_reproduction_exact=True, source_data_through_utc='2026-10-16T12:00:00Z',
        actual_par_candles=True, actual_par_funding_rates=True, historical_oracle_and_mark=False,
        funding_jitter_policy='Forecast at hour t uses settlements only through t-1; payment is charged before same-hour simulated fills.',
        orders_sent=0))
    print('Prepared actual par candles and funding; both target paths reproduced exactly', flush=True)


if __name__ == '__main__': prepare()
