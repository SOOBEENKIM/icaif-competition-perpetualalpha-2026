"""Preserve the baseline and reconcile complete position episodes before changes."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/channel_revision'
BASE = ROOT / 'output/channel_crypto/5d232970183dbd2e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)


def freeze():
    path = OUT / 'baseline.json'
    files = list(BASE.rglob('*')) + list((ROOT/'channel_crypto').glob('*.py'))
    files += [ROOT/'channel_crypto/protocol.json', ROOT/'pullback/broker.py', ROOT/'model.py']
    hashes = {str(f.relative_to(ROOT)): digest(f) for f in files if f.is_file()}
    if path.exists():
        old = json.loads(path.read_text())
        for name, value in old['sha256'].items():
            if digest(ROOT/name) != value:
                raise ValueError('Frozen baseline changed: ' + name)
        return old
    value = dict(identity=BASE.name, frozen_at=datetime.now(timezone.utc).isoformat(), sha256=hashes,
                 rule='Never overwrite these source files or reports. Historical periods are already inspected.')
    write(path, value)
    return value


def analyze(run):
    """Episodes run from flat to flat. Avoid double-counting partially closed lots."""
    episodes = []
    rebalances = []
    for j, symbol in enumerate(('BTC', 'ETH', 'HYPE')):
        current = None
        decisions = {d['time']: d for d in run['decisions'] if d['asset'] == symbol}
        for fill in [f for f in run['fills'] if f['symbol'] == symbol]:
            if fill['position_before'] == 0:
                current = dict(symbol=symbol, start=fill['time'], entry_price=fill['price'],
                               realized=0., fees=0., fills=0, entry_notional=0.)
            current['realized'] += fill['realized']
            current['fees'] += fill['fee']
            current['fills'] += 1
            if not fill['reduce_only']:
                current['entry_notional'] += fill['price'] * fill['qty']
            eligible = [t for t in decisions if t <= fill['time']]
            decision = decisions[max(eligible)] if eligible else None
            if decision and not decision['signal_changed']:
                rebalances.append(fill)
            if fill['position_after'] == 0:
                current['end'] = fill['time']
                current['hours'] = (current['end']-current['start'])/3600
                current['funding'] = sum(-e['signed_notional'][j]*e['rates'][j]
                                        for e in run['funding_events']
                                        if current['start'] < e['time'] <= current['end'])
                current['net'] = current['realized']-current['fees']+current['funding']
                # Highest sampled account PNL for this episode, with actual changing sizes.
                events = [f for f in run['fills'] if f['symbol'] == symbol and current['start'] <= f['time'] <= current['end']]
                pnl_path = []
                for snap in run['snapshots']:
                    if not current['start'] < snap['time'] <= current['end']:
                        continue
                    prior = [f for f in events if f['time'] < snap['time']]
                    # Full per-symbol mark PNL is added below using source candles.
                    pnl_path.append((snap, sum(f['realized']-f['fee'] for f in prior)))
                current['_path'] = pnl_path
                episodes.append(current)
                current = None
        if current is not None:
            raise ValueError('Diagnosis requires fully closed baseline positions')
    return episodes, rebalances


def main():
    import numpy as np
    from ..model import load_history
    freeze()
    data, _ = load_history(ROOT.parent/'history')
    prices = {s: {c.time+3600: c.close for c in data[s]} for s in ('BTC','ETH','HYPE')}
    reports = []
    for window in ('forward-1','forward-2','primary','continuous'):
        path = BASE/'runs'/f'{window}-control-long_only-buffer0p2-base.json'
        run = json.loads(path.read_text())
        episodes, rebalances = analyze(run)
        for e in episodes:
            j = ('BTC','ETH','HYPE').index(e['symbol'])
            path_pnl = [realized + snap['positions'][j]*(prices[e['symbol']][snap['time']]-snap['entry'][j])
                        for snap, realized in e.pop('_path')]
            e['best_sampled_pnl_before_funding'] = max([0., *path_pnl])
            e['giveback_before_funding'] = e['best_sampled_pnl_before_funding']-(e['realized']-e['fees'])
            e['losing'] = e['net'] < 0
            e['fast_loss_72h'] = e['losing'] and e['hours'] <= 72
        net = sum(e['net'] for e in episodes)
        assert abs(net-(run['metrics']['equity']-run['config']['initial'])) < 1e-6
        summary = dict(window=window, episodes=len(episodes), losses=sum(e['losing'] for e in episodes),
                       fast_losses=sum(e['fast_loss_72h'] for e in episodes),
                       gross_realized=run['metrics']['realized'], fees=run['metrics']['fees'],
                       funding=run['metrics']['funding'], net=net,
                       losing_episode_net=sum(e['net'] for e in episodes if e['losing']),
                       volatility_rebalance_fills=len(rebalances),
                       volatility_rebalance_fees=sum(f['fee'] for f in rebalances),
                       winning_then_losing=sum(e['losing'] and e['best_sampled_pnl_before_funding']>0 for e in episodes))
        reports.append(dict(summary=summary, episodes=episodes))
        print(json.dumps(summary), flush=True)
    write(OUT/'diagnosis.json', dict(baseline=BASE.name, reports=reports,
          interpretation='Descriptive attribution, not causal proof. PNL peaks are sampled hourly and cannot all be captured by an executable exit.',
          live_orders_sent=0))


if __name__ == '__main__':
    main()
