"""One paper ledger, one executor, scoped strategy intents, global reservations.

No network or signing interface. A scope can clear its own tasks, never another
scope's tasks. Position changes occur exclusively inside the arrival broker.
"""
from itertools import product
import numpy as np
from experiments.paired_venue_20261017.arrival import broker
from experiments.paired_venue_20261017.engine import PairExecution, quote_pair
from experiments.paired_venue_20261017.forward import current_snapshots
from experiments.cost_horizon_20261016.research import SecondController
from replay_lab.ema_rebuild.engine import Controller
from replay_lab.common_comparison.engine import risk_breaches


def interval_risk(current, targets, pending, coins, prices, equity, sigma, risk):
    """Check every partial-fill corner, not merely the netted final target.

    Gross, absolute net, asset exposure and covariance norm are convex. Their
    maximum over the independent-fill box is attained at a vertex. This is a
    conservative reservation, not an assumption that orders fill atomically.
    """
    if (equity <= 0 or not np.isfinite(equity) or not np.isfinite(prices).all()
            or np.any(prices <= 0) or not np.isfinite(sigma).all()):
        return ['invalid_risk_inputs']
    low = np.array([current.get(c, 0.) for c in coins]); high = low.copy()
    for i, coin in enumerate(coins):
        values = [low[i], targets.get(coin, low[i])]
        values += [low[i] + (o['size'] if o['buy'] else -o['size'])
                   for o in pending if o['coin'] == coin]
        low[i], high[i] = min(values), max(values)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        return ['invalid_target']
    changing = np.flatnonzero(high - low > 1e-12)
    corners = np.tile(low, (2 ** len(changing), 1))
    if len(changing):
        corners[:, changing] += np.array(list(product((0, 1), repeat=len(changing)))) * (high - low)[changing]
    notionals = corners * prices
    # Reserve adverse execution friction before sizing to available equity.
    turnover = float(np.maximum(abs(high - np.array([current.get(c, 0.) for c in coins])),
                                abs(low - np.array([current.get(c, 0.) for c in coins]))) @ prices)
    available = equity - .005 * turnover
    if available <= 0: return ['cost_reserve']
    w = notionals / available
    values = dict(gross=abs(w).sum(axis=1).max(), net=abs(w.sum(axis=1)).max(),
                  asset=abs(w).max(), asset_notional=abs(notionals).max(),
                  daily_volatility=np.sqrt(np.maximum(0., 24 * np.einsum('bi,ij,bj->b', w, sigma, w))).max())
    limits = dict(gross=risk['gross_fraction'], net=risk['net_fraction'], asset=risk['asset_fraction'],
                  asset_notional=risk['asset_notional'], daily_volatility=risk['daily_volatility'])
    return [k for k in limits if values[k] > limits[k] + 1e-10]


class Hub:
    def __init__(self, paper, coins, risk):
        self.paper, self.coins, self.risk = paper, list(coins), risk
        self.inputs = None; self.events = []; self.last_tick = -1

    def update_risk(self, prices, equity, sigma):
        self.inputs = prices, equity, sigma

    def allowed(self, target, now):
        if self.inputs is None: return False
        merged = {c: t['target'] for c, t in self.paper.tasks.items() if t['deadline'] >= now}
        merged.update(target)
        return not interval_risk(self.paper.qty, merged, self.paper.pending, self.coins,
                                 *self.inputs, self.risk)

    def request(self, target, now, deadline, identity, reduction=False):
        if not set(target).issubset(self.coins): raise ValueError('Unknown target market')
        if any(o['coin'] in target for o in self.paper.pending): return False
        if reduction:
            if any(not np.isfinite(q) or abs(q) > abs(self.paper.qty.get(c, 0.)) + 1e-10 or q * self.paper.qty.get(c, 0.) < 0
                   for c, q in target.items()): raise ValueError('Reduction would increase or reverse inventory')
        elif not self.allowed(target, now):
            self.events.append(dict(time=now, event='risk_reservation_rejected', identity=identity))
            return False
        for c, q in target.items():
            self.paper.tasks[c] = dict(target=q, round=identity, created=now, deadline=deadline,
                anchor={}, attempts=dict(reduce=0, increase=0), next=dict(reduce=now, increase=now), terminal=reduction)
        return True

    def clear(self, coins):
        for c in coins: self.paper.tasks.pop(c, None)

    def tick(self, now, books, contexts):
        # Same one-second submission cadence for every comparison arm. Pending
        # arrival fills are still applied on each received event by the caller.
        if now - self.last_tick < 1000: return
        self.last_tick = now
        # Check reservations again at submission time after prices/fills move.
        active = {c: t['target'] for c, t in self.paper.tasks.items() if t['deadline'] >= now}
        if active and not self.allowed({}, now):
            for c, task in self.paper.tasks.items():
                q = self.paper.qty.get(c, 0.); desired = task['target']
                # Cancel future increases, retain strictly reducing tasks.
                task['target'] = 0. if desired * q < 0 else np.sign(q) * min(abs(q), abs(desired))
        self.paper.tick(now, books, contexts)


class PairScope:
    """PairExecution adapter; only Hub.tick may submit or execute orders."""
    def __init__(self, hub, coins): self.hub, self.coins = hub, tuple(coins)
    @property
    def qty(self): return {c: self.hub.paper.qty.get(c, 0.) for c in self.coins}
    @property
    def pending(self): return [o for o in self.hub.paper.pending if o['coin'] in self.coins]
    @property
    def attempts(self): return [o for o in self.hub.paper.attempts if o['coin'] in self.coins]
    @property
    def fatal(self): return self.hub.paper.fatal
    @property
    def tasks(self): return {c: t for c, t in self.hub.paper.tasks.items() if c in self.coins}
    @tasks.setter
    def tasks(self, value):
        if value: raise ValueError('Scoped replacement supports cancellation only')
        self.hub.clear(self.coins)
    def request_target(self, target, now, deadline, round_id):
        if not set(target).issubset(self.coins): raise ValueError('Pair crossed its scope')
        reducing = all(abs(q) <= abs(self.qty[c]) + 1e-10 and q * self.qty[c] >= 0 for c, q in target.items())
        if not self.hub.request(target, now, deadline, round_id, reduction=reducing):
            raise ValueError('Pair intent rejected after its preflight')
    def apply_pending(self, now, books): pass
    def tick(self, now, books, contexts): pass


class Account:
    def __init__(self, name, scenario, plan):
        self.name, self.plan = name, plan
        self.cfg, self.p = plan['paired_protocol'], plan['trading_protocol']
        self.coins, self.pair_coins = plan['coins'], plan['overlay_coins']
        self.paper = broker(plan['metadata'], self.cfg['scenarios'][scenario])
        self.scenario = scenario; self.hub = Hub(self.paper, self.coins, self.p['risk'])
        self.pair = PairExecution(PairScope(self.hub, self.pair_coins), self.cfg) if name == 'second_plus_carry' else None
        self.last_decision = -1; self.decisions = set(); self.last_risk = -1; self.last_pair_bar = -1
        self.peak = 1e6; self.halted = False; self.last_exit = -1; self.events = []

    def baseline_target(self, now, prices, sigma, signal, equity, boundary):
        qty = np.array([self.paper.qty.get(c, 0.) for c in self.coins])
        if self.pair and self.pair.state:
            qty[[self.coins.index(c) for c in self.pair_coins]] = 0.
        when = boundary // 1000
        if self.name == 'first_strategy':
            control = Controller(last_decision=self.last_decision); raw = signal['incumbent']
        else:
            control = SecondController([when], [signal['frame']['retention']])
            control.controller.last_decision = self.last_decision
            raw = np.array(signal['candidate'])
            if self.name == 'second_scaled_control': raw = raw * self.plan['scale_control']
        a = control.propose(when, prices, qty, equity, np.array(raw), sigma,
            np.array(signal['frame']['volume']), np.array(signal['frame']['trades']), self.p, buffer=.2)
        return a

    def step(self, now, books, contexts, signal):
        start, end = [int(self.plan[k] * 1000) for k in ('paper_start_unix', 'capture_end_unix')]
        if now < start: return
        self.paper.apply_pending(now, books)
        fresh = (signal is not None and 0 <= now // 3600000 * 3600 - signal['frame']['source_time'] <= 21600
                 and all(c in contexts and 0 <= now - contexts[c]['received_ms'] <= 15000 for c in self.coins))
        terminal = now >= end - self.plan['end_liquidation_seconds'] * 1000
        if now - self.last_risk >= 30000:
            self.last_risk = now
            if fresh:
                prices = np.array([float(contexts[c]['ctx']['markPx']) for c in self.coins])
                sigma = np.array(signal['frame']['sigma']); equity = self.paper.equity(contexts)
                self.peak = max(self.peak, equity)
                self.halted |= equity / self.peak - 1 <= -self.p['risk']['drawdown_halt']
                self.hub.update_risk(prices, equity, sigma)
                actual = np.array([self.paper.qty.get(c, 0.) for c in self.coins])
                forced = self.halted or bool(risk_breaches(actual, equity, prices, sigma, self.p['risk']))
            else: forced = True; self.hub.inputs = None
            if terminal or forced:
                # In-flight orders are not erased. Resolve them, then flatten.
                self.hub.clear(self.coins)
                if self.pair: self.pair.close(now, 'terminal_or_account_risk')
                excluded = self.pair_coins if self.pair and self.pair.state else []
                zeros = {c: 0. for c in self.coins if c not in excluded}
                self.hub.request(zeros, now, min(end - 1000, now + 300000), f'account_exit:{now}', reduction=True)
            else:
                boundary = now // 21600000 * 21600000
                normal = now - boundary <= 900000 and signal['frame']['source_time'] == now // 3600000 * 3600 and boundary not in self.decisions
                if normal:
                    action = self.baseline_target(now, prices, sigma, signal, equity, boundary)
                    if action:
                        targets = dict(zip(self.coins, action['target_qty']))
                        conflict = self.pair and self.pair.state and any(abs(targets[c]) > 1e-10 for c in self.pair_coins)
                        if conflict:
                            self.pair.close(now, 'baseline_has_priority')
                        else:
                            if self.pair and self.pair.state:
                                targets = {c: q for c, q in targets.items() if c not in self.pair_coins}
                            if self.hub.request(targets, now, now + 900000, f'{self.name}:{boundary}'):
                                self.decisions.add(boundary); self.last_decision = boundary // 1000
                                self.events.append(dict(time=now, event='baseline_target', target=targets))
                            elif self.pair and self.pair.state and not any(o['coin'] in targets for o in self.paper.pending):
                                self.pair.close(now, 'release_budget_for_baseline')
                # No pair competes with a fresh baseline decision or an existing
                # baseline position / pending target in these two markets.
                bar = now // 300000
                if self.pair and self.pair.state is None and bar != self.last_pair_bar and now + 72 * 3600000 <= end - self.plan['end_liquidation_seconds'] * 1000:
                    self.last_pair_bar = bar
                    occupied = any(abs(self.paper.qty.get(c, 0.)) > 1e-10 or abs(self.paper.tasks.get(c, {}).get('target', 0.)) > 1e-10 for c in self.pair_coins)
                    occupied |= any(o['coin'] in self.pair_coins for o in self.paper.pending)
                    snaps = current_snapshots(now, books, contexts, self.cfg)
                    pairs = [v for v in signal['frame']['paired_metrics'] if v['coins'] == self.pair_coins]
                    if not occupied and pairs and pairs[0]['correlation'] >= self.cfg['minimum_correlation'] and all(c in snaps for c in self.pair_coins):
                        proposal = quote_pair(self.pair_coins, snaps, self.plan['metadata'], pairs[0]['beta'], -1, 72,
                                              self.cfg, self.cfg['scenarios'][self.scenario])
                        if proposal and proposal['predicted_net_bps'] > self.cfg['entry_buffer_bps'] and self.hub.allowed(dict(zip(self.pair_coins, proposal['target_qty'])), now):
                            self.pair.enter(proposal, now)
        # Refresh equity and price risk before actual submission, not just at
        # the 30-second strategy/risk decision checkpoint.
        if fresh:
            self.hub.update_risk(np.array([float(contexts[c]['ctx']['markPx']) for c in self.coins]),
                                 self.paper.equity(contexts), np.array(signal['frame']['sigma']))
        else: self.hub.inputs = None
        if self.pair:
            state = self.pair.state
            if state and state['phase'] in ('entering', 'balancing') and now >= state['deadline']:
                self.pair.close(now, 'entry_deadline')
            self.pair.tick(now, books, contexts)
        self.hub.tick(now, books, contexts)
