"""One collateral ledger and causal IOC broker for all strategy assets.

Execution uses the first observed fresh snapshot AFTER latency, not the decision
book. This is a public-book scenario, not a claim about exchange fills.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import math

from .market_data import History


@dataclass(frozen=True)
class Config:
    initial_equity: float = 1_000_000.
    order_notional: float = 10_000.
    gross_limit: float = 100_000.
    taker_fee_bps: float = 9.
    latency_ms: int = 3000
    depth_fraction: float = .1
    slippage_bps: float = 2.
    entry_limit_bps: float = 15.
    exit_limit_bps: float = 30.
    stop_bps: float = 100.
    maximum_loss: float = 5000.
    decision_ms: int = 60000
    maximum_age_ms: int = 5000
    intent_ttl_ms: int = 75000


def floor_quantity(value, decimals):
    lot = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(value)).quantize(lot, rounding=ROUND_FLOOR))


def price_limit(price, decimals, buy):
    d = Decimal(str(price))
    places = max(0, min(6 - decimals, 4 - d.adjusted()))
    return float(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_FLOOR if buy else ROUND_CEILING))


@dataclass
class Position:
    qty: float = 0.
    average: float = 0.
    opened_ms: int = 0
    deadline_ms: int = 0
    cycle_pnl: float = 0.
    cycle_fees: float = 0.
    cycle_funding: float = 0.
    entry_notional: float = 0.


class Portfolio:
    def __init__(self, cfg, strategy, start_ms, end_ms, funding_rates=None, journal=None):
        self.cfg, self.strategy = cfg, strategy
        self.start, self.end = start_ms, end_ms
        self.cash = cfg.initial_equity
        self.positions, self.markets, self.histories = {}, {}, {}
        self.pending, self.last_decisions = {}, {}
        self.last_ms = None
        self.funding_rates = funding_rates or {}
        self.journal = journal
        self.fills, self.cycles, self.trace, self.funding_log, self.decisions = [], [], [], [], []
        self.reasons = {}
        self.halted = False
        self.unresolved = set()
        self.consumed = {}
        self.gap_exposures = []

    def record(self, kind, **fields):
        if self.journal:
            self.journal(dict(kind=kind, **fields))

    def reject(self, reason):
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def disconnect(self, time_ms):
        if self.pending or any(p.qty for p in self.positions.values()):
            self.unresolved.add('connection_gap_with_inventory_or_intent')
            self.gap_exposures.append(dict(time_ms=time_ms,reason='connection_transition'))
        self.record('disconnect',time_ms=time_ms)

    def equity(self, oracle=False):
        return self.cash + sum(p.qty * ((self.markets[c].oracle if oracle else self.markets[c].mark) - p.average)
                               for c, p in self.positions.items() if p.qty)

    def exposure(self):
        return sum(abs(p.qty) * self.markets[c].mark for c, p in self.positions.items())

    def reserve(self):
        return sum(abs(o['qty']) * o['limit'] for o in self.pending.values() if not o['reduce'])

    def quote(self, s, side, quantity, limit=math.inf, consume=False):
        levels = s.asks if side > 0 else s.bids
        # Keep quantities in integer lots: summing float fills can leave an
        # artificial one-lot remainder when a position is closed.
        multiplier = 10 ** s.decimals
        requested = int(Decimal(str(quantity)) * multiplier)
        lots, value = 0, 0.
        version = (s.coin, s.book_ms, side)
        for i, (price, size) in enumerate(levels):
            price *= 1 + side * self.cfg.slippage_bps / 10000
            if (side > 0 and price > limit) or (side < 0 and limit != math.inf and price < limit):
                break
            key = (*version, i)
            available = max(0, int(Decimal(str(size)) * Decimal(str(self.cfg.depth_fraction)) * multiplier) - self.consumed.get(key, 0))
            take_lots = min(requested-lots,available)
            take = take_lots/multiplier
            lots += take_lots
            value += take * price
            if consume:
                self.consumed[key] = self.consumed.get(key, 0) + take_lots
            if lots >= requested:
                break
        qty = lots/multiplier
        return qty, value / qty if qty else 0.

    def funding(self, now):
        if self.last_ms is None:
            self.last_ms = now
            return
        for boundary in range(self.last_ms // 3600000 * 3600000 + 3600000, now // 3600000 * 3600000 + 1, 3600000):
            for coin, p in self.positions.items():
                if not p.qty:
                    continue
                s = self.markets[coin]
                rate = self.funding_rates.get((coin, boundary))
                estimated = rate is None
                if estimated:
                    # Unknown funding is never free or a predicted credit.
                    rate = math.copysign(max(abs(s.funding), .0002), p.qty)
                amount = -p.qty * s.oracle * rate
                self.cash += amount
                p.cycle_funding += amount
                p.cycle_pnl += amount
                row = dict(time_ms=boundary, coin=coin, amount=amount, rate=rate,
                           estimated_rate=estimated, oracle_age_ms=boundary - s.context_ms)
                self.funding_log.append(row)
                self.record('funding', **row)
        self.last_ms = now

    def submit(self, s, side, qty, reduce, reason, deadline=None, predicted_net=None):
        best = s.asks[0][0] if side > 0 else s.bids[0][0]
        buffer = self.cfg.exit_limit_bps if reduce else self.cfg.entry_limit_bps
        limit = price_limit(best * (1 + side * buffer / 10000), s.decimals, side > 0)
        qty = floor_quantity(qty, s.decimals)
        if qty * limit < 10:
            self.reject('below_minimum')
            if reduce:
                self.unresolved.add('position_below_minimum:' + s.coin)
            return
        order = dict(coin=s.coin, side=side, qty=qty, limit=limit, reduce=reduce,
                     submitted_ms=s.time_ms, due_ms=s.time_ms + self.cfg.latency_ms,
                     deadline_ms=deadline, reason=reason, predicted_net_bps=predicted_net)
        self.pending[s.coin] = order
        self.record('intent', **order)

    def execute(self, s):
        o = self.pending.get(s.coin)
        if not o or s.time_ms < o['due_ms']:
            return
        del self.pending[s.coin]
        if s.time_ms - o['submitted_ms'] > self.cfg.intent_ttl_ms:
            self.reject('ioc_observation_timeout')
            return
        p = self.positions.setdefault(s.coin, Position())
        qty = min(o['qty'], abs(p.qty)) if o['reduce'] else o['qty']
        if o['reduce'] and p.qty * o['side'] >= 0:
            raise ValueError('Reduce-only order would increase exposure')
        if not o['reduce'] and (self.halted or p.qty or self.exposure() + qty * o['limit'] > self.cfg.gross_limit + 1e-6):
            self.reject('portfolio_risk_at_fill')
            return
        done, price = self.quote(s, o['side'], qty, o['limit'], consume=True)
        if not done:
            self.reject('ioc_no_fill')
            return
        fee = done * price * self.cfg.taker_fee_bps / 10000
        self.cash -= fee
        p.cycle_pnl -= fee
        p.cycle_fees += fee
        realized = 0.
        if o['reduce']:
            realized = -o['side'] * done * (price - p.average)
            self.cash += realized
            p.cycle_pnl += realized
            p.qty = round(p.qty + o['side'] * done, s.decimals)
        else:
            p.qty, p.average = o['side'] * done, price
            p.opened_ms, p.deadline_ms = s.time_ms, o['deadline_ms']
            p.entry_notional = done * price
        row = dict(time_ms=s.time_ms, submitted_ms=o['submitted_ms'], coin=s.coin, side=o['side'],
                   qty=done, price=price, fee=fee, realized_pnl=realized, reduce_only=o['reduce'],
                   reason=o['reason'], book_ms=s.book_ms, position_after=p.qty)
        self.fills.append(row)
        self.record('fill', **row)
        if o['reduce'] and not p.qty:
            cycle = dict(coin=s.coin, start_ms=p.opened_ms, end_ms=s.time_ms, net_pnl=p.cycle_pnl,
                         fees=p.cycle_fees, funding=p.cycle_funding, entry_notional=p.entry_notional)
            self.cycles.append(cycle)
            self.record('cycle', **cycle)
            self.positions[s.coin] = Position()

    def step(self, s):
        if self.last_ms is not None and s.time_ms < self.last_ms:
            raise ValueError('Out-of-order snapshot')
        self.funding(s.time_ms)
        previous = self.markets.get(s.coin)
        if previous and self.positions.get(s.coin, Position()).qty and s.time_ms-previous.time_ms > self.cfg.intent_ttl_ms:
            self.gap_exposures.append(dict(coin=s.coin,start_ms=previous.time_ms,end_ms=s.time_ms))
        self.markets[s.coin] = s
        if not s.fresh(s.time_ms, self.cfg.maximum_age_ms):
            self.reject('stale_snapshot')
            return
        if self.equity() - self.cfg.initial_equity <= -self.cfg.maximum_loss:
            self.halted = True
        self.execute(s)
        if self.equity() - self.cfg.initial_equity <= -self.cfg.maximum_loss:
            self.halted = True
        history = self.histories.setdefault(s.coin, History())
        history.add(s)
        if hasattr(self.strategy, 'observe'):
            self.strategy.observe(s)
        p = self.positions.setdefault(s.coin, Position())
        if s.coin not in self.pending:
            if p.qty:
                side = -1 if p.qty > 0 else 1
                done, price = self.quote(s, side, abs(p.qty))
                executable_pnl = p.cycle_pnl + p.qty * (price - p.average) - abs(p.qty) * price * self.cfg.taker_fee_bps / 10000 if done >= abs(p.qty) - 1e-10 else None
                stop = executable_pnl is not None and executable_pnl / p.entry_notional * 10000 <= -self.cfg.stop_bps
                x = history.features(s) if self.strategy.persistent else None
                reversal = x is not None and p.qty*x[6]<0
                if self.halted or stop or reversal or s.time_ms >= p.deadline_ms:
                    self.submit(s, side, abs(p.qty), True, 'portfolio_stop' if self.halted else 'stop' if stop else 'trend_reversal' if reversal else 'horizon')
            elif not self.halted and s.time_ms >= self.start and s.time_ms < self.end - (300000 if self.strategy.persistent else self.strategy.horizon_minutes * 60000 + self.cfg.intent_ttl_ms * 2):
                if s.time_ms - self.last_decisions.get(s.coin, 0) >= self.cfg.decision_ms:
                    self.last_decisions[s.coin] = s.time_ms
                    x = history.features(s,self.strategy.current_only)
                    if x is None:
                        self.reject('history_not_ready')
                    else:
                        prediction = self.strategy.forecast(s.coin, x)
                        half_spread = (s.asks[0][0] - s.bids[0][0]) / 2
                        deadline = (min(s.time_ms+240*60000,self.end-self.cfg.intent_ttl_ms*2)
                                    if self.strategy.persistent else s.time_ms + self.strategy.horizon_minutes * 60000)
                        candidates = []
                        for side in (1, -1):
                            qty = floor_quantity(self.cfg.order_notional / (s.mid * (1 + self.cfg.entry_limit_bps / 10000)), s.decimals)
                            n, entry = self.quote(s, side, qty)
                            n2, exit_now = self.quote(s, -side, qty)
                            if min(n, n2) < qty - 1e-10 or qty * entry < 10:
                                continue
                            touch = s.bids[0][0] if side > 0 else s.asks[0][0]
                            impact = max(0., side * (touch - exit_now))
                            if hasattr(self.strategy, 'exit_target'):
                                predicted_exit = self.strategy.exit_target(s.coin, side) - side*impact
                            else:
                                predicted_exit = s.mid * (1 + prediction / 10000) - side * (half_spread + impact)
                            hours = deadline // 3600000 - s.time_ms // 3600000
                            reserve = hours * max(2., 2 * max(0., side * s.funding * 10000))
                            net = (side * (predicted_exit - entry) - self.cfg.taker_fee_bps / 10000 * (entry + predicted_exit)) / entry * 10000 - reserve
                            if net > 0:
                                candidates.append((net, side, qty))
                        d = dict(time_ms=s.time_ms, coin=s.coin, forecast_bps=prediction,
                                 best_net_bps=max((c[0] for c in candidates), default=None))
                        self.decisions.append(d)
                        self.record('decision', **d)
                        if candidates and self.strategy.name != 'cash':
                            net, side, qty = max(candidates)
                            if self.exposure() + self.reserve() + qty * s.mid <= self.cfg.gross_limit:
                                self.submit(s, side, qty, False, 'forecast', deadline, net)
                            else:
                                self.reject('gross_exposure_limit')
                        else:
                            self.reject('no_positive_net_forecast')
        if s.time_ms >= self.start and (not self.trace or s.time_ms - self.trace[-1]['time_ms'] >= 60000):
            self.trace.append(dict(time_ms=s.time_ms, equity=self.equity(), oracle_equity=self.equity(True), gross=self.exposure()))
        history.prune(s.time_ms)
        if len(self.consumed) > 10000:
            self.consumed = {k: v for k, v in self.consumed.items() if k[1] >= s.time_ms - 10000}

    def result(self):
        equity = self.equity(True)
        if self.last_ms is not None and (not self.trace or self.trace[-1]['time_ms'] != self.last_ms):
            self.trace.append(dict(time_ms=self.last_ms,equity=self.equity(),oracle_equity=equity,gross=self.exposure()))
        peak, drawdown = self.cfg.initial_equity, 0.
        daily = {}
        for row in self.trace:
            peak = max(peak, row['oracle_equity'])
            drawdown = min(drawdown, row['oracle_equity'] / peak - 1)
            # Only adjacent UTC-midnight anchors, within 90s, represent full
            # daily returns. Session fragments are never labelled daily risk.
            if row['time_ms'] % 86400000 <= 90000:
                daily.setdefault(row['time_ms'] // 86400000,row['oracle_equity'])
        returns = [daily[d]/daily[d-1]-1 for d in sorted(daily) if d-1 in daily]
        if self.pending:
            self.unresolved.add('pending_orders_at_end')
        if any(p.qty for p in self.positions.values()):
            self.unresolved.add('open_positions_at_end')
        fees = sum(f['fee'] for f in self.fills)
        funding = sum(f['amount'] for f in self.funding_log)
        realized = sum(f['realized_pnl'] for f in self.fills)
        accounting_error = self.cash - (self.cfg.initial_equity + realized - fees + funding)
        return dict(strategy=self.strategy.name, horizon_minutes=self.strategy.horizon_minutes,
                    net_pnl=equity-self.cfg.initial_equity, return_pct=(equity/self.cfg.initial_equity-1)*100,
                    realized_pnl=realized, fees=fees, funding=funding, cash=self.cash,
                    fills=len(self.fills), closed_cycles=len(self.cycles), winning_cycles=sum(c['net_pnl']>0 for c in self.cycles),
                    maximum_drawdown_pct=drawdown*100, daily_returns=returns,
                    daily_volatility=None if len(returns)<2 else float(__import__('statistics').stdev(returns)),
                    estimated_funding=sum(f['estimated_rate'] for f in self.funding_log),
                    funding_oracle_age_max_ms=max((f['oracle_age_ms'] for f in self.funding_log),default=0),
                    accounting_error=accounting_error, unresolved=sorted(self.unresolved),
                    gap_exposure_count=len(self.gap_exposures),
                    positions={c:asdict(p) for c,p in self.positions.items() if p.qty},
                    rejections=self.reasons, actual_orders_sent=0)
