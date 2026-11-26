"""Book-priced holding, paid hedge renewal, and sticky liquidation.

Uses source-fixed 24h forecasts, not rolling spread-z exits. Economic decisions
use current received books. Future closing liquidity remains unknown; its
current reverse-depth reserve is a forecast assumption, never a free fill.
"""
import math
import numpy as np
from experiments.forecast_validity_pairs_20261017.book_session import FixedForecastEntrySession
from experiments.basis_carry_20261017.probe import walk
from replay_lab.common_comparison.engine import rounded, risk_breaches


def hold_value(q, quotes, pair, marks, terminal, maturity, now, sc):
    fee = sc['fee_bps'] / 10000
    reverse, funding, oracle = [], [], []
    for symbol, amount in zip(pair, q):
        quote = quotes.get(symbol)
        if quote is None:
            return None
        price = walk(quote['book']['levels'][0 if amount > 0 else 1], abs(amount), sc['depth_fraction'])
        if price is None:
            return None
        reverse.append(price)
        funding.append(float(quote['ctx']['funding']))
        oracle.append(float(quote['oracle']))
    reverse = np.asarray(reverse)
    gross = float(abs(q) @ marks)
    remaining = max(0., (maturity - now) / 3600)
    cash = -q * np.asarray(oracle) * np.asarray(funding) * remaining
    carry = float(np.where(cash > 0, .5 * cash, cash).sum())
    spread_cost = float(q @ (marks - reverse))
    reserve = spread_cost + gross * sc['exit_impact_bps'] / 10000 + float(abs(q) @ terminal) * fee
    immediate = float(q @ (reverse - marks) - abs(q) @ reverse * fee)
    held = float(q @ (terminal - marks)) + carry - reserve
    if not np.isfinite([carry, reserve, immediate, held]).all():
        return None
    return dict(expected_hold_cash=held, immediate_close_cash=immediate,
                expected_funding_cash=carry, future_exit_reserve=reserve,
                incremental_advantage=held-immediate)


class PairLifecycle(FixedForecastEntrySession):
    def __init__(self, account, intent, cfg, p, scenario, *, on_time=None):
        super().__init__(account, intent, cfg, p, scenario, on_time=on_time)
        self.maturity = intent['forecast_source'] + intent['horizon'] * 3600
        self.latest_source = intent['forecast_source']
        self.renewal_attempts = set()
        self.initial_gross = None
        self.stop_origin = None
        self.stop_bound = None
        self.was_balanced = False
        self.closed_reason = None
        self.baseline_target = None
        self.packet_prefix = f'pair:{intent["forecast_source"]}:{len(account.seen)}'
        self.risk_peak = None

    def enter(self, now, streams, marks, sigma):
        if not self.intent['forecast_source'] <= self.intent['available_time'] <= now:
            raise ValueError('Forecast was not available at the decision')
        return super().enter(now, streams, marks, sigma)

    def submit(self, now, target, limits, streams, marks, sigma, kind):
        # Baseline and pair deltas share one packet so opposite intentions can
        # cross at an observed price; fees and external depth are charged once.
        self.sequence += 1
        base = self.account.owners['base'].qty.copy() if self.baseline_target is None else self.baseline_target.copy()
        pair = self.owner.qty.copy(); pair[self.ids] = target
        raw = dict(packet_id=f'{self.packet_prefix}:{self.sequence}', decision_time=now,
            source_close=now, source_available=now, source_prices=np.asarray(marks).tolist(),
            owner_targets={'base':base.tolist(),'pair':pair.tolist()},
            owner_limits={'pair':{str(j):v for j,v in limits.items()}}, risk_forced=kind=='close',sigma=sigma)
        packet = self.account.seal(raw, self.p, self.sc)
        result = self.account.execute_books(packet, streams, marks, sigma, self.p, self.sc, self.cfg, on_time=self.on_time)
        self.events.append(dict(time=now, kind=kind, result=result))
        if result['unknown_symbols']:
            self.state = 'reconcile_required'
            return result
        known = self.owner.fills[self.fill_start:]
        end = max([o.get('arrival_ms',now*1000) for o in result['orders']]+[f['time']*1000 for f in known]+[now*1000])
        self.ready_at = math.ceil((end+self.cfg['completion_ack_delay_ms'])/1000)
        q = self.owner.qty[self.ids]
        if not np.any(q):
            self.state = 'closed' if kind in ('close','trim') else 'unfilled'
            return result
        if self.first_fill is None:
            self.first_fill = min(f['time'] for f in known)
        ratios = q/self.target
        balanced = min(ratios)>0 and abs(ratios[0]-ratios[1])/max(ratios)<=self.cfg['pair_balance_tolerance']
        if kind == 'close':self.state = 'closing'
        elif balanced:self.state = 'open'
        elif min(ratios)>0:self.state = 'trim'
        elif kind=='entry' and self.completions==0:self.state = 'completion'
        else:self.state = 'closing'
        if self.state == 'open' and self.initial_gross is None:
            q = self.owner.qty[self.ids]
            entry = self.owner.entry[self.ids]
            self.initial_gross = float(abs(q) @ entry)
            unrealized = float(q @ (marks[self.ids] - entry))
            self.stop_origin = self.owner.equity(marks) + self.owner.fees - self.owner.funding - unrealized
            source_gross = float(abs(q) @ np.asarray(self.intent['source_prices']))
            self.stop_bound = (self.cfg['dense_policy']['stop_sigma'] * self.intent['source_vol']
                               * source_gross / self.initial_gross * math.sqrt(self.intent['horizon']))
            self.was_balanced = True
        return result

    def holding(self, now, marks):
        hard = (self.first_fill if self.first_fill is not None else self.intent['forecast_source']) + self.cfg['renewal_policy']['maximum_holding_hours'] * 3600
        lead = self.cfg['exit_lead_seconds']
        price_pnl = None
        if self.stop_origin is not None:
            price_pnl = self.owner.equity(marks) + self.owner.fees - self.owner.funding - self.stop_origin
        reason = None
        if now >= hard - lead:
            reason = 'hard_holding_limit'
        elif price_pnl is not None and price_pnl / self.initial_gross <= -self.stop_bound:
            reason = 'pair_return_stop'
        elif now >= min(self.maturity, hard) - lead:
            reason = 'forecast_maturity'
        return dict(reason=reason, hard_holding_deadline=hard, maturity=self.maturity,
                    episode_price_pnl=price_pnl, original_gross=self.initial_gross, original_stop_bound=self.stop_bound)

    def risk_marks(self, quotes, marks):
        result=marks.copy()
        for symbol,j in zip(self.intent['pair'],self.ids):
            if quotes.get(symbol) is not None:
                result[j]=float(quotes[symbol]['ctx'].get('markPx',quotes[symbol]['mid']))
        return result

    def renewal(self, now, forecast, quotes, marks, sigma, *, risk_blocked=False):
        if forecast is None or self.state != 'open' or risk_blocked:
            return None
        cadence = self.cfg['renewal_policy']['decision_cadence_hours'] * 3600
        boundary = now // cadence * cadence
        if not self.maturity - cadence <= boundary < self.maturity - self.cfg['exit_lead_seconds']:
            return None
        # A source published a few seconds after its completed-hour boundary is
        # usable only after receipt; do not require an impossible zero-latency fit.
        source = forecast['forecast_source']
        if not source <= forecast['available_time'] <= now:
            return None
        if source != boundary or source in self.renewal_attempts or not 0 <= now-source < 3600:
            return None
        self.renewal_attempts.add(source)
        def reject(reason):
            return dict(accepted=False, reason=reason, new_source=source)
        if forecast['pair'] != self.intent['pair'] or forecast['ids'] != self.ids:
            raise ValueError('Renewal pair identity mismatch')
        if source <= self.latest_source or forecast['horizon'] != self.intent['horizon']:
            return reject('not_a_new_same_horizon_forecast')
        state = self.holding(now, marks)
        if state['reason'] is not None:
            return reject('existing_exit_or_stop')
        maturity = source + forecast['horizon'] * 3600
        if maturity <= self.maturity or maturity > state['hard_holding_deadline']:
            return reject('invalid_new_maturity')
        if maturity > self.cfg.get('latest_maturity_unix', float('inf')):
            return reject('new_maturity_outside_evaluation')
        risk_marks = self.risk_marks(quotes,marks)
        equity = self.account.physical.equity(risk_marks)
        if risk_breaches(self.account.physical.qty, equity, risk_marks, sigma, self.p['risk']):
            return reject('global_risk')
        q = self.owner.qty[self.ids].copy()
        anchors = np.asarray(forecast['source_prices'], float)
        terminal = np.asarray(forecast['terminal_prices'], float)
        beta = forecast['source_beta']
        if not np.isfinite([beta, *anchors, *terminal]).all() or min(beta, *anchors, *terminal) <= 0:
            return reject('invalid_forecast')
        ratio = beta * anchors[0] / anchors[1]
        if np.any(q == 0) or np.sign(q[0]) != self.intent['direction'] or q[0]*q[1] >= 0:
            return reject('inventory_not_balanced')
        target = q.copy()
        adjustment = abs(abs(q[1]/q[0])/ratio - 1) > self.cfg['pair_balance_tolerance']
        gross = float(abs(q) @ marks[self.ids])
        if adjustment:
            value = min(gross, self.cfg['pair_max_gross'])
            units = self.intent['direction'] * value / (marks[self.ids[0]] + ratio*marks[self.ids[1]]) * np.array([1., -ratio])
            target = np.array([rounded(v, self.p['execution']['decimals'][j]) for j, v in zip(self.ids, units)])
            if np.any(target == 0) or abs(abs(target[1]/target[0])/ratio-1) > self.cfg['pair_balance_tolerance']:
                return reject('adjusted_hedge_rounding')
            if abs(float(abs(target) @ marks[self.ids])/value-1) > self.cfg['rehedge_policy']['gross_tolerance']:
                return reject('adjustment_gross_mismatch')
        old_value = hold_value(q, quotes, self.intent['pair'], marks[self.ids], terminal, maturity, now, self.sc)
        value = hold_value(target, quotes, self.intent['pair'], marks[self.ids], terminal, maturity, now, self.sc)
        if old_value is None or value is None:
            return reject('missing_visible_exit_depth')
        delta = target-q
        active = np.flatnonzero(abs(delta) > 1e-10)
        previews = marks[self.ids].copy()
        for k in active:
            symbol = self.intent['pair'][k]
            px = walk(quotes[symbol]['book']['levels'][1 if delta[k] > 0 else 0], abs(delta[k]), self.sc['depth_fraction'])
            if px is None or abs(delta[k])*px < self.p['execution']['minimum_order_value']:
                return reject('adjustment_depth_or_minimum')
            previews[k] = px
        fee = self.sc['fee_bps']/10000
        cost = float(delta @ (previews-marks[self.ids]) + abs(delta) @ previews * fee)
        advantage = value['expected_hold_cash'] - cost - old_value['immediate_close_cash']
        buffer = gross*self.cfg['renewal_policy']['incremental_buffer_bps']/10000
        if not np.isfinite(advantage) or advantage <= buffer:
            return reject('incremental_value_below_cost')
        limits = previews.copy()
        for k in active:
            side = np.sign(delta[k])
            distance = (advantage-buffer)/len(active)/(abs(delta[k])*(1+side*fee))
            limits[k] += side*min(distance, previews[k]*.99)
        worst_cost = float(delta @ (limits-marks[self.ids]) + abs(delta) @ limits*fee)
        for mask in ([1, 0], [0, 1], [1, 1]):
            qty = self.account.physical.qty.copy()
            change=delta*np.asarray(mask);qty[self.ids] += change
            after=equity-float(change@(limits-risk_marks[self.ids]))-float(abs(change)@limits)*fee
            if after <= 0 or risk_breaches(qty, after, risk_marks, sigma, self.p['risk']):
                return reject('independent_adjustment_risk')
        return dict(accepted=True, reason='paid_renewal' if adjustment else 'unchanged_inventory_renewal',
                    requires_adjustment=adjustment, new_source=source, new_maturity=maturity,
                    old_qty=q.tolist(), target=target.tolist(), terminal_prices=terminal.tolist(),
                    desired_unit_ratio=ratio, incremental_advantage=advantage, buffer_cash=buffer,
                    limits={str(self.ids[k]): float(limits[k]) for k in active},
                    minimum_advantage_at_limits=value['expected_hold_cash']-worst_cost-old_value['immediate_close_cash'])

    def reconcile_adjustment(self, plan, now, quotes, marks, sigma):
        q = self.owner.qty[self.ids].copy()
        if np.all(abs(q-np.asarray(plan['old_qty'])) < 1e-10):
            self.state = 'open'
            return 'unfilled'
        def unwind(reason):
            self.state = 'closing'
            self.closed_reason = reason
            return 'unwind'
        if self.holding(now, marks)['reason'] is not None:
            return unwind('adjustment_arrival_stop_or_deadline')
        target = np.asarray(plan['target'])
        ratios = q/target
        if min(ratios) <= 0 or abs(ratios[0]-ratios[1])/max(ratios) > self.cfg['pair_balance_tolerance']:
            return unwind('partial_adjustment_hedge_mismatch')
        if abs(abs(q[1]/q[0])/plan['desired_unit_ratio']-1) > self.cfg['pair_balance_tolerance']:
            return unwind('actual_adjustment_hedge_mismatch')
        gross = float(abs(q) @ marks[self.ids])
        if abs(gross/float(abs(target) @ marks[self.ids])-1) > self.cfg['rehedge_policy']['gross_tolerance']:
            return unwind('partial_adjustment_gross_mismatch')
        risk_marks=self.risk_marks(quotes,marks)
        if risk_breaches(self.account.physical.qty, self.account.physical.equity(risk_marks), risk_marks, sigma, self.p['risk']):
            return unwind('adjustment_arrival_risk')
        if self.risk_peak is not None and self.account.physical.equity(risk_marks)/self.risk_peak-1<=-self.p['risk']['drawdown_halt']:
            return unwind('adjustment_arrival_drawdown_halt')
        value = hold_value(q, quotes, self.intent['pair'], marks[self.ids], np.array(plan['terminal_prices']), plan['new_maturity'], now, self.sc)
        if value is None or value['incremental_advantage'] <= gross*self.cfg['renewal_policy']['incremental_buffer_bps']/10000:
            return unwind('adjustment_arrival_value_insufficient')
        self.target = q.copy()
        self.latest_source, self.maturity = plan['new_source'], plan['new_maturity']
        self.state = 'open'
        return 'renewed'

    def advance(self, now, streams, marks, sigma, *, forecast=None, force_reason=None):
        if self.state in ('new', 'rejected', 'unfilled', 'closed', 'reconcile_required') or now < self.ready_at:
            return None
        quotes = self.books(now, streams)
        marks = self.marks(quotes, marks)
        if force_reason:
            self.state, self.closed_reason = 'closing', force_reason
        if self.state == 'open':
            state = self.holding(now, marks)
            if state['reason']:
                self.state, self.closed_reason = 'closing', state['reason']
                self.events.append(dict(time=now, kind='exit_requested', **state))
            elif all(v is not None for v in quotes.values()):
                plan = self.renewal(now, forecast, quotes, marks, sigma)
                if plan is not None:
                    self.events.append(dict(time=now, kind='renewal_decision', **plan))
                    if plan['accepted'] and not plan['requires_adjustment']:
                        self.latest_source, self.maturity = plan['new_source'], plan['new_maturity']
                        return plan
                    if plan['accepted']:
                        old_target = self.target.copy()
                        self.target = np.array(plan['target'])
                        result = self.submit(now, self.target, plan['limits'], streams, marks, sigma, 'rehedge')
                        self.target = old_target
                        if self.state == 'reconcile_required':
                            return result
                        at = self.ready_at
                        if self.on_time is not None:
                            self.on_time(at*1000, self.account)
                        observed = self.books(at, streams)
                        arrival_marks = self.marks(observed, marks)
                        outcome = self.reconcile_adjustment(plan, at, observed, arrival_marks, sigma)
                        self.events.append(dict(time=at, kind='adjustment_receipt', outcome=outcome, maturity=self.maturity))
                        return result
            if self.state == 'open':
                return None
        # Completion/trim and sticky paid closing retain the source-fixed entry
        # adapter. Open holding above uses renewed maturity, never its old exit.
        return super().advance(now, streams, marks, sigma)
