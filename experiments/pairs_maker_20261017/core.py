"""Past-only order-flow inference and an explicitly assumed FIFO queue."""
from dataclasses import dataclass, field
import numpy as np
from experiments.execution_retry_20261016.engine import lot_floor


class FlowFilter:
    """Beta-Bernoulli change-point recursion, adapted from Moret/Lillo (2026).

    This component is not their DQN or a trained market-making strategy.
    The finite run-length tail is truncated and renormalized explicitly.
    """
    def __init__(self, hazard=.02, alpha=1., beta=1., maximum=256):
        self.hazard, self.a0, self.b0, self.maximum = hazard, alpha, beta, maximum
        self.mass = np.ones(1); self.a = np.array([alpha]); self.b = np.array([beta])
        self.length = np.zeros(1); self.seen = 0; self.dropped_mass = 0.

    def predict(self):
        posterior_mean = float(self.mass @ (self.a / (self.a + self.b)))
        return dict(buy_probability=posterior_mean, signed_flow=2*posterior_mean-1,
                    expected_run_length=float(self.mass @ self.length), observations=self.seen)

    def observe(self, buy):
        if buy not in (0,1): raise ValueError('Binary aggressor direction required')
        predictive = self.a / (self.a + self.b)
        evidence = predictive if buy else 1-predictive
        reset_evidence = (self.a0 if buy else self.b0) / (self.a0+self.b0)
        mass = np.r_[self.hazard*reset_evidence, self.mass*(1-self.hazard)*evidence]
        a, b = np.r_[self.a0+buy, self.a+buy], np.r_[self.b0+1-buy, self.b+1-buy]
        length = np.r_[1., self.length+1]
        mass /= mass.sum(); self.dropped_mass += float(mass[self.maximum:].sum())
        self.mass = mass[:self.maximum]; self.mass /= self.mass.sum()
        self.a, self.b, self.length = a[:self.maximum], b[:self.maximum], length[:self.maximum]
        self.seen += 1
        return self.predict()


@dataclass
class PassiveOrder:
    coin: str
    side: int
    price: float
    quantity: float
    queue_ahead: float
    arrival: int
    cancel_effective: int
    decimals: int
    fills: list = field(default_factory=list)
    seen: set = field(default_factory=set)

    def trade(self, received, trade):
        identity = trade['time'], trade['coin'], trade['tid']
        if identity in self.seen: return 0.
        self.seen.add(identity)
        if trade['coin'] != self.coin or not self.arrival < trade['time'] < self.cancel_effective:
            return 0.
        if not self.arrival < received < self.cancel_effective: return 0.
        opposite = 'A' if self.side > 0 else 'B'
        if trade['side'] != opposite or self.side*(float(trade['px'])-self.price)>1e-9:
            return 0.
        quantity = float(trade['sz'])
        consumed = min(self.queue_ahead, quantity)
        self.queue_ahead -= consumed; quantity -= consumed
        remaining = max(0., self.quantity-sum(f['qty'] for f in self.fills))
        fill = lot_floor(min(quantity, remaining), self.decimals)
        if fill:
            self.fills.append(dict(time=received, exchange_time=trade['time'], tid=trade['tid'],
                qty=fill, price=self.price, side=self.side, coin=self.coin))
        return fill


def queue_at_arrival(book, price, side):
    """Return displayed same-price queue; reject crossed or unobserved prices."""
    own = book['levels'][0 if side > 0 else 1]
    opposite = book['levels'][1 if side > 0 else 0]
    if not own or not opposite: return None, 'empty_book'
    if side*(price-float(opposite[0]['px'])) >= -1e-10: return None, 'post_only_rejected'
    if side*(price-float(own[-1]['px'])) < -1e-10: return None, 'queue_outside_observed_depth'
    return sum(float(r['sz']) for r in own if abs(float(r['px'])-price)<1e-9), 'accepted'
