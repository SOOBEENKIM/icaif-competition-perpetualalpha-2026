"""Paper top-k/drop-n allocation, adapted to actual filled inventory and caps."""
import copy
from dataclasses import asdict
import numpy as np
from replay_lab.ema_rebuild.engine import Controller
from .data import DAY


def choose(scores, held, entries, time, top_k=2, drop_n=1, min_days=5):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Missing ranking scores')
    held = set(held)
    order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
    old = [j for j in order if j in held]
    new = [j for j in order if j not in held][:max(0, drop_n + top_k - len(old))]
    combined = sorted(old + new, key=lambda j: (-scores[j], j))
    bottom = set(combined[-drop_n:]) if drop_n else set()
    sell = {j for j in old if j in bottom and time//DAY - entries[j]//DAY >= min_days}
    buy = new[:max(0, len(sell) + top_k - len(old))]
    selected = (held - sell) | set(buy)
    return selected, sell, set(buy)


class PaperAgent:
    """Deterministic order-intent agent; accepts frozen, already causal scores."""
    def __init__(self, scores, scale=1., name='kronos_paper', saved=None, equal_weight=False):
        if not 0 < scale <= 1:
            raise ValueError('Invalid fixed scale')
        self.scores = scores
        self.scale, self.name, self.equal_weight = float(scale), name, equal_weight
        s = copy.deepcopy(saved) if saved is not None else {}
        if s and (s.get('name') != name or s.get('scale') != scale or s.get('equal_weight') != equal_weight):
            raise ValueError('State from a different policy')
        self.controller = Controller(**s.get('controller', {}))
        self.entries = {int(k): int(v) for k, v in s.get('entries', {}).items()}
        self.target = np.array(s.get('target', [0.] * 10), dtype=float)
        self.last_observation = s.get('last_observation', -1)
        self.selections = []

    def decide(self, time, prices, qty, equity, sigma, volume, trades, protocol, halted=False):
        if time <= self.last_observation:
            return None
        qty = np.asarray(qty, dtype=float)
        if (qty.shape != (10,) or not np.isfinite(qty).all() or not np.isfinite(equity) or equity <= 0
                or not all(np.isfinite(v).all() for v in (prices, sigma, volume, trades))
                or np.any(np.asarray(prices) <= 0) or np.any(np.asarray(volume) < 0)
                or np.any(np.asarray(trades) < 0)):
            return None
        if np.any(qty < -1e-10):
            raise ValueError('Paper long-only policy cannot adopt existing shorts')
        self.last_observation = int(time)
        held = set(np.flatnonzero(qty > 1e-10).tolist())
        self.entries = {j: self.entries.get(j, int(time)) for j in held}
        if time % DAY == 0 and not halted:
            if self.equal_weight:
                selected, sell, buy = set(range(10)), set(), set()
                self.target = np.full(10, .027 * self.scale)
            else:
                if int(time) not in self.scores:
                    raise ValueError('No forecast for completed daily origin')
                selected, sell, buy = choose(self.scores[int(time)], held, self.entries, time)
                self.target = np.zeros(10)
                self.target[list(selected)] = .135 * self.scale
            self.selections.append(dict(time=int(time), held=sorted(held), selected=sorted(selected),
                                        sell=sorted(sell), buy=sorted(buy), entries=dict(self.entries)))
        return self.controller.propose(time, prices, qty, equity, self.target, sigma,
                                       volume, trades, protocol, buffer=.2, halted=halted)

    def snapshot(self):
        return dict(name=self.name, scale=self.scale, equal_weight=self.equal_weight,
                    controller=asdict(self.controller), entries=dict(self.entries),
                    target=self.target.tolist(), last_observation=int(self.last_observation))
