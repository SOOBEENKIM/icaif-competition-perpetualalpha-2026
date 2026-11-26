"""Entry/partial-fill reconciliation only; this is not a full-horizon book backtest."""
import numpy as np
from experiments.shared_execution_pairs_20261017.book_session import BookPairSession
from replay_lab.common_comparison.engine import rounded
from .book_plan import plan_entry


class FixedForecastEntrySession(BookPairSession):
    def enter(self, now, streams, marks, sigma):
        if self.state != 'new':
            raise RuntimeError('Session entry already evaluated')
        quotes = self.books(now, streams); marks = self.marks(quotes, marks)
        source = {**self.intent, 'account_marks':marks.tolist()}
        plan = plan_entry(source, quotes, now, self.cfg, self.p, self.sc, self.account.physical.qty,
                          self.account.physical.equity(marks), sigma, owned_qty=self.owner.qty)
        self.plan = plan
        if not plan['accepted']:
            self.state = 'rejected'; return plan
        self.target = np.asarray(plan['target'])
        return self.submit(now, self.target, {o['asset']:o['limit'] for o in plan['orders']}, streams, marks, sigma, 'entry')

    def advance(self, now, streams, marks, sigma):
        if self.state in ('new','rejected','unfilled','closed','reconcile_required','open') or now < self.ready_at:
            return None
        quotes = self.books(now, streams); marks = self.marks(quotes, marks)
        if self.state == 'completion' and now > self.first_fill+self.cfg['completion_window_seconds']:
            self.state = 'closing'
        if now >= self.intent['forecast_source']+self.intent['horizon']*3600:
            self.state = 'closing'
        if self.state == 'completion':
            # Parent completion prices only the missing leg, counts actual sunk
            # fees/funding, and uses this session's fixed source horizon.
            plan = self.completion(now, quotes, marks, sigma); self.completions += 1
            if plan is not None:
                self.events.append(dict(time=now, kind='completion_budget', **{
                    k:v for k,v in plan.items() if k not in ('target','limits')}))
                return self.submit(now, plan['target'], plan['limits'], streams, marks, sigma, 'completion')
            self.state = 'closing'
        if self.state == 'trim':
            ratio = min(self.owner.qty[self.ids]/self.target)
            target = np.array([rounded(v*ratio,self.p['execution']['decimals'][j]) for v,j in zip(self.target,self.ids)])
            if np.any(target == 0):
                self.state = 'closing'
            else:
                return self.submit(now, target, {}, streams, marks, sigma, 'trim')
        if self.state == 'closing':
            return self.submit(now, np.zeros(2), {}, streams, marks, sigma, 'close')
        return None
