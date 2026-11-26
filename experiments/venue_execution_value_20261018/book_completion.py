"""Event-driven original-limit completion; no terminal-price fiction."""
import numpy as np
from replay_lab.common_comparison.engine import rounded, risk_breaches
from experiments.forecast_validity_pairs_20261017.book_session import FixedForecastEntrySession


class OriginalLimitSession(FixedForecastEntrySession):
    def enter(self, now, streams, marks, sigma):
        if self.state != 'new': raise RuntimeError('Already evaluated')
        self.target=np.asarray(self.intent['target'])
        self.plan=self.intent
        return self.submit(now,self.target,self.intent['entry_limits'],streams,marks,sigma,'entry')

    def completion(self, now, quotes, marks, sigma):
        owned=self.owner.qty[self.ids]
        present=np.flatnonzero(owned)
        if len(present)!=1 or any(quotes[s] is None for s in self.intent['pair']):return None
        have=int(present[0]);missing=1-have;ratio=owned[have]/self.target[have]
        if not 0<ratio<=1+1e-8:return None
        target=self.target*ratio;target[have]=owned[have]
        j=self.ids[missing];target[missing]=rounded(target[missing],self.p['execution']['decimals'][j])
        ratios=target/self.target
        if min(ratios)<=0 or abs(ratios[0]-ratios[1])/max(ratios)>self.cfg['pair_balance_tolerance']:return None
        quantity=target[missing];side=np.sign(quantity);limit=self.intent['entry_limits'][str(j)]
        quote=quotes[self.intent['pair'][missing]]
        preview=float(quote['book']['levels'][1 if side>0 else 0][0]['px'])
        if side*(preview-limit)>1e-10 or abs(quantity)*preview<self.p['execution']['minimum_order_value']:return None
        risk_qty=self.account.physical.qty.copy();risk_qty[self.ids]+=target-owned
        equity=self.account.physical.equity(marks)-quantity*(preview-marks[j])-abs(quantity)*preview*self.sc['fee_bps']/10000
        if equity<=0 or risk_breaches(risk_qty,equity,marks,sigma,self.p['risk']):return None
        return dict(target=target,limits={j:limit},policy='one_attempt_original_limit_no_quantity_expansion')
