"""Staged millisecond-preserving ledger, never installed into a live worker.

The old accounting engine truncated fill time to whole seconds. That can put
an after-settlement fill before the payment when the confirmed payment is
published a few milliseconds into that second. No price/quantity rules change.
Older rounded records cannot be repaired by guessing their original times.
"""
import math

from replay_lab.common_comparison.engine import Ledger


class MillisecondLedger(Ledger):
    def fill(self,asset,signed_qty,price,fee_bps,time,**metadata):
        at=float(time)
        if not math.isfinite(at) or at<0:
            raise ValueError('Invalid fill timestamp')
        ms=round(at*1000)
        if abs(at*1000-ms)>.001:
            raise ValueError('Fill timestamp is not on the exchange millisecond grid')
        if 'time_ms' in metadata:
            raise ValueError('Timestamp metadata must not override the observed time')
        super().fill(asset,signed_qty,price,fee_bps,time,**metadata)
        self.fills[-1]['time']=ms/1000
        self.fills[-1]['time_ms']=ms


def install_in_copy(portfolio):
    """Caller must supply a research/staging copy; no filesystem or network."""
    a=portfolio.account
    ledgers=[a.physical,a.reserved,*a.owners.values()]
    if any(type(l) not in (Ledger,MillisecondLedger) for l in ledgers):
        raise TypeError('Unknown ledger behavior; do not replace it')
    if a.pending_orders or a.unknown:
        raise ValueError('Resolve orders before ledger adaptation')
    for l in ledgers:l.__class__=MillisecondLedger
    return portfolio


def ambiguous_funding_records(ledger):
    """Return old integer-time fills whose settlement ordering is unknowable."""
    findings=[]
    for f in ledger.fills:
        if 'time_ms' in f:continue
        for u in ledger.funding_events:
            if ledger.symbols[f['asset']]==u['coin'][4:] and int(f['time'])==int(u['time']):
                findings.append(dict(coin=u['coin'],fill_recorded_time=f['time'],
                    funding_time=u['time'],reason='original_fill_milliseconds_not_retained'))
    return findings
