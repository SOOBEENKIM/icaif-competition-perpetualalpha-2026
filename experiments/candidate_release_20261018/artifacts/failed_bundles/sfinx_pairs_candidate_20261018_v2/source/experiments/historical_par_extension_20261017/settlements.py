"""Preserve exact settlement timestamps and recognize cash after the event."""
import numpy as np

def records(rows,coin,cutoff_ms):
    mapping={}
    for row in rows:
        stamp=int(row['time']);bucket=stamp//3600000*3600
        if row['coin']!=coin or stamp-bucket*1000>=1000 or stamp>cutoff_ms:
            raise ValueError('Wrong asset, unsupported settlement jitter, or future funding')
        if bucket in mapping:raise ValueError('Duplicate hourly settlement')
        rate=float(row['fundingRate'])
        if not np.isfinite(rate):raise ValueError('Nonfinite actual funding')
        mapping[bucket]=(stamp,rate)
    return mapping

def funding_values(rows,times,coin):
    # Overlap/full completed buckets only. Cash uses the original timestamps.
    values=records(rows,coin,2**63-1)
    try:return np.array([values[int(t)][1] for t in times])
    except KeyError as exc:raise ValueError(f'Missing actual funding {coin} {exc}') from exc

def align(rows,times,old_end,coin):
    """Old end-hour payment already exists in the immutable historical prefix."""
    values=records(rows,coin,int(times[-1])*1000)
    required=np.arange(old_end+3600,int(times[-1]),3600)
    if any(t not in values for t in required):raise ValueError('Missing an interior hourly settlement')
    rates=np.zeros(len(times));stamps=np.full(len(times),-1,np.int64);hour_source=np.full(len(times),-1,np.int64)
    for hour,(stamp,rate) in values.items():
        if hour<=old_end:continue
        i=int(np.searchsorted(times*1000,stamp,side='left'))
        if i==len(times):raise ValueError('No observation at/after settlement')
        if stamps[i]>=0:raise ValueError('Multiple settlements in one observation interval')
        assert times[i]*1000>=stamp and (i==0 or times[i-1]*1000<stamp)
        rates[i]=rate;stamps[i]=stamp;hour_source[i]=hour
    return rates,stamps,hour_source

def charge(account,now,prices,rates,observed,stamp,receipt_multiplier):
    """Funding uses inventory held between candles, before next-bar fills."""
    if not np.all(observed) or stamp>now*1000:raise ValueError('Incomplete or future settlement')
    qty=account.physical.qty.copy();cash=-qty*prices*rates
    effective=np.where(cash>0,rates*receipt_multiplier,rates)
    before={key:len(ledger.funding_events) for key,ledger in [('physical',account.physical),*account.owners.items()]}
    account.charge_funding(now,prices,effective,observed)
    for key,ledger in [('physical',account.physical),*account.owners.items()]:
        if len(ledger.funding_events)>before[key]:
            ledger.funding_events[-1].update(settlement_time_ms=int(stamp),recognized_bar_time=int(now))
    return -float(qty@(prices*effective))
