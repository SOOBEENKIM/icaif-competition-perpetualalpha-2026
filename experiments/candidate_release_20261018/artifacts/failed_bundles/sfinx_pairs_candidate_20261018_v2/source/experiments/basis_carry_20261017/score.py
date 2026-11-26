"""Published Oct17 leaderboard arithmetic, with explicit daily input contracts.

Callers supply returns formed at 12:00 UTC, excluding partial today from the
volatility array. This helper does not invent daily observations from quotes.
"""
import numpy as np

def transfer_adjusted_return(end_equity,start_equity,net_transfer):
    if start_equity<=0:raise ValueError('Start equity must be positive')
    return (end_equity-net_transfer)/start_equity-1

def score(total_return,completed_returns,day_spans=None,jury=0.,trading_days=0,start_sample=False):
    r=np.asarray(completed_returns,dtype=float);n=len(r)
    spans=np.ones(n) if day_spans is None else np.asarray(day_spans,dtype=float)
    if spans.shape!=r.shape or np.any(spans<1) or not np.isfinite(spans).all():raise ValueError('Invalid day spans')
    if not np.isfinite(r).all() or not np.isfinite(total_return) or not 0<=jury<=1:raise ValueError('Invalid score inputs')
    s=float(np.std(r/np.sqrt(spans),ddof=1)) if n>=2 else None
    sigma=s*np.sqrt(n) if s is not None else None
    q=(max(total_return,0)+.005)/(abs(total_return)+1.5*sigma+.05) if sigma is not None else None
    eligible=bool(n>=2 and trading_days>=1 and start_sample)
    return dict(return_fraction=total_return,completed_days=n,daily_sample_std=s,volatility_horizon=sigma,
                financial_score=q,score=.8*q+.2*jury if eligible else None,eligible=eligible)
