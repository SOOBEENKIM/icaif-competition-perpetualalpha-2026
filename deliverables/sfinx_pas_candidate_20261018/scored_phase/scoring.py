"""12:00 UTC samples and transfer-adjusted scores; never mixes practice equity."""
import math
import statistics


def evaluate(samples, *, latest=None, trading_days=0, jury=0.):
    """samples: time, equity, net_external_since_previous (USDP).

    First sample is after the initial grant. The grant is capital, not PnL.
    Optional latest partial-day return is display-only. Standings use the last
    completed 12:00 UTC sample; partial-day observations cannot change the score.
    """
    if not 0<=jury<=1:raise ValueError('Jury score outside [0,1]')
    if len(samples)<1:return dict(eligible=False,return_fraction=None,volatility=None,Q=None,score=None)
    if any(s['time']%86400!=43200 for s in samples):raise ValueError('Daily samples must be at 12:00 UTC')
    if any(not math.isfinite(s['equity']) or s['equity']<=0 for s in samples):raise ValueError('Invalid equity')
    adjusted=[];wealth=1.
    for a,b in zip(samples,samples[1:]):
        span=(b['time']-a['time'])/86400
        if span<1 or span!=int(span):raise ValueError('Samples out of order')
        flow=float(b.get('net_external_since_previous',0.))
        if not math.isfinite(flow):raise ValueError('Nonfinite capital flow')
        r=(b['equity']-flow)/a['equity']-1
        wealth*=1+r;adjusted.append(r/math.sqrt(span))
    intraday_wealth=wealth
    if latest is not None:
        if not samples[-1]['time']<latest['time']<samples[-1]['time']+86400:raise ValueError('Invalid partial day')
        if not math.isfinite(latest['equity']) or latest['equity']<=0:raise ValueError('Invalid partial equity')
        intraday_wealth*=(latest['equity']-latest.get('net_external_since_previous',0.))/samples[-1]['equity']
    R=wealth-1;n=len(adjusted)
    sigma=statistics.stdev(adjusted)*math.sqrt(n) if n>=2 else None
    Q=(max(R,0)+.005)/(abs(R)+1.5*sigma+.05) if sigma is not None else None
    eligible=n>=2 and trading_days>=1
    return dict(eligible=eligible,return_fraction=R,intraday_return_fraction=intraday_wealth-1,volatility=sigma,Q=Q,
        score=.8*Q+.2*jury if eligible else None,completed_returns=n)
