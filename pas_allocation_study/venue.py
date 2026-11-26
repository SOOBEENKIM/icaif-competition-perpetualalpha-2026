"""Causal PAS relative-basis correction; no assumption that oracle is tradable."""
import numpy as np


def basis_forecast(times, basis, at, p):
    times=np.asarray(times);basis=np.asarray(basis)
    keep=(times<=at)&(times>at-p['basis_lookback']*3600)&np.isfinite(basis)
    t=times[keep];b=basis[keep]
    consecutive=np.diff(t)==3600
    x=b[:-1][consecutive];dy=np.diff(b)[consecutive]
    if len(x)<p['basis_minimum_pairs']:
        return 0.,dict(reason='insufficient_contiguous_basis_history',pairs=len(x))
    center=x.mean();variance=float(np.mean((x-center)**2))
    if variance<1e-12: return 0.,dict(reason='constant_basis',pairs=len(x))
    # Penalize an increment slope toward 0 (AR coefficient toward 1).
    slope=float(((x-center)@dy)/((x-center)@(x-center)+p['basis_ridge_prior_pairs']*variance))
    phi=float(np.clip(1+slope,0,1))
    # Long-run center estimated from past observations; drift is not extrapolated.
    correction=(1-phi**24)*(center-b[-1])
    # Bound extrapolation by the observed basis range, never tune on outcomes.
    bound=float(np.quantile(abs(b-center),.95))
    correction=float(np.clip(correction,-bound,bound))
    return correction,dict(reason='estimated',pairs=len(x),phi=phi,
        last_observation=int(t[-1]),basis=float(b[-1]),center=float(center),correction=correction)


def net_forecast(reference_mu, correction, funding_rates, *, reference_price=1., venue_price=1., basis_now=0.):
    rates=np.asarray(funding_rates,float)
    known=rates[np.isfinite(rates)][-24:]
    carry=float(known.mean()*12) if len(known)>=12 else 0.
    terminal=reference_price*(1+reference_mu)*np.exp(basis_now+correction)
    return float(terminal/venue_price-1-carry),carry


def executable_edge(reference_mu, reference_price, basis_terminal, book, future_exit_halfspread, funding):
    """Separate quote probe; oracle/ref value is a forecast, never a fill."""
    if not book['levels'][0] or not book['levels'][1]: return None
    bid=float(book['levels'][0][0]['px']);ask=float(book['levels'][1][0]['px'])
    terminal=reference_price*(1+reference_mu)*np.exp(basis_terminal)
    return dict(long_edge=terminal*(1-future_exit_halfspread)/ask-1-funding-.0018,
                short_edge=1-terminal*(1+future_exit_halfspread)/bid+funding-.0018,
                bid=bid,ask=ask,forecast_terminal=terminal,
                minimum_top_notional=min(bid*float(book['levels'][0][0]['sz']),ask*float(book['levels'][1][0]['sz'])))
