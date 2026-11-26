"""Choose exposure first, then the minimum feasible exchange margin leverage.

The selected exposure maximizes the incumbent's forecast utility over a declared
grid, not realized future PnL. This is not a claim of globally optimal leverage.
No signer, account mutation, model fitting, or live deployment occurs here.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Settings:
    max_target_multiplier: float = 3.0
    multiplier_step: float = 0.05
    horizon_hours: int = 168
    risk_aversion: float = 5.0
    entry_cost: float = 0.0019
    exit_cost: float = 0.0024

    def __post_init__(self):
        values = list(vars(self).values())
        if not np.isfinite(values).all() or self.max_target_multiplier < 1:
            raise ValueError('Invalid leverage settings')
        if self.multiplier_step <= 0 or self.horizon_hours <= 0 or self.risk_aversion <= 0:
            raise ValueError('Invalid grid, horizon or risk aversion')
        if min(self.entry_cost, self.exit_cost) < 0:
            raise ValueError('Negative cost')
        if (self.max_target_multiplier - 1) / self.multiplier_step > 10000:
            raise ValueError('Grid too large')


def _vector(value, n, name, *, boolean=False):
    a = np.asarray(value)
    if a.shape != (n,) or (boolean and a.dtype != bool):
        raise ValueError('Invalid ' + name)
    if not boolean:
        a = a.astype(float)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite ' + name)
    return a


def select_exposure(*, base_qty, current_qty, prices, equity, forecast, funding,
                    hourly_covariance, protected, entry_allowed, liquid,
                    decimals, risk, settings=Settings()):
    """Only amplify eligible incumbent targets; mandatory reductions survive.

    Forecast and funding are signed simple returns over the same 168h horizon.
    Hourly covariance is scaled by horizon, as in the incumbent utility model.
    Reference-market forecasts are not thereby validated as PAS forecasts.
    Caller must verify timestamps/availability; no future labels are accepted.
    """
    base = np.asarray(base_qty, dtype=float)
    if base.ndim != 1 or not len(base) or not np.isfinite(base).all():
        raise ValueError('Invalid base target')
    n = len(base)
    old, px, mu, carry = [_vector(v, n, k) for v, k in (
        (current_qty, 'inventory'), (prices, 'prices'),
        (forecast, 'forecast'), (funding, 'funding'))]
    protected, allowed, liquid = [_vector(v, n, k, boolean=True) for v, k in (
        (protected, 'protected'), (entry_allowed, 'entry gate'), (liquid, 'liquidity'))]
    dp = _vector(decimals, n, 'decimals')
    if np.any(dp != np.floor(dp)) or np.any((dp < 0) | (dp > 12)):
        raise ValueError('Invalid lot precision')
    if not math.isfinite(equity) or equity <= 0 or np.any(px <= 0):
        raise ValueError('Invalid equity/price')
    cov = np.asarray(hourly_covariance, dtype=float)
    if (cov.shape != (n, n) or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.T, rtol=0, atol=1e-12)
            or np.linalg.eigvalsh(cov).min() < -1e-12):
        raise ValueError('Invalid covariance')
    names = ('gross_fraction', 'net_fraction', 'asset_fraction',
             'asset_notional', 'daily_volatility', 'sizing_headroom')
    if any(not math.isfinite(risk[k]) or risk[k] <= 0 for k in names):
        raise ValueError('Invalid risk limits')
    if risk['sizing_headroom'] > 1:
        raise ValueError('Invalid risk headroom')
    w0, oldw = base * px / equity, old * px / equity
    headroom = risk['sizing_headroom']
    cap = min(risk['asset_fraction'], risk['asset_notional'] / equity) * headroom

    def values(w):
        return dict(gross=float(abs(w).sum()), net=float(abs(w.sum())),
                    asset=float(abs(w).max()),
                    daily_volatility=float(np.sqrt(max(0., 24 * w @ cov @ w))))

    def hard_feasible(w):
        v = values(w)
        return (v['gross'] <= risk['gross_fraction'] + 1e-10
                and v['net'] <= risk['net_fraction'] + 1e-10
                and v['asset'] <= min(risk['asset_fraction'], risk['asset_notional'] / equity) + 1e-10
                and v['daily_volatility'] <= risk['daily_volatility'] + 1e-10)

    if not hard_feasible(w0):
        raise ValueError('Incumbent target requires risk repair before amplification')
    direction = np.sign(base)
    signed_funding = direction * carry
    edge = direction * mu - np.where(signed_funding > 0, signed_funding, 0.5 * signed_funding)
    # Reductions and reversals must complete first. A leverage overlay must not
    # undo a safety trim, force a gated entry, or enlarge the other side of a pair.
    reducing = (old != 0) & ((old * base <= 0) | (abs(base) < abs(old) - 1e-12))
    eligible = (~protected & allowed & liquid & ~reducing & (base != 0)
                & (edge > settings.entry_cost + settings.exit_cost))
    net_mu = mu - 0.75 * carry
    exit_cost = settings.exit_cost + 0.25 * abs(carry)

    def utility(w):
        return float(net_mu @ w - 0.5 * settings.risk_aversion * settings.horizon_hours * (w @ cov @ w)
                     - settings.entry_cost * abs(w - oldw).sum() - exit_cost @ abs(w))

    baseline_utility = utility(w0)
    best = dict(qty=base.copy(), multiplier=1., utility=baseline_utility)
    trials = []
    grid = np.unique(np.r_[np.arange(1., settings.max_target_multiplier, settings.multiplier_step),
                           settings.max_target_multiplier])
    for multiplier in grid:
        q = base.copy()
        for j in np.flatnonzero(eligible):
            # Only increases face the headroom cap; never alter a protected leg.
            upper = max(abs(base[j]), cap * equity / px[j])
            value = min(abs(base[j]) * multiplier, upper)
            q[j] = direction[j] * math.floor(value * 10 ** int(dp[j]) + 1e-9) / 10 ** int(dp[j])
            if abs(q[j]) < abs(base[j]):
                q[j] = base[j]
        w = q * px / equity
        rv = values(w)
        feasible = hard_feasible(w) and all(rv[k] <= max(values(w0)[k], risk[r] * headroom) + 1e-10
            for k, r in [('gross', 'gross_fraction'), ('net', 'net_fraction'), ('daily_volatility', 'daily_volatility')])
        score = utility(w)
        trials.append(dict(multiplier=float(multiplier), feasible=bool(feasible), utility=score, **rv))
        # Strict improvement only: the incumbent wins ties, avoiding extra churn.
        if feasible and score > best['utility'] + 1e-12:
            best = dict(qty=q, multiplier=float(multiplier), utility=score)
    selected = best['qty']
    if np.any(selected[~eligible] != base[~eligible]):
        raise AssertionError('Overlay changed a protected/ineligible target')
    return dict(target_qty=selected.tolist(), selected_target_multiplier=best['multiplier'],
                eligible=eligible.tolist(), signed_edge=edge.tolist(),
                forecast_utility_gain=best['utility'] - baseline_utility,
                before=values(w0), after=values(selected * px / equity), trials=trials,
                claim='best forecast utility on the declared grid; not optimal future PnL')


def select_margin(*, own_notional, teammate_notional, own_equity, teammate_equity,
                  current_leverage, exchange_max, leverage_ceiling, margin_fraction=0.8,
                  shared_setting_authorized=False):
    """Minimum common integer leverage meeting both separate margin budgets.

    Individual notional here is signed. Opposing virtual ownership consumes
    budget separately: netting never lends the teammate's reserved capital.
    Leverage is a shared per-market exchange setting, not a per-owner setting.
    This returns advice only. Exchange margin/tiers, OI, liquidity and concurrent
    changes must still pass fresh preflight before any setting or order update.
    """
    own = np.asarray(own_notional, float)
    if own.ndim != 1 or not len(own) or not np.isfinite(own).all():
        raise ValueError('Invalid notional')
    n = len(own)
    other = _vector(teammate_notional, n, 'teammate notional')
    current = _vector(current_leverage, n, 'current leverage')
    maximum = _vector(exchange_max, n, 'exchange maximum')
    if (not np.isfinite([own_equity, teammate_equity, margin_fraction, leverage_ceiling]).all()
            or min(own_equity, teammate_equity) <= 0 or not 0 < margin_fraction <= 1
            or leverage_ceiling < 1 or leverage_ceiling != int(leverage_ceiling)
            or np.any(current < 1) or np.any(maximum < 1)
            or np.any(current != np.floor(current)) or np.any(maximum != np.floor(maximum))
            or np.any(current > maximum)):
        raise ValueError('Invalid margin constraints')
    active = (own != 0) | (other != 0)
    budget = np.array([own_equity, teammate_equity]) * margin_fraction
    chosen = None
    for level in range(1, int(leverage_ceiling) + 1):
        proposal = np.minimum(level, maximum)
        proposal[~active] = current[~active]
        margin = np.array([np.sum(abs(own) / proposal), np.sum(abs(other) / proposal)])
        if np.all(margin <= budget + 1e-8):
            chosen = proposal.astype(int)
            break
    if chosen is None:
        return dict(feasible=False, reason='Margin budget exceeds allowed leverage; reduce target',
                    target_leverage=None, live_ready=False)
    changed = active & (chosen != current)
    shared = changed & (other != 0)
    return dict(feasible=True, target_leverage=chosen.tolist(), changes=changed.tolist(),
                shared_setting_changes=shared.tolist(),
                setting_change_permitted=bool(shared_setting_authorized or not shared.any()),
                own_margin=float(np.sum(abs(own) / chosen)),
                teammate_margin=float(np.sum(abs(other) / chosen)),
                physical_margin_estimate=float(np.sum(abs(own + other) / chosen)),
                own_actual_gross_leverage=float(abs(own).sum() / own_equity),
                team_actual_gross_leverage=float(abs(own + other).sum() / (own_equity + teammate_equity)),
                live_ready=False, reason='Unsigned margin proposal; fresh exchange preflight still required')
