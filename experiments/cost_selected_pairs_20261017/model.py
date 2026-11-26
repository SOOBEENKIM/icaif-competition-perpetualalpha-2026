"""Select unchanged expert forecasts using strictly matured costed decisions."""
import numpy as np
from experiments.adaptive_residual_pairs_20261017.model_v2 import build
from experiments.dense_history_pairs_20261017.model import map_state


def proxy(reference, features, forecasts, spec):
    times = reference['market_times']; price = reference['close']
    funding = reference['funding']; observed = reference['observed']
    n, count, experts = forecasts.shape; h = spec['horizon']
    if not np.all(np.diff(times) == 3600) or h < 2:
        raise ValueError('Contiguous hourly reference and horizon >=2 required')
    if n != len(times) or count != len(features['pairs']):
        raise ValueError('Unaligned expert/source arrays')
    fee, impact = spec['fee_bps']/10000, spec['impact_bps']/10000
    actions = np.zeros_like(forecasts, dtype=int)
    preferred = np.zeros_like(actions)
    net = np.full_like(forecasts, np.nan)
    expected = np.full_like(forecasts, np.nan)
    valid = np.isfinite(forecasts).all(-1) & features['fresh']
    for k, ids in enumerate(features['pairs']):
        ids = list(ids)
        for i in np.flatnonzero(valid[:, k]):
            if i == 0 or not observed[i-1, ids].all():
                continue
            beta = features['beta'][i, k]
            if not np.isfinite(beta) or not .1 <= beta <= 10:
                continue
            q0 = np.array([1., -beta])/(1+beta)/price[i, ids]
            for j in range(experts):
                terminal = price[i, ids]*np.array([1+forecasts[i, k, j], 1-forecasts[i, k, j]])
                if not np.isfinite(terminal).all() or np.any(terminal <= 0):
                    continue
                choices = []
                for direction in [-1, 1]:
                    q = q0*direction
                    entry = price[i, ids]*(1+np.sign(q)*impact)
                    exit_price = terminal*(1-np.sign(q)*impact)
                    carry = -q*price[i, ids]*funding[i-1, ids]*(h-1)
                    carry = np.where(carry > 0, carry*spec['forecast_funding_receipt_multiplier'], carry).sum()
                    value = float(q@(exit_price-entry)-fee*(abs(q)@(entry+exit_price))+carry)
                    choices.append((value, direction))
                value, direction = max(choices)
                preferred[i, k, j] = direction
                expected[i, k, j] = value
                action = direction if value > spec['entry_buffer_bps']/10000 else 0
                actions[i, k, j] = action
                if i+h >= n or not (features['fresh'][i+1, k] and features['fresh'][i+h, k]):
                    continue
                if not observed[i+2:i+h+1, ids].all() or not np.isfinite(funding[i+2:i+h+1, ids]).all():
                    continue
                if action == 0:
                    net[i, k, j] = 0.
                    continue
                q = q0*action
                entry = price[i+1, ids]*(1+np.sign(q)*impact)
                exit_price = price[i+h, ids]*(1-np.sign(q)*impact)
                cash = -q[None, :]*price[i+2:i+h+1, ids]*funding[i+2:i+h+1, ids]
                cash = np.where(cash > 0, cash*spec['realized_funding_receipt_multiplier'], cash).sum()
                net[i, k, j] = float(q@(exit_price-entry)-fee*(abs(q)@(entry+exit_price))+cash)
    return dict(actions=actions, preferred=preferred, expected=expected, payoff=net,
                maturity=times+h*3600)


def choose(times, forecasts, ledger, fresh, spec):
    n, pairs, experts = forecasts.shape
    outputs = {kind: np.full((n, pairs), np.nan) for kind in spec['candidates']}
    chosen = {kind: np.full((n, pairs), -1, dtype=int) for kind in spec['candidates']}
    records = []
    maturity = ledger['maturity']
    if not np.array_equal(maturity, times+spec['horizon']*3600):
        raise ValueError('Proxy/forecast horizon mismatch')
    source_valid = np.isfinite(forecasts).all(-1) & fresh & np.isfinite(ledger['expected']).all(-1)
    valid_target = source_valid & np.isfinite(ledger['payoff']).all(-1)
    for i, now in enumerate(times):
        window = (times % (spec['horizon']*3600) == 0) & (maturity < now)
        window &= maturity >= now-spec['lookback_days']*86400
        for k in np.flatnonzero(source_valid[i]):
            ix = np.flatnonzero(window & valid_target[:, k])
            assert np.all(np.diff(times[ix]) >= spec['horizon']*3600)
            detail = {}
            for kind in spec['candidates']:
                scores = np.full(experts, -np.inf); counts = np.zeros(experts, int)
                if len(ix) >= spec['minimum_rows']:
                    for j in range(experts):
                        subset = ix
                        if kind == 'directional':
                            subset = ix[ledger['actions'][ix, k, j] == ledger['preferred'][i, k, j]]
                            if ledger['preferred'][i, k, j] == 0:
                                continue
                        counts[j] = np.count_nonzero(ledger['actions'][subset, k, j])
                        if counts[j] < spec['minimum_active_rows']:
                            continue
                        decay = np.exp2(-(now-maturity[subset])/(spec['half_life_days']*86400))
                        effective = decay.sum()**2/(decay@decay)
                        weights = decay/decay.sum()*effective
                        scores[j] = weights@ledger['payoff'][subset, k, j]/(effective+spec['prior_rows'])
                winner = int(np.argmax(scores))
                if np.isfinite(scores[winner]) and scores[winner] > 0:
                    chosen[kind][i, k] = winner
                    # Preserve original prediction multiplication order exactly.
                    outputs[kind][i, k] = forecasts[i, k, winner]
                detail[kind] = dict(selected=int(chosen[kind][i, k]),
                    scores=[float(s) if np.isfinite(s) else None for s in scores], active_rows=counts.tolist())
            records.append(dict(time=int(now), pair_index=int(k), common_rows=len(ix),
                                last_maturity=int(maturity[ix[-1]]) if len(ix) else None, choices=detail))
    return outputs, chosen, records


def infer(times, reference, features, models, pairs, expert_spec, spec):
    forecasts, _ = build(reference, features, models, pairs, spec['horizon'], expert_spec)
    issued = np.stack([forecasts[k] for k in spec['experts']], axis=-1)
    ledger = proxy(reference, features, issued, spec)
    outputs, chosen, records = choose(reference['market_times'], issued, ledger, features['fresh'], spec)
    states = {kind: map_state(times, reference, features, value, features['pairs'], source_hours=1, lifetime_hours=1)
              for kind, value in outputs.items()}
    controls = {kind: map_state(times, reference, features, forecasts[kind], features['pairs'], source_hours=1, lifetime_hours=1)
                for kind in spec['experts']}
    return states, controls, records, dict(times=reference['market_times'], issued=issued, **ledger,
        **{f'prediction_{kind}': value for kind, value in outputs.items()},
        **{f'selected_{kind}': value for kind, value in chosen.items()})
