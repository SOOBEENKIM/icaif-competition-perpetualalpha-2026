"""Read-only execution compatibility checks for the extended pair universe.

No Exchange, signer, service manager, checkpoint writer or order sender exists
here. The output is a reviewable intent preview, not a live installation.
"""
import copy
import numpy as np
from .selection import PAIRS, IDS, SYMBOLS
from competition_live.profile import load_profile
from experiments.forecast_validity_pairs_20261017.book_plan import plan_entry


def expanded_profile():
    profile = copy.deepcopy(load_profile())
    profile['pairs'] = [list(p) for p in PAIRS]
    profile['pair_selection_revision'] = 'automatic_45_v1'
    for cfg in profile['configs'].values():
        cfg['dense_pairs'] = [list(p) for p in PAIRS]
    return profile


def validate_forecast(f, now):
    pair = tuple(f['pair'])
    if pair not in PAIRS:
        raise ValueError('Unknown/counter-ordered pair identity')
    k = PAIRS.index(pair)
    if tuple(f['ids']) != IDS[k] or f.get('stable_pair_index', k) != k:
        raise ValueError('Pair identity and execution asset IDs differ')
    source = f['forecast_source']; selected_at = f['selection_source']
    if not (source <= f['available_time'] <= now < source+3600
            and selected_at % 86400 == 0 and selected_at <= source < selected_at+86400):
        raise ValueError('Unavailable/expired forecast or selection')
    if f['horizon'] != 24 or f['model'] != 'decision_gate':
        raise ValueError('Unexpected forecast contract')
    anchors = np.asarray(f['source_prices']); terminal = np.asarray(f['terminal_prices'])
    if (anchors.shape != (2,) or terminal.shape != (2,)
            or not np.isfinite([*anchors, *terminal, f['mu'], f['source_beta'], f['source_vol']]).all()
            or np.any(anchors <= 0) or np.any(terminal <= 0)
            or not .25 <= f['source_beta'] <= 4 or f['source_vol'] <= 0):
        raise ValueError('Invalid price/hedge forecast')
    if not np.allclose(terminal, anchors*np.array([1+f['mu'], 1-f['mu']]), rtol=0, atol=1e-10):
        raise ValueError('Source forecast and terminal-price hypothesis differ')
    if f['economic_action'] not in (-1, 0, 1):
        raise ValueError('Unknown economic action')
    if not f['selection_eligible'] and (f['economic_available'] or f['economic_action'] != 0):
        raise ValueError('Excluded pair acquired entry permission')
    return k


def preview_entries(forecasts, quotes, now, cfg, protocol, scenario, qty, equity,
                    sigma, marks, owned_qty, *, active_pair=None, baseline_target=None):
    """Rank current executable expected cash with the retained planner.

    Keeping an active pair takes precedence over selecting another. Actual
    fills, exits and the unknown-outcome lock remain the runner's responsibility.
    """
    from replay_lab.common_comparison.engine import risk_breaches
    reviews, choices, seen = [], [], set()
    for forecast in forecasts:
        k = validate_forecast(forecast, now)
        if k in seen:raise ValueError('Duplicate pair publication')
        seen.add(k)
        if active_pair is not None:
            reviews.append(dict(pair=forecast['pair'], reason='active_pair_lifecycle_retained'))
            continue
        if not forecast['economic_available'] or forecast['economic_action'] == 0:
            reviews.append(dict(pair=forecast['pair'], reason='formation_or_model_gate'))
            continue
        intent = dict(forecast, direction=forecast['economic_action'], account_marks=np.asarray(marks).tolist())
        result = plan_entry(intent, quotes, now, cfg, protocol, scenario, qty, equity, sigma, owned_qty=owned_qty)
        # Include proposed baseline inventory, not merely last filled positions.
        if result['accepted'] and baseline_target is not None:
            q = np.asarray(baseline_target).copy()+np.asarray(owned_qty)
            q[forecast['ids']] += np.asarray(result['target'])
            if risk_breaches(q, equity, np.asarray(marks), sigma, protocol['risk']):
                result = dict(result, accepted=False, reason='combined_base_target_risk', orders=[])
        reviews.append(dict(pair=forecast['pair'], **result))
        if result['accepted']:choices.append((intent, result))
    chosen = max(choices, key=lambda row:(row[1]['expected_net_cash'], row[0]['pair'])) if choices else None
    return dict(selected_intent=chosen[0] if chosen else None,
                selected_preview=chosen[1] if chosen else None,
                candidates=reviews, orders_sent=0,
                mode='preview_only; no exchange action')
