"""Frozen research candidate contract; contains no live sender or signer."""
import copy
from competition_live.profile import load_profile
from auto_pair_selection.selection import PAIRS
from pair_value_learning.candidate import snapshot as coherent_snapshot

NAME='coherent_pairs_only_72'


def research_contract():
    profile=load_profile()
    cfg=copy.deepcopy(profile['configs']['reference'])
    cfg['dense_pairs']=[list(pair) for pair in PAIRS]
    cfg['conditional_completion']={'risk_aversion':0.}
    # Original economic cash limits remain binding. This is the existing
    # 10000bp secondary cap, not an instruction to accept arbitrary prices.
    assert cfg['entry_slippage_cap_bps']==10000
    assert cfg['pair_max_gross']==6000
    return dict(candidate=NAME,arm='pairs_dense_hourlyhedge_individual_72',
        include_directional_core=False,config=cfg,trading=profile['trading'],
        team_initial=100000.,sleeve_initial=50000.,idle_comparison_reserve=50000.,
        forecast_horizon_hours=72,universe_pairs=45,maximum_active_pairs=1,
        nomination='Selected from reused research comparisons, not an untouched test',
        live_connected=False,orders_sent=0)


def snapshot(data,source,received_at):
    result=coherent_snapshot(data,source,received_at)
    result.update(candidate=NAME,research_contract=research_contract(),
        execution_contract='Pair-only; original source hedge and economic limits; '
        '72h source maturity, paid renewal up to168h, one active pair. '
        'Forecasts are not executable orders; current venue/account checks required.')
    return result
