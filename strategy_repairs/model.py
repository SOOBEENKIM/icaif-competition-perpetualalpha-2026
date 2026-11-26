"""Separate directional changes from the risk-gate repair."""
import json
import numpy as np
from all_market_strategy.signals import (
    ROOT, FROZEN, SYMBOLS, channel_trend, ema_trend,
    Market, covariance, funding_weights, funding_signals, components, policies,
    reversion_features, reversion_signal, targets, limits,
)
from replay_lab.v10_revision.core import sizing
from replay_lab.competition_rebuild.regimes import paths
from replay_lab.competition_rebuild.persistence import episode_overlay


def apply_gate_after_sizing(sized_trend, attenuation):
    if not np.isfinite(attenuation).all() or np.any((attenuation<0)|(attenuation>1)):
        raise ValueError('Risk attenuation must be finite and between zero and one')
    return sized_trend * attenuation


def build(data, forecast, probability, arm):
    m = Market(data['market_times'],data['close'],data['volume'],data['trades'],
               data['funding'],data['observed'],SYMBOLS)
    bundle=json.loads((FROZEN/'baseline.json').read_text())
    v7=bundle['signal_bundle']['v7'];core=v7['control_bundle']
    allowed=np.ones(10,bool) if arm['scope']=='all' else np.arange(10)<3
    channel,vol=channel_trend(data,allowed,symmetric=arm['short'])
    policy=dict(core['selected_ema']['policy']);policy['exit_cross']=arm['cross_exit']
    trend,states=ema_trend(data,vol,policy,allowed,symmetric=arm['short'])
    carry=funding_weights(m,.5*funding_signals(m)[1]['forecast'])
    base=((trend+channel)/2+carry)*core['scale']*v7['risk_scale']
    sigma=np.full((len(m.times),10,10),np.nan)
    for i in range(168,len(m.times)):
        sigma[i]=covariance(m,i,bundle['protocol']['risk'])[1]
    parts,ratio=components(m,sigma,base,forecast,probability)
    allocation=json.loads((FROZEN/'allocation_fit.json').read_text())
    legacy_trend=policies(parts,allocation)['v10_frozen']
    sized=sizing(base,sigma,m.times)
    repaired=apply_gate_after_sizing(sized,ratio)
    selected=repaired if arm['risk'] else legacy_trend
    f=reversion_features(m,7)
    reversal=reversion_signal(m,f,7,'all' if arm['scope']=='all' else 'crypto','aligned')
    raw=np.nan_to_num(reversal/np.maximum(f['vol'],.003),nan=0)
    reversion=.9*targets(raw,sigma,m.times,days=3,risk=.001)
    profiles=json.loads((ROOT/'strategy/profiles.json').read_text())
    scale=profiles['original' if arm['original_size'] else 'default']['total_scale']
    pre_gate=limits(sized+.5*reversion)*scale
    final=limits(selected+.5*reversion)*scale
    # This mask is only a genuine reduction of the NET base-owner exposure.
    # A separate opposite-sign reversion position is not mislabelled as a cut.
    protected=(arm['risk'] & (ratio<1-1e-12) & (pre_gate*final>=0)
               & (abs(final)<abs(pre_gate)-1e-12))
    if not np.isfinite(final).all():
        raise ValueError('Invalid final weights')
    return final, dict(times=m.times, ema=states, channel=channel, carry=carry,
        raw_base=base, gate_ratio=ratio, legacy_sized_trend=legacy_trend,
        ungated_sized_trend=sized, repaired_sized_trend=repaired,
        reversion_state=reversal,reversion_weight=reversion,pre_gate=pre_gate,
        final=final,protected=protected,sigma=sigma,allowed=allowed,
        final_scale=scale)
