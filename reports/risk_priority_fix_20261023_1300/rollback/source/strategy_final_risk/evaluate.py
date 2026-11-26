"""Fixed two-arm comparison; conditional historical fills, no live orders."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from competition_live.profile import ROOT,BUNDLE,load_profile
from competition_live.scoring import evaluate
from replay_lab.common_comparison.models import Market
from strategy_repairs.engine import reference_replay
from .model import final_mix

OUT=ROOT/'reports/strategy_repairs_20261021'
OLD=ROOT/'reports/isolated_repairs_20261020'


def main():
    protocol=json.loads((OUT/'risk_protocol.json').read_text())
    frozen=json.loads((OLD/'protocol.json').read_text())
    run=ROOT/frozen['pointer']['run']
    data=dict(np.load(run/'inputs.npz',allow_pickle=False))
    saved=dict(np.load(run/'targets.npz',allow_pickle=False))
    part=dict(np.load(OLD/'risk_only_trace.npz',allow_pickle=False))
    market=Market(data['market_times'],data['close'],data['volume'],data['trades'],
                  data['funding'],data['observed'],tuple(load_profile()['trading']['symbols']))
    weights,trace=final_mix(market,part['ungated_sized_trend'],.5*part['reversion_weight'],
                           float(part['final_scale']),saved['frozen_regime_probability'],
                           saved['chronos_forecast'])
    end=len(market.times)-48
    prefix=Market(*(getattr(market,k)[:end] for k in ('times','close','volume','trades','funding','observed')),market.symbols)
    earlier,_=final_mix(prefix,part['ungated_sized_trend'][:end],.5*part['reversion_weight'][:end],
                       float(part['final_scale']),saved['frozen_regime_probability'][:end],saved['chronos_forecast'][:end])
    np.testing.assert_array_equal(earlier,weights[:end])
    np.savez_compressed(OUT/'final_risk_targets.npz',times=market.times,incumbent=saved['default'],
                        final_mix_risk=weights,sigma=part['sigma'],**trace)
    evidence=dict(prefix_error=float(abs(earlier-weights[:end]).max()),
        last_source=int(market.times[-1]),last_before=trace['pre_gate'][-1].tolist(),
        last_after=weights[-1].tolist(),last_incumbent=saved['default'][-1].tolist(),
        last_ratio=trace['ratio'][-1].tolist(),orders_sent=0,independent_holdout=False,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (run/'inputs.npz',run/'targets.npz',OLD/'risk_only_trace.npz',OUT/'risk_protocol.json')})
    (OUT/'risk_construction.json').write_text(json.dumps(evidence,indent=2))
    ref=dict(np.load(BUNDLE/'data/reference.npz',allow_pickle=False))
    state=dict(np.load(BUNDLE/'models/reference_gate_state.npz',allow_pickle=False))
    ix=np.searchsorted(market.times,ref['times'])
    np.testing.assert_array_equal(market.times[ix],ref['times'])
    np.testing.assert_allclose(saved['default'][ix],ref['first'],rtol=1e-11,atol=1e-13)
    profile=load_profile()
    expected=json.loads((BUNDLE/'expected.json').read_text())
    windows=[r for r in expected['runs'] if r['venue']=='reference' and r['arm']==profile['candidate']]
    rows=[]
    for window in windows:
        for name,w,protected,enabled in [
            ('incumbent',saved['default'],np.zeros_like(weights,dtype=bool),False),
            ('final_mix_risk',weights,trace['protected'],True)]:
            d=dict(ref);d['first']=w[ix]
            replay=reference_replay(d['times'],d['first'],protected[ix],enabled)
            cfg=copy.deepcopy(profile['configs']['reference'])
            cfg['conditional_completion']={'risk_aversion':0.}
            r=replay(d,state,None,cfg,profile['trading'],profile['candidate'],
                     window['scenario'],window['start'],window['end'])
            m=r['metrics']
            daily=[dict(time=s['time'],equity=s['equity']+50000) for s in r['snapshots'] if s['time']%86400==43200]
            team=evaluate(daily,trading_days=len({f['time']//86400 for f in r['fills']}))
            row=dict(variant=name,phase=window['phase'],scenario=window['scenario'],
                net_pnl=m['net_pnl'],Q=team['Q'],fees=m['fees'],base_pnl=m['base_owned_pnl'],
                pair_pnl=m['pair_owned_pnl'],completed_pairs=m['completed_pairs'],fills=m['fills'],
                accounting_error=m['maximum_accounting_error'],risk_breach_bars=m['risk_breach_bars'],
                timing_pass=all(f['source_available']<=f['decision_time']<f['time'] for f in r['fills']))
            assert row['accounting_error']<1e-6 and row['timing_pass']
            rows.append(row)
            (OUT/'risk_history_results.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
    index={(r['phase'],r['scenario'],r['variant']):r for r in rows}
    passes=all(index[phase,sc,'final_mix_risk']['net_pnl']>=index[phase,sc,'incumbent']['net_pnl']
               and index[phase,sc,'final_mix_risk']['Q']>=index[phase,sc,'incumbent']['Q']
               and index[phase,sc,'final_mix_risk']['risk_breach_bars']==0
               for phase,sc,_ in index)
    (OUT/'risk_decision.json').write_text(json.dumps(dict(
        historical_comparison_pass=passes,activate=False,
        reason='PAS net alpha and prospective execution not established; '+('historical criterion passed' if passes else 'historical criterion failed'),
        protocol=protocol,orders_sent=0),indent=2))


if __name__=='__main__':
    main()
