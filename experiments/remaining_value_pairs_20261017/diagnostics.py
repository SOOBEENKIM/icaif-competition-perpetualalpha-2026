"""Attribute completed candidate runs; no hindsight strategy changes."""
import json
import time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from .research import ROOT,HERE,OUT,PARENT,CONTROL,runpath,readrun,save,sha,verify,protected


def main():
    results=json.loads((OUT/'results.json').read_text())
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
    d,_,p=inputs(cfg)
    paths=[Path(__file__),OUT/'results.json',OUT/'selection.json',OUT/'reference_remaining_state.npz',
           ROOT/'experiments/risk_neutral_pairs_20261017/attribution.py',*sorted((OUT/'runs').glob('reference-*.json.gz'))]
    paths += [runpath('reference',phase,sc) for phase in ['validation','comparison'] for sc in ['base','stress']]
    sources={str(path.relative_to(ROOT)):sha(path) for path in paths}
    verify(json.loads((OUT/'frozen.json').read_text())['sources']);verify(sources)
    save(OUT/'diagnostics_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),post_evaluation_only=True))
    rows=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            parent=readrun(runpath('reference',phase,sc));old=parent['metrics']
            for kind in ['renewal','remaining']:
                arm=f'first_dense_hourlyhedge_remaining_{kind}_24'
                run=readrun(OUT/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz');m=run['metrics']
                decomposition={k:m[k]-old[k] for k in ['price_move_pnl','execution_cost','fees','funding','net_pnl','base_owned_pnl','pair_owned_pnl']}
                assert abs(decomposition['net_pnl']-(decomposition['price_move_pnl']-decomposition['execution_cost']-decomposition['fees']+decomposition['funding']))<1e-6
                attribution=episode_attribution(run,d,cfg['scenarios'][sc])
                original_attribution=episode_attribution(parent,d,cfg['scenarios'][sc])
                snaps={s['time']:s for s in run['snapshots']};observations=[]
                for event in run['events']:
                    early=event['event']=='remaining_value_check' and event.get('available')
                    veto=event.get('reason')=='remaining_model_renewal_veto'
                    if not early and not veto:continue
                    now=event['time'];i=int(np.searchsorted(d['times'],now));assert d['times'][i]==now
                    ids=[p['symbols'].index(symbol) for symbol in event['pair']]
                    if early:
                        maturity=event['source']+event['horizon_hours']*3600
                        q=np.array(event['owned_qty']);terminal=np.array(event['terminal_prices'])
                    else:
                        maturity=event['proposed_maturity']
                        q=np.array(snaps[now]['pair_qty'])[ids];terminal=np.array(event['joint_terminal_prices'])
                    j=int(np.searchsorted(d['times'],maturity))
                    if maturity>run['end'] or j>=len(d['times']) or d['times'][j]!=maturity:continue
                    marks=d['par_close'][i,ids];gross=float(abs(q)@marks)
                    if gross<=0:continue
                    predicted=float(q@(terminal-marks));observed=float(q@(d['par_close'][j,ids]-marks))
                    observations.append(dict(time=now,maturity=maturity,pair=event['pair'],
                        kind='holding_review' if early else 'renewal_veto_current_inventory_probe',
                        requested_close=bool(event.get('close',False)),owned_qty=q.tolist(),gross=gross,
                        predicted_price_cash=predicted,observed_fixed_inventory_price_cash=observed,
                        predicted_price_bps=predicted/gross*10000,observed_fixed_inventory_price_bps=observed/gross*10000,
                        is_executable_counterfactual=False))
                summaries=[]
                for name in ['holding_review','renewal_veto_current_inventory_probe']:
                    obs=[o for o in observations if o['kind']==name]
                    if not obs:continue
                    predicted=np.array([o['predicted_price_bps'] for o in obs]);actual=np.array([o['observed_fixed_inventory_price_bps'] for o in obs])
                    summaries.append(dict(kind=name,count=len(obs),rmse_bps=float(np.sqrt(np.mean((actual-predicted)**2))),
                        zero_price_change_rmse_bps=float(np.sqrt(np.mean(actual**2))),
                        mean_predicted_bps=float(predicted.mean()),mean_observed_bps=float(actual.mean()),
                        close_requested=sum(o['requested_close'] for o in obs),
                        closes_followed_by_positive_fixed_inventory_price_cash=sum(o['requested_close'] and o['observed_fixed_inventory_price_cash']>0 for o in obs)))
                rows.append(dict(phase=phase,scenario=sc,kind=kind,arm=arm,metrics=m,decomposition=decomposition,
                                 episodes=attribution['episodes'],control_episodes=original_attribution['episodes'],
                                 forecast_summaries=summaries,observations=observations))
                print(phase,sc,kind,'PnL delta',round(decomposition['net_pnl'],4),'price delta',round(decomposition['price_move_pnl'],4),
                      'cost+fee delta',round(decomposition['execution_cost']+decomposition['fees'],4),flush=True)
    verify(sources)
    save(OUT/'diagnostics.json',dict(rows=rows,protected=protected(),historical_independence=False,orders_sent=0,
         distinction='Forecast diagnostics hold actual current inventory fixed to the stated horizon and use future marks only. '
                     'They exclude simulated exit fills, future rehedges, fees and funding and are not realized trading PnL. '
                     'For rejected renewal adjustments, actual current inventory can differ from the proposed adjusted target; '
                     'the probe cannot validate the rejected target decision directly. Full-path cost decomposition uses actual replay ledgers.'))
    print('DONE attribution and conditional fixed-inventory price diagnostics',flush=True)


if __name__=='__main__':main()
