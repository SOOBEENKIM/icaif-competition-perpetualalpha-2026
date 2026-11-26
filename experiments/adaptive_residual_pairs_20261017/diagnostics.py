"""Post-evaluation attribution and costly close/re-entry pattern counts."""
import gzip,json
from pathlib import Path
import numpy as np
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts_v2'


def same_direction_reentries(run,maximum_gap=21600):
    intents=[e for e in run['events'] if e['event']=='pair_intent'];result=[]
    step=run['snapshots'][1]['time']-run['snapshots'][0]['time']
    for previous,new in zip(intents,intents[1:]):
        closes=[e for e in run['events'] if e['event']=='closed' and previous['time']<e['time']<=new['time']]
        if not closes or not closes[-1]['balanced']:continue
        close=closes[-1]
        if previous['pair']!=new['pair'] or previous['direction']!=new['direction'] or new['time']-close['time']>maximum_gap:continue
        # A new forecast at the close timestamp was NOT known one step earlier
        # when the old scheduled close was submitted. Do not use this hindsight
        # pattern to manufacture a renewal decision at that earlier time.
        result.append(dict(pair=new['pair'],direction=new['direction'],old_intent=previous['time'],
            closed_at=close['time'],new_intent=new['time'],gap_seconds=new['time']-close['time'],
            old_source=previous['forecast_source'],new_source=new['forecast_source'],
            new_source_available_at_previous_close_decision=new['forecast_source']<=close['time']-step))
    return result


def main():
    results=json.loads((ART/'results.json').read_text());frozen=json.loads((ART/'frozen.json').read_text())
    for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    rd,_,_=inputs(cfgs['reference'])
    with np.load(INPUT/'inputs.npz') as z:pd={k:z[k].copy() for k in z.files}
    data={'reference':rd,'par':pd};attributions=[];reentries=[]
    for row in results['rows']:
        if row['arm'] not in cfgs[row['venue']]['candidates']:continue
        key='-'.join(row[k] for k in ['venue','phase','arm','scenario'])
        with gzip.open(ART/'runs'/f'{key}.json.gz','rt') as f:run=json.load(f)
        audit=episode_attribution(run,data[row['venue']],cfgs[row['venue']]['scenarios'][row['scenario']])
        intents=[e for e in run['events'] if e['event']=='pair_intent']
        groups={name:dict(price_move=0.,execution_cost=0.,fees=0.,funding=0.,net_pnl=0.,fills=0,episodes=0)
            for name in ['unfilled','balanced_closed','aborted_closed','residual_open']}
        for episode,intent in zip(audit['episodes'],intents):
            closes=[e for e in run['events'] if e['event']=='closed' and e['time']==episode['end']]
            category='unfilled' if episode['fills']==0 else ('residual_open' if not closes else ('balanced_closed' if closes[0]['balanced'] else 'aborted_closed'))
            episode.update(category=category,expected_net_cash_at_intent=intent['expected_net_cash'],source_price_alpha=intent['source_price_alpha'])
            for name in groups[category]:groups[category][name]+=1 if name=='episodes' else episode[name]
        aggregate={k:sum(e[k] for e in audit['episodes']) for k in ['price_move','execution_cost','fees','funding','net_pnl','fills']}
        assert abs(aggregate['net_pnl']-run['metrics']['pair_owned_pnl'])<1e-6
        identity={k:row[k] for k in ['venue','phase','arm','scenario']}
        attributions.append(dict(**identity,**audit,aggregate=aggregate,groups=groups))
        repeats=same_direction_reentries(run)
        reentries.append(dict(**identity,patterns=repeats,pattern_count=len(repeats),
            not_a_counterfactual_backtest=True,no_avoided_cost_claim=True))
    fits=json.loads((ART/'fits.json').read_text());summaries=[]
    for venue in cfgs:
        for h in [24,72]:
            for k,pair in enumerate(cfgs[venue]['dense_pairs']):
                rr=[x for x in fits if x['venue']==venue and x['horizon']==h and x['pair_index']==k]
                assert all(x['last_maturity']<x['time'] for x in rr)
                summaries.append(dict(venue=venue,horizon=h,pair=pair,decisions=len(rr),
                    min_effective_rows=min(x['effective_rows'] for x in rr),max_effective_rows=max(x['effective_rows'] for x in rr),
                    minimum_slope=min(x['affine_slope'] for x in rr),maximum_slope=max(x['affine_slope'] for x in rr),
                    minimum_bias=min(x['bias'] for x in rr),maximum_bias=max(x['bias'] for x in rr)))
    save(ART/'attribution.json',dict(rows=attributions,not_used_for_selection=True,orders_sent=0))
    save(ART/'reentry_patterns.json',dict(rows=reentries,descriptive_only=True,orders_sent=0))
    save(ART/'diagnostic_status.json',dict(attribution_runs=len(attributions),calibration_summary=summaries,
        maximum_episode_accounting_error=max(x['accounting_error'] for x in attributions),protected=protected(),orders_sent=0))
    for x in attributions:
        if x['venue']=='reference' and x['phase']=='comparison' and x['arm'] in ['first_dense_bias_24','first_dense_bias_72','first_dense_affine_72']:
            print(x['arm'],x['scenario'],'aggregate',x['aggregate'],'groups',x['groups'],flush=True)


if __name__=='__main__':main()
