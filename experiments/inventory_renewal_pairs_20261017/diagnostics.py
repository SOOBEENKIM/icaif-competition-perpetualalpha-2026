"""Attribute renewal gains without treating avoided costs as synthetic PnL."""
import gzip,json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.complementary_alpha_20261017.research import protected
from .research import control_directory

HERE=Path(__file__).resolve().parent;ART=HERE/'artifacts'


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def main():
    results=json.loads((ART/'results.json').read_text());frozen=json.loads((ART/'frozen.json').read_text())
    for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    rd,_,_=inputs(cfgs['reference'])
    with np.load(INPUT/'inputs.npz') as z:pd={k:z[k].copy() for k in z.files}
    data={'reference':rd,'par':pd};rows=[];audit_error=0.
    for row in results['rows']:
        arm=row['arm']
        if '_renew_' not in arm:continue
        venue,phase,scenario=row['venue'],row['phase'],row['scenario'];matched=arm.replace('_renew_','_')
        key=f'{venue}-{phase}-{arm}-{scenario}';prior_key=f'{venue}-{phase}-{matched}-{scenario}'
        run=read(ART/'runs'/f'{key}.json.gz');prior=read(control_directory(matched)/'runs'/f'{prior_key}.json.gz')
        audits={name:episode_attribution(r,data[venue],cfgs[venue]['scenarios'][scenario]) for name,r in [('renewed',run),('fixed',prior)]}
        totals={name:{k:sum(e[k] for e in a['episodes']) for k in ['price_move','execution_cost','fees','funding','net_pnl','fills']} for name,a in audits.items()}
        audit_error=max(audit_error,*(a['accounting_error'] for a in audits.values()))
        delta=dict(pair_price_change=totals['renewed']['price_move']-totals['fixed']['price_move'],
            pair_execution_cost_saved=totals['fixed']['execution_cost']-totals['renewed']['execution_cost'],
            pair_fees_saved=totals['fixed']['fees']-totals['renewed']['fees'],
            pair_funding_change=totals['renewed']['funding']-totals['fixed']['funding'],
            pair_net_change=totals['renewed']['net_pnl']-totals['fixed']['net_pnl'],
            base_owned_change=run['metrics']['base_owned_pnl']-prior['metrics']['base_owned_pnl'],
            account_net_change=run['metrics']['net_pnl']-prior['metrics']['net_pnl'])
        assert abs(delta['pair_net_change']-sum(delta[k] for k in ['pair_price_change','pair_execution_cost_saved','pair_fees_saved','pair_funding_change']))<1e-6
        assert abs(delta['account_net_change']-delta['pair_net_change']-delta['base_owned_change'])<1e-6
        renewals=[e for e in run['events'] if e['event']=='renewed'];rejected=Counter(e['reason'] for e in run['events'] if e['event']=='renewal_rejected')
        for e in renewals:
            # Renewal is a state decision, never a fresh pair fill packet.
            assert not any(d.get('pair_entry') and d['decision_time']==e['time'] for d in run['decisions'])
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,matched=matched,
            renewal_events=len(renewals),rejected=dict(rejected),delta=delta,owner_totals=totals,
            actual_completed_pairs=run['metrics']['completed_pairs'],
            residual_gross=run['metrics']['residual_gross'],pnl_after_exit_reserve=run['metrics']['pnl_after_liquidation_reserve'],
            episodes=audits['renewed']['episodes']))
    save(ART/'attribution.json',dict(rows=rows,post_evaluation_only=True,no_retrospective_fee_subtraction=True,
        maximum_episode_accounting_error=audit_error,protected=protected(),orders_sent=0))
    for row in rows:
        if row['venue']=='reference' and row['phase']=='comparison' and row['arm'] in ['first_dense_renew_ridge_24','first_dense_renew_bias_24','first_dense_renew_affine_24']:
            print(row['arm'],row['scenario'],row['delta'],'rejected',row['rejected'],flush=True)


if __name__=='__main__':main()
