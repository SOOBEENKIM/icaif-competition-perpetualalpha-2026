"""Post-selection causal ablation: same joint mu, original +/-mu representation."""
import copy,gzip,json,time
from pathlib import Path
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as PAR_INPUT,save,sha
from experiments.conditional_completion_pairs_20261017.portfolio import replay
from .research import HERE,OUT,EXT,verify


def main():
    out=OUT/'projection_ablation';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    result=json.loads((OUT/'results.json').read_text());cfgs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    cfgs['par_extended']=copy.deepcopy(cfgs['par'])
    files=[Path(__file__),OUT/'results.json',OUT/'selection.json',*(OUT/f'{v}_joint_state.npz' for v in cfgs)]
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy();sources.update({str(p.relative_to(ROOT)):sha(p) for p in files})
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,purpose='Diagnostic only after original candidate selection. Keep all newly predicted scalar pair returns unchanged but remove explicit per-leg prices, reverting to original +/-mu terminal prices in the unchanged conditional-cash engine. Do not select or promote this ablation from later-period outcomes.',orders_sent=0))
    d,_,p=inputs(cfgs['reference']);data={'reference':d,'par':load(PAR_INPUT/'inputs.npz'),'par_extended':load(EXT/'inputs.npz')}
    states={v:load(OUT/f'{v}_joint_state.npz') for v in cfgs}
    for s in states.values():s.pop('terminal_prices')
    rows=[]
    for venue,phase in result['windows']:
        cfg=copy.deepcopy(cfgs[venue]);cfg['conditional_completion']={'risk_aversion':0.}
        for sc in ['base','stress']:
            with gzip.open(OUT/'runs'/f'{venue}-{phase}-first_dense_hourlyhedge_joint_24-{sc}.json.gz','rt') as f:joint=json.load(f)
            arm='first_dense_hourlyhedge_jointscalar_24'
            run=replay(data[venue],states[venue],None,cfg,p,arm,sc,joint['start'],joint['end'])
            with gzip.open(out/'runs'/f'{venue}-{phase}-{sc}.json.gz','wt') as f:json.dump(run,f,allow_nan=False)
            changes={k:[joint['metrics'][k],run['metrics'][k]] for k in run['metrics'] if run['metrics'][k]!=joint['metrics'][k]}
            row=dict(venue=venue,phase=phase,scenario=sc,metrics=run['metrics'],changed_metrics=changes,
                     fills_identical=run['fills']==joint['fills'],snapshots_identical=run['snapshots']==joint['snapshots'])
            rows.append(row);print(venue,phase,sc,'PnL',run['metrics']['net_pnl'],'changed_metrics',changes,'same_fills',row['fills_identical'],flush=True)
    verify(sources)
    save(out/'results.json',dict(rows=rows,new_diagnostic_accounts=12,not_candidate_selection=True,not_eligible_for_promotion=True,
         orders_sent=0,historical_independence=False,live_strategy_changed=False,reserved_raw_read=False))
    print('DONE 12 diagnostic accounts',flush=True)


if __name__=='__main__':main()
