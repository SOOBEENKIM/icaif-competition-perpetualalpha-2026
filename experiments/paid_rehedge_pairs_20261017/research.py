"""Frozen paid hedge adjustment comparisons; same forecasts and risk budget."""
import gzip,json,time
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs as reference_inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,ts,save,sha
from experiments.complementary_alpha_20261017.research import protected
from experiments.dense_history_pairs_20261017.research import load
from .portfolio import replay

HERE=Path(__file__).resolve().parent
DENSE=HERE.parent/'dense_history_pairs_20261017'
ADAPTIVE=HERE.parent/'adaptive_residual_pairs_20261017'
RENEWAL=HERE.parent/'inventory_renewal_pairs_20261017'


def control_directory(arm):
    if '_renew_' in arm:return RENEWAL/'artifacts'
    return DENSE/'artifacts' if arm in ['first_strategy','second_strategy'] or '_ridge_' in arm else ADAPTIVE/'artifacts_v2'


def main():
    out=HERE/'artifacts';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    for parent in [DENSE/'artifacts',ADAPTIVE/'artifacts_v2',RENEWAL/'artifacts']:
        frozen=json.loads((parent/'frozen.json').read_text())
        for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    paths=[*HERE.glob('*.py'),*HERE.glob('*protocol.json'),INPUT/'inputs.npz',
        ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',ROOT/'experiments/risk_preserving_20261016/artifacts/targets.npz',
        ROOT/cfgs['reference']['reference_run']/'inputs.npz',
        *[DENSE/n for n in ['portfolio.py','policy.py','completion.py']],
        ROOT/'experiments/shared_execution_pairs_20261017/account_v2.py',RENEWAL/'policy.py',HERE/'test_results.txt']
    for parent in [DENSE/'artifacts',ADAPTIVE/'artifacts_v2',RENEWAL/'artifacts']:paths.extend([parent/'frozen.json',parent/'results.json'])
    state_paths={}
    for venue,cfg in cfgs.items():
        prior=json.loads((RENEWAL/f'{venue}_protocol.json').read_text());excluded={'controls','candidates','experiment','selection','rehedge_policy'}
        assert {k:v for k,v in cfg.items() if k not in excluded}=={k:v for k,v in prior.items() if k not in excluded}
        for kind in ['ridge','bias','affine','recent']:
            for h in [24,72]:
                parent=DENSE/'artifacts' if kind=='ridge' else ADAPTIVE/'artifacts_v2'
                state_paths[venue,kind,h]=parent/f'{venue}_{kind}_{h}_state.npz';paths.append(state_paths[venue,kind,h])
        for phase in ['validation','comparison']:
            for arm in cfg['controls']:
                for scenario in cfg['scenarios']:paths.append(control_directory(arm)/'runs'/f'{venue}-{phase}-{arm}-{scenario}.json.gz')
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),historical_independence=False,
        orders_sent=0,previous_goal_turn='progress_inventory_renewal_compared_and_cost_attribution_complete_no_qualified_replacement'))
    rd,_,protocol=reference_inputs(cfgs['reference']);pd=load(INPUT/'inputs.npz');data={'reference':rd,'par':pd}
    states={key:load(path) for key,path in state_paths.items()};controls={};rows=[];reuse=[];runs={}
    def get_state(venue,arm):
        clean=arm.replace('_renewhedge_','_').replace('_renew_','_');_,_,kind,h=clean.split('_')
        return states[venue,kind,int(h)]
    def read_controls(venue,phase):
        cfg=cfgs[venue]
        for arm in cfg['controls']:
            for scenario in cfg['scenarios']:
                key=(venue,phase,arm,scenario);path=control_directory(arm)/'runs'/('-'.join(key)+'.json.gz')
                with gzip.open(path,'rt') as f:run=json.load(f)
                controls[key]=run;runs[key]=run
                rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=run['metrics']))
                reuse.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,path=str(path.relative_to(ROOT)),sha256=sha(path)))
    verification=[('reference','validation','first_dense_renew_bias_24','base'),
        ('reference','comparison','first_dense_renew_ridge_72','stress'),('par','validation','first_dense_renew_affine_72','base'),
        ('par','comparison','second_dense_renew_recent_72','stress')]
    verified=[]
    def verify_controls(venue,phase):
        cfg=cfgs[venue]
        for key in verification:
            if key[:2]!=(venue,phase):continue
            _,_,arm,scenario=key;old=controls[key]
            run=replay(data[venue],get_state(venue,arm),None,cfg,protocol,arm,scenario,old['start'],old['end'])
            for field in ['fills','snapshots','metrics','events']:assert run[field]==old[field],(key,field)
            verified.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,exact_fields=['fills','snapshots','metrics','events']))
    def record(venue,phase,arm,scenario):
        cfg=cfgs[venue];start=cfg['validation_start'] if phase=='validation' else cfg['comparison_start']
        end=cfg['comparison_start'] if phase=='validation' else cfg['end']
        run=replay(data[venue],get_state(venue,arm),None,cfg,protocol,arm,scenario,ts(start),ts(end));key=(venue,phase,arm,scenario);runs[key]=run
        renewals=[e for e in run['events'] if e['event']=='renewed'];assert len(renewals)==run['metrics']['renewals']
        assert all(e['new_source']<=e['time']<e['old_maturity']<e['new_maturity']<=e['hard_holding_deadline'] for e in renewals)
        assert all(e['entry_fees_charged_again']==0 for e in renewals)
        adjusted=[e for e in renewals if e['orders_required']>0]
        assert all(e['adjustment_decision_time']<e['time'] for e in adjusted)
        assert len(adjusted)==run['metrics']['hedge_adjustments_completed']
        with gzip.open(out/'runs'/('-'.join(key)+'.json.gz'),'wt') as f:
            json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
        rows.append(dict(venue=venue,phase=phase,arm=arm,scenario=scenario,metrics=run['metrics']))
        m=run['metrics'];print('-'.join(key),'PnL',round(m['net_pnl'],3),'Q',round(m['financial_score'],7),
            'closed',m['completed_pairs'],'renewals',m['renewals'],'adjusted',m['hedge_adjustments_completed'],'failed',m['hedge_adjustments_failed'],flush=True)
    read_controls('reference','validation');verify_controls('reference','validation')
    for arm in cfgs['reference']['candidates']:
        for scenario in cfgs['reference']['scenarios']:record('reference','validation',arm,scenario)
    ranking=[]
    for arm in cfgs['reference']['candidates']:
        checks=[];diff=[];strict_renewal=False
        matched=arm.replace('_renewhedge_','_');renewed=arm.replace('_renewhedge_','_renew_')
        for scenario in cfgs['reference']['scenarios']:
            m=runs['reference','validation',arm,scenario]['metrics']
            checks.extend([m['net_pnl']>0,m['completed_pairs']>=3,m['risk_breach_bars']==0])
            for base in ['first_strategy','second_strategy']:
                b=runs['reference','validation',base,scenario]['metrics'];diff.append(m['net_pnl']-b['net_pnl'])
                checks.extend([m['net_pnl']>b['net_pnl'],m['financial_score']>b['financial_score']])
            for base in [matched,renewed]:
                b=runs['reference','validation',base,scenario]['metrics']
                checks.extend([m['net_pnl']>=b['net_pnl'],m['financial_score']>=b['financial_score']])
                if base==renewed:strict_renewal|=m['net_pnl']>b['net_pnl'] and m['financial_score']>b['financial_score']
        ranking.append(dict(arm=arm,eligible=all(checks) and strict_renewal,minimum_pnl_advantage=min(diff),strict_renewal_improvement=strict_renewal))
    ranking.sort(key=lambda x:(-x['eligible'],-x['minimum_pnl_advantage'],x['arm']))
    selection=dict(selected=ranking[0]['arm'],eligible=ranking[0]['eligible'],ranking=ranking,
        frozen_before_later_account_comparison=True,historical_independence=False)
    save(out/'selection.json',selection);selection_hash=sha(out/'selection.json')
    for venue,phase in [('reference','comparison'),('par','validation'),('par','comparison')]:
        read_controls(venue,phase);verify_controls(venue,phase)
        for arm in cfgs[venue]['candidates']:
            for scenario in cfgs[venue]['scenarios']:record(venue,phase,arm,scenario)
    assert sha(out/'selection.json')==selection_hash
    for rel,digest in sources.items():assert sha(ROOT/rel)==digest,rel
    save(out/'control_reuse.json',reuse);save(out/'control_verification.json',verified)
    save(out/'results.json',dict(rows=rows,selection=selection,new_candidate_replays=len(rows)-len(reuse),
        control_verification_replays=len(verified),reused_control_accounts=len(reuse),
        matched_forecast_inputs_unchanged=True,protected=protected(),orders_sent=0,
        automatic_live_replacement=False,historical_independence=False))
    print('DONE',len(rows)-len(reuse),'candidate replays;',len(verified),'control verification;',len(reuse),'controls; eligible',selection['eligible'],flush=True)


if __name__=='__main__':main()
