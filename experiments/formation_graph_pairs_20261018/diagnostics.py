"""Explain measured changes; all diagnostic removals are explicitly post hoc."""
import copy
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from .research import ROOT,HERE,OUT,GATE,STABILITY,inputs,load,readrun,plain,replay,save,sha,verify,protected
from .formation import mask_entries


def main():
    spec=json.loads((HERE/'protocol.json').read_text())
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text())
    d,_,p=inputs(cfg)
    files=[Path(__file__),OUT/'results.json',OUT/'selection.json',
           ROOT/'experiments/risk_neutral_pairs_20261017/attribution.py',
           *sorted((OUT/'runs').glob('reference-*.json.gz'))]
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy()
    sources.update({str(path.relative_to(ROOT)):sha(path) for path in files});verify(sources)
    save(OUT/'diagnostics_frozen.json',dict(at_unix=time.time(),sources=sources,post_evaluation_only=True))
    rows=[];runs={}
    for phase in ['validation','comparison','continuous']:
        for sc in ['base','stress']:
            for kind,arm,directory in [('candidate',spec['candidate'],OUT),
                ('retained',spec['control'],STABILITY if phase=='continuous' else GATE)]:
                run=readrun(directory/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz')
                runs[phase,sc,kind]=run
                audit=episode_attribution(run,d,cfg['scenarios'][sc])
                entries=[e for e in run['events'] if e['event']=='pair_intent']
                totals=defaultdict(lambda:defaultdict(float))
                for episode,entry in zip(audit['episodes'],entries):
                    episode['expected_net_cash_at_entry']=entry['expected_net_cash']
                    episode['duration_hours']=(episode['end']-episode['time'])/3600
                    pair='/'.join(episode['pair'])
                    for key in ['price_move','execution_cost','fees','funding','net_pnl','fills']:
                        totals[pair][key]+=episode[key]
                    totals[pair]['episodes']+=1
                rows.append(dict(phase=phase,scenario=sc,kind=kind,metrics=run['metrics'],
                                 by_pair={k:dict(v) for k,v in totals.items()},**audit))
                print('ATTRIBUTION',phase,sc,kind,{k:round(v['net_pnl'],6) for k,v in totals.items()},flush=True)
    # A causal implementation check, not a newly selected strategy: undo only
    # the two added equity/ETF entry permissions and verify the retained path.
    six=load(ROOT/'experiments/decision_sector_pairs_20261018/artifacts/reference_sector_state.npz')
    ablated=mask_entries(six,[True,True,False,False,False,False])
    ablations=[]
    for sc in ['base','stress']:
        old=runs['comparison',sc,'retained']
        c=copy.deepcopy(cfg);c['conditional_completion']={'risk_aversion':0.};c['dense_pairs']=spec['pairs']
        result=plain(replay(d,ablated,None,c,p,spec['candidate'],sc,old['start'],old['end']))
        for key in old:
            if key!='arm':assert result[key]==old[key],(sc,key)
        ablations.append(dict(scenario=sc,all_account_fields_exact_except_arm=True,
                              restored_pnl=old['metrics']['net_pnl'],restored_q=old['metrics']['financial_score']))
    # Report old control entry times while an added ETF episode was occupying
    # the sole pair slot. This timing observation does not itself prove that
    # a replacement would have filled profitably.
    overlaps=[]
    for phase in ['validation','comparison','continuous']:
        for sc in ['base','stress']:
            candidate=next(r for r in rows if (r['phase'],r['scenario'],r['kind'])==(phase,sc,'candidate'))
            retained=runs[phase,sc,'retained']
            for e in candidate['episodes']:
                if not set(e['pair']).intersection({'SP500','EWY','EWJ'}):continue
                rival=[x for x in retained['events'] if x['event']=='pair_intent' and e['time']<x['time']<e['end']]
                if rival:
                    overlaps.append(dict(phase=phase,scenario=sc,episode=e,
                      retained_entries=[{k:x[k] for k in ['time','pair','expected_net_cash']} for x in rival]))
    verify(sources)
    save(OUT/'diagnostics.json',dict(rows=rows,post_hoc_ablation=ablations,overlapping_control_entries=overlaps,
        extra_diagnostic_replays=2,protected=protected(),orders_sent=0,
        scope='Attribution includes realized and final marked inventory. Sum of episode cash/price/fee/funding equals pair-owned equity. Removing ETF entry permission restores retained account exactly in both later reference scenarios. Overlap observations are not forecasts or proof that arbitrary preemption improves PnL. No revised selection.',
        historical_independence=False))
    print('DONE attribution and 2 exact diagnostic ablations',flush=True)


if __name__=='__main__':main()
