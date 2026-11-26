"""Pair attribution and risk-trigger audit without changing the tested universe."""
import json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from .research import ROOT,HERE,OUT,GATE,readrun,save,sha,verify,protected


def main():
    spec=json.loads((HERE/'protocol.json').read_text());result=json.loads((OUT/'results.json').read_text())
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/reference_protocol.json').read_text());d,_,p=inputs(cfg)
    files=[Path(__file__),OUT/'results.json',OUT/'selection.json',
           ROOT/'experiments/risk_neutral_pairs_20261017/attribution.py',*sorted((OUT/'runs').glob('reference-*.json.gz'))]
    sources={str(path.relative_to(ROOT)):sha(path) for path in files}
    verify(json.loads((OUT/'frozen.json').read_text())['sources']);verify(sources)
    save(OUT/'diagnostics_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),post_evaluation_only=True))
    rows=[]
    for phase in ['validation','comparison']:
        for sc in ['base','stress']:
            for kind in ['crypto','sector','ungated']:
                arm=f'first_dense_hourlyhedge_decision_{kind}_24'
                run=readrun(OUT/'runs'/f'reference-{phase}-{arm}-{sc}.json.gz')
                attribution=episode_attribution(run,d,cfg['scenarios'][sc]);totals=defaultdict(lambda:defaultdict(float))
                for e in attribution['episodes']:
                    pair='/'.join(e['pair'])
                    for k in ['price_move','execution_cost','fees','funding','net_pnl','fills']:totals[pair][k]+=e[k]
                    totals[pair]['episodes']+=1;totals[pair]['closed']+=int(e['closed'])
                assert abs(sum(v['net_pnl'] for v in totals.values())-run['metrics']['pair_owned_pnl'])<1e-6
                snaps={s['time']:s for s in run['snapshots']};breaches=[]
                for b in run['risk_breaches']:
                    now=b['time'];s=snaps[now]
                    intents=[x for x in run['decisions'] if x['decision_time']==now]
                    following=next((x for x in run['snapshots'] if x['time']>now),None)
                    breaches.append(dict(**b,account_snapshot=s,decision=intents[-1] if intents else None,
                         next_snapshot=following,forced_order_requested=bool(intents and intents[-1]['risk_forced'])))
                rows.append(dict(phase=phase,scenario=sc,kind=kind,metrics=run['metrics'],
                    by_pair={k:dict(v) for k,v in totals.items()},episodes=attribution['episodes'],risk_breaches=breaches))
                print(phase,sc,kind,'pair PnL',{k:round(v['net_pnl'],2) for k,v in totals.items()},
                      'risk breach bars',len(breaches),flush=True)
    verify(sources)
    save(OUT/'diagnostics.json',dict(rows=rows,protected=protected(),orders_sent=0,historical_independence=False,
         scope='Actual reference account pair-owner price/fee/impact/funding decomposition. Risk snapshot and next requested forced order are distinct from a guaranteed instant return inside limits. No post-hoc pair removal or candidate reselection.'))
    print('DONE pair and risk attribution',flush=True)


if __name__=='__main__':main()
