"""Post-result mechanism checks, never used to change the candidate."""
import copy,gzip,json
import numpy as np
from .prepare import ROOT,OUT,START,END
from .engine import simulate
from pas_adaptive_development.study import save,native


def main():
    from competition_live.profile import load_profile
    from pas_strategy_audit.data import load
    from joint_wallet.audit import reconstruct
    p=json.loads((OUT/'protocol.json').read_text());profile=load_profile()
    cfg,d=load(ROOT/'reports/pas_allocation_study_20261021/pas');cfg['cutoff_ms']=END*1000
    events=json.loads((OUT/'events.json').read_text());tape=json.loads((OUT/'trade_tape.json').read_text())
    funding=json.loads((OUT/'funding_tape.json').read_text());known={(r['coin'],r['time']//3600000) for r in funding}
    for r in d['funding_events']:
        coin='pas:'+cfg['symbols'][r['j']]
        if (coin,r['time']//3600000) not in known:funding.append(dict(coin=coin,time=r['time'],rate=r['rate'],cash_per_unit=None))
    save(OUT/'diagnostic_protocol.json',dict(post_result=True,rule='Exactly2x incumbent raw weights, same hard risk caps and same execution. No parameter search.',
        purpose='Determine whether PAS gains are merely higher exposure. Does not replace the frozen candidate or primary result.'))
    scenarios=copy.deepcopy(p['scenarios']);scenarios['optimistic_100pct']={**scenarios['base_1pct'],'participation':1.}
    results=[]
    for name,sc in scenarios.items():
        signal=copy.deepcopy(events)
        for e in signal:e['weights']=(2*np.asarray(e['weights'])).tolist()
        r=simulate(cfg,d,signal,tape,funding,profile['trading'],sc)
        results.append(dict(arm='incumbent_2x_diagnostic',scenario=name,**r['metrics']))
        with gzip.open(OUT/f'incumbent_2x_diagnostic_{name}.json.gz','wt') as f:json.dump(r,f,default=native,allow_nan=False)
    save(OUT/'diagnostic_results.json',results)
    # Actual executed quantities/prices/fees, with the SAME end trade-price
    # valuation as simulations, to check execution fidelity without mixing marks.
    own=ROOT/'reports/joint_wallet_20261022';fills=[r for r in json.loads((own/'public_fills.json').read_text()) if r['time']<=END*1000]
    userfund=[r for r in json.loads((own/'public_funding.json').read_text()) if r['time']<=END*1000]
    b=reconstruct(profile['trading']['symbols'],{'ours':50000.,'teammate':50000.},fills,userfund,set(),{63128571918,63128575185,63240249201})
    index=int(np.flatnonzero(d['times']==END*1000)[0]);prices=d['close'][index]
    ledger=b.owners['ours'];actual=dict(mode='actual own fills/fees/funding; endpoint trade-price valuation proxy',
        net_pnl=ledger.equity(prices)-50000.,fees=ledger.fees,funding=ledger.funding,
        qty=ledger.qty.tolist(),prices=prices.tolist(),first_fill=min(f['time'] for f in fills if f['oid']==63128571918),
        actual_fill_rows=sum(f['oid'] in {63128571918,63128575185,63240249201} for f in fills),
        teammate_excluded=True,official_mark_equity=False)
    save(OUT/'actual_execution_reference.json',actual)
    component=[]
    for name in scenarios:
        with gzip.open(OUT/f'budget_lifecycle_{name}.json.gz','rt') as f:r=json.load(f)
        trace=r['allocation_trace'];additions=[]
        for v in trace:
            free=np.asarray(v['free']);target=np.asarray(v['target']);old=np.asarray(v['old'])
            if np.any(free&(abs(target)>abs(old)+1e-9)):
                additions.append(dict(source=v['time'],symbols=[profile['trading']['symbols'][j] for j in np.flatnonzero(free&(abs(target)>abs(old)+1e-9))]))
        component.append(dict(scenario=name,free_sleeve_additions=additions,nonzero_pair_entries=0,
            short_opening_fills=sum(f['signed_qty']<0 and not f['reduce_only'] for f in r['fills']),
            final_target=r['decisions'][-1]['target_qty']))
    save(OUT/'component_attribution.json',component)
    print(json.dumps(dict(diagnostic=results,actual=actual,components=component)))


if __name__=='__main__':main()
