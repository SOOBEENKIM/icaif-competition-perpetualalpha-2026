"""Post-outcome description of frozen forecast edges, not parameter selection."""
import json
import numpy as np
from .collect import HERE, ROOT, OUT, save
from .probe import load_tape, asof, walk, expected_net, funding_events_between


def main():
    cfg=json.loads((HERE/'protocol.json').read_text())
    meta=json.loads((OUT/'raw/metadata.json').read_text())['response'][0]
    decimals={r['name'].split(':')[-1]:r['szDecimals'] for r in meta['universe']}
    rows=[]
    for path in cfg['tapes']:
        streams,_=load_tape(ROOT/path)
        for symbol,s in streams.items():
            if not s['book'] or not s['ctx']:continue
            step=cfg['decision_interval_seconds']*1000
            for t in range(((s['book_times'][0]+step-1)//step)*step,s['book_times'][-1]+1,step):
                now=asof(s,t,cfg)
                if now is None:continue
                qty=np.floor(cfg['target_notional']/now['oracle']*10**decimals[symbol])/10**decimals[symbol]
                for arm in cfg['arms']:
                    for scenario,sc in cfg['scenarios'].items():
                        vals=[]
                        count=len(funding_events_between(t+sc['latency_ms'],t+sc['latency_ms']+cfg['horizon_minutes']*60000))
                        for side in (1,-1):
                            preview=walk(now['book']['levels'][1 if side==1 else 0],qty,sc['depth_fraction'])
                            if preview is None:continue
                            edge=expected_net(side,preview,now['oracle'],now['mid'],now['halfspread'],
                                0. if arm=='basis_only' else float(now['ctx']['funding']),count,sc['fee_bps'],
                                1. if arm=='full_convergence_diagnostic' else cfg['basis_reversion_fraction'],
                                cfg['positive_funding_forecast_haircut'])
                            vals.append((edge,side))
                        if vals:
                            best,side=max(vals)
                            rows.append(dict(tape=path,symbol=symbol,time=t,arm=arm,scenario=scenario,
                                forecast_net_bps=best,side=side,spread_bps=2*now['halfspread']/now['mid']*10000,
                                mid_basis_bps=(now['mid']/now['oracle']-1)*10000))
    summary=[]
    for sc in cfg['scenarios']:
        for arm in cfg['arms']:
            selected=[r for r in rows if r['scenario']==sc and r['arm']==arm]
            values=np.array([r['forecast_net_bps'] for r in selected])
            summary.append(dict(scenario=sc,arm=arm,decisions=len(values),min=float(values.min()),
                median=float(np.median(values)),max=float(values.max()),positive=int((values>0).sum()),
                above_entry_buffer=int((values>=cfg['minimum_forecast_net_bps']).sum())))
    result=dict(post_outcome_descriptive_only=True,entry_rules_changed=False,summary=summary,decisions=rows)
    dest=OUT/'quote_probe/edge_diagnostic.json'
    if dest.exists():
        # Verify reproduction, preserving the first descriptive output.
        assert json.loads(dest.read_text())==result, 'Descriptive diagnostic changed'
        print('Diagnostic reproduced exactly; original output preserved')
    else:
        save(dest,result)


if __name__=='__main__':
    main()
