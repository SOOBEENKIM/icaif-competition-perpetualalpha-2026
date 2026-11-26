"""Descriptive six-hour PAS direction labels, never fitting or selecting rules."""
import json
from datetime import datetime
import numpy as np
from competition_live.profile import ROOT
from pas_strategy_audit.data import load
from .protocol import OUT,freeze


def main():
    spec=freeze();cfg,d=load(OUT/'pas')
    z=dict(np.load(OUT/'targets.npz',allow_pickle=False));pub={}
    for path in (ROOT/'practice_live/updates').glob('*/signals.json'):
        s=json.loads(path.read_text());source=int(datetime.fromisoformat(s['input_completed_bar_utc']).timestamp())
        at=int(datetime.fromisoformat(s['created_at_utc']).timestamp()*1000)
        if cfg['start_ms']<=source*1000<cfg['cutoff_ms']:pub[source]=min(at,pub.get(source,at))
    rows=[]
    for name in spec['arms']:
        labels=[]
        for i,t in enumerate(z['times']):
            t=int(t);end=(t+21600)*1000
            if t%21600 or t not in pub or end>cfg['cutoff_ms']:continue
            a=int(np.searchsorted(d['times'],pub[t]));b=int(np.searchsorted(d['times'],end))
            for j,s in enumerate(cfg['symbols']):
                weight=z[name][i,j]
                if not weight:continue
                observed=(np.isfinite(d['close'][a,j]) and np.isfinite(d['close'][b,j]) and
                          max(d['age_ms'][a,j],d['age_ms'][b,j])<=900000)
                label=dict(source_time=t,prediction_available_ms=pub[t],outcome_ms=end,symbol=s,
                           direction=int(np.sign(weight)),weight=float(weight),observed=bool(observed))
                if observed:
                    r=float(d['close'][b,j]/d['close'][a,j]-1)
                    label.update(pas_price_return=r,directional_return=float(np.sign(weight)*r),
                                 correct=bool(np.sign(weight)*r>0))
                labels.append(label)
        valid=[x for x in labels if x['observed']]
        nonflat=[x for x in valid if x['pas_price_return']!=0]
        rows.append(dict(variant=name,available_labels=len(valid),unavailable_labels=len(labels)-len(valid),
            nonflat_labels=len(nonflat),direction_accuracy=sum(x['correct'] for x in nonflat)/len(nonflat) if nonflat else None,
            mean_signed_price_return=float(np.mean([x['directional_return'] for x in valid])) if valid else None,
            labels=labels))
    (OUT/'pas_direction_labels.json').write_text(json.dumps(dict(
        role='Descriptive diagnosis only, not a trading simulation or selection metric.',
        horizon='Next existing six-hour rebalance; entry valuation after actual publication.',
        filter='Completed labels only; endpoint last-trade age <=15 minutes; missing rows disclosed.',
        independence=False,rows=rows),indent=2))


if __name__=='__main__':
    main()
