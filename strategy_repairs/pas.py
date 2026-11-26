import gzip
import json
from datetime import datetime
import numpy as np
from competition_live.profile import ROOT,load_profile
from competition_live.runtime import private_load
from pas_strategy_audit.data import load
from .protocol import OUT,freeze
from .engine import pas_replay
from .history import native


def main():
    spec=freeze();cfg,d=load(OUT/'pas');p=load_profile()['trading']
    targets=dict(np.load(OUT/'targets.npz',allow_pickle=False))
    source=dict(np.load(ROOT/spec['pointer']['inputs'],allow_pickle=False))
    publications={}
    for path in (ROOT/'practice_live/updates').glob('*/signals.json'):
        value=json.loads(path.read_text())
        t=int(datetime.fromisoformat(value['input_completed_bar_utc']).timestamp())
        at=int(datetime.fromisoformat(value['created_at_utc']).timestamp()*1000)
        if cfg['start_ms']<=t*1000<cfg['cutoff_ms']:
            publications[t]=min(at,publications.get(t,at))
    portfolio=private_load(ROOT/'.runtime/pas_competition_live/coordinator.pkl')['portfolio']
    gates=[e for e in portfolio.events if e['kind']=='entry_check' and e['time']*1000<=cfg['cutoff_ms']]
    assert gates and all(e['reason']=='decision_gate_rejected' for e in gates)
    scenarios=dict(base=dict(fee_bps=9.,impact_bps=10.,participation=.01,receipt_multiplier=1.),
                   stress=dict(fee_bps=12.,impact_bps=20.,participation=.0025,receipt_multiplier=.5))
    rows=[]
    for name,arm in spec['arms'].items():
        events=[]
        for i,t in enumerate(targets['times']):
            t=int(t)
            if cfg['start_ms']<=t*1000<cfg['cutoff_ms']:
                assert t in publications
                events.append(dict(time=publications[t],source_time=t,weights=targets[name][i].tolist(),
                                   sigma=targets['sigma'][i].tolist(),volume=source['volume'][i].tolist(),
                                   trades=source['trades'][i].tolist()))
        # Only use information whose publication has arrived. The simulator
        # invokes the controller with the source timestamp after availability.
        run=pas_replay(targets['times'],targets[name],targets[name+'_protected'],arm['risk'])
        for scenario,assumptions in scenarios.items():
            result=run(cfg,d,events,p,assumptions)
            with gzip.open(OUT/'pas'/f'{name}_{scenario}.json.gz','wt') as f:json.dump(result,f,default=native)
            row=dict(variant=name,scenario=scenario,**result['metrics']);rows.append(row)
            print(json.dumps(row),flush=True)
    (OUT/'pas_results.json').write_text(json.dumps(rows,indent=2))
    (OUT/'pas_gate_check.json').write_text(json.dumps(dict(checks=len(gates),all_rejected=True),indent=2))


if __name__=='__main__':
    main()
