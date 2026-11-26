"""Reproduce all sealed results without overwriting the originals.

The first-run launcher seals its then-existing directory. Later reporting and
diagnostic files do not change that seal; this verifier checks its named files.
"""
import copy,json
import numpy as np
from .prepare import ROOT,OUT,END
from .engine import simulate
from pas_economic_allocation.lifecycle import LifecycleController
from pas_adaptive_development.study import sha,save


def main():
    from competition_live.profile import load_profile
    from pas_strategy_audit.data import load
    def read(name):return json.loads((OUT/name).read_text())
    for name in ('source_freeze.json','execution_freeze.json'):
        for path,digest in read(name).items():assert sha(ROOT/path)==digest,path
    cfg,d=load(ROOT/'reports/pas_allocation_study_20261021/pas');cfg['cutoff_ms']=END*1000
    p=read('protocol.json');profile=load_profile();events=read('events.json');tape=read('trade_tape.json');funding=read('funding_tape.json')
    known={(r['coin'],r['time']//3600000) for r in funding}
    for r in d['funding_events']:
        coin='pas:'+cfg['symbols'][r['j']]
        if (coin,r['time']//3600000) not in known:funding.append(dict(coin=coin,time=r['time'],rate=r['rate'],cash_per_unit=None))
    z=dict(np.load(OUT/'candidate_inputs.npz'));scenarios=copy.deepcopy(p['scenarios'])
    scenarios['optimistic_100pct']={**scenarios['base_1pct'],'participation':1.}
    verified=[]
    for expected in read('results.json')+read('diagnostic_results.json'):
        arm=expected['arm'];sc=scenarios[expected['scenario']];signals=copy.deepcopy(events);controller=None
        if arm=='size_only':
            for e in signals:e['weights']=e['size_only']
        elif arm=='incumbent_2x_diagnostic':
            for e in signals:e['weights']=(2*np.asarray(e['weights'])).tolist()
        elif arm=='budget_lifecycle':
            controller=LifecycleController(z['times'],z['forecast'],z['carry'],z['anchor'],z['protected'],5.)
        result=simulate(cfg,d,signals,tape,funding,profile['trading'],sc,controller=controller)
        for k,v in result['metrics'].items():assert v==expected[k],(arm,expected['scenario'],k,v,expected[k])
        verified.append(dict(arm=arm,scenario=expected['scenario'],all_metrics_equal=True))
    for path,digest in p['live_original'].items():assert sha(ROOT/path)==digest,path
    save(OUT/'reproduction.json',dict(runs=verified,model_refit=False,original_results_overwritten=False,orders_sent=0))
    print('All16 frozen result rows reproduced exactly; originals unchanged.')


if __name__=='__main__':main()
