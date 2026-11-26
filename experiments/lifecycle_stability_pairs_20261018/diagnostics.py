"""Exact-path comparisons and observed entry checks, without new selection."""
import json
import time
from collections import Counter
from pathlib import Path
from .research import ROOT,HERE,OUT,GATE,PARENT,readrun,save,sha,verify,protected


def main():
    result=json.loads((OUT/'results.json').read_text());spec=json.loads((HERE/'protocol.json').read_text())
    files=[Path(__file__),OUT/'results.json',OUT/'models.json',*sorted((OUT/'runs').glob('*.json.gz'))]
    sources={str(path.relative_to(ROOT)):sha(path) for path in files}
    verify(json.loads((OUT/'frozen.json').read_text())['sources']);verify(sources)
    save(OUT/'diagnostics_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    identical=[];checks=[]
    windows=[*result['windows'],['reference','continuous']]
    for venue,phase in windows:
        for scenario in ['base','stress']:
            path=OUT/'runs'/f'{venue}-{phase}-{spec["candidate"]}-{scenario}.json.gz';run=readrun(path)
            old_path=(OUT if phase=='continuous' else GATE)/'runs'/f'{venue}-{phase}-{spec["control"]}-{scenario}.json.gz'
            old=readrun(old_path)
            equal={field:run[field]==old[field] for field in ['metrics','events','orders','fills','owner_fills','snapshots','decisions']}
            identical.append(dict(venue=venue,phase=phase,scenario=scenario,equal=equal))
            records=run['stability_checks'];counts=Counter(r['reason'] for r in records)
            negative=[r for r in records if r['full'] is not None and r['full']<0]
            actual={(e['time'],tuple(e['pair']),e['direction']) for e in run['events'] if e['event']=='pair_intent'}
            entry_checks=[r for r in records if (r['time'],tuple(r['pair']),r['direction']) in actual]
            checks.append(dict(venue=venue,phase=phase,scenario=scenario,total=len(records),counts=dict(counts),
                               full_negative=len(negative),negative_but_preserved=sum(r['allowed'] for r in negative),
                               vetoed=sum(not r['allowed'] for r in records),actual_entry_checks=entry_checks))
            print('CHECKS',venue,phase,scenario,dict(counts),'full negative',len(negative),'equal',all(equal.values()),flush=True)
    models=json.loads((OUT/'models.json').read_text());audits=[]
    parent_labels=json.loads((PARENT/'labels.json').read_text())['rows'];by_id={r['sample_id']:r for r in parent_labels}
    from .model import overlaps
    for model in models.values():
        for block in model['blocks']:
            child=block['model']
            if child is None:continue
            assert child['last_outcome']<model['fit_time']
            assert all(not overlaps(by_id[sample],block['left'],block['right']) for sample in child['training_ids'])
            audits.append((model['fit_time'],block['left']))
    assert len(audits)==result['successful_deleted_block_fits']
    verify(sources)
    save(OUT/'diagnostics.json',dict(exact_paths=identical,checks=checks,verified_deleted_block_fits=len(audits),
         semantics='A proposal check is not a fill. Deletion-sign agreement is model influence stability, not a calibrated predictive confidence interval. All causal block fits exclude overlapping labels.',
         protected=protected(),orders_sent=0,historical_independence=False))
    print('DONE stability diagnostics',flush=True)


if __name__=='__main__':main()
