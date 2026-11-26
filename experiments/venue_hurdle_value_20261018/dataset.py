"""Augment immutable execution episodes with source features and initial labels."""
import gzip,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.venue_execution_value_20261018.dataset import setup as old_setup,ROOT,load,save,sha,verify,protected
from .features import activity,features,initial_state

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'
PARENT=ROOT/'experiments/venue_execution_value_20261018/artifacts'


def setup():
    oldspec,cfg,p,d,de=old_setup()
    spec=json.loads((HERE/'protocol.json').read_text())
    de['execution_x'],de['execution_names']=activity(d,de['pairs'])
    return spec,cfg,p,d,de


def main():
    text=(HERE/'test_results.txt').read_text();assert 'Ran 7 tests' in text and text.rstrip().endswith('OK')
    OUT.mkdir(exist_ok=False)
    spec,cfg,p,d,de=setup()
    sources=json.loads((PARENT/'conclusion_frozen.json').read_text())['sources'].copy();verify(sources)
    files=[*HERE.glob('*.py'),HERE/'protocol.json',HERE/'test_results.txt',PARENT/'labels.json']
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in files});verify(sources)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),previous_goal_turn=spec['previous_goal_turn']))
    previous=json.loads((PARENT/'labels.json').read_text());rows=[]
    for row in previous['rows']:
        if row['status'] not in ('closed','unfilled','censored'):continue
        path=PARENT/'episodes'/f"{row['sample_id']}.json.gz"
        with gzip.open(path,'rt') as f:run=json.load(f)
        source=row['source'];i=int(np.searchsorted(d['times'],source))
        intent=next(e for e in run['events'] if e['event']=='pair_intent')
        snapshot=next(s for s in run['snapshots'] if s['time']==source+300)
        x,gross=features(intent,d,de,i,p)
        assert abs(gross-row['intended_gross'])<1e-8
        initial=initial_state(np.asarray(snapshot['pair_qty'])[intent['ids']],intent['target'],cfg['pair_balance_tolerance'])
        if initial==0:assert row['status']=='unfilled' and row['net_pnl']==0.
        rows.append(dict(row,features=x.tolist(),initial_class=initial,initial_time=snapshot['time']))
        sources[str(path.relative_to(ROOT))]=sha(path)
    np.savez_compressed(OUT/'state.npz',**de)
    save(OUT/'labels.json',dict(rows=rows,initial_classes=dict(Counter(r['initial_class'] for r in rows)),
        cash_statuses=dict(Counter(r['status'] for r in rows)),activity_features=de['execution_names'],
        initial_labels_from_censored=sum(r['status']=='censored' for r in rows),source_features=len(rows[0]['features']),
        actual_new_episode_replays=0,existing_episode_paths_read=len(rows),historical_independence=False,orders_sent=0))
    verify(sources)
    save(OUT/'dataset_frozen.json',dict(at_unix=time.time(),sources=sources))
    save(HERE/'work_state.json',dict(status='progress_initial_execution_labels_and_source_features_prepared',
        rows=len(rows),next_action='Fit fixed-date mixture and direct tree; reproduce parent accounts before comparing later periods.',
        goal_complete=False,no_progress_streak=0,orders_sent=0,live_strategy_changed=False))
    print('DONE labels',len(rows),dict(Counter(r['initial_class'] for r in rows)),flush=True)


if __name__=='__main__':main()
