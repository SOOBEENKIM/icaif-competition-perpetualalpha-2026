"""Finish report from saved standard-JSON results, without rerunning research.

The original process completed every replay and saved results, then its HTML
renderer encountered numpy.bool_ in the in-memory selection object. Loading
the already-saved JSON normalizes scalar types without changing any result.
Keep the frozen original source and its exit-1 log as the audit trail.
"""
import gzip
import json
import time
from pathlib import Path
from .research import ROOT, HERE, OUT, ARMS, path_for, verify, render, save, sha


def main():
    result = json.loads((OUT / 'results.json').read_text())
    assert result['new_account_replays'] == 36 and len(result['exact_control_replays']) == 12
    selected = json.loads((OUT / 'selection.json').read_text())
    assert result['selection'] == selected
    paths = [Path(__file__), OUT / 'results.json', OUT / 'selection.json', HERE / 'research.log']
    checked = 0
    for row in result['rows']:
        if row['arm'] not in ARMS.values(): continue
        path = OUT / 'runs' / f"{row['venue']}-{row['phase']}-{row['arm']}-{row['scenario']}.json.gz"
        paths.append(path)
        with gzip.open(path, 'rt') as f: run = json.load(f)
        with gzip.open(path_for(row['venue'], row['phase'], row['scenario']), 'rt') as f: parent = json.load(f)
        assert run['metrics'] == row['metrics']
        assert (run['start'], run['end']) == (parent['start'], parent['end'])
        checked += 1
    assert checked == 36
    verify(json.loads((OUT / 'frozen.json').read_text())['sources'])
    save(OUT / 'render_recovery_frozen.json', dict(at_unix=time.time(),
         sources={str(p.relative_to(ROOT)): sha(p) for p in paths},
         reason='Only HTML serialization failed after all 36 replays/results save. Render standard JSON; do not retrain, rerun, or change selection.'))
    report = render(result)
    save(HERE / 'work_state.json', dict(status='progress_shared_factor_accounts_and_report_completed',
        report=report, selection=selected, new_account_replays=36, exact_controls=12, tests_passed=9,
        render_recovered_from_saved_json=True, orders_sent=0, live_strategy_changed=False,
        reserved_raw_read=False, historical_independence=False, goal_complete=False, no_progress_streak=0))
    print('DONE report recovery; verified 36 saved accounts; no recomputation', flush=True)


if __name__ == '__main__': main()
