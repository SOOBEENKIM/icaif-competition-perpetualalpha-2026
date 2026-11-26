"""Fixed-size terminal-anchor sensitivity; no invented future executions."""
import json
import numpy as np
from experiments.basis_carry_20261017.probe import load_tape,asof
from experiments.par_direct_forecast_20261017.data import ROOT,save


def main():
    art=ROOT/'experiments/daily_tree_pairs_20261017/artifacts'
    book=json.loads((art/'books/results.json').read_text())
    cfg=json.loads((ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json').read_text())
    with np.load(ROOT/'experiments/par_direct_forecast_20261017/artifacts/inputs.npz') as z:
        times=z['times'];reference=z['reference_close']
    rows=[];loaded={}
    for trial in book['trials']:
        tape=trial['tape']
        if tape not in loaded:loaded[tape]=load_tape(ROOT/tape)[0]
        intent,plan=trial['intent'],trial['plan'];q=np.array(plan['target'])
        anchors=np.array(intent['source_prices']);terminal=np.array(intent['terminal_prices']);entries=np.array(plan['entry_preview'])
        at=trial['time'];ids=intent['ids'];ix=np.searchsorted(times,at,side='right')-1
        quotes=[asof(loaded[tape][s],at*1000,cfg) for s in intent['pair']]
        mids=np.array([v['mid'] for v in quotes]);oracles=np.array([v['oracle'] for v in quotes])
        fee=cfg['scenarios'][trial['scenario']]['fee_bps']/1e4
        def expected(end):
            return float(q@(end-entries)-abs(q)@entries*fee+plan['expected_funding_cash']-plan['exit_cost_reserve'])
        assert abs(expected(terminal)-plan['expected_net_cash'])<1e-7
        rows.append(dict(model=trial['model'],horizon=intent['horizon'],pair=intent['pair'],scenario=trial['scenario'],time=at,source=intent['forecast_source'],
            expected_net_cash=expected(terminal),without_model_increment=expected(anchors),
            without_source_aging=expected(reference[ix,ids]+terminal-anchors),
            current_oracle_anchor=expected(oracles+terminal-anchors),
            without_venue_basis_or_source_aging=expected(mids+terminal-anchors),entry_buffer=plan['buffer_cash'],
            source_reference_vs_venue_cash=float(q@(anchors-mids)),model_increment_cash=float(q@(terminal-anchors)),
            known_entry_fees=trial['known_fees'],realized_horizon_pnl=None))
    save(art/'book_value_attribution.json',dict(rows=rows,
        scope='Same selected source forecast and pre-fill planned rounded quantities. Change terminal-anchor hypotheses only. These are pre-order estimates, not predictions recomputed after partial-fill repair, not actual future exits and not reselection.',
        scenario_cases=len(rows),source_tape_pair_groups=len({(t['tape'],tuple(t['intent']['pair']),t['intent']['forecast_source']) for t in book['trials']}),orders_sent=0))
    print('BOOK VALUE ATTRIBUTION',len(rows),'scenario cases')


if __name__=='__main__':main()
