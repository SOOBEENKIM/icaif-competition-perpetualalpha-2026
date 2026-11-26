"""Separate model alpha, reference aging, venue basis and projected funding."""
import json
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.data import ROOT, OUT as INPUT, save
from experiments.dense_history_pairs_20261017.research import load

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'artifacts/books';b=json.loads((out/'results.json').read_text());d=load(INPUT/'inputs.npz')
    cfg=json.loads((HERE/'par_protocol.json').read_text());states={};rows=[]
    for trial in b['trials']:
        kind,h=trial['model'],trial['horizon'];key=(kind,h)
        if key not in states:
            parent=HERE/'prepared'
            states[key]=load(parent/f'par_{kind}_{h}_state.npz')
        state=states[key];i=int(np.searchsorted(d['times'],trial['decision_time'],side='right')-1)
        k=cfg['dense_pairs'].index(trial['pair']);ids=list(map(int,state['pairs'][k]))
        source=state['source_prices'][i,ids];current=d['reference_close'][i,ids]
        plan=trial['plan'];q=np.array(plan['target']);entry=np.array(plan['entry_preview']);terminal=np.array(plan['terminal_prices'])
        fee=cfg['scenarios'][trial['scenario']]['fee_bps']/10000
        model=float(q@(terminal-source));age=float(q@(source-current));basis=float(q@(current-entry))
        funding=plan['expected_funding_cash'];entry_fee=float(abs(q)@entry*fee);exit_cost=plan['exit_cost_reserve']
        total=model+age+basis+funding-entry_fee-exit_cost
        assert abs(total-plan['expected_net_cash'])<1e-7,(total,plan['expected_net_cash'])
        rows.append(dict(tape=trial['tape'],model=kind,horizon=h,validity_hours=trial['validity_hours'],scenario=trial['scenario'],
                         pair=trial['pair'],decision_time=trial['decision_time'],source_age_hours=trial['source_age_hours'],
                         planned_model_cash=model,reference_age_cash=age,venue_to_current_reference_cash=basis,
                         expected_funding_cash=funding,entry_fee=entry_fee,exit_cost_reserve=exit_cost,
                         expected_net_cash=total,net_without_reference_age=total-age,
                         net_without_model=total-model,buffer=plan['buffer_cash'],status=trial['status']))
    good=[r for r in rows if r['status']=='open_maturity_exit_unobserved']
    summary=dict(known_paired_entry_probes=len(good),
                 positive_above_buffer_without_reference_age=sum(r['net_without_reference_age']>r['buffer'] for r in good),
                 positive_above_buffer_without_model=sum(r['net_without_model']>r['buffer'] for r in good),
                 source_age_min_hours=min(r['source_age_hours'] for r in good),
                 source_age_max_hours=max(r['source_age_hours'] for r in good),
                 unique_tape_pair_decision_directions=len({(r['tape'],tuple(r['pair']),r['decision_time'],next(t['direction'] for t in b['trials'] if all(t[k]==r[k] for k in ['model','horizon','scenario','validity_hours','decision_time','pair','tape']))) for r in good}),
                 not_actual_future_pnl=True,not_independent_opportunity_count=True)
    save(out/'edge_attribution.json',dict(rows=rows,summary=summary,orders_sent=0,post_evaluation_only=True))
    print(json.dumps(summary,ensure_ascii=False))
    for r in good[:10]:
        print(r['model'],r['horizon'],r['scenario'],r['source_age_hours'],{k:round(r[k],3) for k in [
            'planned_model_cash','reference_age_cash','venue_to_current_reference_cash','expected_funding_cash','expected_net_cash','net_without_reference_age']})


if __name__=='__main__':main()
