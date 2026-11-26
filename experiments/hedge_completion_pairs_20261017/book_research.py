"""Reused quote probes test execution mechanics, not the new strategy's alpha."""
import json,time
from collections import Counter
import numpy as np
from strategy.agent import TAIL
from experiments.basis_carry_20261017.probe import load_tape
from experiments.pair_execution_calibration_20261017.research import HISTORY
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.complementary_alpha_20261017.research import protected
from .research import HERE
from .book_lifecycle import audit_probe


def main():
    out=HERE/'book_artifacts';out.mkdir(exist_ok=False)
    source=ROOT/'experiments/reference_anchor_pairs_20261017/artifacts/quotes/results.json'
    old=json.loads(source.read_text());cfg=json.loads((ROOT/'experiments/pair_execution_calibration_20261017/lifecycle_protocol.json').read_text())
    cfg.update(completion_ack_delay_ms=1000,completion_buffer_bps=5)
    tapes=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    paths=[HERE/'book_research.py',HERE/'book_lifecycle.py',HERE/'book_completion.py',HERE/'test_book.py',source,INPUT/'inputs.npz',*[ROOT/t for t in tapes]]
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(out/'frozen.json',dict(at_unix=time.time(),sources=sources,protocol=cfg,protected=protected(),
        purpose='All 37 previous known hypothetical maker-fill chunks from seven forecast arms; compare an extra hedge attempt with identical entry selections. Not the FRAM entry signals and not an account backtest.',orders_sent=0))
    with np.load(INPUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    symbols=json.loads((TAIL/'baseline.json').read_text())['protocol']['symbols']
    meta=json.loads((HISTORY/'metadata.json').read_text())['response'][0]['universe']
    decimals={r['name'].split(':')[-1]:r['szDecimals'] for r in meta};rows=[]
    for tape in tapes:
        streams,_=load_tape(ROOT/tape)
        for prior in old['trials']:
            if prior['tape']!=tape:continue
            direction=prior['lead_fill']['side']*(1 if prior['pair'][0]==prior['lead'] else -1)
            matches=[r for r in old['decisions'] if r['status']=='selected' and r['model']==prior['model'] and
                r['scenario']==prior['scenario'] and r['tape']==tape and r['quote_decision']==prior['quote_decision'] and
                r['pair']==prior['pair'] and r['lead']==prior['lead'] and r['direction']==direction]
            assert len(matches)==1,(prior['model'],prior['quote_decision'],len(matches))
            probe={**matches[0], 'lead_fill':prior['lead_fill'],'phase':prior['phase'],'forecast_aligned':prior['forecast_aligned']}
            for enabled in (False,True):
                life={**cfg,'completion_enabled':enabled};sc=cfg['scenarios'][prior['scenario']]
                r=audit_probe(probe,streams,decimals,d,symbols,life,sc)
                if not enabled:
                    for key in ('net_pnl','positions','funding_events','hedge_status','notional','exit_decision'):
                        assert r[key]==prior[key],key
                r.update(model=prior['model'],scenario=prior['scenario'],completion_enabled=enabled);rows.append(r)
        print('tape',tape,'paths',len(rows),flush=True)
    summary=[]
    for model in sorted({r['model'] for r in rows}):
        for scenario in cfg['scenarios']:
            for enabled in (False,True):
                rr=[r for r in rows if r['model']==model and r['scenario']==scenario and r['completion_enabled']==enabled]
                if not rr:continue
                gross=sum(r['notional'] for r in rr)
                summary.append(dict(model=model,scenario=scenario,completion_enabled=enabled,probes=len(rr),
                    net_cash=sum(r['net_pnl'] for r in rr),conditional_net_bps=sum(r['net_pnl'] for r in rr)/gross*10000,
                    completion_plans=sum(bool(r['completion'] and 'limit' in r['completion']) for r in rr),
                    completed_hedges=sum(r['hedge_status']=='completed_full' for r in rr),
                    unknown_outcomes=sum(r['unobserved_hedge_outcome'] for r in rr),
                    residual_gross=sum(r['residual_gross'] for r in rr),
                    statuses=dict(Counter(r['hedge_status'] for r in rr))))
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(out/'results.json',dict(rows=rows,summaries=summary,original_paths_reproduced_exactly=True,
        new_entry_strategy_evidence=False,account_return=None,Q=None,historical_independence=False,orders_sent=0))
    print('DONE',len(rows),'paired paths',flush=True)


if __name__=='__main__':main()
