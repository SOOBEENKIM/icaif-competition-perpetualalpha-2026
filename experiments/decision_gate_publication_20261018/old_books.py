"""Old received-book entry diagnostics; no invented future holding outcomes."""
import copy
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.basis_carry_20261017.probe import load_tape
from experiments.published_pair_lifecycle_20261017.portfolio import Portfolio as OriginalPortfolio
from experiments.dense_history_pairs_20261017.research import load
from experiments.par_direct_forecast_20261017.data import OUT as INPUT
from experiments.complementary_alpha_20261017.research import protected
from experiments.conditional_completion_pairs_20261017.research import verify
from .portfolio import Portfolio
from .forward import ROOT, HERE, OUT, ORIGINAL, sha, save


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()))


def main():
    OUT.mkdir(exist_ok=True)
    target = OUT / 'old_book_audit.json'
    if target.exists(): raise FileExistsError('Do not overwrite the diagnostic')
    allowed = json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    statepath = ROOT/'experiments/decision_gate_pairs_20261017/artifacts/par_gate_state.npz'
    state = load(statepath); data = load(INPUT/'inputs.npz')
    parent = json.loads(ORIGINAL.read_text())
    sources = parent['sources'].copy()
    paths = [*HERE.glob('*.py'), HERE/'protocol.json', ORIGINAL, statepath, INPUT/'inputs.npz', *(ROOT/t for t in allowed)]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths}); verify(sources)
    save(OUT/'old_book_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    records = []; exact = 0
    for tape in allowed:
        streams, meta = load_tape(ROOT/tape)
        cadence = parent['book_protocol']['signal_cadence_seconds']
        first = math.ceil(meta['start_ms']/1000/cadence)*cadence
        for now in range(first,meta['end_ms']//1000+1,cadence):
            i = int(np.searchsorted(data['times'],now,side='right')-1)
            if i<0 or now-data['times'][i]>=300: continue
            forecasts = []
            for k, pair in enumerate(state['pairs']):
                if not state['directions'][i,k]: continue
                ids = list(map(int,pair)); source = int(state['signal_times'][i,k])
                anchors = state['source_prices'][i,ids]; mu = float(state['mu'][i,k])
                direct = dict(model='direct',pair=[parent['trading_protocol']['symbols'][j] for j in ids],
                    ids=ids,horizon=24,forecast_source=source,available_time=source+5,
                    source_prices=anchors.tolist(),source_beta=float(state['beta'][i,k]),
                    source_vol=float(state['vol'][i,k]),mu=mu,
                    terminal_prices=(anchors*np.array([1+mu,1-mu])).tolist())
                forecasts.extend([direct,dict(direct,model='decision_gate',
                    economic_available=bool(state['economic_available'][i,k]),
                    economic_action=int(state['economic_action'][i,k]))])
            # Flat-account entry diagnostic. No invented historical baseline
            # publication or trading PnL: only risk covariance is used here.
            publication = dict(received_ms=now*1000,data=dict(
                frame=dict(source_time=now//3600*3600,sigma=data['sigma'][i].tolist(),
                    retention=[1.]*10,volume=[0.]*10,trades=[0.]*10),
                incumbent=[0.]*10,candidate=[0.]*10))
            local = copy.deepcopy(parent)
            local.update(paper_start_unix=now,capture_end_unix=now+7*86400)
            for scenario in local['book_protocol']['scenarios']:
                outputs = {}
                for tag, cls, name in [('original_direct',OriginalPortfolio,'first_plus_direct'),
                                       ('adapted_direct',Portfolio,'first_plus_direct'),
                                       ('decision_gate',Portfolio,'first_plus_decision_gate')]:
                    def no_unobserved_funding(at, account):
                        if int(at)//3600000 != now//3600:
                            raise ValueError('Funding outside entry-only observation')
                    arm = cls(name,scenario,local,no_unobserved_funding)
                    arm.last_decision=now//21600*21600
                    arm.step(now,streams,publication,forecasts)
                    payload=plain(dict(events=arm.events,snapshots=arm.snapshots,
                        fills=arm.account.physical.fills,qty=arm.account.physical.qty,
                        cash=arm.account.physical.cash,fees=arm.account.physical.fees,fatal=arm.fatal,
                        pair_state=arm.pair.state if arm.pair else None))
                    outputs[tag]=payload
                assert outputs['original_direct']==outputs['adapted_direct']
                exact+=1
                records.append(dict(tape=tape,time=now,scenario=scenario,outputs=outputs))
    reasons={kind:dict(Counter(e['reason'] for r in records for e in r['outputs'][kind]['events']
                              if e['kind']=='entry_check')) for kind in ['adapted_direct','decision_gate']}
    entries={kind:sum(r['outputs'][kind]['pair_state'] is not None for r in records)
             for kind in ['adapted_direct','decision_gate']}
    verify(sources)
    save(target,dict(records=records,exact_ungated_accounts=exact,reasons=reasons,pair_sessions=entries,
        scope='Fresh flat-account entry checks only; historical source states plus explicit 5s allowance, not captured publication envelopes. No baseline-trading performance claim.',
        full_strategy_pnl=None,full_strategy_q=None,complete_future_exit_observed=False,
        protected=protected(),orders_sent=0,reserved_raw_read=False))
    print('OLD_BOOKS',exact,'exact original/adapted direct entry accounts; sessions',entries,flush=True)


if __name__=='__main__':main()
