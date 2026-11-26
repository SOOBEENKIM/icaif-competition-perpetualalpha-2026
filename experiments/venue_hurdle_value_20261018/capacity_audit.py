"""Measure a proxy mismatch: past traded volume is not standing executable depth.

Independent hypothetical IOC packets at a fixed 120k gross ceiling, not alpha
signals or a portfolio. Never sum their fills into a realizable trading result.
"""
import json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from .dataset import ROOT,HERE,OUT,setup,save,sha,verify,protected
from .features import initial_state
from experiments.basis_carry_20261017.probe import load_tape,asof
from experiments.published_pair_lifecycle_20261017.account import BookAccount
from replay_lab.common_comparison.engine import rounded


def main():
    dest=OUT/'capacity';dest.mkdir(exist_ok=False)
    spec,cfg,p,d,de=setup();books_path=ROOT/'experiments/shared_execution_pairs_20261017/book_lifecycle_settings.json'
    books=json.loads(books_path.read_text())
    allowed=json.loads((ROOT/'experiments/basis_carry_20261017/protocol.json').read_text())['tapes']
    sources=json.loads((OUT/'evaluation_frozen.json').read_text())['sources'].copy()
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),books_path,*(ROOT/t for t in allowed)]});verify(sources)
    save(dest/'frozen.json',dict(sources=sources,at_unix=time.time(),gross_ceiling=120000.,source_book_limit_buffer_bps=10,
        purpose='Diagnostic, no profit selection. Fixed source-only reference return hedge; actual received books and independent legs.',
        latency_not_matched_to_5minute_candle_proxy=True,not_alpha_signals=True,orders_sent=0))
    rows=[];counts=Counter()
    for tape in allowed:
        streams,meta=load_tape(ROOT/tape);first=math.ceil(meta['start_ms']/1000/900)*900
        for now in range(first,meta['end_ms']//1000+1,900):
            i=int(np.searchsorted(d['times'],now))
            if i==len(d['times']) or d['times'][i]!=now:continue
            for k,raw_ids in enumerate(de['pairs']):
                ids=list(map(int,raw_ids));beta=float(de['beta'][i,k])
                if not np.isfinite(beta) or not .1<=beta<=10 or de['correlation'][i,k]<cfg['venue_cash']['minimum_correlation']:continue
                pair=spec['pairs'][k];quotes={s:asof(streams[s],now*1000,books) if s in streams else None for s in pair}
                if any(v is None for v in quotes.values()):continue
                marks=d['par_close'][i].copy()
                for s,j in zip(pair,ids):marks[j]=quotes[s]['mid']
                w=np.array([1.,-beta])/(1+beta)
                for scenario,sc in books['scenarios'].items():
                    participation=cfg['scenarios'][scenario]['participation']
                    candle_gross=float(np.min(participation*d['par_volume'][i,ids]*marks[ids]/abs(w)))
                    active=bool(de['fresh'][i,k])
                    for direction in [-1,1]:
                        q=np.array([rounded(120000*direction*weight/marks[j],p['execution']['decimals'][j]) for j,weight in zip(ids,w)])
                        if np.any(abs(q)*marks[ids]<p['execution']['minimum_order_value']):continue
                        limits={}
                        for symbol,j,quantity in zip(pair,ids,q):
                            side=np.sign(quantity);top=float(quotes[symbol]['book']['levels'][1 if side>0 else 0][0]['px'])
                            limits[str(j)]=top*(1+side*.001)
                        account=BookAccount(p['initial'],p['symbols']);target=np.zeros(10);target[ids]=q
                        packet=dict(packet_id=f'{now}:{k}:{direction}',decision_time=now,source_close=now,source_available=now,
                            source_prices=marks.tolist(),owner_targets={'base':np.zeros(10).tolist(),'pair':target.tolist()},
                            owner_limits={'pair':limits},risk_forced=False,sigma=d['sigma'][i])
                        packet=account.seal(packet,p,sc)
                        def no_unobserved_funding(at,ledger):assert int(at)//3600000==now//3600
                        receipt=account.execute_books(packet,streams,marks,d['sigma'][i],p,sc,books,on_time=no_unobserved_funding)
                        owned=account.owners['pair'].qty[ids];state=None if receipt['unknown_symbols'] else initial_state(owned,q,.05)
                        counts['probes']+=1;counts[f'initial_state_{state}']+=1
                        if not active:counts['source_inactive']+=1
                        if not active and state in [1,2]:counts['source_inactive_but_book_filled']+=1
                        if not active and state==2:counts['source_inactive_but_book_balanced']+=1
                        rows.append(dict(tape=tape,time=now,pair=pair,direction=direction,scenario=scenario,
                            source_both_traded=active,candle_volume_gross_capacity=candle_gross,
                            intended_gross=float(abs(q)@marks[ids]),initial_state=state,
                            received_filled_gross=float(abs(owned)@marks[ids]),orders=receipt,fills=account.physical.fills,
                            qty=owned.tolist(),fees=account.physical.fees,account_return=None,financial_score=None,orders_sent=0))
        print('CAPACITY',tape,dict(counts),flush=True)
    verify(sources)
    save(dest/'results.json',dict(rows=rows,counts=dict(counts),independent_hypothetical_entries=True,
        unique_source_pair_directions=len({(r['time'],tuple(r['pair']),r['direction']) for r in rows}),
        not_alpha_signals=True,full_exits_unobserved=True,candle_and_book_latency_differ=True,
        conclusion='Report counts; do not infer profitability or manufacture a long historical L2 replay from candles.',
        protected=protected(),orders_sent=0,live_strategy_changed=False,reserved_raw_read=False))
    print('DONE capacity',dict(counts),flush=True)


if __name__=='__main__':main()
