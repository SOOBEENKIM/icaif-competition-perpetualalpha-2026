"""Audit all episode owners, pending exits and added entries after replay."""
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .research import ROOT,HERE,OUT,readrun,save,sha,verify,protected


def main():
    sources=json.loads((OUT/'frozen.json').read_text())['sources'].copy()
    paths=sorted((OUT/'runs').glob('*.json.gz'))
    files=[Path(__file__),OUT/'results.json',OUT/'selection.json',*paths,
           OUT/'books/results.json',OUT/'books_partial/results.json']
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in files});verify(sources)
    save(OUT/'audit_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    rows=[]
    for path in paths:
        run=readrun(path);snap={s['time']:s for s in run['snapshots']};by_pair=defaultdict(lambda:defaultdict(float))
        for episode in run['episodes']:
            pair='/'.join(episode['pair'])
            for key in ['net_pnl','fees','funding','realized']:by_pair[pair][key]+=episode[key]
            by_pair[pair]['episodes']+=1
        assert abs(sum(e['net_pnl'] for e in run['episodes'])-run['metrics']['pair_owned_pnl'])<1e-6
        for name in ['fees','funding']:
            actual=(sum(f['fee'] for owner in ['pair','residual'] for f in run['owner_fills'][owner]) if name=='fees'
                    else sum(f['signed_amount']+f['adverse_amount'] for owner in ['pair','residual'] for f in run['owner_funding'][owner]))
            assert abs(sum(e[name] for e in run['episodes'])-actual)<1e-6,(path,name)
        for s in run['snapshots']:
            np.testing.assert_allclose(np.array(s['base_qty'])+s['active_pair_qty']+np.array(s['residual_qty']),s['qty'],rtol=0,atol=1e-10)
            assert not np.any((np.array(s['active_pair_qty'])!=0)&(np.array(s['residual_qty'])!=0))
        residual_fills=[f for f in run['owner_fills']['residual'] if f['allocation_type']!='residual_transfer']
        for fill in residual_fills:
            assert fill['episode_id'] is not None
            assert abs(fill['position_after'])<=abs(fill['position_before'])+1e-10
            assert fill['position_after']*fill['position_before']>=0
        concurrent=[]
        for event in run['events']:
            if event['event']=='residual_handoff':
                assert any(d['execution_time']==event['time'] and d['active_exit_attempt'] for d in run['decisions'])
            if event['event']=='pair_intent':
                residual=np.asarray(snap[event['time']]['residual_qty'])
                assert not np.any(residual[event['ids']])
                if np.any(residual):concurrent.append(dict(time=event['time'],pair=event['pair'],
                   expected_net_cash=event['expected_net_cash'],residual_qty=residual.tolist(),
                   residual_notional=snap[event['time']]['residual_notional']))
        complete=sum(int(e.get('balanced',False)) for e in run['events'] if e['event']=='closed')
        complete+=sum(int(e['residual']['balanced']) for e in run['events'] if e['event']=='residual_closed')
        assert complete==run['metrics']['completed_pairs']
        rows.append(dict(file=str(path.relative_to(ROOT)),metrics=run['metrics'],
                        by_pair={k:dict(v) for k,v in by_pair.items()},entries_while_residual=concurrent,
                        residual_reductions=len(residual_fills),episode_accounting_verified=True))
    verify(sources)
    save(OUT/'audit.json',dict(rows=rows,accounts_verified=len(rows),
        accounts_with_new_entry_during_residual=sum(bool(r['entries_while_residual']) for r in rows),
        risk_breach_bars=sum(r['metrics']['risk_breach_bars'] for r in rows),protected=protected(),
        orders_sent=0,reserved_raw_read=False,historical_independence=False))
    print('AUDIT',len(rows),'accounts; concurrent entry accounts',sum(bool(r['entries_while_residual']) for r in rows),flush=True)


if __name__=='__main__':main()
