"""Validate a proposed state handoff without transferring execution ownership."""
import json,sqlite3,time
import numpy as np
from .observe_live import account
from .prepare import ROOT,OUT,save


def main():
    observed=account()
    db=sqlite3.connect('file:'+str(ROOT/'practice_live/live_runtime/state.sqlite3')+'?mode=ro',uri=True)
    db.execute('BEGIN')
    kv={k:json.loads(v) for k,v in db.execute('select key,value from kv')}
    orders=[{'cloid':c,'status':s,'plan':json.loads(p)} for c,s,p in db.execute('select cloid,status,plan from orders')]
    fills=[json.loads(r) for r, in db.execute("select raw from records where kind='fills'")]
    db.rollback();db.close()
    unknown=[o for o in orders if o['status'] not in ('filled','canceled','rejected','triggered') and not o['status'].endswith(('Canceled','Rejected'))]
    if unknown:raise ValueError('Unresolved orders block handoff')
    assert {(f['coin'],f['tid']) for f in fills}=={(f['coin'],f['tid']) for f in observed['raw']['fills']}
    assert float(kv['anchor']['initial_equity'])==1e6
    rounds=[r for k,r in kv.items() if k.startswith('round:normal:') and r['status']=='complete']
    normal=max(r['decision_time'] for r in rounds)
    proposed={'control':{'controller':{'last_decision':normal}},'peak':max(kv.get('peak',1e6),observed['equity']),
              'halted':bool(kv.get('halted',False)),'last_tick':observed['time'],'last_normal':normal}
    save(OUT/'handoff_check.json',{'checked_at':time.time(),'passed':True,'actual_positions':observed['qty'],
        'equity':observed['equity'],'fees':observed['ledger']['fees'],'funding':observed['ledger']['funding'],
        'matched_fills':len(fills),'terminal_orders':len(orders),'unresolved_orders':0,'proposed_supervisor_state':proposed,
        'actual_writer_transferred':False,'orders_sent':0,
        'activation_requirement':'Before any future activation, stop/reconcile the incumbent writer, recheck this snapshot, and reuse its durable order store/claims. This file does not activate the candidate.'})
    print(json.dumps({'state_handoff_preparation':'passed','matched_fills':len(fills),'orders_sent':0}))

if __name__=='__main__':main()
