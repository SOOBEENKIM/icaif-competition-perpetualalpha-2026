"""Read-only venue checks using a COPY of the durable live order store."""
import json,sqlite3,tempfile,time
from pathlib import Path
from bot.connection import build_client,check_connection,close_client,load_settings
from practice_live.live_store import Store
from practice_live.refresh import ROOT,OUT as LIVE,save_json,iso
from practice_live.infer import protected_hashes
from practice_live.live_logic import identity
from .live import BudgetRunner,epoch_snapshot
from .live_signal import DIRECTORY,read_publication

def main():
    DIRECTORY.mkdir(exist_ok=True)
    frame,obj=read_publication()
    if frame.source_time!=int(time.time()//3600)*3600:raise ValueError('Prepare a current signal first')
    with tempfile.TemporaryDirectory(prefix='icaif-budget-preflight-') as td:
        d=Path(td);src=sqlite3.connect('file:'+str(LIVE/'live_runtime/state.sqlite3')+'?mode=ro',uri=True);dest=sqlite3.connect(d/'state.sqlite3')
        src.backup(dest);dest.close();src.close()
        store=Store(d/'state.sqlite3');client=None
        try:
            if store.get('frozen_fingerprint')!=identity(protected_hashes()):raise ValueError('Incumbent hash mismatch')
            settings=load_settings();client=build_client(settings,execution=False);check_connection(client,settings)
            runner=BudgetRunner(client,settings,store,d,False);runner.recover();account=runner.account()
            pending=[o for o in store.orders() if o['status'] in ('sending','unknown','responded','open')]
            if pending:raise ValueError('Unresolved orders')
            if any(json.loads(v)['status']!='complete' for v, in store.db.execute("select value from kv where key like 'round:%'")):raise ValueError('Pending rounds')
            result={'at_utc':iso(),'passed':True,'source_hour':frame.source_time,'candidate_weights':frame.weights,
                'retention':frame.retention,'candidate_publication_hash':obj['sha256'],'proposed_epoch':epoch_snapshot(account,store),
                'orders_sent':0,'production_database_written':False,'existing_orders_reconciled':len(store.orders()),
                'positions_reconciled':account['qty'].tolist(),'account':account['ledger']}
            save_json(DIRECTORY/'preflight.json',result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:
            if client:close_client(client)
            store.close()

if __name__=='__main__':main()
