"""Run a staged tick against public data; no signer or order client exists."""
import json
import shutil
import tempfile
import time
from pathlib import Path
from competition_live.profile import ROOT, load_profile
from competition_live.runner import LiveRunner, PublicPAS
from competition_live.runtime import json_save

profile=load_profile()
source=ROOT/'.runtime/pas_competition_live'
out=Path(__file__).resolve().parent.parent

class NoOrders:
    def market(self,coin):
        raise RuntimeError('Shadow tick requested an order; no order client exists: '+coin)
    def submit(self,*args):
        raise AssertionError('No orders permitted')

with tempfile.TemporaryDirectory() as directory:
    directory=Path(directory)
    for name in ('coordinator.pkl','signal.json'):
        shutil.copy2(source/name,directory/name)
    public=PublicPAS(profile)
    live=LiveRunner(profile,directory,public,NoOrders())
    before=dict(qty=live.portfolio.account.physical.qty.tolist(),
                fills=len(live.portfolio.account.physical.fills),
                pending=len(live.portfolio.account.pending_orders),
                unknown=sorted(live.portfolio.account.unknown),
                reserve=live.portfolio.account.reserved.cash)
    result=live.tick()
    assert before['qty']==result['qty']
    assert before['fills']==result['fills']
    assert not before['pending'] and not before['unknown']
    json_save(out/'checkpoint_shadow.json',dict(at=time.time(),before=before,
        after=result,signer_loaded=False,order_client_created=False,orders_sent=0))
    print(json.dumps(dict(state=result['state'],qty=result['qty'],fills=result['fills'],
        source=result['last_source'],signal_error=result['signal_error'],
        reserve=result['reserved_cash'],orders_sent=0)))
