"""Read-only post-end check: incumbent paper path versus actual owner ledger."""
import gzip,json,time,sys
from datetime import datetime,timezone
import numpy as np
from .protocol import ROOT,OUT,save,sha

DEST=OUT/'incumbent_real_execution_audit';START=1792897500;END=1792915500


def ledger_value(ledger,marks,at):
    fills=[f for f in ledger.fills if f['time']<=at]
    funding=[f for f in ledger.funding_events if f['time']<=at]
    qty=np.zeros(len(marks));cash=ledger.initial
    for f in fills:
        qty[f['asset']]+=f['signed_qty'];cash-=f['signed_qty']*f['price']+f['fee']
    cash+=sum(f['signed_amount'] for f in funding)
    return dict(time=at,equity=float(cash+qty@marks),qty=qty.tolist(),marks=marks.tolist(),
                fees=sum(f['fee'] for f in fills),funding=sum(f['signed_amount'] for f in funding),
                boundary_fill_timestamp_ambiguities=sum(abs(f['time']-at)<1 for f in ledger.fills))


def freeze():
    assert time.time()<END and not (DEST/'protocol.json').exists()
    save(DEST/'protocol.json',dict(at=time.time(),start=START,end=END,
        purpose='After the fixed end only, compare simulated incumbent to actual confirmed OWN ledger, excluding teammate returns. Reconstruct both at the same received marks and timestamps. Do not treat simulated receipts as actual fills.',
        interpretation='Report fill counts, signed quantity changes, actual fees/funding and PnL mismatch. If candidate paired advantage is not larger than this baseline execution mismatch, paper improvement is not resolved above observed modelling error. This is a diagnostic of simulator fidelity, not tuning a new strategy.',
        source_sha256=sha(__file__),orders_sent=0))


def check():
    p=json.loads((DEST/'protocol.json').read_text());assert sha(__file__)==p['source_sha256'];return p


def run():
    check()
    while time.time()<END+1200:
        save(DEST/'status.json',dict(at=time.time(),state='waiting_until_after_fixed_end',orders_sent=0))
        time.sleep(min(30,max(1,END+1200-time.time())))
    check()
    from competition_live.profile import load_profile
    from competition_live.runtime import private_load
    from .book_replay import load_books,BookVenue
    path=ROOT/'.runtime/pas_budget_lifecycle/coordinator.pkl'
    for _ in range(5):
        before=sha(path);p=private_load(path)['portfolio'];after=sha(path)
        if before==after:break
        time.sleep(1)
    else:raise ValueError('Could not read a consistent live checkpoint')
    profile=load_profile();streams,_=load_books((OUT/'capture').glob('*.jsonl.gz'),END)
    sc=dict(fee_bps=9.,depth_fraction=1.,latency_ms=1000,receipt_multiplier=1.)
    a=p.account.physical
    conservation=a.initial-sum(f['signed_qty']*f['price']+f['fee'] for f in a.fills)+sum(f['signed_amount'] for f in a.funding_events)
    error=abs(conservation-a.shadow_cash)
    if error>1e-6:raise ValueError('Actual owner ledger history does not reconstruct its cash')
    endpoints=[]
    for at in [START,END]:
        marks=BookVenue(streams,profile,sc,at,at,[]).marks()
        if marks is None:raise ValueError('Actual comparison lacks common endpoint marks')
        endpoints.append(ledger_value(a,marks,at))
    first,last=endpoints
    fills=[f for f in a.fills if START<f['time']<=END]
    audit=dict(at=time.time(),checkpoint_sha256=after,portfolio_class=type(p).__name__,
        execution_revision=getattr(p,'execution_revision',None),execution_policy=getattr(p,'execution_policy',None),
        initial=first,terminal=last,net_pnl=last['equity']-first['equity'],
        fees=last['fees']-first['fees'],funding=last['funding']-first['funding'],fills=fills,
        accounting_error=error,unknown=list(p.account.unknown),pending=len(p.account.pending_orders),fatal=p.fatal,
        orders_sent_by_audit=0,actual_owned_ledger=True)
    save(DEST/'actual_path.json',audit)
    while time.time()<END+2100:
        save(DEST/'status.json',dict(at=time.time(),state='waiting_for_corrected_paper_control',orders_sent=0))
        time.sleep(min(30,max(1,END+2100-time.time())))
    from .liquidation_audit import value_at
    file=OUT/'exact_forward_review/entry_lifetime/incumbent_base.json.gz'
    if not file.exists():raise ValueError('Corrected incumbent comparison is not ready')
    with gzip.open(file,'rt') as fp:paper=json.load(fp)
    x,y=[value_at(streams,profile,sc,paper,at) for at in [START,END]]
    signed_actual=np.array(last['qty'])-np.array(first['qty']);signed_paper=np.array(y['qty'])-np.array(x['qty'])
    comparison=dict(actual_net_pnl=audit['net_pnl'],paper_net_pnl=y['equity']-x['equity'],
        absolute_pnl_mismatch=abs(audit['net_pnl']-(y['equity']-x['equity'])),
        actual_fees=audit['fees'],paper_fees=y['fees']-x['fees'],actual_funding=audit['funding'],paper_funding=y['funding']-x['funding'],
        actual_ledger_fill_fragments=len(fills),paper_order_fills=len(paper['fills']),
        quantity_delta={s:dict(actual=float(signed_actual[j]),paper=float(signed_paper[j])) for j,s in enumerate(profile['trading']['symbols'])},
        timestamp_ambiguities=first['boundary_fill_timestamp_ambiguities']+last['boundary_fill_timestamp_ambiguities'],
        source='Actual owner ledger versus conditional received-book sweep, identical mark endpoints.',orders_sent=0)
    save(DEST/'comparison.json',comparison)
    save(DEST/'status.json',dict(at=time.time(),state='complete',orders_sent=0))


if __name__=='__main__':globals()[sys.argv[1]]()
