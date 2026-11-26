"""Reproduce the observed paper receipt exception without live I/O."""
import json,time
from decimal import Decimal
from unittest.mock import patch
from .protocol import OUT,save
from competition_live.profile import load_profile
from competition_live.runtime import private_load
from . import book_replay,cash_replay,data_aware_book_replay
from .delayed_forward_review import delayed_submit
from .exact_settlement import run_exact,read_records
from .full_forward_review import prepare,EXPERIMENTS


def main():
    original=book_replay.BookVenue.send
    def traced(self,order):
        response=original(self,order);rows=self.rows[response['filled']['oid']]
        got=sum((Decimal(x['sz']) for x in rows),Decimal(0))
        reported=Decimal(str(response['filled']['totalSz']))
        if got!=reported or got>Decimal(order.size):
            save(OUT/'simulation_receipt_mismatch_reproduction.json',dict(at=time.time(),order=vars(order),rows=rows,
                reported=str(reported),sum_rows=str(got),requested=order.size,excess=str(got-Decimal(order.size)),
                simulated_order=self.orders[-1],orders_sent=0))
        return response
    p=load_profile();streams,_=book_replay.load_books([OUT/'sealed_forward_capture/received_through_fixed_end.jsonl.gz'],1792915500)
    source=sorted([json.loads(f.read_text()) for f in (OUT/'forward_signals').glob('*.json')],key=lambda x:x['received_at'])
    funding=[]
    for symbol in p['trading']['symbols']:
        funding.extend(dict(r,coin='pas:'+symbol) for r in json.loads((OUT/'forward_evaluation'/f'{symbol}_funding.json').read_text())['rows'])
    initial=[]
    for f in (OUT/'initial_funding').glob('*.json'):
        r=json.loads(f.read_text())
        if 'request' in r:initial.extend(dict(x,coin=r['request']['coin']) for x in r['data'])
    seed=private_load(OUT/'forward_seed.pkl')['portfolio'];e=EXPERIMENTS[3]
    state,direct=prepare(seed,e,'frozen_side_quote')
    sc=dict(fee_bps=12.,depth_fraction=.5,latency_ms=3000,receipt_multiplier=.5)
    with patch.object(book_replay.BookVenue,'send',traced),patch.object(book_replay,'run_book',data_aware_book_replay.run_book_data_aware),patch.object(data_aware_book_replay,'PHASE_SECONDS',45),patch.object(cash_replay.CashVenue,'submit',delayed_submit):
        try:run_exact(streams,source,p,sc,e['replay_start'],1792915500,state,direct,funding,cash_units=json.loads((OUT/'settlement_check/cash_units.json').read_text()),records=read_records(OUT/'settlement_check/raw'),initial_settlements=initial)
        except Exception as error:print(type(error).__name__,str(error))
    f=OUT/'simulation_receipt_mismatch_reproduction.json'
    if f.exists():
        r=json.loads(f.read_text());print(json.dumps({k:r[k] for k in ['reported','sum_rows','requested','excess']},indent=2))


if __name__=='__main__':main()
