"""Unsigned budgeted /info reads, preserving original collection envelopes."""
import json,time
from .protocol import *

def main():
 from candidate_handoff.public import PublicPAS
 from competition_live.profile import load_profile
 api=PublicPAS(load_profile())
 try:
  for s in SYMBOLS:
   for start in range(SPLIT-3600,END,24*3600):
    path=OUT/'collection'/f'{s}_{start}.json'
    if path.exists():continue
    req=dict(type='candleSnapshot',req=dict(coin='pas:'+s,interval='1m',startTime=start*1000,endTime=min(END,start+86400)*1000))
    rows=api.post(req);save(path,dict(request=req,received_at=time.time(),data=rows))
    print(s,start,'candles',len(rows),flush=True)
   path=OUT/'collection'/f'{s}_funding.json'
   if not path.exists():
    req=dict(type='fundingHistory',coin='pas:'+s,startTime=(START-86400)*1000,endTime=END*1000)
    rows=api.post(req)
    if len(rows)>=500:raise ValueError('Funding truncated')
    save(path,dict(request=req,received_at=time.time(),data=rows))
  path=OUT/'current_market.json'
  if not path.exists():
   meta,ctx=api.market();books={}
   for s in SYMBOLS:
    b=api.post(dict(type='l2Book',coin='pas:'+s));books[s]=dict(received_at=time.time(),book=b)
   save(path,dict(received_at=time.time(),meta=meta,contexts=ctx,books=books,orders_sent=0))
 finally:api.session.close()
 save(OUT/'collection_finished.json',dict(at=time.time(),read_only=True,orders_sent=0))

if __name__=='__main__':main()
