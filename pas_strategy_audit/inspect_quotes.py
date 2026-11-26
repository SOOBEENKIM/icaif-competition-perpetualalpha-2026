"""Unsigned public quote smoke check; does not construct an execution client."""
import json,time
from .execution import PublicQuoteProvider
from .data import OUT
from competition_live.profile import load_profile
from experiments.basis_carry_20261017.probe import asof
p=load_profile();provider=PublicQuoteProvider(p)
cfg=json.loads(open('competition_live/strategy_plan.json').read())['book_protocol'];rows=[]
for s in p['trading']['symbols']:
 stream=provider(s);at=int(time.time()*1000);q=asof(stream,at,cfg)
 rows.append({'symbol':s,'received_ms':at,'valid':q is not None,'book_age_ms':at-stream['book'][0][2]['time'],'context_age_ms':at-stream['ctx'][0][0]})
(OUT/'candidate_quote_check.json').write_text(json.dumps({'orders_sent':0,'signer_loaded':False,'markets':rows},indent=2));print(json.dumps(rows))
assert all(x['valid'] for x in rows)
