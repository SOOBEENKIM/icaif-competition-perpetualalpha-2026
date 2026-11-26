"""Unsigned quote-ingestion smoke check after candidate source freeze."""
import hashlib
import json
import time
from competition_live.profile import ROOT,load_profile
from pas_strategy_audit.execution import PublicQuoteProvider
from experiments.basis_carry_20261017.probe import asof
from .protocol import OUT


def main():
    p=load_profile();provider=PublicQuoteProvider(p)
    cfg=dict(book_max_age_ms=15000,context_max_age_ms=3000)
    files=list((ROOT/'strategy_repairs').glob('*.py'))
    files += [ROOT/'pas_strategy_audit/execution.py',ROOT/'all_market_strategy/signals.py']
    files += [OUT/'targets.npz',OUT/'protocol.json',OUT/'nomination.json']
    frozen=dict(at_ms=int(time.time()*1000),files={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (OUT/'candidate_freeze.json').write_text(json.dumps(frozen,indent=2))
    rows=[]
    for symbol in p['trading']['symbols']:
        stream=provider(symbol);now=int(time.time()*1000)
        quote=asof(stream,now,cfg)
        if quote is None:raise ValueError('Freshness check failed: '+symbol)
        bid=float(quote['book']['levels'][0][0]['px']);ask=float(quote['book']['levels'][1][0]['px'])
        rows.append(dict(symbol=symbol,received_ms=now,bid=bid,ask=ask,
            book_age_ms=now-int(quote['book']['time']),context_age_ms=now-stream['ctx_times'][-1],
            bid_notional=bid*float(quote['book']['levels'][0][0]['sz']),
            ask_notional=ask*float(quote['book']['levels'][1][0]['sz']),valid=True))
        print(symbol,'valid',flush=True)
    (OUT/'quote_probe.json').write_text(json.dumps(dict(source_frozen_ms=frozen['at_ms'],
        rows=rows,orders_sent=0,evidence='Public quote ingestion only; not fills or strategy PnL.'),indent=2))


if __name__=='__main__':
    main()
