"""Independent fill arithmetic, chronology and balance reconciliation.

This checker does not call Portfolio or strategy functions and cannot generate
orders. It reconstructs each position from immutable fills and funding records.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path


def audit(document):
    summary=document['summary']
    positions={}
    realized=Decimal(0);fees=Decimal(0)
    previous=-1
    errors=[]
    for index,f in enumerate(document['fills']):
        coin=f['coin'];side=Decimal(f['side'])
        qty=Decimal(str(f['qty']));price=Decimal(str(f['price']))
        fee=Decimal(str(f['fee']))
        if f['time_ms']<previous or f['time_ms']<=f['submitted_ms']:
            errors.append([index,'invalid_fill_chronology'])
        previous=f['time_ms']
        if qty<=0 or price<=0 or side not in (-1,1):
            errors.append([index,'invalid_fill_values'])
        if abs(fee-qty*price*Decimal('0.0009'))>Decimal('.000001'):
            errors.append([index,'fee_does_not_match_9bp'])
        old,average=positions.get(coin,(Decimal(0),Decimal(0)))
        realized_fill=Decimal(0)
        if old*side<0:
            if qty>abs(old)+Decimal('0.000000001'):
                errors.append([index,'close_crosses_zero'])
            realized_fill=-side*qty*(price-average)
            new=old+side*qty
            if abs(new)<Decimal('0.000000001'):new=Decimal(0)
        else:
            if f['reduce_only']:
                errors.append([index,'reduce_only_increases_risk'])
            new=old+side*qty
            average=(abs(old)*average+qty*price)/abs(new)
        if abs(Decimal(str(f['realized_pnl']))-realized_fill)>Decimal('.000001'):
            errors.append([index,'realized_fill_pnl_mismatch'])
        if abs(Decimal(str(f['position_after']))-new)>Decimal('0.000000001'):
            errors.append([index,'position_after_mismatch'])
        positions[coin]=(new,average)
        realized+=realized_fill;fees+=fee
    funding=sum((Decimal(str(r['amount'])) for r in document['funding']),Decimal(0))
    expected_cash=Decimal('1000000')+realized-fees+funding
    residual=Decimal(str(summary['cash']))-expected_cash
    if abs(residual)>Decimal('.00001'):
        errors.append(['balance','cash_identity_mismatch'])
    for coin,(qty,_) in positions.items():
        reported=Decimal(str(summary['positions'].get(coin,{}).get('qty',0)))
        if abs(qty-reported)>Decimal('0.000000001'):
            errors.append([coin,'terminal_position_mismatch'])
    if abs(realized-Decimal(str(summary['realized_pnl'])))>Decimal('.00001'):
        errors.append(['summary','realized_total_mismatch'])
    return dict(passed=not errors,fill_count=len(document['fills']),cash_residual=float(residual),
                reconstructed_realized=float(realized),fees=float(fees),funding=float(funding),errors=errors,
                scope='Arithmetic and fill timing only; this does not validate the economic hypothesis or actual venue execution.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--output',required=True)
    args=p.parse_args();results={}
    for name in args.paths:
        path=Path(name);results[str(path)]=audit(json.loads(path.read_text()))
    result=dict(passed=all(x['passed'] for x in results.values()),ledgers=results)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=result['passed'],ledgers=len(results),failures={k:v['errors'] for k,v in results.items() if not v['passed']})))
    if not result['passed']:raise SystemExit(1)
