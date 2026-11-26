"""Lot-integer paper fills; never relax the live receipt consistency check."""
from decimal import Decimal,ROUND_FLOOR,ROUND_HALF_EVEN


def lots(value,factor):
    x=Decimal(str(value))*factor
    nearest=x.to_integral_value(rounding=ROUND_HALF_EVEN)
    # Same sub-lot numerical tolerance as the original sweep. Real fractions
    # of a lot are still floored; this only absorbs binary depth arithmetic.
    if abs(x-nearest)<=Decimal('1e-9'):return int(nearest)
    return int(x.to_integral_value(rounding=ROUND_FLOOR))


def sweep(levels,quantity,fraction,side,decimals,limit=None):
    factor=Decimal(10)**decimals
    remaining=max(0,lots(quantity,factor));filled=0;value=Decimal(0)
    for row in levels:
        price=Decimal(str(row['px']))
        if limit is not None and side*(float(price)-float(limit))>1e-10:break
        available=max(0,lots(Decimal(str(row['sz']))*Decimal(str(fraction)),factor))
        take=min(remaining,available)
        value+=price*take;filled+=take;remaining-=take
        if not remaining:break
    return float(Decimal(filled)/factor),float(value/filled) if filled else None
