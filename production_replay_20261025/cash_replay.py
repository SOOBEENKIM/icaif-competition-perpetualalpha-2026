"""Exact public settlement amounts inside the unchanged paper decision path.

Cash is applied only at its settlement timestamp. The public records are
evaluation labels, never forecast features. No real exchange client is used.
"""
import copy
from unittest.mock import patch
from . import book_replay


class CashVenue(book_replay.BookVenue):
    def __init__(self,*args,cash_units,**kwargs):
        super().__init__(*args,**kwargs)
        self.cash_units={(u['coin'],u['hour']):u for u in cash_units}

    def settle(self,at_ms,account):
        while self.fidx<len(self.funding) and self.funding[self.fidx]['time']<=at_ms:
            f=self.funding[self.fidx];self.fidx+=1
            coin=f['coin'];j=self.profile['trading']['symbols'].index(coin[4:])
            qty=sum(x['signed_qty'] for x in account.physical.fills if x['asset']==j and x['time']<f['time']/1000)
            if not qty:continue
            unit=self.cash_units.get((coin,f['time']//3600000))
            if unit is None and float(f['fundingRate']):
                self.failures.append(dict(time=f['time']/1000,kind='missing_verified_settlement',symbol=coin[4:]))
                continue
            if unit is not None and abs(unit['rate']-float(f['fundingRate']))>1e-12:
                self.failures.append(dict(time=f['time']/1000,kind='verified_settlement_rate_mismatch',symbol=coin[4:]))
                continue
            amount=qty*unit['cash_per_unit'] if unit is not None else 0.
            multiplier=self.sc['receipt_multiplier'] if amount>0 else 1.
            amount*=multiplier
            row=dict(time=f['time'],delta=dict(coin=coin,szi=str(qty),usdc=str(amount)),synthetic=True)
            account.apply_actual_funding([row])
            self.cash_records.append(dict(row,verified_public_settlement=True,
                cash_unit=unit,maximum_rounding_error=abs(qty)*unit['maximum_unit_error']*multiplier if unit else 0.))


def run_cash(*args,cash_units,**kwargs):
    def factory(*a,**kw):return CashVenue(*a,cash_units=cash_units,**kw)
    # Decision/account/evaluator source is unchanged; replace only the
    # settlement adapter, including any quantity feedback from different cash.
    with patch.object(book_replay,'BookVenue',side_effect=factory):
        result=book_replay.run_book(*args,**kwargs)
    result['funding_mode']='verified_settlement_cash_policy_replayed'
    return result
