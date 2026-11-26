"""Isolated historical broker for all ten competition assets.

This module imports no wallet, exchange SDK, networking or secret configuration.
Positions, cash and orders exist only in the in-memory HistoricalBroker.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math

import numpy as np

from ..model import SYMBOLS
COINS=SYMBOLS


def quantity(value,decimals):
    return float(Decimal(str(max(0.,value))).quantize(Decimal(1).scaleb(-decimals),rounding=ROUND_FLOOR))


def execution_price(price,sign,spread,decimals):
    value=Decimal(str(price*(1+sign*spread/10000)))
    places=max(0,min(6-decimals,4-value.adjusted()))
    return float(value.quantize(Decimal(1).scaleb(-places),rounding=ROUND_CEILING if sign>0 else ROUND_FLOOR))


class HistoricalBroker:
    mode='historical'
    def __init__(self,config):
        self.config=config;self.cash=float(config['initial'])
        self.qty=np.zeros(len(COINS));self.entry=np.zeros(len(COINS))
        self.realized=self.fees=self.funding=self.turnover=0.
        self.fills=[];self.funding_events=[];self.orders=[];self.next_id=1

    def equity(self,prices):return float(self.cash+self.qty@(prices-self.entry))

    def order(self,j,signed_qty,price,capacity,time,reduce_only,reason,basket_id):
        if not math.isfinite(signed_qty) or not math.isfinite(price) or price<=0:raise ValueError('Invalid order')
        sign=1 if signed_qty>0 else -1;s=COINS[j];before=self.qty[j]
        requested=quantity(abs(signed_qty),self.config['decimals'][s]);oid=self.next_id;self.next_id+=1
        order=dict(id=oid,symbol=s,requested_qty=requested,sign=sign,reduce_only=reduce_only,
                   time=int(time),reason=reason,basket_id=basket_id)
        if reduce_only:
            available=abs(before) if before*sign<0 else 0.
        else:
            available=max(0,self.config['per_market_notional_cap']/price-abs(before))
            if before*sign<0:available=0.
        size=quantity(min(requested,max(0,capacity),available),self.config['decimals'][s])
        # Closing dust is allowed in this adapter, explicitly tagged rather than flipped.
        if not reduce_only and size*price<self.config['minimum_order_value']:size=0.
        realized=0.
        if size:
            if reduce_only:realized=size*(price-self.entry[j])*(1 if before>0 else -1)
            else:self.entry[j]=(abs(before)*self.entry[j]+size*price)/(abs(before)+size)
            self.qty[j]=round(before+sign*size,self.config['decimals'][s])
            if not self.qty[j]:self.entry[j]=0.
            fee=size*price*self.config['fee_bps']/10000
            self.realized+=realized;self.fees+=fee;self.cash+=realized-fee;self.turnover+=size*price
            self.fills.append(dict(order_id=oid,time=int(time),symbol=s,side='BUY' if sign>0 else 'SELL',
                                   qty=size,price=price,fee=fee,realized=realized,reduce_only=reduce_only,
                                   position_before=float(before),position_after=float(self.qty[j]),
                                   reason=reason,basket_id=basket_id))
        self.orders.append(dict(**order,filled_qty=size,status='FILLED' if size>=requested and size>0 else 'IOC_REMAINDER_CANCELLED'))
        return size

    def charge_funding(self,time,prices,rates,observed,adverse_daily=0):
        cash=-float(np.sum(self.qty*prices*rates))
        stress=-float(np.sum(abs(self.qty)*prices))*adverse_daily/10000/24
        total=cash+stress
        self.cash+=total;self.funding+=total
        if np.any(self.qty):
            self.funding_events.append(dict(time=int(time),amount=total,rate_amount=cash,stress_amount=stress,
                                            rates=rates.tolist(),observed=observed.tolist(),
                                            oracle_proxy='reference open',signed_notional=(self.qty*prices).tolist()))


