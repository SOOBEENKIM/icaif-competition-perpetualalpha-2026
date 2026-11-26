"""PAS public preflight and explicit injected-client IOC boundary.

No key lookup. Existing isolated ETH executor is deliberately NOT reused: its
net-position risk assumptions are wrong for overlapping virtual owners.
"""
import time
from decimal import Decimal
import numpy as np
from competition_live.execution import prepare_pas_order
from competition_live.venue import phase_state,check_oi,account_equity
from hyperliquid.utils.types import Cloid
from .book import ReconciliationError
from bot.execution import OrderRejected


class PASGateway:
    def __init__(self,profile,public,client=None):
        self.profile=profile;self.public=public;self.client=client;self.checked={}
        if profile['dex']!='pas' or profile['rest_url']!='https://api.hyperliquid-testnet.xyz':
            raise ValueError('Only PAS testnet supported')

    def require_client(self):
        if self.client is None:raise ReconciliationError('No order client supplied; read-only gateway')
        if self.client.base_url!=self.profile['rest_url']:raise ReconciliationError('Wrong client network')
        if getattr(self.client,'account_address',None)!=self.profile['team_wallet']:
            raise ReconciliationError('Client is not explicitly bound to the registered team wallet')
        if not getattr(self.client,'orders_enabled',False):raise ReconciliationError('Order client is not explicitly enabled')

    def leverage_for_entry(self,coin,book,active):
        if active.get('leverage')=={'type':'cross','value':1}:return active
        j=book.symbols.index(coin[4:])
        # Configure an unused market on demand. Opposing virtual positions are
        # still exposure even when the physical net is zero.
        if any(abs(owner.qty[j])>1e-12 for owner in book.owners.values()):
            raise ReconciliationError('Shared-market leverage changed while an owner is exposed')
        self.require_client()
        response=self.client.update_leverage(1,coin,is_cross=True)
        if response.get('status')!='ok':raise ReconciliationError('Cross 1x setup rejected')
        active=self.public.post({'type':'activeAssetData','user':self.profile['team_wallet'],'coin':coin})
        if active.get('leverage')!={'type':'cross','value':1}:
            raise ReconciliationError('Cross 1x setup not confirmed')
        return active

    def preflight(self,plan,book):
        profile=self.profile;now=time.time();phase=phase_state(now,profile)
        if phase in ('before_start','ended'):raise ReconciliationError('Outside competition phase')
        if phase=='closing_only' and not plan['reduce_only']:
            raise ReconciliationError('Physical-increasing virtual close needs coordinated team end liquidation')
        state=self.public.account()
        if state['orders']:raise ReconciliationError('Outstanding wallet orders must be reconciled')
        if not -2000<=now*1000-int(state['positions']['time'])<=15000:
            raise ReconciliationError('Stale exchange account')
        held={r['position']['coin']:float(r['position']['szi']) for r in state['positions']['assetPositions']}
        coins=['pas:'+s for s in book.symbols]
        if set(held)-set(coins) or not np.allclose([held.get(c,0.) for c in coins],book.physical.qty,rtol=0,atol=1e-9):
            raise ReconciliationError('Teammate/our fills changed; refresh ownership before sending')
        value=account_equity(state,profile)
        if value['equity']<=0:raise ReconciliationError('Unfunded wallet')
        meta,contexts=self.public.market();j=coins.index(plan['coin'])
        if (meta['collateralToken']!=profile['collateral_token'] or
            [m['name'] for m in meta['universe']]!=coins):raise ReconciliationError('PAS identity changed')
        order=prepare_pas_order(profile,plan['coin'],plan['delta']>0,abs(plan['delta']),plan['price'],
            meta['universe'][j],contexts[j],current_qty=book.physical.qty[j],
            reduce_only=plan['reduce_only'],slot=plan['cloid'])
        if Decimal(order.size)!=abs(Decimal(str(plan['delta']))) or Decimal(order.price)!=Decimal(str(plan['price'])):
            raise ReconciliationError('Plan must already satisfy exchange precision')
        check_oi([order],state['positions']['assetPositions'],meta,contexts,
            self.public.post({'type':'perpDexLimits','dex':'pas'}))
        if not plan['reduce_only']:
            active=self.public.post({'type':'activeAssetData','user':profile['team_wallet'],'coin':plan['coin']})
            active=self.leverage_for_entry(plan['coin'],book,active)
            capacity=Decimal(active['maxTradeSzs'][0 if plan['delta']>0 else 1])
            if abs(Decimal(str(plan['delta'])))>capacity:raise ReconciliationError('Insufficient physical margin capacity')
        self.checked[plan['cloid']]=time.time()

    def send(self,plan):
        self.require_client()
        if time.time()-self.checked.get(plan['cloid'],-1e30)>2:
            raise ReconciliationError('Expired preflight; refresh before any order')
        self.checked.pop(plan['cloid'])
        # Exactly one submission. Unknown/rejected outcomes remain in the
        # dispatch journal until reconciled, never treated as an empty fill.
        response=self.client.order(plan['coin'],plan['delta']>0,abs(plan['delta']),plan['price'],
            {'limit':{'tif':'Ioc'}},reduce_only=plan['reduce_only'],cloid=Cloid.from_str(plan['cloid']))
        if response.get('status')!='ok':raise ReconciliationError('Submission requires reconciliation')
        statuses=response.get('response',{}).get('data',{}).get('statuses',[])
        if len(statuses)==1 and isinstance(statuses[0],dict) and set(statuses[0])=={'error'}:
            raise OrderRejected(str(statuses[0]['error']))
        if len(statuses)!=1 or 'filled' not in statuses[0]:raise ReconciliationError('No confirmed terminal IOC receipt')
        row=statuses[0]['filled']
        return dict(oid=row['oid'],filled_size=row['totalSz'])
