"""PAR-only order adapter. Imported code does not enable or send any orders."""
import hashlib
import time
import numpy as np
from decimal import Decimal, ROUND_FLOOR
from dataclasses import replace

from hyperliquid.exchange import Exchange
from hyperliquid.utils.constants import TESTNET_API_URL
from hyperliquid.utils.types import Cloid
from bot.execution import Executor, LimitOrder, OrderRejected, OutcomeUnknown, rounded_price, response_status
from .profile import CANDIDATE
from .venue import PublicPAR, account_equity, check_oi, phase_state
from replay_lab.common_comparison.engine import risk_breaches


class PARExchange(Exchange):
    """Reject wrong network, market, schedule, and administrative actions."""
    orders_enabled=False

    def post(self,url_path,payload=None):
        if self.base_url!=TESTNET_API_URL:raise ValueError('Testnet only')
        if url_path=='/exchange':
            if not self.orders_enabled:raise ValueError('Order transmission disabled')
            state=phase_state(time.time(),self.scored_profile)
            if state in ('before_start','ended'):raise ValueError('Outside scored trading window')
            action=(payload or {}).get('action',{})
            kind=action.get('type')
            valid=set(self.par_asset_ids)
            if kind=='order':
                if not action.get('orders') or action.get('builder'):raise ValueError('Unsupported order action')
                for order in action['orders']:
                    if order.get('a') not in valid:raise ValueError('Non-PAR asset')
                    if state=='closing_only' and not order.get('r'):raise ValueError('Only reductions near phase end')
            elif kind in ('cancel','cancelByCloid'):
                field='a' if kind=='cancel' else 'asset'
                if any(c.get(field) not in valid for c in action.get('cancels',[])):raise ValueError('Non-PAR cancellation')
            elif kind=='updateLeverage':
                if action.get('asset') not in valid:raise ValueError('Non-PAR leverage change')
            else:raise ValueError('Administrative actions are disabled')
        elif url_path!='/info':raise ValueError('Unexpected API path')
        return super().post(url_path,payload)


def build_order_client(profile, signer, *, enable_orders=False):
    """Signer is supplied by the eventual launcher; no key lookup or account switch."""
    if profile['dex']!='par' or profile['candidate']!=CANDIDATE or profile['rest_url']!=TESTNET_API_URL:
        raise ValueError('Wrong scored strategy/profile')
    client=PARExchange(signer,TESTNET_API_URL,account_address=profile['team_wallet'],perp_dexs=['par'],timeout=15)
    client.scored_profile=profile
    client.par_asset_ids={client.info.coin_to_asset['par:'+s] for s in profile['trading']['symbols']}
    client.orders_enabled=bool(enable_orders)
    return client


def prepare_par_order(profile, coin, is_buy, size, price, market, context, *, current_qty=0., reduce_only=False, slot):
    if coin!='par:'+coin.split(':')[-1] or coin not in {'par:'+s for s in profile['trading']['symbols']}:
        raise ValueError('Only PAR markets are permitted')
    if market['name']!=coin or type(is_buy) is not bool or type(reduce_only) is not bool:
        raise ValueError('Order identity/type mismatch')
    q,px,old=Decimal(str(size)),Decimal(str(price)),Decimal(str(current_qty))
    if not all(v.is_finite() for v in (q,px,old)) or min(q,px)<=0:raise ValueError('Invalid order numbers')
    decimals=market['szDecimals'];lot=Decimal(1).scaleb(-decimals)
    q=q.quantize(lot,rounding=ROUND_FLOOR);px=rounded_price(px,decimals,is_buy)
    if min(q,px)<=0:raise ValueError('Rounded quantity/price is zero')
    delta=q*(1 if is_buy else -1)
    full_close=reduce_only and old*delta<0 and q==abs(old)
    if reduce_only and (old*delta>=0 or q>abs(old)):raise ValueError('Reduction exceeds held position')
    if q*px<10 and not full_close:raise ValueError('Below minimum; exception only for complete reduce-only close')
    if q*px>Decimal(str(profile['execution']['maximum_single_order_notional'])):
        raise ValueError('Per-order capital limit')
    mark=Decimal(str(context['markPx']))
    if mark<=0 or not mark.is_finite() or abs(px/mark-1)>Decimal('.05'):raise ValueError('Invalid price band')
    key=f"{profile['candidate']}:{profile['start_unix']}:par:{slot}"
    cloid='0x'+hashlib.sha256(key.encode()).hexdigest()[:32]
    return LimitOrder(coin,is_buy,format(q,'f'),format(px,'f'),'Ioc',reduce_only,cloid)


class PARExecutor(Executor):
    def __init__(self,client,profile,journal):
        super().__init__(client,{'team_wallet':profile['team_wallet']},
            max_notional=profile['execution']['maximum_single_order_notional'],journal=journal)
        self.profile=profile;self.public=PublicPAR(profile)
        self.expected_positions=None;self.risk_covariance=None

    def set_risk_state(self,qty,sigma):
        n=len(self.profile['trading']['symbols'])
        qty=np.asarray(qty,float);sigma=np.asarray(sigma,float)
        if qty.shape!=(n,) or sigma.shape!=(n,n) or not np.isfinite(qty).all() or not np.isfinite(sigma).all():
            raise ValueError('Invalid actual position/covariance state')
        if not np.allclose(sigma,sigma.T) or np.linalg.eigvalsh(sigma).min() < -1e-12:
            raise ValueError('Invalid covariance matrix')
        self.expected_positions=qty.copy();self.risk_covariance=sigma.copy()

    def market(self,coin):
        meta,ctx=self.public.market()
        for m,c in zip(meta['universe'],ctx):
            if m['name']==coin:return m,c
        raise ValueError('Unknown PAR market')

    def submit(self,order):
        self._require_orders()
        now=time.time();phase=phase_state(now,self.profile)
        if phase in ('before_start','ended') or (phase=='closing_only' and not order.reduce_only):
            raise ValueError('Order blocked by competition schedule')
        account=self.public.account();value=account_equity(account,self.profile)
        if not value['funded']:raise ValueError('USDPR funding has not arrived')
        if account['orders']:raise OutcomeUnknown('Reconcile open PAR orders before sending')
        server_ms=int(account['positions']['time'])
        if not -2000<=time.time()*1000-server_ms<=15000:raise ValueError('Stale account snapshot')
        meta,contexts=self.public.market()
        if meta['collateralToken']!=self.profile['collateral_token']:raise ValueError('Collateral changed')
        index=next(i for i,m in enumerate(meta['universe']) if m['name']==order.coin)
        held={p['position']['coin']:float(p['position']['szi']) for p in account['positions']['assetPositions']}
        coins=['par:'+s for s in self.profile['trading']['symbols']]
        if set(held)-set(coins) or self.expected_positions is None:
            raise ValueError('Selected coordinator position state required')
        actual=np.array([held.get(c,0.) for c in coins])
        if not np.allclose(actual,self.expected_positions,rtol=0,atol=1e-9):
            raise OutcomeUnknown('Exchange positions differ from reconciled strategy positions')
        fresh=prepare_par_order(self.profile,order.coin,order.is_buy,order.size,order.price,
            meta['universe'][index],contexts[index],current_qty=held.get(order.coin,0),
            reduce_only=order.reduce_only,slot=order.cloid)
        if fresh.size!=order.size or fresh.price!=order.price:raise ValueError('Order precision changed')
        limits=self.public.post({'type':'perpDexLimits','dex':'par'})
        check_oi([order],account['positions']['assetPositions'],meta,contexts,limits)
        if not order.reduce_only:
            active=self.public.post({'type':'activeAssetData','user':self.profile['team_wallet'],'coin':order.coin})
            if active.get('leverage')!={'type':'cross','value':1}:
                self.record('leverage_intent',coin=order.coin,value=1,is_cross=True)
                response=self.client.update_leverage(1,order.coin,is_cross=True)
                if response.get('status')!='ok':raise ValueError('Cross 1x leverage update rejected')
                active=self.public.post({'type':'activeAssetData','user':self.profile['team_wallet'],'coin':order.coin})
                if active.get('leverage')!={'type':'cross','value':1}:
                    raise ValueError('Cross 1x leverage update not yet confirmed')
                self.record('leverage_confirmed',coin=order.coin,value=1,is_cross=True)
            if Decimal(order.size)>Decimal(active['maxTradeSzs'][0 if order.is_buy else 1]):
                raise ValueError('Insufficient PAR trading capacity')
            projected=abs(held.get(order.coin,0)+(1 if order.is_buy else -1)*float(order.size))*float(contexts[index]['markPx'])
            cap=min(self.profile['trading']['risk']['asset_notional'],value['equity']*self.profile['trading']['risk']['asset_fraction'])
            if projected>cap+1e-8:raise ValueError('Per-asset risk limit')
            marks=np.array([float(c['markPx']) for c in contexts])
            delta=(1 if order.is_buy else -1)*float(order.size)
            proposed=actual.copy();proposed[index]+=delta
            conservative_equity=value['equity']-delta*(float(order.price)-marks[index])-float(order.notional)*self.profile['trading']['execution']['fee_bps']/10000
            if risk_breaches(proposed,conservative_equity,marks,self.risk_covariance,self.profile['trading']['risk']):
                raise ValueError('Aggregate account risk limit')
        # Caller must journal/claim the packet and reconcile by cloid after any
        # uncertain outcome; this adapter never retries an order automatically.
        self.record('submit_intent',strategy=CANDIDATE,dex='par',collateral='USDPR',order=order.__dict__)
        try:
            response=self.client.order(order.coin,order.is_buy,float(order.size),float(order.price),
                {'limit':{'tif':'Ioc'}},reduce_only=order.reduce_only,cloid=Cloid.from_str(order.cloid))
        except Exception:
            self.record('submit_unknown',cloid=order.cloid)
            raise OutcomeUnknown('Reconcile PAR cloid before any retry: '+order.cloid) from None
        self.record('submit_response',cloid=order.cloid,response=response)
        status=response_status(response,'order')
        if not isinstance(status,dict) or not ({'filled','resting'}&status.keys()):
            raise OutcomeUnknown('No confirmed PAR order outcome')
        return status

    def cancel(self,coin,cloid):
        self._require_orders()
        if coin not in {'par:'+s for s in self.profile['trading']['symbols']}:raise ValueError('Non-PAR cancel')
        self.record('cancel_intent',coin=coin,cloid=cloid)
        try:response=self.client.cancel_by_cloid(coin,Cloid.from_str(cloid))
        except Exception:raise OutcomeUnknown('Uncertain PAR cancellation') from None
        if response_status(response,'cancel')!='success':raise OutcomeUnknown('Cancellation not confirmed')

    def account_state(self):
        a=self.public.account()
        return {'positions':a['positions']['assetPositions'],'open_orders':a['orders']}
