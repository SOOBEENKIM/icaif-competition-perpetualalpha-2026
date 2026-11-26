"""PAR public adapter and account valuation, with strict market isolation."""
from decimal import Decimal
import math
import time
import requests


class PublicPAR:
    def __init__(self, profile):
        if profile['dex']!='par' or profile['rest_url']!='https://api.hyperliquid-testnet.xyz':
            raise ValueError('PAR testnet required')
        self.profile=profile
        self.session=requests.Session()

    def post(self, payload):
        allowed={'perpDexs','meta','metaAndAssetCtxs','perpDexLimits','spotMeta','userAbstraction',
                 'clearinghouseState','spotClearinghouseState','openOrders','userFees','extraAgents',
                 'userRateLimit','activeAssetData','l2Book','orderStatus','userFillsByTime',
                 'userFunding','userNonFundingLedgerUpdates','fundingHistory'}
        kind=payload.get('type')
        if kind not in allowed: raise ValueError('Only public info queries are allowed')
        if kind in {'meta','metaAndAssetCtxs','perpDexLimits','clearinghouseState','openOrders'} and payload.get('dex')!='par':
            raise ValueError('PAR dex selector is mandatory')
        if 'coin' in payload and payload['coin'] not in {'par:'+s for s in self.profile['trading']['symbols']}:
            raise ValueError('Wrong execution market')
        response=self.session.post(self.profile['rest_url']+'/info',json=payload,timeout=15)
        response.raise_for_status()
        return response.json()

    def market(self):
        return self.post({'type':'metaAndAssetCtxs','dex':'par'})

    def account(self):
        user=self.profile['team_wallet']
        mode=self.post({'type':'userAbstraction','user':user})
        before=self.post({'type':'clearinghouseState','user':user,'dex':'par'})
        spot=self.post({'type':'spotClearinghouseState','user':user})
        after=self.post({'type':'clearinghouseState','user':user,'dex':'par'})
        return dict(mode=mode,positions=after,positions_before=before,spot=spot,
            orders=self.post({'type':'openOrders','user':user,'dex':'par'}))


def account_equity(snapshot, profile):
    """Observed unified USDPR total already includes the PAR mark-to-market PnL."""
    positions=snapshot['positions']['assetPositions']
    expected={'par:'+s for s in profile['trading']['symbols']}
    if any(p['position']['coin'] not in expected for p in positions):
        raise ValueError('Unexpected position market')
    mode=snapshot['mode']
    if mode=='unifiedAccount':
        matching=[b for b in snapshot['spot']['balances'] if b['coin']=='USDPR'
                  and b['token']==profile['collateral_token']]
        if len(matching)>1: raise ValueError('Duplicate USDPR balances')
        value=Decimal(matching[0]['total']) if matching else Decimal(0)
        source='spotClearinghouseState.USDPR.total (already marked to market)'
    elif mode in ('disabled','default','standard'):
        value=Decimal(snapshot['positions']['marginSummary']['accountValue'])
        source='clearinghouseState(dex=par).marginSummary.accountValue'
    else:
        raise ValueError('Unsupported account abstraction mode: '+str(mode))
    if not value.is_finite(): raise ValueError('Nonfinite account equity')
    return dict(equity=float(value),source=source,mode=mode,
        funded=bool(value>0),positions=len(positions),open_orders=len(snapshot['orders']))


def validate_metadata(snapshot,profile):
    meta=snapshot['meta'];universe=meta['universe']
    coins=['par:'+s for s in profile['trading']['symbols']]
    if [m['name'] for m in universe]!=coins or meta['collateralToken']!=profile['collateral_token']:
        raise ValueError('Market/collateral identity changed')
    token=next(t for t in snapshot['spot_meta']['tokens'] if t['index']==meta['collateralToken'])
    if token['name']!='USDPR': raise ValueError('Wrong collateral name')
    for m,precision in zip(universe,profile['trading']['execution']['decimals']):
        if m['szDecimals']!=precision or m.get('isDelisted') or m.get('onlyIsolated'):
            raise ValueError('PAR market specifications changed: '+m['name'])
    caps=dict(snapshot['limits']['coinToOiCap'])
    if set(caps)!=set(coins):raise ValueError('Missing PAR market caps')
    if any(float(v)!=profile['venue_caps']['per_market'] for v in caps.values()):
        raise ValueError('Per-market caps differ from announcement; review required')
    if float(snapshot['limits']['totalOiCap'])!=profile['venue_caps']['total']:
        raise ValueError('Global cap differs from announcement; review required')
    # At fee scale 1 and normal mode positive base rates are doubled.
    rates=[]
    for m in universe:
        if m.get('growthMode',False) or float(m.get('deployerFeeScale','1'))!=1:
            raise ValueError('Fee mode changed; cannot reuse the calibrated 9bp assumption')
        rates.append(float(snapshot['fees']['userCrossRate'])*2*10000)
    if any(not math.isclose(r,profile['gate_spec']['fee_bps']) for r in rates):
        raise ValueError('Actual fee tier differs from selected model cost assumption')
    return dict(markets=len(universe),collateral=token['name'],collateral_token=token['index'],
        size_decimals={m['name']:m['szDecimals'] for m in universe},
        taker_bps=rates[0],maker_bps=float(snapshot['fees']['userAddRate'])*2*10000,
        per_market_cap=profile['venue_caps']['per_market'],global_cap=profile['venue_caps']['total'])


def phase_state(now,profile):
    if now<profile['start_unix']:return 'before_start'
    if now>=profile['end_unix']:return 'ended'
    if now>=profile['end_unix']-profile['execution']['end_liquidation_seconds']:return 'closing_only'
    return 'trading'


def check_oi(orders,positions,meta,contexts,limits):
    """Conservative cap preview: each incremental leg may increase market OI.

    Venue checks remain authoritative because other teams can trade after this read.
    """
    if len(meta['universe'])!=len(contexts): raise ValueError('Unaligned market contexts')
    ctx={m['name']:c for m,c in zip(meta['universe'],contexts)}
    caps={k:Decimal(v) for k,v in limits['coinToOiCap']}
    qty={p['position']['coin']:Decimal(p['position']['szi']) for p in positions}
    valuation={c:max(Decimal(v['oraclePx']),Decimal(v.get('markPx',v['oraclePx']))) for c,v in ctx.items()}
    if any(not x.is_finite() or x<=0 for x in valuation.values()):raise ValueError('Invalid OI valuation')
    current={c:Decimal(v['openInterest'])*valuation[c] for c,v in ctx.items()}
    if any(not x.is_finite() or x<0 for x in current.values()):raise ValueError('Invalid OI data')
    added={c:Decimal(0) for c in ctx}
    for o in orders:
        coin=o.coin
        if coin not in ctx or coin not in caps:raise ValueError('Unknown OI market')
        old=qty.get(coin,Decimal(0));delta=Decimal(o.size)*(1 if o.is_buy else -1)
        if o.reduce_only:
            if old*delta>=0 or abs(delta)>abs(old):raise ValueError('Invalid reduce-only quantity')
        increment=max(Decimal(0),abs(old+delta)-abs(old))*valuation[coin]
        added[coin]+=increment;qty[coin]=old+delta
    for coin,increment in added.items():
        if increment>0 and current[coin]+increment>caps[coin]:raise ValueError('Market OI cap: '+coin)
    if sum(added.values())>0 and sum(current.values())+sum(added.values())>Decimal(limits['totalOiCap']):
        raise ValueError('DEX OI cap')
    return dict(current_oi=float(sum(current.values())),increment_upper_bound=float(sum(added.values())))
