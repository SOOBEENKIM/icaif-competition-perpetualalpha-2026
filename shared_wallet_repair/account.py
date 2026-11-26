"""Disjoint market ownership, confirmed fills only; no exchange mutations.

ETH is reserved for the confirmed teammate strategy. This deliberately does not
pretend two independent writers can safely own netted inventory in one market.
"""
import copy
import hashlib
import json
import numpy as np
from competition_live.live_account import LiveAccount
from replay_lab.common_comparison.engine import Ledger
from bot.execution import OutcomeUnknown

PROTECTED='pas:ETH'


def external_ledger(profile,fills,funding,own_cloids,cutoff_ms):
    """Rebuild instead of incrementally crediting a partially saved reserve."""
    ledger=Ledger(profile['reserved_capital'],profile['trading']['symbols'])
    j=ledger.symbols.index('ETH');events=[];seen=set()
    for n,f in enumerate(fills):
        if f['coin']!=PROTECTED or f['time']>cutoff_ms:continue
        identity=f['coin'],f['tid']
        if identity in seen:continue
        seen.add(identity)
        if f.get('cloid') in own_cloids:raise OutcomeUnknown('Own ETH order conflicts with teammate market reservation')
        events.append((f['time'],1,n,f))
    seen=set()
    for n,f in enumerate(funding):
        if f['delta'].get('coin')!=PROTECTED or f['time']>cutoff_ms:continue
        identity=hashlib.sha256(json.dumps(f,sort_keys=True).encode()).hexdigest()
        if identity in seen:continue
        seen.add(identity);events.append((f['time'],0,n,f))
    for time,kind,_,f in sorted(events):
        if kind:
            if f['feeToken']!='USDP' or abs(ledger.qty[j]-float(f['startPosition']))>1e-8:
                raise OutcomeUnknown('Teammate ETH fill history is incomplete or uses another collateral')
            q=float(f['sz'])*(1 if f['side']=='B' else -1);price=float(f['px']);fee=float(f['fee'])
            if not np.isfinite([q,price,fee]).all() or not q or price<=0:
                raise OutcomeUnknown('Invalid teammate ETH fill')
            rate=fee/(abs(q)*price)*10000
            if ledger.qty[j]*q<0 and abs(q)>abs(ledger.qty[j]):
                old=float(ledger.qty[j]);ledger.fill(j,-old,price,rate,time/1000,source='teammate_confirmed')
                q+=old
            if q:ledger.fill(j,q,price,rate,time/1000,source='teammate_confirmed')
        else:
            d=f['delta'];cash=float(d['usdc'])
            if any(x['coin']==PROTECTED and x['time']==time for x in fills):
                raise OutcomeUnknown('Teammate fill and funding ordering is ambiguous')
            if abs(ledger.qty[j]-float(d['szi']))>1e-8 or not np.isfinite(cash):
                raise OutcomeUnknown('Teammate funding does not match confirmed inventory')
            ledger.cash+=cash;ledger.shadow_cash+=cash;ledger.funding+=cash;ledger.signed_funding+=cash
            ledger.funding_events.append(dict(time=time/1000,signed_amount=cash,source='actual_userFunding',coin=PROTECTED))
    return ledger


class SharedWalletAccount(LiveAccount):
    def apply_actual_funding(self,rows):
        # The reserve is rebuilt with actual ETH funding first by SharedRunner.
        super().apply_actual_funding([r for r in rows if r['delta'].get('coin')!=PROTECTED])

    def execute_books(self,intent,*args,**kwargs):
        j=self.physical.symbols.index('ETH')
        if self.physical.qty[j] or any(o.qty[j] for o in self.owners.values()):
            raise OutcomeUnknown('User-owned ETH conflicts with protected teammate inventory')
        if any(abs(v[j])>1e-12 for v in intent['owner_targets'].values()):
            self._take(intent)
            # Reject the whole packet before either leg; never create a naked
            # BTC leg when the paired ETH order is prohibited.
            self.checkpoint('teammate_market_packet_blocked',{'coin':PROTECTED})
            return dict(orders=[dict(symbol='ETH',status='TEAMMATE_MARKET_RESERVED',filled_qty=0.)],
                        unknown_symbols=[],orders_sent=0)
        return super().execute_books(intent,*args,**kwargs)


def market_marks(state,profile):
    marks=np.zeros(len(profile['trading']['symbols']))
    for row in state['assetPositions']:
        p=row['position'];s=p['coin'].split(':',1)[1]
        if s not in profile['trading']['symbols']:raise OutcomeUnknown('Unexpected position market')
        q=float(p['szi']);value=float(p.get('positionValue',0))
        if q:
            marks[profile['trading']['symbols'].index(s)]=value/abs(q) if value else float(p['entryPx'])+float(p['unrealizedPnl'])/q
    return marks


def reconcile_shared(account,snapshot,profile):
    from competition_live.venue import account_equity
    from competition_live.runner import BalanceMismatch
    value=account_equity(snapshot,profile)
    if account.pending_orders or account.unknown:
        raise OutcomeUnknown('Unresolved user orders still require reconciliation')
    if any(o['coin']!=PROTECTED for o in snapshot['orders']):
        raise OutcomeUnknown('Unrecognized order outside teammate ETH market')
    j=account.physical.symbols.index('ETH')
    if account.physical.qty[j] or any(o.qty[j] for o in account.owners.values()):
        raise OutcomeUnknown('Overlapping user/teammate ETH inventory')
    combined=account.physical.qty+account.reserved.qty
    rows={r['position']['coin']:r['position'] for r in snapshot['positions']['assetPositions']}
    actual=np.array([float(rows.get('pas:'+s,{}).get('szi',0)) for s in account.physical.symbols])
    if not np.allclose(actual,combined,rtol=0,atol=1e-9):
        raise OutcomeUnknown('Combined user/teammate inventory differs from exchange')
    for k,s in enumerate(account.physical.symbols):
        owner=account.reserved if k==j else account.physical
        if actual[k] and abs(float(rows['pas:'+s]['entryPx'])-owner.entry[k])*abs(actual[k])>.05:
            raise OutcomeUnknown('Combined average entry does not match confirmed fills')
    expected=[];user_values=[];external_values=[]
    for state in [snapshot.get('positions_before',snapshot['positions']),snapshot['positions']]:
        marks=market_marks(state,profile)
        user_values.append(account.reconcile(marks));external_values.append(account.reserved.reconcile(marks))
        expected.append(user_values[-1]+external_values[-1])
    if not min(expected)-.05<=value['equity']<=max(expected)+.05:
        raise BalanceMismatch('Combined cash/funding differs from observed USDP equity')
    value.update(ledger_team_equity=expected[-1],strategy_equity=user_values[-1],teammate_equity=external_values[-1])
    return value


def isolated_snapshot(snapshot,profile,reserve):
    """Explicit risk view of our sleeve for the unchanged order-size checks.

    The real snapshot is still checked by the runner. The executor independently
    checks teammate ETH quantity/entry here before constructing this view.
    """
    result=copy.deepcopy(snapshot);j=reserve.symbols.index('ETH')
    state=snapshot['positions'];rows={r['position']['coin']:r['position'] for r in state['assetPositions']}
    eth=rows.get(PROTECTED,{});qty=float(eth.get('szi',0))
    if abs(qty-reserve.qty[j])>1e-9 or (qty and abs(float(eth['entryPx'])-reserve.entry[j])*abs(qty)>.05):
        raise OutcomeUnknown('Teammate inventory changed after reconciliation; refresh before ordering')
    external=[]
    for key in ['positions_before','positions']:
        if key not in snapshot:continue
        external.append(reserve.equity(market_marks(snapshot[key],profile)))
        result[key]['assetPositions']=[r for r in result[key]['assetPositions'] if r['position']['coin']!=PROTECTED]
    deduction=max(external)-profile['reserved_capital']
    if snapshot['mode']!='unifiedAccount':raise OutcomeUnknown('Shared adapter requires verified unified USDP accounting')
    for balance in result['spot']['balances']:
        if balance['coin']=='USDP':balance['total']=str(float(balance['total'])-deduction)
    result['orders']=[o for o in result['orders'] if o['coin']!=PROTECTED]
    return result
