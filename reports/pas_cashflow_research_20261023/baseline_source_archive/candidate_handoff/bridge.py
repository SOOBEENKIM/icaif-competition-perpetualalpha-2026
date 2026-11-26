"""Actual-fill adapter between the retained base/pair account and team netting."""
import copy,json,time
from decimal import Decimal
from pathlib import Path
import numpy as np
from competition_live.runner import history
from competition_live.live_account import LiveAccount
from competition_live.runtime import fingerprint
from joint_wallet.audit import reconstruct
from joint_wallet.book import ReconciliationError,credit
from joint_wallet.dispatch import Dispatcher,SubmissionNotAttempted
from joint_wallet.pas_gateway import PASGateway
from bot.execution import OrderRejected


def owned_ids(root,dispatch_directory):
    cloids=set();oids=set()
    journal=Path(root)/'.runtime/pas_competition_live/orders.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            e=json.loads(line)
            if e.get('event')=='submit_intent':cloids.add(e['order']['cloid'])
            if e.get('event')=='submit_response':
                for s in e.get('response',{}).get('response',{}).get('data',{}).get('statuses',[]):
                    if 'filled' in s:oids.add(s['filled']['oid'])
    path=Path(dispatch_directory)/'dispatch.json'
    if path.exists():
        state=json.loads(path.read_text())
        for key,row in state['completed'].items():
            if row['owner']=='ours':cloids.add(key);oids.add(row['oid'])
        pending=state.get('pending')
        if pending and pending['plan']['owner']=='ours':
            cloids.add(pending['plan']['cloid'])
            if pending.get('receipt'):oids.add(pending['receipt']['oid'])
    return cloids,oids


class HistoryBridge:
    def __init__(self,root,directory,profile,public,*,settled_units=None):
        self.root=Path(root);self.directory=Path(directory);self.profile=profile;self.public=public
        self.settled_units=settled_units or {};self.latest=None;self.rows=None;self.funding=None

    def refresh(self):
        p=self.profile;end=int(time.time()*1000);start=p['start_unix']*1000
        cloids,oids=owned_ids(self.root,self.directory)
        fills=history(self.public,'userFillsByTime',p,start,end)
        funding=history(self.public,'userFunding',p,start,end)
        self.latest=reconstruct(p['trading']['symbols'],{'ours':50000.,'teammate':50000.},fills,funding,cloids,oids,settled_units=self.settled_units)
        self.rows=fills;self.funding=funding;return self.latest

    def reconcile_exchange(self,book):
        from competition_live.venue import account_equity
        state=self.public.account();symbols=self.profile['trading']['symbols']
        if state['orders']:raise ReconciliationError('Outstanding orders must be resolved before handoff')
        if not -2000<=time.time()*1000-int(state['positions']['time'])<=15000:raise ReconciliationError('Stale account')
        rows={r['position']['coin']:r['position'] for r in state['positions']['assetPositions']}
        if set(rows)-{'pas:'+s for s in symbols}:raise ReconciliationError('Unknown physical market')
        actual=np.array([float(rows.get('pas:'+s,{}).get('szi',0)) for s in symbols])
        if not np.allclose(actual,book.physical.qty,rtol=0,atol=1e-9):raise ReconciliationError('Exchange inventory changed during history read')
        meta,ctx=self.public.market();marks=np.array([float(x['markPx']) for x in ctx])
        for j,s in enumerate(symbols):
            if actual[j]:
                pos=rows['pas:'+s]
                if abs(float(pos['entryPx'])-book.physical.entry[j])*abs(actual[j])>.05:raise ReconciliationError('Physical average entry differs')
                marks[j]=float(pos['entryPx'])+float(pos['unrealizedPnl'])/actual[j]
        errors=book.reconcile(marks);equity=account_equity(state,self.profile)['equity']
        # In unified mode the marked spot balance was read BETWEEN two
        # position snapshots. Compare against both observed valuations, not
        # only the later mark. Quantities and entries must agree in both.
        valuations=[]
        for snapshot in (state.get('positions_before',state['positions']),state['positions']):
            positions={r['position']['coin']:r['position'] for r in snapshot['assetPositions']}
            if set(positions)-{'pas:'+s for s in symbols}:raise ReconciliationError('Unknown physical market')
            q=np.array([float(positions.get('pas:'+s,{}).get('szi',0)) for s in symbols])
            if not np.allclose(q,book.physical.qty,rtol=0,atol=1e-9):
                raise ReconciliationError('Exchange inventory changed across valuation snapshots')
            for j,s in enumerate(symbols):
                if q[j] and abs(float(positions['pas:'+s]['entryPx'])-book.physical.entry[j])*abs(q[j])>.05:
                    raise ReconciliationError('Physical average entry differs across snapshots')
            valuations.append(book.physical.cash+sum(float(r['unrealizedPnl']) for r in positions.values()))
        if state['mode']!='unifiedAccount':valuations=valuations[-1:]
        if not min(valuations)-.05<=equity<=max(valuations)+.05:
            raise ReconciliationError('Equity outside observed valuation interval; funding/transfer or snapshot race')
        return dict(snapshot=state,marks=marks,conservation=errors,
            equity_error=equity-book.physical.equity(marks),observed_valuation_interval=[min(valuations),max(valuations)])

    def sync(self,account,book,marks):
        if account.pending_orders or account.unknown:raise ReconciliationError('Unresolved old order; do not migrate')
        if not np.allclose(account.physical.qty,book.owners['ours'].qty,rtol=0,atol=1e-9):raise ReconciliationError('Own confirmed fills differ from old checkpoint')
        if abs(account.physical.fees-book.owners['ours'].fees)>1e-6:raise ReconciliationError('Own fees differ')
        if np.max(abs(account.physical.entry-book.owners['ours'].entry)*abs(account.physical.qty))>.05:
            raise ReconciliationError('Own cost basis differs from confirmed history')
        known=set(account.applied_funding)
        for event in book.events:
            if event['kind']!='funding' or event['identity'] in known:continue
            coin=event['coin'];j=account.physical.symbols.index(coin[4:]);at=event['time']/1000
            unit=event['cash']/event['net_qty'] if abs(event['net_qty'])>1e-12 else event['cash_per_unit']
            if unit is None:raise ReconciliationError('Missing observed settlement for offsetting owners')
            quantities={name:sum(f['signed_qty'] for f in ledger.fills if f['asset']==j and f['time']<at) for name,ledger in account.owners.items()}
            own=sum(quantities.values())
            credit(account.physical,own*unit,at,coin,event['identity'])
            for name,qty in quantities.items():credit(account.owners[name],qty*unit,at,coin,event['identity'])
            account.applied_funding.add(event['identity'])
        account.reserved=copy.deepcopy(book.owners['teammate'])
        ours=account.reconcile(marks);external=book.owners['ours'].equity(marks)
        if abs(ours-external)>1e-6 or abs(account.physical.funding-book.owners['ours'].funding)>1e-6:
            raise ReconciliationError('Own PnL/funding does not match inherited event book')
        return dict(own_equity=ours,teammate_equity=account.reserved.equity(marks),conservation=book.reconcile(marks))

    def settlement(self,at,account):
        # A public fill/funding publication can race the following account
        # snapshot. Retry only reads; never retry a submission here.
        for attempt in range(3):
            try:
                book=self.refresh();checked=self.reconcile_exchange(book);break
            except ReconciliationError:
                if attempt==2:raise
        self.sync(account,book,checked['marks'])
        account.checkpoint('joint_confirmed_settlement',dict(book_version=book.version))


class PreviewOnly(RuntimeError):
    def __init__(self,plan):self.plan=plan;super().__init__('Unsigned preview stopped before transmission')


class JointExecutor:
    """Same LiveAccount path for preview and actual dispatch, with explicit I/O."""
    def __init__(self,profile,directory,bridge,*,lease=None,client=None,preview=False):
        self.profile=profile;self.bridge=bridge;self.lease=lease;self.preview=preview
        self.gateway=PASGateway(profile,bridge.public,client)
        self.dispatch=Dispatcher(directory,profile);self.expected=None;self.sigma=None
        self.confirmed_rows={}

    def market(self,coin):
        meta,ctx=self.bridge.public.market()
        j=[m['name'] for m in meta['universe']].index(coin);return meta['universe'][j],ctx[j]

    def set_risk_state(self,qty,sigma):
        self.expected=np.asarray(qty,float).copy();self.sigma=np.asarray(sigma,float).copy()

    def plan(self,order):
        b=self.bridge.refresh();checked=self.bridge.reconcile_exchange(b)
        if self.expected is None or not np.allclose(b.owners['ours'].qty,self.expected,rtol=0,atol=1e-9):raise ReconciliationError('Own holdings changed before send')
        j=b.symbols.index(order.coin[4:]);delta=float(order.size)*(1 if order.is_buy else -1)
        plan=b.plan('ours',order.coin[4:],float(Decimal(str(self.expected[j]))+Decimal(str(delta))),float(order.price),
            self.profile['trading']['execution']['decimals'][j],max_notional=self.profile['execution']['maximum_single_order_notional'],slot=order.cloid)
        if plan is None:raise ReconciliationError('Requested owner change is not an executable exchange order')
        # Keep the original durable pending-order cloid for receipt recovery.
        plan['cloid']=order.cloid
        return b,plan,checked['marks']

    def submit(self,order):
        if not self.preview:
            if self.lease is None:raise ReconciliationError('No shared writer lease')
            self.lease.require()
        try:book,plan,marks=self.plan(order)
        except ReconciliationError as e:
            if self.preview:raise
            # This path has not invoked the dispatcher or venue sender.
            raise OrderRejected(str(e)) from e
        if self.preview:
            self.dispatch.validate_request(plan,book,marks,self.sigma)
            self.gateway.preflight(plan,book)
            raise PreviewOnly(plan)
        def preflight(p,b):self.lease.require();return self.gateway.preflight(p,b)
        def send(p):self.lease.require();return self.gateway.send(p)
        def confirmed(oid,cloid):
            for _ in range(6):
                rows=history(self.bridge.public,'userFillsByTime',self.profile,self.profile['start_unix']*1000,int(time.time()*1000))
                matching={(r['coin'],r['tid']):r for r in rows if r['oid']==oid and r['coin']==plan['coin']}
                if matching:
                    self.confirmed_rows[oid]=list(matching.values())
                    return self.confirmed_rows[oid]
                time.sleep(1)
            return []
        try:result=self.dispatch.execute(plan,marks,self.sigma,refresh=self.bridge.refresh,preflight=preflight,send=send,confirmed_fills=confirmed)
        except SubmissionNotAttempted as e:raise OrderRejected(str(e)) from e
        return dict(filled=dict(oid=result['oid'],totalSz=result['filled_size']))
