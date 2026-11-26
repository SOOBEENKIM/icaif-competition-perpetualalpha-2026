"""Conserved physical account, separately owned active and liquidating inventory."""
import copy
from itertools import product
import numpy as np
from replay_lab.common_comparison.engine import Ledger,risk_breaches
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount,difference
from experiments.published_pair_lifecycle_20261017.account import BookAccount


class ResidualOwnership:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.owners['residual']=Ledger(0.,self.physical.symbols)
        self.active_episode=None;self.episode_books={};self.episode_meta={}
        self.residual_by_asset={};self.residuals=[];self.handoffs=[]

    def start_episode(self,proposal,now):
        if self.unknown:raise ValueError('Unreconciled orders prevent new episode')
        if np.any(self.owners['pair'].qty):raise ValueError('Active owner not flat')
        if any(self.owners['residual'].qty[j] for j in proposal['ids']):raise ValueError('Residual symbol overlap')
        eid=len(self.episode_books)
        self.active_episode=eid;self.episode_books[eid]=Ledger(0.,self.physical.symbols)
        self.episode_meta[eid]=dict(episode=eid,pair=proposal['pair'],ids=list(proposal['ids']),start=now)
        return eid

    def owner_fill(self,name,j,amount,price,fee,time,source):
        before=len(self.owners[name].fills)
        eid=self.active_episode if name=='pair' else self.residual_by_asset.get(j) if name=='residual' else None
        if name!='base' and eid is None:raise ValueError('Missing episode ownership')
        if name=='residual' and source!='residual_transfer':
            old=self.owners[name].qty[j]
            if old*amount>=0 or abs(amount)>abs(old)+1e-10:raise ValueError('Residual can only reduce')
        super().owner_fill(name,j,amount,price,fee,time,source)
        for fill in self.owners[name].fills[before:]:
            fill['episode_id']=eid
            if eid is not None and source!='residual_transfer':
                self.episode_books[eid].fill(j,fill['signed_qty'],price,fee,time,allocation_type=source,owner=name)
        if name=='residual' and self.owners[name].qty[j]==0:self.residual_by_asset.pop(j,None)

    def charge_funding(self,time,prices,rates,observed):
        super().charge_funding(time,prices,rates,observed)
        for book in self.episode_books.values():book.charge_funding(time,prices,rates,observed,0.)

    def reconcile(self,marks):
        equity=super().reconcile(marks)
        if not hasattr(self,'episode_books'):return equity
        pair=self.owners['pair'];residual=self.owners['residual']
        books=list(self.episode_books.values())
        qty=sum((b.qty for b in books),np.zeros_like(pair.qty))
        errors=[float(np.max(abs(qty-pair.qty-residual.qty))),
            abs(sum(b.reconcile(marks) for b in books)-pair.equity(marks)-residual.equity(marks)),
            abs(sum(b.fees for b in books)-pair.fees-residual.fees),
            abs(sum(b.funding for b in books)-pair.funding-residual.funding)]
        self.maximum_error=max(self.maximum_error,*errors)
        assert max(errors)<1e-6,('episode conservation',errors)
        return equity

    def seal(self,intent,p,sc):
        targets=intent['owner_targets']
        if 'residual' not in targets:raise ValueError('Residual owner target required')
        if np.any(targets['residual']):raise ValueError('Residual target must remain zero')
        if np.any((np.asarray(targets['pair'])!=0)&(self.owners['residual'].qty!=0)):
            raise ValueError('Active target overlaps residual inventory')
        return super().seal(intent,p,sc)

    def cross(self,intent,j,price,time):
        if self.needs(intent)['residual'][j]==0:return super().cross(intent,j,price,time)
        priority=['residual','pair','base']
        for buyer in priority:
            for seller in priority:
                if buyer==seller:continue
                needs=self.needs(intent);buy=needs[buyer][j];sell=needs[seller][j]
                if buy<=0 or sell>=0:continue
                limits=intent['sealed_owner_limits']
                a=limits.get((buyer,j));b=limits.get((seller,j))
                if (a is not None and price>a) or (b is not None and price<b):continue
                quantity=min(buy,-sell)
                self.owner_fill(buyer,j,quantity,price,0.,time,'internal_transfer')
                self.owner_fill(seller,j,-quantity,price,0.,time,'internal_transfer')
                self.transfers.append(dict(time=time,asset=j,buyer=buyer,seller=seller,quantity=quantity,price=price,exchange_fee=0.))

    def allocate(self,intent,fill,fee_bps):
        j=fill['asset'];needs=self.needs(intent)
        if needs['residual'][j]==0:return super().allocate(intent,fill,fee_bps)
        side=np.sign(fill['signed_qty']);remaining=abs(fill['signed_qty'])
        priority={'residual':0,'pair':1,'base':2}
        eligible=[n for n,v in needs.items() if side*v[j]>0]
        eligible.sort(key=lambda n:(not(self.owners[n].qty[j]*needs[n][j]<0),priority[n]))
        for name in eligible:
            take=min(remaining,abs(needs[name][j]))
            if take:self.owner_fill(name,j,side*take,fill['price'],fee_bps,fill['time'],'external_fill')
            remaining=difference(remaining,take)
        assert abs(remaining)<1e-8

    def handoff(self,pair,marks,now):
        if not pair.get('closing') or not pair.get('observed_close_attempt'):
            raise ValueError('Handoff requires an observed closing attempt')
        if self.unknown or now*1000<getattr(self,'last_packet_arrival_ms',-1):
            raise ValueError('Unreconciled orders prevent handoff')
        q=self.owners['pair'].qty.copy();ids=np.flatnonzero(q)
        if not len(ids) or any(j not in pair['ids'] for j in ids):raise ValueError('Unexpected active residual')
        if any(self.owners['residual'].qty[j] for j in ids):raise ValueError('Residual owners overlap')
        eid=self.active_episode;before=self.reconcile(marks)
        physical=(self.physical.cash,self.physical.shadow_cash,self.physical.fees,len(self.physical.fills),self.physical.qty.copy())
        for j in ids:
            self.residual_by_asset[int(j)]=eid
            self.owner_fill('pair',int(j),-q[j],marks[j],0.,now,'residual_transfer')
            self.owner_fill('residual',int(j),q[j],marks[j],0.,now,'residual_transfer')
        item=dict(episode=eid,pair=pair['pair'],ids=list(map(int,ids)),time=now,qty=q.tolist(),
                  balanced=pair.get('balanced',False),aborted=pair.get('aborted',False),done=False)
        self.residuals.append(item);self.handoffs.append(copy.deepcopy(item));self.active_episode=None
        assert abs(self.reconcile(marks)-before)<1e-6
        assert physical[:4]==(self.physical.cash,self.physical.shadow_cash,self.physical.fees,len(self.physical.fills))
        np.testing.assert_array_equal(physical[4],self.physical.qty)
        return copy.deepcopy(item)

    def finished_residuals(self,now):
        done=[]
        for item in self.residuals:
            if not item['done'] and not np.any(self.episode_books[item['episode']].qty):
                item.update(done=True,closed_at=now);done.append(copy.deepcopy(item))
        return done

    def episode_results(self,marks):
        return [dict(**self.episode_meta[eid],qty=b.qty.tolist(),net_pnl=b.equity(marks),fees=b.fees,
                     funding=b.funding,realized=b.realized,marked_open=bool(np.any(b.qty)),
                     fills=b.fills,funding_events=b.funding_events) for eid,b in self.episode_books.items()]


class ResidualAccount(ResidualOwnership,SharedAccount):pass
class ResidualBookAccount(ResidualOwnership,BookAccount):pass


def safe_with_residual(proposal,residual_qty,base_qty,base_target,equity,prices,sigma,p,sc):
    residual=np.asarray(residual_qty);ids=np.flatnonzero(residual)
    if any(residual[j] for j in proposal['ids']):return False
    if not len(ids):return True
    q=np.asarray(proposal['target']);new_ids=proposal['ids']
    close_cost=float(abs(residual)@prices)*(sc['fee_bps']+sc['impact_bps'])/10000
    new_cost=float(abs(q)@prices[new_ids])*2*(sc['fee_bps']+sc['impact_bps'])/10000
    if equity<=close_cost+new_cost:return False
    for remain in product((0.,1.),repeat=len(ids)):
        for base in (base_qty,base_target):
            owned=np.asarray(base).copy();owned[ids]+=residual[ids]*remain
            for mask in ([1,0],[0,1],[1,1]):
                check=owned.copy();check[new_ids]+=q*np.asarray(mask)
                if risk_breaches(check,equity-close_cost-new_cost,prices,sigma,p['risk']):return False
    return True
