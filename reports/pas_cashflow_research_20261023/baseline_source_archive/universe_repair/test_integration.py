import copy, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .core import REVISION,LEGACY,SYMBOLS
from candidate_handoff.producer import validate,configured_revision
from candidate_handoff.portfolio import adopt
from competition_live.runtime import fingerprint
from competition_live.profile import load_profile,START
from competition_live.live_account import LiveAccount
from competition_live.candidate import build_candidate


class IntegrationTests(unittest.TestCase):
    def test_live_all_market_requirement_cannot_silently_fall_back_to_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)
            (directory/'team_execution_contract.json').write_text(json.dumps(dict(
                required_core_revision=REVISION,required_core_symbols=list(SYMBOLS))))
            with self.assertRaisesRegex(ValueError,'fallback is forbidden'):configured_revision(directory)
            (directory/'strategy_revision.json').write_text(json.dumps(dict(revision=LEGACY)))
            with self.assertRaisesRegex(ValueError,'fallback is forbidden'):configured_revision(directory)
            (directory/'strategy_revision.json').write_text(json.dumps(dict(revision=REVISION)))
            self.assertEqual(configured_revision(directory),REVISION)

    def packet(self):
        from candidate_handoff.test_handoff import HandoffTests
        f=HandoffTests();f.setUp();packet=f.packet(START+130);body=packet['body']
        body['core_revision']=REVISION;body['frame']['symbols']=list(SYMBOLS)
        body['universe_audit']=[dict(symbol=s,considered=True) for s in SYMBOLS]
        body['frame']['anchor_weights']=[.01,-.01]*5
        packet['sha256']=fingerprint(body)
        return packet

    def test_revision_contract_rejects_a_stale_or_incomplete_producer(self):
        p=load_profile();packet=self.packet();validate(packet,p,START+130,expected_revision=REVISION)
        with self.assertRaises(ValueError):validate(packet,p,START+130,expected_revision=LEGACY)
        broken=copy.deepcopy(packet);broken['body']['universe_audit'][4]['considered']=False
        broken['sha256']=fingerprint(broken['body'])
        with self.assertRaises(ValueError):validate(broken,p,START+130,expected_revision=REVISION)

    def test_all_ten_signed_core_targets_reach_order_sender_and_fill_accounting(self):
        p=load_profile();clock=[START+130.];fills=[];orders=[]
        def stream(symbol):
            at=int(clock[0]*1000)-5
            book=dict(time=at,levels=[[dict(px='100',sz='10000')],[dict(px='100',sz='10000')]])
            return dict(book_times=[at],book=[(at,0,book)],ctx_times=[at],
                ctx=[(at,0,dict(markPx='100',oraclePx='100',funding='0'))],resets=[])
        class Venue:
            def market(self,coin):
                return dict(name=coin,szDecimals=p['trading']['execution']['decimals'][p['trading']['symbols'].index(coin[4:])]),dict(markPx='100')
            def set_risk_state(self,*args):pass
            def submit(self,order):
                oid=len(orders)+1;orders.append(order);clock[0]+=4
                fills.append(dict(coin=order.coin,oid=oid,tid=oid,side='B' if order.is_buy else 'A',
                    sz=order.size,px='100',fee=str(float(order.size)*100*.0009),feeToken='USDP',time=int(clock[0]*1000)))
                return {'filled':dict(oid=oid,totalSz=order.size,avgPx='100')}
        account=LiveAccount(p,Venue(),lambda *args:None,lambda oid,*args:[f for f in fills if f['oid']==oid])
        account.refresh_market=stream
        portfolio=adopt(build_candidate(p,lambda *args:None,account=account)['portfolio'])
        # adopt() preserves a detached account; attach the recording mock again.
        portfolio.account.executor=account.executor;portfolio.account.refresh_market=stream
        packet=self.packet()
        with patch('competition_live.live_account.time.time',side_effect=lambda:clock[0]):
            portfolio.step_candidate(clock[0],{s:stream(s) for s in SYMBOLS},packet,p)
        self.assertEqual({o.coin for o in orders},{'pas:'+s for s in SYMBOLS})
        self.assertEqual(len(orders),10)
        for o in orders:self.assertEqual(o.is_buy,SYMBOLS.index(o.coin[4:])%2==0)
        np.testing.assert_array_equal(np.sign(portfolio.account.physical.qty),[1,-1]*5)
        self.assertAlmostEqual(portfolio.account.reconcile(np.full(10,100.)),50000-portfolio.account.physical.fees)
        self.assertEqual(portfolio.events[-1].get('core_revision',REVISION),REVISION)


if __name__=='__main__':unittest.main()
