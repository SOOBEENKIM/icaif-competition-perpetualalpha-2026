import copy,json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from .portfolio import adopt,target_from_frame
from .locks import WriterLease,WriterBusy
from .bridge import HistoryBridge,JointExecutor,PreviewOnly
from .producer import validate
from competition_live.profile import load_profile,START
from competition_live.runtime import fingerprint
from competition_live.live_account import LiveAccount
from competition_live.candidate import build_candidate
from joint_wallet.book import JointBook
from joint_wallet.test_book import fill
from pas_economic_allocation.lifecycle import LifecycleController


class HandoffTests(unittest.TestCase):
    def setUp(self):self.p=load_profile()

    def fixture(self):
        from pas_strategy_audit.test_execution import ExecutionTests
        f=ExecutionTests();a,p,t,orders,stream,fetches=f.setup_case(LiveAccount)
        a.refresh_market=stream
        return f,a,p,t,orders,stream

    def packet(self,at):
        frame=dict(source_time=START,published_time=START+100,weights=[.04]+[0.]*9,retention=[1.]*10,
            sigma=(np.eye(10)*1e-8).tolist(),volume=[1000.]*10,trades=[10.]*10,source_hash='fixture')
        event=dict(channel='strategySignal',received_ms=int((START+100)*1000)+1,
            data=dict(frame=frame,publication_sha256=fingerprint(frame),incumbent=frame['weights'],candidate=frame['weights']))
        cf=dict(source_time=START,source_hash='fixture',anchor_weights=[.04]+[0.]*9,protected=[False]*10,
            price_forecast_168h=[0.]*10,reference_funding_forecast_168h=[0.]*10,
            sigma=frame['sigma'],volume=frame['volume'],trades=frame['trades'],risk_aversion=5.)
        old=dict(publication=event,forecasts=[],profile_hash=fingerprint(self.p),candidate=self.p['candidate'],published_at=START+100)
        body=dict(candidate='budget_lifecycle',profile_hash=fingerprint(self.p),published_at=START+120,frame=cf,original_signal=old)
        return dict(body=body,sha256=fingerprint(body))

    def test_migration_preserves_actual_inventory_entries_pnl_and_control_state(self):
        f,a,p,t,orders,stream=self.fixture();f.execute(a,p,t,stream,[.01]+[0.]*9,'seed')
        p.counter=17;p.last_decision=START-21600;p.peak=50042.;p.used={('BTC','ETH'):START-3600}
        p.base_task={'target':np.full(10,99.),'expires':START+999}
        saved=copy.deepcopy(a);new=adopt(p)
        np.testing.assert_array_equal(new.account.physical.qty,saved.physical.qty)
        np.testing.assert_array_equal(new.account.physical.entry,saved.physical.entry)
        for name in ('cash','fees','funding','realized'):self.assertEqual(getattr(new.account.physical,name),getattr(saved.physical,name))
        self.assertEqual(new.counter,17);self.assertEqual(new.peak,50042.);self.assertEqual(new.used,p.used)
        self.assertIsNone(new.base_task);self.assertIsNotNone(p.base_task)

    def test_selected_policy_one_row_matches_original_clock_and_allows_short(self):
        cf=self.packet(START+130)['body']['frame'];cf['anchor_weights']=[0.]*10;cf['price_forecast_168h'][3]=-.05
        prices=np.full(10,100.);qty=np.zeros(10)
        actual,_=target_from_frame(cf,prices,qty,50000.,self.p['trading'])
        c=LifecycleController([START],np.array([cf['price_forecast_168h']]),np.zeros((1,10)),np.zeros((1,10)),np.zeros((1,10),bool),5.)
        expected=c.propose(START,prices,qty,50000.,np.zeros(10),np.array(cf['sigma']),np.array(cf['volume']),np.array(cf['trades']),self.p['trading'],buffer=.2)
        self.assertEqual(actual['target_qty'],expected['target_qty']);self.assertLess(actual['target_qty'][3],0.)

    def test_exact_candidate_target_reaches_retained_live_executor_without_reset(self):
        f,a,p,t,orders,stream=self.fixture();f.execute(a,p,t,stream,[.01]+[0.]*9,'seed')
        p=adopt(p);p.account.executor=a.executor;p.account.refresh_market=stream;p.account.checkpoint=lambda *x:None
        packet=self.packet(t[0]);prior=p.account.physical.qty.copy();marks=np.full(10,80000.)
        target,_=target_from_frame(packet['body']['frame'],marks,prior,p.account.reconcile(marks),self.p['trading'])
        with patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):
            p.step_candidate(t[0],{s:stream(s) for s in self.p['trading']['symbols']},packet,self.p)
        self.assertGreater(len(orders),1)
        self.assertAlmostEqual(sum(float(x.size) for x in orders[1:]),target['target_qty'][0]-prior[0],places=8)
        self.assertEqual(p.last_decision,START)
        np.testing.assert_allclose(p.account.physical.qty,target['target_qty'],atol=1e-9)

    def test_does_not_reschedule_same_round_or_backdate_off_cadence_entry(self):
        f,a,p,t,orders,stream=self.fixture();p=adopt(p);p.account.executor=a.executor;p.account.refresh_market=stream
        t[0]=START+3600+130;packet=self.packet(t[0]);n=len(orders)
        with patch('competition_live.live_account.time.time',side_effect=lambda:t[0]):
            p.step_candidate(t[0],{s:stream(s) for s in self.p['trading']['symbols']},packet,self.p)
        self.assertEqual(len(orders),n);self.assertEqual(p.last_decision,-1)

    def test_tampered_forecast_or_unavailable_signal_is_rejected(self):
        p=self.packet(START+130);validate(p,self.p,START+130)
        bad=copy.deepcopy(p);bad['body']['frame']['anchor_weights'][0]=.9
        with self.assertRaises(ValueError):validate(bad,self.p,START+130)
        with self.assertRaises(ValueError):validate(p,self.p,START+119)

    def test_same_wallet_lock_excludes_old_and_new_and_releases_after_error(self):
        with tempfile.TemporaryDirectory() as d:
            with WriterLease(d) as first:
                first.require()
                with self.assertRaises(WriterBusy):
                    with WriterLease(d):self.fail('Two writers acquired the same wallet')
            with WriterLease(d) as second:second.require()
            with self.assertRaises(WriterBusy):second.require()

    def test_joint_executor_closes_ours_without_closing_teammate_and_no_double_send(self):
        from bot.execution import LimitOrder,OrderRejected
        b=JointBook(self.p['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        b.apply_fill('teammate',fill(1,'A',.02,0,price=80000.))
        b.apply_fill('ours',fill(2,'B',.01,-.02,price=80000.));rows=[];sent=[]
        public=SimpleNamespace(post=lambda payload:rows)
        bridge=SimpleNamespace(public=public,refresh=lambda:b,reconcile_exchange=lambda book:dict(marks=np.full(10,80000.)))
        order=LimitOrder('pas:ETH',False,'.01','80000','Ioc',True,'0x'+'1'*32)
        with tempfile.TemporaryDirectory() as d,WriterLease(d) as lease:
            e=JointExecutor(self.p,Path(d)/'dispatch',bridge,lease=lease)
            def send(plan):
                sent.append(plan);row=fill(3,'A',abs(plan['delta']),-.01,price=80000.,fee=.72)
                row['cloid']=plan['cloid'];rows.append(row);b.apply_fill('ours',row)
                return dict(oid=3,filled_size=str(abs(plan['delta'])))
            e.gateway=SimpleNamespace(preflight=lambda p,b:b.validate_plan(p),send=send)
            e.set_risk_state(b.owners['ours'].qty,np.eye(10)*1e-8)
            result=e.submit(order)
            self.assertEqual(float(result['filled']['totalSz']),.01)
            self.assertFalse(sent[0]['reduce_only']);self.assertEqual(b.owners['teammate'].qty[1],-.02)
            self.assertEqual(b.owners['ours'].qty[1],0.)
            b.reconcile(np.full(10,80000.))
            with self.assertRaises(OrderRejected):e.submit(order)
            self.assertEqual(len(sent),1)

    def test_preview_executes_preflight_but_cannot_call_sender(self):
        from bot.execution import LimitOrder
        b=JointBook(self.p['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        bridge=SimpleNamespace(public=None,refresh=lambda:b,reconcile_exchange=lambda book:dict(marks=np.full(10,80000.)))
        with tempfile.TemporaryDirectory() as d:
            e=JointExecutor(self.p,d,bridge,preview=True);calls=[]
            e.gateway=SimpleNamespace(preflight=lambda p,b:calls.append('checked'),send=lambda *x:self.fail('Preview must not send'))
            e.set_risk_state(np.zeros(10),np.eye(10)*1e-8)
            order=LimitOrder('pas:BTC',True,'.01','80000','Ioc',False,'0x'+'2'*32)
            with self.assertRaises(PreviewOnly) as raised:e.submit(order)
            self.assertEqual(calls,['checked']);self.assertEqual(raised.exception.plan['delta'],.01)
            self.assertFalse((Path(d)/'dispatch.json').exists());np.testing.assert_array_equal(b.physical.qty,0.)

    def test_joint_funding_sync_preserves_opposing_owner_quantities_and_is_idempotent(self):
        from .bridge import HistoryBridge
        b=JointBook(self.p['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        b.apply_fill('teammate',fill(1,'A',2,0,price=100.,fee=0.))
        b.apply_fill('ours',fill(2,'B',1,-2,price=100.,fee=0.))
        b.apply_funding('pas:ETH',3000,.2,-1,'settlement')
        a=LiveAccount(self.p,None,lambda *x:None,lambda *x:[])
        a.physical_fill(1,1.,100.,0.,2.);a.owner_fill('base',1,1.,100.,0.,2.,'known')
        bridge=HistoryBridge(Path('/tmp'),Path('/tmp'),self.p,None)
        first=bridge.sync(a,b,np.full(10,100.));second=bridge.sync(a,b,np.full(10,100.))
        self.assertEqual(first,second);self.assertAlmostEqual(a.physical.funding,-.2)
        self.assertAlmostEqual(a.reserved.funding,.4);self.assertEqual(a.physical.qty[1],1.)

    def test_teammate_receipts_are_not_reclassified_as_our_orders(self):
        from .bridge import owned_ids
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'dispatch.json').write_text(json.dumps(dict(
                completed={'our':dict(owner='ours',oid=11),'team':dict(owner='teammate',oid=22)},
                pending=dict(plan=dict(owner='teammate',cloid='pending_team'),receipt=dict(oid=33)))))
            cloids,oids=owned_ids(path,path)
            self.assertEqual(cloids,{'our'});self.assertEqual(oids,{11})

    def test_inbox_is_unsigned_idempotent_and_rejects_changed_or_expired_requests(self):
        from .inbox import publish
        with tempfile.TemporaryDirectory() as d:
            now=time.time();args=dict(request_id='one',symbol='ETH',target=-.4,expected_owned=-.4064,source_time=now,expires_at=now+60)
            path=publish(d,**args);self.assertEqual(path,publish(d,**args))
            with self.assertRaises(ValueError):publish(d,**{**args,'target':-.1})
            with self.assertRaises(ValueError):publish(d,**{**args,'request_id':'expired','expires_at':now-1})
            self.assertEqual(len(list((Path(d)/'teammate_inbox').glob('*.json'))),1)
            self.assertFalse((Path(d)/'dispatch.json').exists())

    def test_teammate_queue_uses_same_dispatcher_and_preserves_our_ownership(self):
        from .inbox import publish,process
        from .bridge import owned_ids
        now=time.time();b=JointBook(self.p['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        b.apply_fill('teammate',fill(1,'A',2,0,price=100.));b.apply_fill('ours',fill(2,'B',1,-2,price=100.))
        rows=[];sent=[]
        def post(payload):
            if payload['type']=='userFillsByTime':return rows
            if payload['type']=='l2Book':return dict(coin='pas:ETH',time=int(time.time()*1000),levels=[[dict(px='100',sz='100')],[dict(px='100',sz='100')]])
            raise AssertionError(payload)
        meta=dict(universe=[dict(name='pas:'+s,szDecimals=d) for s,d in zip(self.p['trading']['symbols'],self.p['trading']['execution']['decimals'])])
        public=SimpleNamespace(post=post,market=lambda:(meta,[dict(markPx='100')]*10))
        bridge=SimpleNamespace(public=public,refresh=lambda:b,reconcile_exchange=lambda book:dict(marks=np.full(10,100.)))
        with tempfile.TemporaryDirectory() as d,WriterLease(d) as lease:
            e=JointExecutor(self.p,d,bridge,lease=lease)
            def send(plan):
                sent.append(plan);r=fill(3,'B',.5,-1,price=100.,fee=.045);r['cloid']=plan['cloid']
                rows.append(r);b.apply_fill('teammate',r);return dict(oid=3,filled_size='.5')
            e.gateway=SimpleNamespace(preflight=lambda p,b:b.validate_plan(p),send=send)
            publish(d,request_id='team-close',symbol='ETH',target=-1.5,expected_owned=-2.,source_time=now,expires_at=now+60)
            process(d,e,self.p,np.eye(10)*1e-6);process(d,e,self.p,np.eye(10)*1e-6)
            self.assertEqual(len(sent),1);self.assertEqual(b.owners['ours'].qty[1],1.)
            self.assertEqual(b.owners['teammate'].qty[1],-1.5);b.reconcile(np.full(10,100.))
            self.assertEqual(owned_ids(Path(d),Path(d)),(set(),set()))

    def test_restart_reads_new_checkpoint_and_keeps_pending_base_task(self):
        from .runner import connect
        _,a,p,_,_,_=self.fixture();new=adopt(p);new.strategy_id='budget_lifecycle'
        new.counter=41;new.base_task=dict(target=np.zeros(10),expires=START+900)
        fake=SimpleNamespace(refresh=lambda:object(),reconcile_exchange=lambda b:dict(marks=np.full(10,80000.)),
            sync=lambda *x:dict(own_equity=50000.),settlement=lambda *x:None)
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);(directory/'coordinator.pkl').write_bytes(b'fixture')
            with patch('candidate_handoff.runner.HistoryBridge',return_value=fake),patch('candidate_handoff.runner.load_existing',return_value=(new,'digest')) as load:
                got,_,_,_=connect(directory,directory,self.p,None,lambda b:None)
            self.assertEqual(load.call_args.args[0],directory/'coordinator.pkl')
            self.assertEqual(got.counter,41);self.assertIsNotNone(got.base_task)

    def test_pair_exit_state_and_owner_binding_survive_adoption(self):
        _,a,p,_,_,_=self.fixture()
        p.pair=SimpleNamespace(state='closing',intent=dict(forecast_source=START-3600),account=a,owner=a.owners['pair'],on_time=None)
        new=adopt(p)
        self.assertEqual(new.pair.state,'closing');self.assertEqual(new.pair.intent,p.pair.intent)
        self.assertIs(new.pair.account,new.account);self.assertIs(new.pair.owner,new.account.owners['pair'])

    def test_missing_ownership_policy_stops_before_loading_signer(self):
        from .runner import main
        with tempfile.TemporaryDirectory() as d,patch('sys.argv',['runner','--directory',d,'--execute']),patch('candidate_handoff.runner.explicit_signer') as signer:
            with self.assertRaisesRegex(ValueError,'Execution ownership policy is missing'):main()
            signer.assert_not_called()

    def test_explicit_external_mode_does_not_claim_teammate_is_integrated(self):
        from .runner import require_contract,ROOT
        c=dict(schema='owner_delta_v1',team_wallet=self.p['team_wallet'],
            owner_capital={'ours':50000.,'teammate':50000.},global_lock=str(ROOT/'.runtime/wallet_writer.lock'),
            transport='external_fill_reconciliation',authorized_without_teammate_adapter=True,
            unattributed_fills_owner='teammate',teammate_direct_orders_disabled=False,verified_teammate_adapter=False)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'team_execution_contract.json';path.write_text(json.dumps(c))
            self.assertEqual(require_contract(Path(d),self.p),c)
            for field,value in [('authorized_without_teammate_adapter',False),('verified_teammate_adapter',True),('unattributed_fills_owner','ours')]:
                path.write_text(json.dumps({**c,field:value}))
                with self.assertRaises(ValueError):require_contract(Path(d),self.p)

    def test_changed_external_inventory_before_send_is_rejected_without_unknown_order(self):
        from bot.execution import LimitOrder,OrderRejected
        from joint_wallet.book import ReconciliationError
        b=JointBook(self.p['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        bridge=SimpleNamespace(public=None,refresh=lambda:b,reconcile_exchange=lambda book:dict(marks=np.full(10,100.)))
        order=LimitOrder('pas:ETH',True,'1','100','Ioc',False,'0x'+'9'*32)
        with tempfile.TemporaryDirectory() as d,WriterLease(d) as lease:
            e=JointExecutor(self.p,d,bridge,lease=lease);e.set_risk_state(np.zeros(10),np.eye(10)*1e-6)
            def reject(*args):raise ReconciliationError('Teammate fill changed the physical position')
            def forbidden(*args):self.fail('Must not send a stale order')
            e.gateway=SimpleNamespace(preflight=reject,send=forbidden)
            with self.assertRaises(OrderRejected):e.submit(order)
            self.assertIsNone(e.dispatch.state()['pending'])
            self.assertEqual(e.dispatch.state()['completed'],{})

    def test_unified_equity_uses_bracketing_marks_and_still_rejects_missing_cash(self):
        from joint_wallet.book import ReconciliationError
        b=JointBook(self.p['trading']['symbols'],{'ours':50000.,'teammate':50000.})
        b.apply_fill('ours',fill(1,'B',1,0,price=100.,fee=.1))
        now=int(time.time()*1000)
        def positions(pnl):return dict(time=now,assetPositions=[dict(position=dict(coin='pas:ETH',szi='1',entryPx='100',unrealizedPnl=str(pnl)))])
        state=dict(mode='unifiedAccount',positions_before=positions(0),positions=positions(2),orders=[],
            spot=dict(balances=[dict(coin='USDP',token=self.p['collateral_token'],total=str(b.physical.cash+1))]))
        public=SimpleNamespace(account=lambda:state,market=lambda:(None,[dict(markPx='100')]*10))
        with tempfile.TemporaryDirectory() as d:
            bridge=HistoryBridge(d,d,self.p,public)
            result=bridge.reconcile_exchange(b)
            self.assertEqual(result['observed_valuation_interval'],[b.physical.cash,b.physical.cash+2])
            self.assertAlmostEqual(result['equity_error'],-1.)
            state['spot']['balances'][0]['total']=str(b.physical.cash+5)
            with self.assertRaisesRegex(ReconciliationError,'outside observed valuation interval'):bridge.reconcile_exchange(b)
            state['spot']['balances'][0]['total']=str(b.physical.cash+1)
            state['positions_before']['assetPositions'][0]['position']['szi']='2'
            with self.assertRaisesRegex(ReconciliationError,'inventory changed across'):bridge.reconcile_exchange(b)


if __name__=='__main__':unittest.main()
