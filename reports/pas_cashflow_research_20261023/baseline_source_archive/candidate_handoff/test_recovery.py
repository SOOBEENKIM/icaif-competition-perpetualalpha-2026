import copy,json,shutil,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
from pathlib import Path
import numpy as np
from . import producer
from .recovery import recover_confirmed
from competition_live.profile import load_profile
from competition_live.runtime import private_load
from joint_wallet.book import ReconciliationError

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'reports/handoff_recovery_20261023_0138/backup'
RAW=ROOT/'reports/market_peer_refresh_20261022_2348/raw/wallets/0x125778065a578d598d88bddaa3fdd635280d0e47.json'

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.p=private_load(FIXTURE/'coordinator.pkl')['portfolio']
        self.dispatch=json.loads((FIXTURE/'dispatch.json').read_text())
        self.rows=json.loads(RAW.read_text())['fills']['rows']
        self.marks=np.array([81678.,2406.6,78.032,7632,150.64,94.482,107.57,4065,1.12,157.2])
        self.key=next(iter(self.p.account.pending_orders))

    def recover(self,p=None,dispatch=None,rows=None):
        return recover_confirmed(p or self.p,dispatch if dispatch is not None else self.dispatch,
            self.rows if rows is None else rows,self.marks,now=1792685000.)

    def test_exact_crash_after_confirmed_gold_fill_recovers_without_sending(self):
        result,detail=self.recover()
        self.assertEqual(self.p.account.physical.qty[7],0.) # work on a copy
        self.assertAlmostEqual(result.account.physical.qty[7],-.187)
        self.assertEqual(len(result.account.physical.fills),14)
        self.assertAlmostEqual(result.account.physical.fees-self.p.account.physical.fees,.683939)
        self.assertEqual(result.account.pending_orders,{})
        self.assertEqual(result.account.unknown,set())
        np.testing.assert_array_equal(result.account.reserved.qty,self.p.account.reserved.qty)
        np.testing.assert_array_equal(result.account.owners['pair'].qty,self.p.account.owners['pair'].qty)
        self.assertEqual(result.last_decision,self.p.last_decision)
        self.assertEqual(result.counter,self.p.counter)
        self.assertEqual(result.account.seen,self.p.account.seen)
        self.assertIsNone(result.base_task) # expired USDEUR/USDJPY intent is not replayed
        self.assertIsNone(result.account.executor)
        self.assertEqual(detail['applied_fill_count'],2)
        result.account.reconcile(self.marks)

    def test_repeat_recovery_never_doubles_inventory_or_fees(self):
        first,_=self.recover();second,detail=self.recover(first)
        np.testing.assert_array_equal(first.account.physical.qty,second.account.physical.qty)
        self.assertEqual(first.account.physical.fees,second.account.physical.fees)
        self.assertEqual(detail['applied_fill_count'],0)

    def test_incomplete_receipt_wrong_owner_and_wrong_side_are_rejected(self):
        for mutate in [lambda d:d['completed'].pop(self.key),
                       lambda d:d['completed'][self.key].update(owner='teammate'),
                       lambda d:d['completed'][self.key].update(filled_size='.2')]:
            d=copy.deepcopy(self.dispatch);mutate(d)
            with self.assertRaises(ReconciliationError):self.recover(dispatch=d)
        rows=copy.deepcopy(self.rows)
        next(r for r in rows if r['oid']==63389799110)['side']='B'
        with self.assertRaises(ReconciliationError):self.recover(rows=rows)
        with self.assertRaises(ReconciliationError):self.recover(rows=[r for r in self.rows if r['tid']!=517431097463069])
        self.assertEqual(self.p.account.physical.qty[7],0.)

    def test_unknown_other_market_is_not_erased(self):
        p=copy.deepcopy(self.p);p.account.unknown.add('BTC')
        with self.assertRaises(ReconciliationError):self.recover(p)

    def test_duplicate_response_rows_are_idempotent_but_conflicts_fail(self):
        rows=self.rows+self.rows
        result,detail=self.recover(rows=rows)
        self.assertEqual(detail['applied_fill_count'],2)
        row=copy.deepcopy(next(r for r in self.rows if r['oid']==63389799110));row['sz']='1.0'
        with self.assertRaises(ReconciliationError):self.recover(rows=self.rows+[row])

    def test_restart_connection_reconciles_real_checkpoint_before_wiring_sender(self):
        from .runner import connect
        capture=ROOT/'reports/handoff_recovery_20261023_0138/preview'
        snapshot=json.loads((capture/'account_snapshot.json').read_text())
        funding=json.loads((capture/'public_funding.json').read_text())
        fills=json.loads((capture/'public_fills.json').read_text())
        market=json.loads((capture/'public/009_metaAndAssetCtxs.json').read_text())['response']
        book=json.loads((capture/'joint_book.json').read_text())
        owned=[e['row'] for e in book['events'] if e['kind']=='fill' and e['owner']=='ours']
        calls=[]
        def post(req):
            calls.append(req['type'])
            return fills if req['type']=='userFillsByTime' else funding
        public=SimpleNamespace(post=post,account=lambda:snapshot,market=lambda:market)
        executor=SimpleNamespace(confirmed_rows={63389799110:[r for r in fills if r['oid']==63389799110]})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            for n in ['coordinator.pkl','dispatch.json']:shutil.copy2(FIXTURE/n,path/n)
            with patch('candidate_handoff.bridge.owned_ids',return_value=({r['cloid'] for r in owned},{r['oid'] for r in owned})),patch('time.time',return_value=snapshot['positions']['time']/1000+.1):
                p,bridge,_,detail=connect(path,path,load_profile(),public,lambda b:executor)
            self.assertAlmostEqual(p.account.physical.qty[7],-.187)
            self.assertEqual(p.account.pending_orders,{})
            self.assertEqual(detail['recovery']['applied_fill_count'],2)
            before=len(calls)
            got=p.account.collect_fills(63389799110,0,.187)
            self.assertEqual(len(got),2);self.assertEqual(len(calls),before)

    def test_public_read_error_reconnects_before_next_tick(self):
        from .runner import main
        from .public import PublicReadUnavailable
        import time
        now=time.time();profiles=load_profile();events=[]
        def portfolio(fail):
            account=SimpleNamespace(physical=SimpleNamespace(qty=np.zeros(10),fills=[],fees=0.,funding=0.),reserved=SimpleNamespace(qty=np.zeros(10)))
            def step(*args):
                events.append('tick_fail' if fail else 'tick_success')
                if fail:raise PublicReadUnavailable('fixture 429',60)
            return SimpleNamespace(account=account,step_candidate=step,last_decision=0,fatal=[],pair=None,events=[])
        first,second=portfolio(True),portfolio(False)
        bridge=SimpleNamespace(settlement=lambda *args:None,latest=SimpleNamespace(serialize=lambda:{}))
        def connect(*args):
            events.append('reconnect' if 'tick_fail' in events else 'connect')
            return (second if 'tick_fail' in events else first),bridge,{},{}
        with tempfile.TemporaryDirectory() as d:
            Path(d,'signal.json').write_text(json.dumps({'body':{'frame':{'source_time':int(now//3600)*3600}}}))
            with patch('sys.argv',['runner','--directory',d,'--execute','--once']),patch('candidate_handoff.runner.WriterLease'),patch('candidate_handoff.runner.require_contract',return_value={'transport':'external_fill_reconciliation'}),patch('candidate_handoff.runner.PublicPAS'),patch('candidate_handoff.runner.producer.validate'),patch('candidate_handoff.runner.producer.configured_revision',return_value='all_market_signed_core_v1'),patch('candidate_handoff.runner.fresh_preflight'),patch('candidate_handoff.runner.explicit_signer'),patch('candidate_handoff.runner.build_order_client'),patch('candidate_handoff.runner.connect',side_effect=connect),patch('candidate_handoff.runner.collect_books',return_value={}),patch('candidate_handoff.runner.save_coordinator'),patch('candidate_handoff.runner.time.sleep'):
                main()
        self.assertEqual(events,['connect','tick_fail','reconnect','tick_success'])

if __name__=='__main__':unittest.main()
