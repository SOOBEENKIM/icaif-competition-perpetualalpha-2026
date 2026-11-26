"""Regression cases use only fake venues and temporary private checkpoints."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import runner
from .candidate import build_candidate
from .live_account import LiveAccount
from .profile import load_profile, START
from .runtime import save_coordinator, private_load
from .test_runtime import account_snapshot
from pas_strategy_audit.test_execution import ExecutionTests


class OperationalRepairs(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile()
        self.case = ExecutionTests()

    def inventory(self):
        a, p, clock, orders, stream, fetches = self.case.setup_case(LiveAccount)
        for ledger in (a.physical, a.owners['base']):
            ledger.fill(0, .01, 80000, 9, START)
        return a, p, clock, orders, stream, fetches

    def test_restart_restores_risk_loop_without_signal(self):
        a, p, *_ = self.inventory()
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            directory = workspace / '.runtime/pas_competition_live'
            save_coordinator(directory / 'coordinator.pkl', p, self.profile, 'fixture')
            snapshot = account_snapshot(self.profile, cash='99999.28', qty='.01')
            with patch.object(runner, 'verify_deployment'), \
                 patch.object(runner, 'verify_bundle'), \
                 patch.object(runner, 'fresh_preflight', return_value=snapshot), \
                 patch.object(runner, 'load_publication', return_value=(None, [])), \
                 patch.object(runner.time, 'time', return_value=START+7*3600), \
                 patch.object(runner, 'explicit_signer'), \
                 patch.object(runner, 'build_order_client'), \
                 patch.object(runner, 'PASExecutor'), \
                 patch.object(runner, 'LiveRunner') as live:
                runner.main(True, True, workspace, directory)
            live.return_value.tick.assert_called_once()

    def test_new_flat_ledger_waits_for_signal(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            with patch.object(runner, 'verify_deployment'), \
                 patch.object(runner, 'verify_bundle'), \
                 patch.object(runner, 'fresh_preflight', return_value=account_snapshot(self.profile)), \
                 patch.object(runner, 'load_publication', return_value=(None, [])), \
                 patch.object(runner.time, 'time', return_value=START+7*3600), \
                 patch.object(runner, 'explicit_signer') as signer:
                runner.main(True, True, workspace)
            signer.assert_not_called()

    def test_sequential_orders_refresh_after_four_second_fill_wait(self):
        a, p, clock, orders, stream, fetches = self.case.setup_case(LiveAccount)
        a.refresh_market = stream
        self.case.execute(a, p, clock, stream, [.01, .01]+[0.]*8, 'fresh')
        self.assertEqual(len(orders), 2)
        np.testing.assert_allclose(a.physical.qty[:2], [.01, .01])
        self.assertEqual(fetches[10:], ['BTC', 'ETH'])
        self.assertAlmostEqual(a.reconcile(np.full(10, 80000.)), 50000-a.physical.fees)

    def test_stale_refresh_is_rejected(self):
        a, p, clock, orders, stream, _ = self.case.setup_case(LiveAccount)
        stale = stream('BTC')
        a.refresh_market = lambda _: stale
        clock[0] += 10
        self.case.execute(a, p, clock, stream, [.01]+[0.]*9, 'stale')
        self.assertEqual(orders, [])

    def test_missing_flat_market_does_not_block_actual_reduce_only_exit(self):
        a, p, clock, orders, stream, _ = self.inventory()
        a.refresh_market = stream
        clock[0] = START+7*3600
        streams = {s:stream(s) for s in p.p['symbols'] if s != 'USDJPY'}
        with patch('competition_live.live_account.time.time', side_effect=lambda:clock[0]):
            p.step(clock[0], streams, None, [])
        self.assertEqual(len(orders), 1)
        self.assertTrue(orders[0].reduce_only)
        self.assertTrue(np.all(a.physical.qty == 0))
        self.assertEqual(p.unpriced_flat_assets, ['USDJPY'])
        self.assertEqual(p.fatal, [])
        self.assertEqual(a.reserved.cash, 50000)

    def test_missing_held_or_offset_owner_market_still_blocks_valuation(self):
        a, p, clock, orders, stream, _ = self.inventory()
        streams = {s:stream(s) for s in p.p['symbols'] if s != 'BTC'}
        marks, _ = p.prices(clock[0], streams)
        self.assertIsNone(marks)
        a.physical.fill(0, -.01, 80000, 9, START)
        a.owners['pair'].fill(0, -.01, 80000, 9, START)
        marks, _ = p.prices(clock[0], streams)
        self.assertIsNone(marks)

    def test_missing_flat_market_blocks_outstanding_increase(self):
        a, p, clock, orders, stream, _ = self.inventory()
        clock[0] = START+1000
        p.base_task = dict(target=np.array([.02]+[0.]*9), expires=clock[0]+100)
        pub = dict(received_ms=(clock[0]-1)*1000, data=dict(frame=dict(
            source_time=START, sigma=np.eye(10)*1e-8)))
        streams = {s:stream(s) for s in p.p['symbols'] if s != 'USDJPY'}
        with patch.object(p, 'submit_baseline') as submit:
            p.step(clock[0], streams, pub, [])
        submit.assert_not_called()
        self.assertIsNone(p.base_task)

    def test_runtime_does_not_pickle_quote_client(self):
        a, p, *_ = self.inventory()
        a.refresh_market = lambda _: None
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'coordinator.pkl'
            save_coordinator(path, p, self.profile, 'fixture')
            self.assertIsNone(private_load(path)['portfolio'].account.refresh_market)
            live = runner.LiveRunner(self.profile, Path(temp), None, a.executor)
            self.assertTrue(callable(live.portfolio.account.refresh_market))
            self.assertAlmostEqual(live.portfolio.account.physical.qty[0], .01)

    def test_invalid_signal_still_runs_existing_risk_management(self):
        a, p, clock, orders, stream, _ = self.inventory()
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            save_coordinator(directory/'coordinator.pkl',p,self.profile,'fixture')
            with patch.object(runner,'PublicQuoteProvider',return_value=stream):
                live=runner.LiveRunner(self.profile,directory,None,a.executor)
            live.public=unittest.mock.Mock()
            live.public.account.return_value=account_snapshot(self.profile,cash='99999.28',qty='.01')
            live.settlement=lambda *args:None
            live.portfolio.settlement=lambda *args:None
            live.portfolio.account.collect_fills=a.collect_fills
            clock[0]=START+2000
            streams={s:stream(s) for s in p.p['symbols']}
            with patch.object(runner,'collect_books',return_value=streams), \
                 patch.object(runner,'load_publication',side_effect=ValueError('digest mismatch')), \
                 patch.object(runner.time,'time',side_effect=lambda:clock[0]):
                report=live.tick()
            self.assertEqual(report['signal_error'],'digest mismatch')
            self.assertEqual(report['qty'],[0.]*10)
            self.assertTrue(orders[0].reduce_only)


if __name__ == '__main__':
    unittest.main()
