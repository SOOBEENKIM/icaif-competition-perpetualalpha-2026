import copy,json,tempfile,unittest
from types import SimpleNamespace
from pathlib import Path
from . import test_quote_integration as fixtures
from .runtime_adapter import stage,wire_candidate,account_class
from candidate_handoff.portfolio import CandidatePortfolio
from candidate_handoff.migration import wire as old_wire
from competition_live.live_account import LiveAccount
from .quote_guard import QuoteAccount
from .quote_archive import make_record
from competition_live.profile import START
from .protocol import OUT


class RuntimeAdapterTests(unittest.TestCase):
    def incumbent(self):
        p,*_=fixtures.QuoteIntegrationTests().case();p.__class__=CandidatePortfolio
        return p

    def wire(self,p,fn):
        original=p.account;bridge=SimpleNamespace(settlement=lambda *x:None)
        fn(p,bridge,original.executor,lambda *x:None,original.collect_fills,original.refresh_market)

    def test_old_reconnect_would_erase_new_account_guard(self):
        p=stage(self.incumbent(),'frozen_side_quote')
        self.assertIs(type(p.account),QuoteAccount)
        self.wire(p,old_wire)
        self.assertIs(type(p.account),LiveAccount)

    def test_new_reconnect_preserves_each_candidate_guard(self):
        for arm in ['fixed_entry_1h','fresh_block_catchup','frozen_side_quote','quote_inputs','size_only_2']:
            with self.subTest(arm=arm):
                p=stage(self.incumbent(),arm,quote_archive_directory=Path.cwd()/'reports/production_replay_20261025/quote_feature_archive')
                expected=account_class(p)
                for _ in range(2):
                    self.wire(p,wire_candidate);self.assertIs(type(p.account),expected)

    def test_outstanding_or_unknown_orders_cannot_be_discarded(self):
        p=self.incumbent();p.account.unknown.add('BTC')
        with self.assertRaises(ValueError):stage(p,'fixed_entry_1h')

    def test_unsupported_diagnostic_is_not_promoted(self):
        with self.assertRaises(ValueError):stage(self.incumbent(),'hourly_fresh')

    def test_pair_callback_rebinds_to_retained_owner_after_wire(self):
        p=self.incumbent()
        p.pair=SimpleNamespace(state='holding',account=p.account,owner=p.account.owners['pair'],on_time=None)
        q=stage(p,'frozen_side_quote');self.wire(q,wire_candidate)
        self.assertIs(q.pair.account,q.account);self.assertIs(q.pair.owner,q.account.owners['pair'])
        self.assertIsNotNone(q.pair.on_time);self.assertEqual(q.pair.state,'holding')

    def test_archived_runtime_matches_full_stream_decision_and_orders(self):
        helper=fixtures.QuoteIntegrationTests();reference=helper.case();staged=helper.case()
        p,clock,orders,full,packet,profile=staged;p.__class__=CandidatePortfolio
        with tempfile.TemporaryDirectory(dir=OUT) as directory:
            for point in [START-10800,START]:
                record=make_record(full,p.p['symbols'],point,p.cfg,START+1)
                (Path(directory)/f'{point}.json').write_text(json.dumps(record))
            q=stage(p,'quote_inputs',quote_archive_directory=directory)
            # Reconnect through the adapter; preserve this fixture's mocked
            # executor and clock, with no actual order client anywhere.
            self.wire(q,wire_candidate)
            latest=copy.deepcopy(full)
            for stream in latest.values():
                for field in ['book','ctx']:
                    stream[field]=[r for r in stream[field] if r[0]>START*1000]
                    stream[field+'_times']=[r[0] for r in stream[field]]
            helper.step(reference);helper.step((q,clock,orders,latest,packet,profile))
            import numpy as np
            np.testing.assert_array_equal(q.account.physical.qty,reference[0].account.physical.qty)
            self.assertEqual([(o.coin,o.is_buy,o.size,o.price) for o in orders],
                             [(o.coin,o.is_buy,o.size,o.price) for o in reference[2]])
            self.assertEqual(set(q.last_quote_archive_evidence['hashes']),{str(START-10800),str(START)})


if __name__=='__main__':unittest.main()
