import unittest
from . import test_engine as fixtures
from .catchup_portfolio import adopt as catchup
from .quote_guard import adopt as quotes
from .historical_wiring_repair import rewire


class HistoricalWiringTests(unittest.TestCase):
    def test_adoption_rebinds_executor_clock_callbacks_and_ledger(self):
        for adopter in [catchup,quotes]:
            with self.subTest(candidate=adopter.__module__):
                profile,data,venue,p,target=fixtures.VenueTests().case()
                q=adopter(p)
                self.assertIsNot(q.account.executor,venue)  # reproduces old harness defect
                rewire(q,p)
                self.assertIs(q.account.executor,venue)
                self.assertIs(q.account.checkpoint.__self__,venue)
                self.assertIs(q.account.collect_fills.__self__,venue)
                self.assertIs(q.account.refresh_market.__self__,venue)
                self.assertIs(q.settlement.__self__,venue)
                self.assertIs(venue.account,q.account)
                venue.now+=60
                self.assertEqual(q.account.executor.now,venue.now)


if __name__=='__main__':unittest.main()
