import unittest
from .economics import evaluate_size,maximum_eligible_size


class EconomicsTests(unittest.TestCase):
    def setUp(self):
        self.book=dict(levels=[[dict(px='99.9',sz='100')],[dict(px='100.1',sz='100')]])
        self.kw=dict(fee_bps=9,depth_fraction=1.,exit_impact_bps=10.,buffer_bps=5.)

    def test_weight_or_good_funding_is_not_a_validated_price_forecast(self):
        row=evaluate_size(self.book,1000,1,funding_rate=-.001,**self.kw)
        self.assertEqual(row['reason'],'no_validated_pas_alpha')
        self.assertEqual(row['adverse_funding_scenario_bps'],0)
        self.assertEqual(maximum_eligible_size([row]),0)

    def test_size_requires_both_entry_and_exit_depth(self):
        row=evaluate_size(self.book,20000,1,expected_pas_return_bps=100,
                          forecast_validated=True,**self.kw)
        self.assertEqual(row['reason'],'insufficient_visible_depth')

    def test_cost_uses_both_fees_and_adverse_funding(self):
        row=evaluate_size(self.book,1000,1,funding_rate=.00001,**self.kw)
        self.assertAlmostEqual(row['roundtrip_book_and_fees_bps'],38.)
        self.assertAlmostEqual(row['adverse_funding_scenario_bps'],2.4)
        self.assertAlmostEqual(row['required_price_edge_bps'],55.4)

    def test_profitable_prediction_must_cover_cost_and_buffer(self):
        a=evaluate_size(self.book,1000,-1,expected_pas_return_bps=-60,forecast_validated=True,**self.kw)
        b=evaluate_size(self.book,5000,-1,expected_pas_return_bps=-50,forecast_validated=True,**self.kw)
        self.assertTrue(a['eligible']);self.assertFalse(b['eligible'])
        self.assertEqual(maximum_eligible_size([a,b]),1000)


if __name__=='__main__':
    unittest.main()
