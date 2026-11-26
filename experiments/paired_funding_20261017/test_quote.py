import unittest
from .quote import funding_value


class FundingCashTests(unittest.TestCase):
    def test_positive_long_pays_and_receipt_is_haircut(self):
        self.assertAlmostEqual(funding_value([1,-2],[100,50],[100,0],.5),-1.)
        self.assertAlmostEqual(funding_value([-1,2],[100,50],[100,0],.5),.5)

    def test_signed_oracle_notional_is_used_for_each_leg(self):
        self.assertAlmostEqual(funding_value([-2,1],[100,300],[50,10],1.),.7)


if __name__=='__main__':unittest.main()
