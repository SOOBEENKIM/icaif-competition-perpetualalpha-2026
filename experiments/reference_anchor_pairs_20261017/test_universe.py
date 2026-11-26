import unittest
import numpy as np
from .universe_audit import hurst_rs_walk,screen


class HurstTests(unittest.TestCase):
    def test_random_walk_and_reverting_process_are_distinguished(self):
        rng=np.random.default_rng(20261017);noise=rng.normal(size=16384)
        walk=np.cumsum(noise);ou=np.zeros(len(noise))
        for i in range(1,len(ou)):ou[i]=.85*ou[i-1]+noise[i]
        hw,ho=hurst_rs_walk(walk),hurst_rs_walk(ou)
        self.assertGreater(hw,.4);self.assertLess(hw,.7)
        self.assertLess(ho,.5);self.assertLess(ho,hw)

    def test_translation_and_scaling_do_not_change_estimate(self):
        x=np.cumsum(np.random.default_rng(12).normal(size=4096))
        self.assertAlmostEqual(hurst_rs_walk(x),hurst_rs_walk(7*x+15),places=10)

    def test_formation_ranking_ignores_future_prices(self):
        rng=np.random.default_rng(56);prices=np.exp(np.cumsum(rng.normal(0,.001,(1440,3)),axis=0))
        times=np.arange(len(prices))*3600
        a=screen(times,prices,['A','B','C'],1200*3600,0)
        prices[1200:]*=100
        b=screen(times,prices,['A','B','C'],1200*3600,0)
        self.assertEqual(a,b);self.assertEqual(len(a),3)
        self.assertTrue(all(isinstance(r['structural_pass'],bool) for r in a))


if __name__=='__main__':unittest.main()
