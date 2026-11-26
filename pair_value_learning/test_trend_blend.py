import unittest
import numpy as np
from .trend_blend import build


class TrendBlendTests(unittest.TestCase):
    def test_trend_sign_and_future_data_invariance(self):
        t=np.arange(200)*3600
        p=np.ones((200,10));p[:,0]=np.exp(np.arange(200)*.001)
        shape=(200,1)
        s=dict(times=t,pairs=np.array([[0,1]]),beta=np.ones(shape),vol=np.ones(shape)*.01,mu=np.zeros(shape),economic_available=np.ones(shape,bool),economic_action=np.zeros(shape,int))
        a=build(dict(market_times=t,close=p),s)
        self.assertGreater(a['coherent_trend_72']['mu'][180,0],0)
        p[181:,0]=1e6
        b=build(dict(market_times=t,close=p),s)
        np.testing.assert_array_equal(a['coherent_blend_72']['mu'][:181],b['coherent_blend_72']['mu'][:181])
        s['pairs']=np.array([[1,0]])
        c=build(dict(market_times=t,close=p),s)
        self.assertLess(c['coherent_trend_72']['mu'][180,0],0)
        self.assertAlmostEqual(c['coherent_trend_72']['mu'][180,0],-a['coherent_trend_72']['mu'][180,0])


if __name__=='__main__':unittest.main()
