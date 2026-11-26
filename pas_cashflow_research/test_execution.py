import copy,unittest
import numpy as np
from .economic_execution import execute,simulate
from pas_counterfactual_20261022.engine import simulate as original
from competition_live.profile import load_profile
from experiments.shared_execution_pairs_20261017.account_v2 import SharedAccount

class Contracts(unittest.TestCase):
    def setup_case(self):
        p=load_profile()['trading'];a=SharedAccount(50000,p['symbols']);sc=dict(fee_bps=9,impact_bps=10,participation=.01,receipt_multiplier=1)
        price=np.ones(10)*100;cov=np.eye(10)*1e-6
        return p,a,sc,price,cov

    def test_buy_limit_blocks_adverse_fill_without_erasing_order(self):
        p,a,sc,price,cov=self.setup_case();pending=dict(qty=np.r_[1.,np.zeros(9)],time=0,source=0,source_prices=price,entry_limits={0:101.},economic_execution=True)
        batch=dict(coin='pas:BTC',qty=1000.,buy_price=102.,sell_price=101.,time=60000)
        r=execute(a,pending,batch,price,cov,p,sc);self.assertEqual(r['filled'],0);self.assertTrue(r['limit_blocked'])
        batch['buy_price']=100.;r=execute(a,pending,batch,price,cov,p,sc);self.assertGreater(r['filled'],0)

    def test_risk_reduction_does_not_reuse_expired_entry_price(self):
        p,a,sc,price,cov=self.setup_case();a.physical_fill(0,1.,100.,9,0);a.owner_fill('base',0,1.,100.,9,0,'initial')
        pending=dict(qty=np.zeros(10),time=0,source=0,source_prices=price,entry_limits={0:200.},economic_execution=True)
        r=execute(a,pending,dict(coin='pas:BTC',qty=1000.,buy_price=90.,sell_price=90.,time=60000),price,cov,p,sc)
        self.assertEqual(r['filled'],1.);self.assertEqual(a.physical.qty[0],0.)

    def test_default_engine_matches_original_cash_and_fills(self):
        p,a,sc,price,cov=self.setup_case();start=43200000;end=start+7200000
        d=dict(times=np.array([start,start+60000,end]),close=np.tile(price,(3,1)),age_ms=np.zeros((3,10)),active=np.ones((3,10),bool))
        signal=dict(time=start+120000,source_time=start//1000,sigma=cov.tolist(),weights=[.1]+[0.]*9,volume=[1.]*10,trades=[1.]*10)
        tape=[dict(time=start+180000,coin='pas:BTC',qty=1000.,vwap=100.,buy_price=100.,sell_price=100.)]
        class Fixed:
            def propose(self,*args,**kwargs):return dict(target_qty=[1.]+[0.]*9)
        cfg=dict(start_ms=start,cutoff_ms=end)
        x=original(cfg,d,[signal],tape,[],p,sc,controller=Fixed());y=simulate(cfg,d,[signal],tape,[],p,sc,controller=Fixed())
        self.assertEqual(x,y)

if __name__=='__main__':unittest.main()
