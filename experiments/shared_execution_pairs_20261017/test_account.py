import copy,json,unittest
import numpy as np
from strategy.agent import TAIL
from replay_lab.common_comparison.models import Market
from .account import SharedAccount


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.p=json.loads((TAIL/'baseline.json').read_text())['protocol'];self.p['execution']['fee_bps']=9
        self.sc=dict(fee_bps=9,impact_bps=10,participation=.1,depth_fraction=1.,latency_ms=1000)
        self.a=SharedAccount(1e6,self.p['symbols']);self.price=np.full(10,100.);self.sigma=np.eye(10)*1e-7
        self.market=Market(np.array([0,300,600]),np.full((3,10),100.),np.full((3,10),10000.),np.full((3,10),10),np.zeros((3,10)),np.ones((3,10),bool),tuple(self.p['symbols']))
        self.cfg=dict(book_max_age_ms=15000,context_max_age_ms=3000,future_exit_observation_grace_ms=5000)

    def intent(self,b=0.,q=0.,at=0,**kwargs):
        base=np.zeros(10);pair=np.zeros(10);base[0]=b;pair[0]=q
        raw=dict(decision_time=at,source_close=at,source_available=at,execution_time=at+300,
            source_volume=np.full(10,10000.),source_trades=np.full(10,10),source_prices=self.price.tolist(),
            owner_targets={'base':base.tolist(),'pair':pair.tolist()},risk_forced=False,sigma=self.sigma,**kwargs)
        return self.a.seal(raw,self.p,self.sc)

    def book(self,at=1000,price=100.,size=100.):
        value={'time':at,'levels':[[{'px':str(price-.01),'sz':str(size)}],[{'px':str(price+.01),'sz':str(size)}]]}
        return dict(book=[(at,0,value)],ctx=[(at,0,{'oraclePx':str(price),'funding':'0'})],book_times=[at],ctx_times=[at],resets=[])

    def test_opposing_owners_do_not_create_external_cash_or_fees(self):
        self.a.execute_candle(self.intent(10,-10),self.market,1,self.p,self.sc)
        self.assertEqual(self.a.physical.fills,[]);self.assertEqual(self.a.physical.fees,0)
        self.assertEqual(self.a.owners['base'].qty[0],10);self.assertEqual(self.a.owners['pair'].qty[0],-10)
        self.price[0]=120;self.assertEqual(self.a.reconcile(self.price),1e6)
        self.assertAlmostEqual(self.a.owners['base'].equity(self.price)-1e6,-self.a.owners['pair'].equity(self.price))

    def test_same_direction_sends_one_net_order_and_counts_fees_once(self):
        self.a.execute_candle(self.intent(10,5),self.market,1,self.p,self.sc)
        self.assertEqual(len(self.a.physical.fills),1);self.assertEqual(self.a.physical.qty[0],15)
        self.assertEqual(self.a.owners['base'].qty[0],10);self.assertEqual(self.a.owners['pair'].qty[0],5)
        self.assertAlmostEqual(self.a.physical.fees,15*100.1*.0009)

    def test_partial_capacity_prioritizes_pair_and_can_retry_without_double_fills(self):
        self.market.volume[1,0]=40
        self.a.execute_candle(self.intent(10,5),self.market,1,self.p,self.sc)
        self.assertEqual(self.a.physical.qty[0],4);self.assertEqual(self.a.owners['pair'].qty[0],4)
        self.a.execute_candle(self.intent(10,5,at=300),self.market,2,self.p,self.sc)
        self.assertEqual(self.a.physical.qty[0],15);self.assertEqual(self.a.owners['pair'].qty[0],5)

    def test_pair_limit_still_binds_when_physical_order_reduces(self):
        self.a.execute_candle(self.intent(-10,0),self.market,1,self.p,self.sc)
        self.a.execute_candle(self.intent(-10,5,at=300,owner_limits={'pair':{'0':99.}}),self.market,2,self.p,self.sc)
        self.assertEqual(self.a.physical.qty[0],-10);self.assertEqual(self.a.owners['pair'].qty[0],0)

    def test_disagreeing_internal_limits_do_not_create_virtual_fill(self):
        self.a.execute_candle(self.intent(10,-10,owner_limits={'base':{'0':99.},'pair':{'0':101.}}),self.market,1,self.p,self.sc)
        self.assertEqual(self.a.transfers,[]);self.assertEqual(self.a.owners['pair'].qty[0],0)

    def test_net_funding_matches_owner_sum_under_effective_stress_rate(self):
        self.a.execute_candle(self.intent(10,-5),self.market,1,self.p,self.sc)
        rates=np.zeros(10);rates[0]=-.00005
        self.a.charge_funding(3600,self.price,rates,np.ones(10,bool));self.a.reconcile(self.price)
        self.assertAlmostEqual(self.a.physical.funding,.025)
        self.assertAlmostEqual(sum(x.funding for x in self.a.owners.values()),.025)

    def test_virtual_reversal_is_split_and_conserved(self):
        self.a.execute_candle(self.intent(10,-10),self.market,1,self.p,self.sc)
        self.a.execute_candle(self.intent(-5,5,at=300),self.market,2,self.p,self.sc)
        self.assertEqual(self.a.physical.fills,[]);self.assertEqual(self.a.owners['base'].qty[0],-5)
        self.a.reconcile(self.price)

    def test_duplicate_packet_is_rejected(self):
        packet=self.intent(10,5);self.a.execute_candle(packet,self.market,1,self.p,self.sc)
        with self.assertRaises(RuntimeError):self.a.execute_candle(packet,self.market,1,self.p,self.sc)

    def test_missing_book_outcome_blocks_new_order_instead_of_assuming_no_fill(self):
        r=self.a.execute_books(self.intent(10,0),{},self.price,self.sigma,self.p,self.sc,self.cfg)
        self.assertEqual(r['unknown_symbols'],['BTC']);self.assertIsNone(r['orders'][0]['filled_qty'])
        with self.assertRaises(RuntimeError):self.a.execute_books(self.intent(10,0,at=1),{},self.price,self.sigma,self.p,self.sc,self.cfg)

    def test_no_candle_volume_gate_in_received_book_execution(self):
        packet=self.intent(10,5);packet['source_volume'][:]=0;packet['source_trades'][:]=0
        r=self.a.execute_books(packet,{'BTC':self.book()},self.price,self.sigma,self.p,self.sc,self.cfg)
        self.assertEqual(r['unknown_symbols'],[]);self.assertEqual(self.a.physical.qty[0],15)
        self.assertAlmostEqual(self.a.physical.fees,15*100.01*.0009)

    def test_book_limits_are_sealed_before_future_price_jump(self):
        packet=self.intent(10,0)
        self.a.execute_books(packet,{'BTC':self.book(price=105.)},self.price,self.sigma,self.p,self.sc,self.cfg)
        self.assertEqual(self.a.physical.qty[0],0)

    def test_hour_boundary_requires_settlement_processing(self):
        packet=self.intent(10,0,at=3599)
        with self.assertRaises(ValueError):self.a.execute_books(packet,{'BTC':self.book(at=3600000)},self.price,self.sigma,self.p,self.sc,self.cfg)
        self.assertEqual(self.a.physical.fills,[])


if __name__=='__main__':unittest.main()
