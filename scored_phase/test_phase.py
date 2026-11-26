import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from .profile import START, END, load_profile
from .venue import account_equity, validate_metadata, phase_state, check_oi, PublicPAS
from .execution import prepare_pas_order, PASExchange
from .scoring import evaluate


class PhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile=load_profile()
        cls.snapshot=json.loads((Path(__file__).parent/'artifacts/public_preflight/snapshot.json').read_text())

    def order(self,**kwargs):
        params=dict(profile=self.profile,coin='pas:BTC',is_buy=True,size='.00123',price=80000,
            market={'name':'pas:BTC','szDecimals':5},context={'markPx':'80000'},slot='test')
        params.update(kwargs)
        return prepare_pas_order(**params)

    def test_capital_and_absolute_caps_scaled_not_model(self):
        p=self.profile
        self.assertEqual(p['trading']['initial'],100000)
        self.assertEqual(p['trading']['risk']['asset_notional'],15000)
        self.assertEqual(p['configs']['reference']['pair_max_gross'],12000)
        self.assertEqual(p['trading']['risk']['gross_fraction'],.6)
        self.assertEqual(p['gate_spec']['horizon'],24)

    def test_live_metadata_confirms_usdp_and_changed_btc_precision(self):
        report=validate_metadata(self.snapshot,self.profile)
        self.assertEqual(report['size_decimals']['pas:BTC'],5)
        self.assertEqual(report['collateral'],'USDP')
        self.assertAlmostEqual(report['taker_bps'],9)

    def test_changed_collateral_or_cap_rejected(self):
        for key in ['collateral','cap']:
            s=copy.deepcopy(self.snapshot)
            if key=='collateral':s['meta']['collateralToken']=-1
            else:s['limits']['totalOiCap']='500000000'
            with self.assertRaises(ValueError):validate_metadata(s,self.profile)

    def test_unified_ignores_practice_balance_adds_pnl_once(self):
        s=copy.deepcopy(self.snapshot)
        s['spot']['balances']=[{'coin':'USDPR','token':999,'total':'1000000'},
            {'coin':'USDP','token':self.profile['collateral_token'],'total':'99995'}]
        s['positions']['assetPositions']=[{'position':{'coin':'pas:BTC','szi':'.01','unrealizedPnl':'15'}}]
        s['positions']['marginSummary']['accountValue']='100010'
        self.assertEqual(account_equity(s,self.profile)['equity'],100010)

    def test_standard_uses_dex_equity_without_spot_double_count(self):
        s=copy.deepcopy(self.snapshot);s['mode']='disabled'
        s['positions']['marginSummary']['accountValue']='100012'
        s['spot']['balances']=[{'coin':'USDP','token':self.profile['collateral_token'],'total':'100000'}]
        self.assertEqual(account_equity(s,self.profile)['equity'],100012)

    def test_zero_funding_and_unsupported_mode(self):
        s=copy.deepcopy(self.snapshot);s['spot']['balances']=[];s['positions']['assetPositions']=[]
        self.assertFalse(account_equity(s,self.profile)['funded'])
        s['mode']='futureMode'
        with self.assertRaises(ValueError):account_equity(s,self.profile)

    def test_phase_boundaries(self):
        self.assertEqual(phase_state(START-1,self.profile),'before_start')
        self.assertEqual(phase_state(START,self.profile),'trading')
        self.assertEqual(phase_state(END-900,self.profile),'closing_only')
        self.assertEqual(phase_state(END,self.profile),'ended')

    def test_pas_lot_and_stable_order_identity(self):
        a=self.order();b=self.order()
        self.assertEqual(a.size,'0.00123');self.assertEqual(a.cloid,b.cloid)

    def test_wrong_dex_and_oversized_order_rejected(self):
        for coin in ['par:BTC','pa:BTC','BTC']:
            with self.assertRaises(ValueError):self.order(coin=coin)
        with self.assertRaises(ValueError):self.order(size='1')

    def test_sub_minimum_only_entire_reduce_only_close(self):
        o=self.order(is_buy=False,size='.00001',current_qty=.00001,reduce_only=True)
        self.assertEqual(o.size,'0.00001')
        with self.assertRaises(ValueError):self.order(size='.00001')
        with self.assertRaises(ValueError):self.order(is_buy=False,size='.00001',current_qty=.00002,reduce_only=True)

    def test_market_and_global_oi_caps(self):
        meta={'universe':[{'name':'pas:BTC'},{'name':'pas:ETH'}]}
        ctx=[{'openInterest':'124.999','oraclePx':'80000'},{'openInterest':'10000','oraclePx':'4000'}]
        limits={'coinToOiCap':[['pas:BTC','10000000'],['pas:ETH','40000000']],'totalOiCap':'50000000'}
        with self.assertRaisesRegex(ValueError,'Market OI'):check_oi([self.order()],[],meta,ctx,limits)
        limits['coinToOiCap'][0][1]='20000000'
        with self.assertRaisesRegex(ValueError,'DEX OI'):check_oi([self.order()],[],meta,ctx,limits)
        positions=[{'position':{'coin':'pas:BTC','szi':'.00123'}}]
        reduction=self.order(is_buy=False,current_qty=.00123,reduce_only=True)
        self.assertEqual(check_oi([reduction],positions,meta,ctx,limits)['increment_upper_bound'],0)

    def test_public_adapter_cannot_query_wrong_dex_or_send_action(self):
        public=PublicPAS(self.profile)
        with self.assertRaises(ValueError):public.post({'type':'clearinghouseState','dex':'par'})
        with self.assertRaises(ValueError):public.post({'type':'order'})
        public.session.close()

    def test_low_level_transmission_cannot_bypass_phase_and_market(self):
        client=object.__new__(PASExchange);client.base_url=self.profile['rest_url']
        client.scored_profile=self.profile;client.pas_asset_ids={123};client.orders_enabled=True
        good={'action':{'type':'order','orders':[{'a':123,'r':False}]}}
        with patch('scored_phase.execution.time.time',return_value=START-1):
            with self.assertRaises(ValueError):client.post('/exchange',good)
        with patch('scored_phase.execution.time.time',return_value=END):
            with self.assertRaises(ValueError):client.post('/exchange',good)
        with patch('scored_phase.execution.time.time',return_value=START+10):
            with self.assertRaises(ValueError):client.post('/exchange',{'action':{'type':'order','orders':[{'a':999}]}})
            with self.assertRaises(ValueError):client.post('/exchange',{'action':{'type':'withdraw3'}})

    def test_scoring_uses_daily_12utc_and_ignores_transfers(self):
        samples=[{'time':START,'equity':100000},
            {'time':START+86400,'equity':111000,'net_external_since_previous':10000},
            {'time':START+172800,'equity':109890}]
        result=evaluate(samples,trading_days=1)
        self.assertAlmostEqual(result['return_fraction'],1.01*.99-1)
        self.assertAlmostEqual(result['volatility'],.02)
        self.assertAlmostEqual(result['score'],.8*(.005/(.0001+1.5*.02+.05)))
        self.assertTrue(result['eligible'])
        with self.assertRaises(ValueError):evaluate([{'time':START+1,'equity':100000}])

    def test_missing_day_normalization_and_partial_day(self):
        samples=[{'time':START,'equity':100000},
            {'time':START+2*86400,'equity':102000},
            {'time':START+3*86400,'equity':102000}]
        plain=evaluate(samples,trading_days=1)
        partial=evaluate(samples,trading_days=1,latest={'time':START+3*86400+3600,'equity':103000})
        self.assertAlmostEqual(plain['volatility'],.02/math.sqrt(2))
        self.assertEqual(plain['volatility'],partial['volatility'])
        self.assertAlmostEqual(partial['return_fraction'],.02)
        self.assertAlmostEqual(partial['intraday_return_fraction'],.03)
        self.assertEqual(plain['score'],partial['score'])
        self.assertFalse(evaluate(samples[:2],trading_days=1)['eligible'])
        self.assertFalse(evaluate(samples,trading_days=0)['eligible'])


if __name__=='__main__':unittest.main()
