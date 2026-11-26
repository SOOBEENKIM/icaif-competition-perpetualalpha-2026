"""Settlement fixtures: returns must survive deposits, reversals and ordering."""
from decimal import Decimal as D
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('wallet_accounting', Path(__file__).parents[1]/'scripts/build_public_wallet_leaderboard.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def fill(time, start, side, size, price, pnl='0', fee='0'):
    return dict(time=time,startPosition=str(start),side=side,sz=str(size),px=str(price),closedPnl=str(pnl),
                fee=str(fee),feeToken='USDPR',coin='par:X',crossed=True)

class PublicWalletAccountingTests(unittest.TestCase):
    def test_partial_exit_then_reverse_has_settlement_basis_and_actual_costs(self):
        fs=[fill(1,0,'B',2,10,fee=1),fill(2,2,'A',1,12,pnl=2,fee='.2'),
            fill(3,1,'A',2,11,pnl=1,fee='.3')]
        ordered,issues=m.ordered_fills(fs)
        result,symbols,errors=m.reconstruct(ordered,[{'delta':{'coin':'par:X','usdc':'.5'}}],[],{'par:X':D(9)})
        self.assertEqual(issues+errors,[])
        self.assertEqual(result['net'],D(4))  # 3 closed - 1.5 fees + .5 funding + 2 open.
        self.assertEqual(symbols['par:X']['quantity'],D(-1))
        self.assertEqual(symbols['par:X']['entry'],D(11))

    def test_reversed_same_millisecond_chain_and_round_trip(self):
        fs=[fill(1,1,'A',1,11,pnl=1),fill(1,0,'B',1,10)]
        ordered,issues=m.ordered_fills(fs)
        self.assertEqual(issues,[])
        self.assertEqual([f['side'] for f in ordered],['B','A'])
        self.assertEqual(m.reconstruct(ordered,[],[],{})[0]['net'],D(1))

    def test_missing_history_is_not_treated_as_complete(self):
        _,issues=m.ordered_fills([fill(1,5,'A',2,11,pnl=2)])
        self.assertTrue(issues)

    def test_displayed_closed_pnl_rounding_does_not_change_executed_price_return(self):
        fs=[fill(1,0,'B',1,'10.12345'),fill(2,1,'A',1,'11',pnl='.8766')]
        result,_,issues=m.reconstruct(fs,[],[],{})
        self.assertEqual(issues,[])
        self.assertEqual(result['net'],D('.87655'))
        self.assertEqual(result['closed'],D('.8766'))
        self.assertEqual(result['reported_pnl_rounding_adjustment'],D('-.00005'))

    def test_deposits_and_internal_transfers_are_not_trading_profit(self):
        wallet={'address':'alice','fills':{'rows':[]},'ledger':{'rows':[
            {'time':1,'delta':{'type':'spotTransfer','token':'USDPR','user':'funder','destination':'alice','amount':'1000000','feeToken':'USDPR','fee':'1'}},
            {'time':2,'delta':{'type':'send','token':'USDPR','user':'alice','destination':'alice','amount':'90000','feeToken':'','fee':'0'}},
            {'time':3,'delta':{'type':'spotTransfer','token':'USDPR','user':'alice','destination':'bob','amount':'100','feeToken':'USDPR','fee':'1'}}]}}
        flows,costs,issues=m.cashflows(wallet,3,[])
        self.assertEqual(issues,[])
        self.assertEqual(sum(v for _,v,_ in flows),D(999900))
        self.assertEqual(costs,[(3,D(1))])
        self.assertEqual(m.reconstruct([],[],costs,{})[0]['net'],D(-1))

if __name__ == '__main__':
    unittest.main()
