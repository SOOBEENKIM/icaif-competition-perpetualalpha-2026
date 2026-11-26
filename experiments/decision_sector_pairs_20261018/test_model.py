import copy,unittest
import numpy as np
from .model import restrict_pairs,assert_original_pair_fields
from experiments.decision_gate_pairs_20261017.entry import filter_proposals
from experiments.decision_gate_pairs_20261017.test_model import DecisionGateTests


class UniverseTests(unittest.TestCase):
    def test_fixed_crypto_universe_cannot_trade_equity_signal(self):
        state=dict(pairs=np.arange(12).reshape(6,2),directions=np.ones((2,6),int),
                   economic_available=np.ones((2,6),bool),economic_action=np.ones((2,6),int),mu=np.ones((2,6)))
        saved=copy.deepcopy(state);restricted=restrict_pairs(state,3)
        proposals=[dict(pair_index=k,direction=1) for k in range(6)]
        self.assertEqual([p['pair_index'] for p in filter_proposals(proposals,restricted['economic_action'][0],restricted['economic_available'][0])],[0,1,2])
        for key in saved:np.testing.assert_array_equal(state[key],saved[key])
        np.testing.assert_array_equal(restricted['mu'],state['mu'])

    def test_pair_identity_not_confused_with_asset_price_axis(self):
        old=dict(pairs=np.array([[0,1],[0,2]]),signal_times=np.zeros((3,2)),source_prices=np.ones((3,10)),
                 economic_action=np.ones((3,2)),economic_available=np.ones((3,2),bool))
        new=dict(old,pairs=np.array([[0,1],[0,2],[1,2]]),signal_times=np.zeros((3,3)),
                 economic_action=np.ones((3,3)),economic_available=np.ones((3,3),bool))
        assert_original_pair_fields(new,old)
        new['pairs']=np.array([[1,0],[0,2],[1,2]])
        with self.assertRaises(AssertionError):assert_original_pair_fields(new,old)


if __name__=='__main__':unittest.main()
