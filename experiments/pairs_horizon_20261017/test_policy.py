import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from .policy import HorizonMatchedController


class TimingTests(unittest.TestCase):
    def test_exit_order_is_issued_before_label_maturity(self):
        cfg = json.loads(Path('experiments/pairs_forecast_20261017/protocol.json').read_text())
        cfg['pairs'] = [['ETH', 'BTC']]
        market = SimpleNamespace(symbols=['ETH', 'BTC'], exact=lambda t: int(t / 3600))
        c = HorizonMatchedController(market, np.full((100, 1), -.1), np.ones((100, 1)), np.ones((100, 1)) * .1,
                                    cfg, 72, {'impact_bps': 10}, 999)
        c.states[0] = dict(phase='holding', planned=np.array([1., -1.]), actual=np.array([1., -1.]),
            entry=4, entry_prices=np.array([100., 100.]), beta=1., direction=1, entry_vol=.1)
        protocol = dict(risk={}, execution={'fee_bps': 9})
        with patch('experiments.pairs_horizon_20261017.policy.risk_breaches', return_value=[]):
            def decision(i):
                return c.propose(i*3600, np.array([100.,100.]), np.array([1.,-1.]), 1e6, None,
                    np.eye(2), np.ones(2), np.ones(2), protocol)
            self.assertIsNone(decision(74))
            close = decision(75)
        self.assertEqual(close['target_qty'], [0., 0.])
        self.assertEqual(c.events[-1]['reason'], 'label_maturity')
        self.assertEqual(75 + 1 - 4, 72)  # next-step fill minus actual entry.


if __name__ == '__main__': unittest.main()
