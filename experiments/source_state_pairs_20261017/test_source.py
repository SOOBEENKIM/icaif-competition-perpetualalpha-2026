import copy,unittest
import numpy as np
from strategy.agent import TAIL
import json
from experiments.adaptive_mean_pairs_20261017.signals import design
from .audit import inputs
from .signals import reference_confirmation


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg,cls.d,cls.de,cls.ref=inputs()

    def test_future_truncation_preserves_every_prior_confirmation(self):
        cfg,d,de,ref=self.cfg,self.d,self.de,self.ref
        cut=int(d['times'][len(d['times'])//2]);dm=d['times']<=cut;rm=ref['market_times']<=cut
        full=reference_confirmation(d,de,ref,cfg['entry_z'])
        subd={k:v[dm] for k,v in d.items()};subde={k:(v if k=='pairs' else v[dm]) for k,v in de.items()}
        subref={k:v[rm] for k,v in ref.items()}
        short=reference_confirmation(subd,subde,subref,cfg['entry_z'])
        for k in ('standard','delayed','reference_z','z'):np.testing.assert_array_equal(full[k][dm],short[k])

    def test_venue_price_changes_do_not_change_reference_confirmation(self):
        cfg,d,de,ref=self.cfg,self.d,self.de,self.ref
        symbols=json.loads((TAIL/'baseline.json').read_text())['protocol']['symbols']
        altered={k:v.copy() for k,v in d.items()};altered['par_close'][:,0]*=1.01
        altered_de=design(altered,ref,cfg,symbols)
        original=reference_confirmation(d,de,ref,cfg['entry_z'])
        revised=reference_confirmation(altered,altered_de,ref,cfg['entry_z'])
        np.testing.assert_array_equal(original['standard'],revised['standard'])
        np.testing.assert_array_equal(original['delayed'],revised['delayed'])
        self.assertTrue(np.any(original['z']!=revised['z']))

    def test_current_venue_stop_and_cost_state_is_retained(self):
        out=reference_confirmation(self.d,self.de,self.ref,self.cfg['entry_z'])
        for k in ('z','beta','mean','sd','mean_log_prices','forecast_times'):
            np.testing.assert_array_equal(out[k],self.de[k])
        self.assertTrue(np.all(out['forecast_times']<=self.d['times']))


if __name__=='__main__':unittest.main()
