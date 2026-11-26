import copy
import json
import unittest
from pathlib import Path
import numpy as np
from auto_pair_selection.test_selection import data
from auto_pair_selection.signals import prepare_features
from auto_pair_selection.selection import PAIRS, IDS
from .policy import ValidationSpec, fit_calibration, calibrate, current_action, monitor, apply_entry_gate
from .study import fit_raw


class ValidationTests(unittest.TestCase):
    def test_calibration_ignores_future_and_unmatured_labels(self):
        t=np.arange(100)*86400;x=np.linspace(-1,1,100);y=.05+.4*x;m=t+86400
        a=fit_calibration(t,x,y,m,80*86400)
        poison=y.copy();poison[m>=80*86400]=np.nan
        self.assertEqual(a,fit_calibration(t,x,poison,m,80*86400))
        self.assertLess(a['last_label_maturity'],80*86400)
        self.assertEqual(a,fit_calibration(t[:80],x[:80],y[:80],m[:80],80*86400))

    def test_uncertain_or_insufficient_forecast_abstains(self):
        self.assertEqual(current_action(.01,.009),0)
        self.assertEqual(current_action(.01,.001),1)
        self.assertEqual(current_action(-.01,.001),-1)
        self.assertIsNone(fit_calibration(np.arange(4)*86400,np.ones(4),np.ones(4),np.arange(4)*86400+86400,10*86400))
        self.assertIsNone(calibrate(None,.01,.02))

    def test_bad_prediction_history_is_rejected_good_long_and_short_can_pass(self):
        t=np.arange(60)*86400;m=t+86400;p=np.tile([.01,-.01],30);se=np.full(60,.0001)
        good=monitor(t,p,se,p,m,50*86400)
        bad=monitor(t,p,se,-p,m,50*86400)
        self.assertTrue(good['eligible']);self.assertFalse(bad['eligible'])
        self.assertIn('forecast_not_better_than_zero',bad['reasons'])
        self.assertIn('past_net_proxy_not_robustly_positive',bad['reasons'])
        poison=p.copy();poison[m>=50*86400]=1e9
        self.assertEqual(good,monitor(t,p,se,poison,m,50*86400))

    def test_monitor_cannot_relabel_old_bad_forecasts_with_new_good_fit(self):
        t=np.arange(60)*86400;p=np.full(60,.01);actual=-p;se=np.zeros(60)
        # The monitor consumes immutable publications, not current fit output.
        wrong=monitor(t,p,se,actual,t+86400,50*86400)
        hindsight=monitor(t,-p,se,actual,t+86400,50*86400)
        self.assertFalse(wrong['eligible']);self.assertTrue(hindsight['eligible'])

    def test_gate_uses_same_logic_for_all45_and_keeps_held_forecasts(self):
        n=len(PAIRS)
        s=dict(mu=np.ones((1,n))*.01,raw_mu=np.ones((1,n))*.02,
          directions=np.ones((1,n),int),pairs=np.asarray(IDS),
          gate_available=np.ones((1,n),bool),gate_action=np.ones((1,n),int),
          calibration_available=np.ones((1,n),bool),validated=np.ones((1,n),bool),
          calibrated_action=-np.ones((1,n),int))
        k=PAIRS.index(('EWY','EWJ'));s['validated'][0,k]=False
        a=apply_entry_gate(s,'validated_all45')
        self.assertEqual(np.count_nonzero(a['economic_action']),44)
        self.assertEqual(a['economic_action'][0,k],0);self.assertEqual(a['mu'][0,k],.01)
        self.assertEqual(a['directions'][0,k],1)
        self.assertEqual(a['economic_action'][0,PAIRS.index(('USDEUR','USDJPY'))],-1)
        b=apply_entry_gate(s,'expanded_only')
        self.assertTrue((b['economic_action']==1).all());np.testing.assert_array_equal(b['mu'],s['raw_mu'])

    def test_future_prices_do_not_change_historical_features(self):
        d=data();f,y,m,terminal=prepare_features(d);cut=90*24
        altered=copy.deepcopy(d);altered['close'][cut+1:]*=np.arange(1,11)*20
        g,_,_,_=prepare_features(altered)
        for key in ('x','beta','vol','fresh'):
            np.testing.assert_allclose(f[key][:cut+1],g[key][:cut+1],equal_nan=True)

    def test_all45_models_and_training_maturity(self):
        d=data();f,y,m,terminal=prepare_features(d);source=90*86400
        ds=json.loads(Path('competition_live/strategy_plan.json').read_text())['direct_spec']
        gs=json.loads(Path('competition_live/config.json').read_text())['gate_spec']
        a=fit_raw(d,f,y,m,terminal,source,ds,gs,False)
        poison=y.copy();poison[m>=source]=1e9
        b=fit_raw(d,f,poison,m,terminal,source,ds,gs,False)
        self.assertEqual(a,b);self.assertEqual(len(a['direct']),45)
        self.assertTrue(all(x['last_maturity']<source for x in a['direct'].values()))


if __name__=='__main__':unittest.main()
