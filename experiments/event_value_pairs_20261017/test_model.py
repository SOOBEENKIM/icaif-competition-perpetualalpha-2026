import copy,unittest
from unittest.mock import patch
import numpy as np
from .model import source_design,completed_events,fit_at,map_state


class EventValueTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(entry_z=3.,stop_multiplier=2.,maximum_holding_hours=168,time_decay_start_hours=84,pairs=[['A','B']])
        self.spec=dict(training_window_days=180,minimum_events=24,minimum_group_events=12,prior_events=24)

    def test_statistical_recross_without_price_motion_is_not_confirmed(self):
        ref=dict(market_times=np.array([0,3600]),close=np.full((2,2),100.))
        raw=dict(pairs=[[0,1]],beta=np.ones((2,1)),mean=np.array([[-.031],[-.029]]),sd=np.full((2,1),.01),
            z=np.array([[3.1],[2.9]]),delayed=np.array([[0],[-1]]))
        with patch('experiments.event_value_pairs_20261017.model.design',return_value=raw):d=source_design(ref,self.cfg,['A','B'])
        self.assertFalse(d['confirmed'][1,0]);self.assertFalse(d['price_recovery'][1,0])

    def test_real_price_recross_is_confirmed_under_current_parameters(self):
        ref=dict(market_times=np.array([0,3600]),close=np.array([[100*np.exp(.031),100],[100*np.exp(.029),100]]))
        raw=dict(pairs=[[0,1]],beta=np.ones((2,1)),mean=np.zeros((2,1)),sd=np.full((2,1),.01),
            z=np.array([[3.1],[2.9]]),delayed=np.array([[0],[-1]]))
        with patch('experiments.event_value_pairs_20261017.model.design',return_value=raw):d=source_design(ref,self.cfg,['A','B'])
        self.assertTrue(d['confirmed'][1,0]);self.assertTrue(d['price_recovery'][1,0])

    def paths(self):
        price=np.array([[100*np.exp(-.029),100],[98.,100],[100.,100],[110.,100],[99.,100]])
        ref=dict(market_times=np.arange(5)*3600,close=price)
        raw=dict(pairs=[[0,1]],delayed=np.ones((5,1),int),beta=np.ones((5,1)),sd=np.full((5,1),.01),
            mean=np.zeros((5,1)),z=np.full((5,1),-2.9),confirmed=np.ones((5,1),bool),price_recovery=np.ones((5,1),bool))
        return ref,raw

    def test_label_matures_after_exit_observation_and_does_not_overlap(self):
        ref,raw=self.paths();e=completed_events(ref,raw,self.cfg)
        self.assertEqual(len(e),1);self.assertEqual(e[0]['maturity'],10800);self.assertEqual(e[0]['exit_reason'],'mean_cross')
        q=np.array([.5,-.5])/ref['close'][0]
        self.assertAlmostEqual(e[0]['realized_price_return'],float(q@(ref['close'][3]-ref['close'][0])))

    def test_censored_path_never_becomes_a_zero_return_training_label(self):
        ref,raw=self.paths();ref={k:v[:3] for k,v in ref.items()};raw={k:v[:3] if isinstance(v,np.ndarray) else v for k,v in raw.items()}
        events=completed_events(ref,raw,self.cfg);self.assertEqual(len(events),1);self.assertIsNone(events[0]['maturity'])
        self.assertEqual(fit_at(events,20000,self.spec)['events'],0)

    def test_fit_excludes_same_time_or_future_labels_and_uses_zero_prior(self):
        history=[dict(time=i*3600,maturity=(i+1)*3600,pair_index=0,confirmed=True,expected_full_convergence=.01,realized_price_return=.02) for i in range(24)]
        at=25*3600;future=dict(time=at-3600,maturity=at,pair_index=0,confirmed=True,expected_full_convergence=.01,realized_price_return=-100)
        a=fit_at(history+[future],at,self.spec);self.assertEqual(a['events'],24)
        self.assertAlmostEqual(a['groups']['True']['ratio'],1.)
        future['realized_price_return']=100;self.assertEqual(a,fit_at(history+[future],at,self.spec))
        b=fit_at(history,24*3600,self.spec);self.assertIsNone(b['groups']['True']['ratio'])

    def test_shrink_keeps_negative_evidence_and_never_increases_full_target(self):
        history=[dict(time=i,maturity=i+1,pair_index=0,confirmed=False,expected_full_convergence=.01,realized_price_return=-.01) for i in range(30)]
        r=fit_at(history,100,self.spec);self.assertEqual(r['groups']['False']['ratio'],0)
        self.assertLess(r['groups']['False']['unshrunk_ratio'],0)

    def test_source_anchor_and_source_fit_survive_later_venue_price_and_fit_changes(self):
        ref=dict(market_times=np.array([0,3600]),close=np.array([[100.,200.],[999.,999.]]))
        state=dict(directions=np.array([[1],[1]]),signal_times=np.array([[0],[0]]),terminal_prices=np.array([[[110.,180.]],[[110.,180.]]]))
        raw=dict(pairs=[[0,1]],confirmed=np.ones((2,1),bool))
        fits=[dict(time=0,last_maturity=-1,groups={'True':{'ratio':.25}}),dict(time=3600,last_maturity=3599,groups={'True':{'ratio':1.}})]
        mapped,audit=map_state(state,np.array([0,3600]),ref,raw,fits,'calibrated')
        np.testing.assert_array_equal(mapped['terminal_prices'],np.array([[[102.5,195.]],[[102.5,195.]]]))
        self.assertEqual(len(audit),1);self.assertEqual(audit[0]['fit_time'],0)
        np.testing.assert_array_equal(state['terminal_prices'][0,0],[110.,180.])


if __name__=='__main__':unittest.main()
