import copy
import unittest
import numpy as np
from .formation import forest, select, mask_entries
from experiments.decision_gate_pairs_20261017.entry import filter_proposals


class FormationTests(unittest.TestCase):
    def fixture(self):
        from datetime import datetime, timezone
        cutoff = 2000 * 3600
        t = np.arange(1900, 2020) * 3600
        rng = np.random.default_rng(10)
        x = rng.normal(size=(len(t), 3)) * .01
        x[:, 1] = .9*x[:, 0] + .1*x[:, 1]
        d = dict(market_times=t, close=100*np.exp(x.cumsum(0)),
                 volume=np.ones_like(x), trades=np.ones_like(x))
        s = dict(formation_end_exclusive=datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
                 formation_days=3, minimum_observations=20, minimum_correlation=.3,
                 groups=[['A','B','C']], pairs=[['A','B'],['A','C'],['B','C']])
        return d, s, cutoff

    def test_future_mutation_has_no_effect(self):
        d,s,cut = self.fixture(); expected = select(d,['A','B','C'],s)
        changed = copy.deepcopy(d)
        for k in ['close','volume','trades']:
            changed[k][d['market_times'] >= cut] = np.nan
        self.assertEqual(expected, select(changed,['A','B','C'],s))
        self.assertTrue(all(e['last_observation'] < cut for e in expected['edges']))

    def test_activity_is_required_at_both_endpoints(self):
        d,s,_ = self.fixture(); original = select(d,['A','B','C'],s)['edges'][0]
        d['volume'][55,0] = 0
        changed = select(d,['A','B','C'],s)['edges'][0]
        self.assertEqual(original['observations'] - changed['observations'], 2)

    def test_insufficient_data_does_not_force_pair(self):
        d,s,_ = self.fixture(); s['minimum_observations']=1000
        self.assertEqual(select(d,['A','B','C'],s)['selected_pairs'], [])

    def test_ties_deterministic_and_no_cycle(self):
        edges=[dict(pair=p,distance=1) for p in [['B','C'],['A','C'],['A','B']]]
        self.assertEqual([e['pair'] for e in forest(edges,['A','B','C'])], [['A','B'],['A','C']])
        self.assertEqual(forest(edges,['A','B','C']), forest(edges[::-1],['A','B','C']))

    def test_missing_hour_is_not_a_single_hour_return(self):
        d,s,_ = self.fixture(); before=select(d,['A','B','C'],s)['edges'][0]['observations']
        d={k:np.delete(v,55,axis=0) for k,v in d.items()}
        after=select(d,['A','B','C'],s)['edges'][0]['observations']
        self.assertEqual(before-after,2)

    def test_entry_mask_keeps_held_forecasts_and_input(self):
        state=dict(pairs=np.array([[0,1],[0,2]]), economic_available=np.ones((3,2),bool),
                   economic_action=np.ones((3,2)),directions=np.ones((3,2)),mu=np.arange(6).reshape(3,2))
        changed=mask_entries(state,[True,False])
        self.assertTrue(state['economic_available'].all())
        for key in state:
            if key!='economic_available':np.testing.assert_array_equal(changed[key],state[key])
        choices=[dict(pair_index=0,direction=1),dict(pair_index=1,direction=1)]
        self.assertEqual(filter_proposals(choices,changed['economic_action'][0],changed['economic_available'][0]),choices[:1])


if __name__=='__main__':unittest.main()
