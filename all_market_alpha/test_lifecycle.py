import unittest
import numpy as np
from .wave2 import vacant_overlay
from .wave3 import allocate
from .wave4 import reverse_targets


class LifecycleTests(unittest.TestCase):
    def test_opposite_trend_reverses_every_asset_without_named_filter(self):
        n=72;t=1775001600+np.arange(n)*3600
        base=np.ones((n,10))*.03
        auxiliary=np.ones((n,10))*-.01
        quality=np.ones((n,10))*-.8
        before=allocate(base,quality,t)
        self.assertTrue(np.all(before[-1]>0))
        after=reverse_targets(base,before,auxiliary,quality,t,"reverse")
        np.testing.assert_allclose(after[-1],-.01)

    def test_one_negative_bar_cannot_reverse_a_position(self):
        n=72;t=1775001600+np.arange(n)*3600
        base=np.ones((n,10))*.03;aux=-base
        quality=np.ones((n,10))*.8;quality[24]=-.8
        after=reverse_targets(base,base,aux,quality,t,"reverse")
        np.testing.assert_array_equal(after,base)

    def test_exit_control_never_invents_a_short(self):
        n=72;t=1775001600+np.arange(n)*3600
        base=np.ones((n,10))*.03;aux=-base;quality=np.ones((n,10))*-.8
        out=reverse_targets(base,base,aux,quality,t,"exit")
        self.assertTrue(np.all(out>=0))
        np.testing.assert_array_equal(out[-1],0)

    def test_inherited_target_has_priority_in_vacant_layer(self):
        n=72;t=1775001600+np.arange(n)*3600
        base=np.zeros((n,10));base[1:,0]=.03
        aux=np.ones((n,10))*.02
        out=vacant_overlay(base,aux,t)
        self.assertEqual(out[0,0],.005)
        self.assertEqual(out[1,0],.03)
        self.assertTrue(np.all(out[-1,1:]==.005))

    def test_appending_future_quality_cannot_change_past_reversal(self):
        rng=np.random.default_rng(92);n=720
        t=1775001600+np.arange(n)*3600
        base=rng.normal(0,.02,(n,10));q=np.tanh(rng.normal(size=(n,10)))
        aux=-base
        a=reverse_targets(base,base,aux,q,t,"reverse")
        stop=n-48
        b=reverse_targets(base[:stop],base[:stop],aux[:stop],q[:stop],t[:stop],"reverse")
        np.testing.assert_array_equal(a[:stop],b)


if __name__=="__main__":
    unittest.main()
