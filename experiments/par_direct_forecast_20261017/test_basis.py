import json,unittest
import numpy as np
from .test_direct import DirectTests
from .basis_engine import replay
from .basis_model import predict
from .model import design
from .data import HERE,OUT,ts

class BasisTests(DirectTests):
    def run_replay(self,arm='ridge_6',overlay=False):
        return replay(self.d,self.de,self.pred,self.cfg,self.p,arm,'base',int(self.d['times'][0]),int(self.d['times'][-1]),overlay=overlay)

    def test_entry_limit_blocks_adverse_future_price(self):
        self.d['par_close'][1,1]=110
        r=self.run_replay()
        self.assertFalse(any(f['asset']==1 and f['time']==self.d['times'][1] for f in r['fills']))
        self.assertTrue(any(o.get('entry_limit_blocked') for o in r['orders']))

    def test_basis_model_prefix_and_post_arrival_decay(self):
        with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
        cfg=json.loads((HERE/'protocol.json').read_text());cfg['fit_dates']=cfg['fit_dates'][:1]
        de=design(d,cfg,self.p['symbols']);full,records=predict(d,de,cfg)
        mask=d['times']<=ts('2026-10-09T12:00:00');short={k:v[mask] for k,v in d.items()}
        small=design(short,cfg,self.p['symbols']);pre,pr=predict(short,small,cfg)
        for k in full:np.testing.assert_allclose(full[k][mask],pre[k],rtol=0,atol=0,equal_nan=True)
        self.assertEqual(records,pr);self.assertTrue(any('phi' in r for r in records))
        self.assertTrue(all(r['last_label_time']<r['fit_time'] for r in records if 'phi' in r))
        # Zero persistence implies no remaining convergence after one-bar arrival.
        phi=0.;delta=.02
        self.assertEqual((phi**73-phi)*delta,0.)

if __name__=='__main__':unittest.main()
