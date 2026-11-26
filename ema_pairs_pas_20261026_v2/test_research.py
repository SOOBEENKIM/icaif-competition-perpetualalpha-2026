import copy,unittest
import numpy as np
from .model import estimates,basis_projection,PAIRS

def fixture():
 rng=np.random.default_rng(4026);n=1100
 factors=rng.normal(0,.002,(n,1));ret=factors*np.linspace(.6,1.4,10)+rng.normal(0,.0003,(n,10))
 p=np.exp(np.cumsum(ret,axis=0))*np.arange(1,11)*10
 times=1700002800+np.arange(n)*3600
 ref=dict(market_times=times,close=p,volume=np.ones_like(p))
 basis=np.zeros_like(p)
 for i in range(1,n):basis[i]=.9*basis[i-1]+rng.normal(0,.001,10)
 pas=p*np.exp(basis);d=dict(times=times*1000,close=pas,age_ms=np.zeros_like(p))
 return ref,d

class ResearchTests(unittest.TestCase):
 def test_all45_not_symbol_whitelist(self):
  r,d=fixture();f,a=estimates(r['market_times'][1000],r,d,np.zeros(10),'ema_basis_blend')
  self.assertEqual(len(a),45);self.assertEqual(len(PAIRS),45)
  self.assertTrue(any(x['pair']==['EWY','EWJ'] for x in f))
  self.assertTrue(any(x['pair']==['BRENT','GOLD'] for x in f))
 def test_poison_future_invariance(self):
  r,d=fixture();t=r['market_times'][1000]
  for arm in ['basis_reversion','relative_trend','ema_basis_blend']:
   before=estimates(t,r,d,np.zeros(10),arm);rr=copy.deepcopy(r);dd=copy.deepcopy(d)
   rr['close'][1001:]*=1000;rr['volume'][1001:]=0;dd['close'][1001:]*=.001;dd['age_ms'][1001:]=1e20
   self.assertEqual(before,estimates(t,rr,dd,np.zeros(10),arm))
 def test_missing_interval_not_bridged(self):
  r,d=fixture();d['age_ms'][990]=1e20
  f,a=estimates(r['market_times'][1000],r,d,np.zeros(10),'basis_reversion')
  self.assertTrue(f)
  self.assertTrue(all(x['basis']['n']==94 for x in a if x['eligible']))
  d['age_ms'][1000]=1e20
  f,a=estimates(r['market_times'][1000],r,d,np.zeros(10),'basis_reversion')
  self.assertEqual(f,[])
 def test_constant_basis_no_fake_alpha(self):
  v,a=basis_projection(np.ones(90)*.1);self.assertEqual(v,0)
 def test_short_basis_no_fake_alpha(self):
  v,a=basis_projection(np.arange(30)*.01);self.assertEqual(v,0)
 def test_price_scale_invariant(self):
  r,d=fixture();t=r['market_times'][1000];f,a=estimates(t,r,d,np.zeros(10),'basis_reversion')
  r['close'][:,3]*=100;d['close'][:,3]*=100
  ff,aa=estimates(t,r,d,np.zeros(10),'basis_reversion')
  np.testing.assert_allclose([x['mu'] for x in f],[x['mu'] for x in ff],atol=1e-12)
 def test_funding_receipt_haircut_and_direction(self):
  r,d=fixture();d['close']=r['close'].copy();carry=np.linspace(-.1,.1,10)
  f,a=estimates(r['market_times'][1000],r,d,carry,'basis_reversion')
  self.assertEqual(next(x for x in f if x['ids']==[0,9])['economic_action'],1)
  reverse,_=estimates(r['market_times'][1000],r,d,-carry,'basis_reversion')
  self.assertEqual(next(x for x in reverse if x['ids']==[0,9])['economic_action'],-1)
  self.assertTrue(all(x['source_prices']==x['terminal_prices'] for x in f))
 def test_beta_risk_not_price_ratio(self):
  r,d=fixture();f,a=estimates(r['market_times'][1000],r,d,np.zeros(10),'basis_reversion')
  self.assertTrue(all(.1<=x['source_beta']<=10 for x in f))
  self.assertTrue(all(x['source_vol']>0 for x in f))

if __name__=='__main__':unittest.main()
