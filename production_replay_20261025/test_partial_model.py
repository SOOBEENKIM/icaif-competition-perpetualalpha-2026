import numpy as np
from .partial_model import filtered,python_filter,null_nll,fit,validate,Spec


def synthetic(n=2200,seed=7):
    rng=np.random.default_rng(seed);x=np.cumsum(rng.normal(0,.006,n));r=np.cumsum(rng.normal(0,.0002,n));m=np.zeros(n)
    for i in range(1,n):m[i]=.91*m[i-1]+rng.normal(0,.004)
    return 1.6*x+r+m,x,np.ones(n,bool)


def test_c_kernel_matches_matrix_joseph_and_missing_hours():
    y,x,good=synthetic(600);good[:3]=False;good[15:96]=False;good[110::5]=False
    p=[1.6,.91,4e-8,16e-6]
    np.testing.assert_allclose(filtered(y,x,good,p),python_filter(y,x,good,p),rtol=1e-8,atol=1e-9,equal_nan=True)


def test_filter_prefix_cannot_see_later_prices():
    y,x,g=synthetic();p=[1.6,.91,4e-8,16e-6]
    a=filtered(y,x,g,p);y[1000:]+=99;x[1000:]-=17
    np.testing.assert_array_equal(a[:1000],filtered(y,x,g,p)[:1000])


def test_missing_observation_never_used_and_elapsed_time_not_compressed():
    y,x,g=synthetic(100);g[10:20]=False;p=[1.6,.91,4e-8,16e-6]
    a=filtered(y,x,g,p);y[10:20]=np.nan;x[10:20]=np.nan
    np.testing.assert_array_equal(a,filtered(y,x,g,p))
    assert abs(a[19,1]-a[9,1]*.91**10)<1e-12


def test_null_density_counts_gap():
    y=np.array([0.,999.,2.]);x=np.zeros(3);g=np.array([1,0,1],bool)
    q=2.;r=null_nll(y,x,g,1.,q)
    assert abs(r[2]-.5*(np.log(2*np.pi*4)+1.))<1e-12


def test_joint_fit_recovers_synthetic_positive_hedge():
    y,x,g=synthetic();m=fit(y[:1440],x[:1440],g[:1440])
    assert m and abs(m['params'][0]-1.6)<.12
    assert m['nll']<m['null_nll']
    a=validate(y[:1944],x[:1944],g[:1944],m)
    assert a['last_maturity_offset']<1944
    assert a['labels']>=4 and a['calibration']>=0


def test_validation_only_uses_disjoint_fixed_share_return_labels():
    y,x,g=synthetic();m=fit(y[:1440],x[:1440],g[:1440])
    a=validate(y[:1944],x[:1944],g[:1944],m)
    origins=np.array(a['origin_offsets']);assert np.all(np.diff(origins)>=72)
    beta=m['params'][0];truth=(np.expm1(y[origins+72]-y[origins])-beta*np.expm1(x[origins+72]-x[origins]))/(1+beta)
    assert abs(a['zero_mse']-np.mean(truth**2))<1e-14
