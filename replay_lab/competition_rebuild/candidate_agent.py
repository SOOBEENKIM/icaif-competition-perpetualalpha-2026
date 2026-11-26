"""Research agent: frozen v7 + independent JM/Chronos exposure filters.

No endpoints, keys, signing or order submission. Caller supplies completed bars,
settled funding and actual current virtual-account state to decide().
"""
import json,os
from pathlib import Path
import numpy as np
from .common import sha
from .regimes import regime_features,fit_jump,paths
from .persistence import episode_overlay
from .causal_risk import normalize_risk
from ..portfolio_development.candidate_agent import FundingTrendAgent
from ..common_comparison.models import covariance


def jm_probability(m):
    ids,x,r=regime_features(m);prob=np.full_like(m.close,.5);first=np.searchsorted(ids,1440)
    smooth=np.full(10,.5);alpha=1-np.exp(np.log(.5)/4)
    for begin in range(first,len(ids),56):
        end=min(begin+56,len(ids));aa=max(28,begin-360)
        for j in range(10):
            state=fit_jump(x[aa:begin,j],r[aa:begin,j]);cost=state['cost'].copy()
            for k in range(begin,end):
                xx=(x[k,j]-state['mean'])/state['std'];loss=.5*((state['centres']-xx)**2).sum(1)
                cost=loss+(cost[:,None]+10.*(1-np.eye(2))).min(0);cost-=cost.min()
                smooth[j]=(1-alpha)*smooth[j]+alpha*float(cost.argmin()==state['bull'])
                prob[ids[k]:min(ids[k]+6,len(m.times)),j]=smooth[j]
    return prob


class CausalHybridAgent:
    def __init__(self,bundle,checkpoint,saved=None):
        if bundle['kind']!='jm_chronos_risk_filter_v1':raise ValueError('Unsupported agent')
        self.bundle=bundle;self.checkpoint=Path(checkpoint);self.base=FundingTrendAgent(bundle['v7'],saved)
        self.model=None;self.cached={}
        manifest=json.loads((self.checkpoint/'provenance.json').read_text())
        assert manifest['revision']=='95a9710e2596287d08352589f42634fa5abdf0a7'
        for name,record in manifest['files'].items():assert sha(self.checkpoint/name)==record['sha256']

    def _model(self):
        if self.model is None:
            os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
            import torch
            from chronos import Chronos2Pipeline
            torch.set_num_threads(4)
            self.model=Chronos2Pipeline.from_pretrained(str(self.checkpoint),device_map='cpu',local_files_only=True)
        return self.model

    def chronos_predictions(self,m):
        origins=np.flatnonzero((np.arange(len(m.times))>=1008)&(m.times%21600==0));point=np.zeros_like(m.close);missing=[]
        for i in origins:
            context=np.log(m.close[i-167:i+1].T).astype('float32');key=(int(m.times[i]),context.tobytes())
            if key not in self.cached:missing.append((i,key,context))
        for k in range(0,len(missing),8):
            batch=missing[k:k+8]
            q,_=self._model().predict_quantiles([r[2] for r in batch],prediction_length=25,quantile_levels=[.1,.5,.9],context_length=168,batch_size=80,cross_learning=False)
            for (i,key,_),value in zip(batch,q):
                v=value.detach().cpu().numpy().astype('float64');assert np.isfinite(v).all();self.cached[key]=v[:,-1,1]-v[:,0,1]
        for i in origins:
            context=np.log(m.close[i-167:i+1].T).astype('float32');key=(int(m.times[i]),context.tobytes())
            point[i:i+6]=self.cached[key]
        return point

    def prepare_history(self,m,original,*,funding_source):
        self.base.prepare_history(m,original,funding_source=funding_source);base=self.base.weights.copy()
        probability=jm_probability(m);_,regime_gate=paths(m,probability,base)
        forecast=self.chronos_predictions(m);forecast_gate=episode_overlay(m,base,forecast,'sticky')
        sigmas=np.full((len(m.times),10,10),np.nan)
        for i in range(168,len(m.times)):sigmas[i]=covariance(m,i,self.bundle['v7']['control_bundle']['protocol']['risk'])[1]
        weights=normalize_risk(base,(regime_gate+forecast_gate)/2,sigmas,m.times)
        self.base.weights=weights*self.bundle.get('exposure_multiplier',1.)
        self.weights=self.base.weights;self.market=m;self.probability=probability;self.forecast=forecast

    def decide(self,time,qty,equity,valid_input=True):return self.base.decide(time,qty,equity,valid_input)
    def snapshot(self):return self.base.snapshot()
