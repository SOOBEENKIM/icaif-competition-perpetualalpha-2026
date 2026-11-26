"""Count activity at the endpoints of an existing par training target."""
import json
import time
from pathlib import Path
import numpy as np
from experiments.par_direct_forecast_20261017.model import design,labels
from experiments.par_direct_forecast_20261017.data import ROOT,save,sha,ts
from experiments.dense_history_pairs_20261017.research import load
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'


def main():
    folder=ROOT/'experiments/par_direct_forecast_20261017'
    cfg=json.loads((folder/'protocol.json').read_text());d=load(folder/'artifacts/inputs.npz')
    plan=ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json'
    symbols=json.loads(plan.read_text())['trading_protocol']['symbols']
    paths=[Path(__file__),folder/'model.py',folder/'protocol.json',folder/'artifacts/inputs.npz',plan]
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    save(OUT/'par_label_activity_frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    f=design(d,cfg,symbols);rows=[];n=len(d['times'])
    for horizon in cfg['horizons_hours']:
        target,maturity=labels(d,f,horizon);delta=horizon*12+1
        next_active=np.zeros_like(f['fresh']);terminal_active=np.zeros_like(f['fresh'])
        next_active[:-1]=f['fresh'][1:];terminal_active[:-delta]=f['fresh'][delta:]
        valid=np.isfinite(f['x']).all(-1)&f['fresh']&np.isfinite(target)&(f['beta']>=.1)&(f['beta']<=10)
        for date in cfg['fit_dates']:
            now=ts(date)
            window=(d['times']>=now-cfg['training_window_days']*86400)&(maturity<now)&(d['times']%cfg['fit_sampling_seconds']==0)
            used=valid&window[:,None]
            rows.append(dict(horizon=horizon,fit_time=now,original_training_rows=int(used.sum()),
                 next_bar_both_active=int((used&next_active).sum()),terminal_both_active=int((used&terminal_active).sum()),
                 both_endpoints_active=int((used&next_active&terminal_active).sum()),
                 per_pair_original=used.sum(0).tolist(),per_pair_both_endpoints_active=(used&next_active&terminal_active).sum(0).tolist()))
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    save(OUT/'par_label_activity.json',dict(rows=rows,pairs=cfg['pairs'],orders_sent=0,protected=protected(),
        interpretation='Existing labels are next-bar last-trade close to horizon last-trade close. Source activity is required but endpoint activity is not. Counts diagnose stale execution proxies; both-active candles still do not establish quoted depth or paired fills. Filtering on future activity would change the conditional target and must not be used as an inference-time liquidity guarantee. This audit changes no models or candidate selection.'))
    print('DONE existing par label activity audit',rows,flush=True)


if __name__=='__main__':main()
