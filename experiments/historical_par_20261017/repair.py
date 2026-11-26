"""Only partial-fill handling changes; no alpha or risk-budget retuning."""
import difflib
import gzip
import json
import time
import numpy as np
from strategy.agent import TAIL
from experiments.complementary_alpha_20261017.research import protected
from .data import HERE,OUT,ROOT,ts,save,sha
from .repair_engine import run


def main():
    dest=OUT/'repair';dest.mkdir(exist_ok=True)
    if (dest/'frozen.json').exists():raise FileExistsError('Repair experiment already evaluated')
    cfg=json.loads((OUT/'continuous/protocol.json').read_text())['cfg']
    p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    cfg['variants']=['pairs_carry','pairs_basis_carry','pairs_ridge_basis_carry','second_plus_basis_carry']
    save(dest/'protocol.json',dict(cfg=cfg,repair='When both legs partially fill, reduce the overfilled leg on future candles to the lesser common fill fraction. Abort if not repaired within two hours. One-leg fills still abort. Original alpha, horizon, fees and risk limits unchanged.',
        model_or_parameter_selection=False,historical_independence=False,prior_results_exposed=True,orders_sent=0))
    diff=''.join(difflib.unified_diff((HERE/'continuous_engine.py').read_text().splitlines(True),(HERE/'repair_engine.py').read_text().splitlines(True)))
    (dest/'engine.diff').write_text(diff)
    sources={str(q.relative_to(ROOT)):sha(q) for q in [HERE/'repair.py',HERE/'repair_engine.py',dest/'protocol.json',OUT/'inputs.npz',OUT/'signals.npz']}
    save(dest/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected()))
    with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(OUT/'signals.npz') as z:s={k:z[k].copy() for k in z.files}
    rows=[];(dest/'runs').mkdir(exist_ok=True)
    for phase,start,end in [('full',cfg['start_utc'],cfg['end_utc']),('later',cfg['split_utc'],cfg['end_utc'])]:
        for arm in cfg['variants']:
            for scenario in cfg['scenarios']:
                r=run(d,s,cfg,p,arm,scenario,ts(start),ts(end));r['phase']=phase
                with gzip.open(dest/'runs'/f'{phase}-{arm}-{scenario}.json.gz','wt') as f:
                    json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
                repairs=sum(e['event']=='balanced_after_repair' for e in r['events'])
                rows.append(dict(phase=phase,arm=arm,scenario=scenario,metrics=r['metrics'],successful_hedge_repairs=repairs))
                m=r['metrics'];print(phase,arm,scenario,'PnL',round(m['net_pnl'],4),'pairs',m['completed_pairs'],
                                    'repairs',repairs,'aborts',m['aborted_pairs'],flush=True)
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    save(dest/'results.json',dict(rows=rows,orders_sent=0,historical_independence=False,protected_sources_unchanged=True))


if __name__=='__main__':main()
