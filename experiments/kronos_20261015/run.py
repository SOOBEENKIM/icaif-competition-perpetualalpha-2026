"""Frozen historical comparison. No strategy selection, network or orders."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import platform
import numpy as np
from replay_lab.common_comparison.models import covariance, timestamp
from replay_lab.macro_development.engine import simulate as existing_simulate
from replay_lab.pairs_rl.verify import independent_accounting
from .data import HERE, ROOT, SOURCE, DAY, HOUR, daily_bars, load, make_plan, sha, write
from .engine import simulate
from .policy import PaperAgent


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen artifact changed: ' + str(path))
    else:
        write(path, value)


def forecast_files(kind, plan, bars, start, end):
    directory = HERE/'forecasts'/kind
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['plan_sha256'] != sha(HERE/'plan.json'):
        raise ValueError('Forecast belongs to another plan')
    if kind != 'chronos_daily':
        checks = [('forecast_adapter_sha256', 'forecast.py'), ('data_adapter_sha256', 'data.py'),
                  ('cached_decoder_sha256', 'cached_decoder.py')]
        if manifest['source_sha256'] != sha(SOURCE):
            raise ValueError('Forecast input changed')
    else:
        checks = [('adapter_sha256', 'chronos_control.py')]
    for key, file in checks:
        if manifest[key] != sha(HERE/file):
            raise ValueError('Forecast adapter changed: ' + file)
    result, hashes = {}, {}
    origins = range(start, end-end%DAY+1, DAY)
    horizon = 12 if kind == 'hourly' else 10
    for origin in origins:
        path = directory/f'{origin}.npz'
        with np.load(path, allow_pickle=False) as f:
            values, last = f['forecast'].copy(), f['last_close'].copy()
            if int(f['origin']) != origin:
                raise ValueError('Mismatched origin')
        shape = (10,horizon) if kind == 'chronos_daily' else (10,horizon,4)
        if values.shape != shape or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Invalid forecast')
        i = np.searchsorted(bars['known'], origin)
        if bars['known'][i] != origin or not np.array_equal(last, bars['close'][i]):
            raise ValueError('Forecast reference close mismatched')
        result[origin] = dict(forecast=values, last=last)
        hashes[str(path.relative_to(HERE))] = sha(path)
    return result, hashes


def scoring(forecasts):
    return {t:(f['forecast'][:,:,3] if f['forecast'].ndim == 3 else f['forecast']).mean(1)/f['last']-1
            for t,f in forecasts.items()}


def preservation():
    before = json.loads((HERE/'preserved_before.json').read_text())
    changed = [name for name,digest in before.items() if not (ROOT/name).exists() or sha(ROOT/name) != digest]
    upstream = json.loads((HERE/'upstream.json').read_text())['files']
    modified = [name for name,digest in upstream.items()
                if not (HERE/'vendor/Kronos'/name).exists() or sha(HERE/'vendor/Kronos'/name) != digest]
    return dict(passed=not changed and not modified, protected_files=len(before), changed=changed,
                official_files=len(upstream), official_modified=modified)


def save_run(run, phase, market):
    # The shared ledger includes NumPy scalar booleans. Normalize its record,
    # without changing any calculations, before serializing/auditing.
    run = json.loads(json.dumps(run, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist(), allow_nan=False))
    run['phase'] = phase
    run['audit'] = independent_accounting(run, market)
    if 'selections' in run:
        for item in run['selections']:
            if len(item['sell']) > 1:
                raise ValueError('More than one rank replacement')
            for j in item['sell']:
                if item['time']//DAY - item['entries'][str(j)]//DAY < 5:
                    raise ValueError('Rank replacement before minimum holding period')
    path = HERE/'runs'/phase/f"{run['policy']}-{run['scenario']}.json.gz"
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.partial.gz')
    with gzip.open(tmp,'wt',encoding='utf-8') as f:
        json.dump(run,f,allow_nan=False)
    tmp.replace(path)
    return {k:run[k] for k in ['phase','policy','scenario','start','end','metrics','gates','audit']} | {
        'file':str(path.relative_to(HERE)), 'sha256':sha(path)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibrate-only',action='store_true')
    args=parser.parse_args()
    plan=make_plan(); market,z,protocol=load(); bars=daily_bars(z)
    phases={key:(timestamp(plan[key]['start']),timestamp(plan[key]['end']))
            for key in ['calibration','validation','comparison']}
    start,end=phases['calibration'] if args.calibrate_only else (phases['calibration'][0],phases['comparison'][1])
    forecasts,hashes={},{}
    for kind in ['daily','chronos_daily']:
        forecasts[kind],h=forecast_files(kind,plan,bars,start,end); hashes.update(h)
    scores={name:scoring(forecasts[kind]) for name,kind in [('kronos_paper','daily'),('chronos2_paper','chronos_daily')]}
    sigma=np.full((len(market.times),10,10),np.nan)
    for i in range(market.exact(start),market.exact(end)+1):
        sigma[i]=covariance(market,i,protocol['risk'])[1]
    code={p.name:sha(p) for p in sorted(HERE.glob('*.py')) if p.name != 'report.py'}
    freeze(HERE/'evaluation_implementation.json', dict(code=code,plan_sha256=sha(HERE/'plan.json'),
        implementation_note='KV cache added for CPU feasibility before any return evaluation; official sampler parity verified.',
        cache_parity_sha256=sha(HERE/'model_parity.json'), historical_reuse=True, independent_holdout=False))
    calibrated={}
    for name in scores:
        run=simulate(market,PaperAgent(scores[name],name=name+'_uncalibrated'),protocol,'base',*phases['calibration'],sigma)
        row=save_run(run,'raw_calibration',market)
        std=run['metrics']['daily_vol']
        scale=min(1.,plan['calibration']['daily_std_budget']/std) if std>0 else 1.
        calibrated[name]=dict(scale=scale,development_uncalibrated_metrics=row['metrics'],
                              audit=row['audit'],raw_run=row['file'])
    # Freeze before validation/comparison runs. No later outcomes affect scales.
    calibration=dict(models=calibrated,period=plan['calibration'],
                     inputs={p:h for p,h in hashes.items() if int(p.split('/')[-1].split('.')[0])<=phases['calibration'][1]},
                     implementation_sha256=sha(HERE/'evaluation_implementation.json'))
    freeze(HERE/'calibration.json',calibration)
    print(json.dumps({'calibrated_scales':{k:v['scale'] for k,v in calibrated.items()} }),flush=True)
    if args.calibrate_only:
        return
    freeze(HERE/'comparison_inputs.json',dict(forecasts=hashes,calibration_sha256=sha(HERE/'calibration.json')))
    rows=[]
    weights={}
    for name,folder,key in [('selected_default','tail_calibration','tail_calibrated'),
                            ('selected_original','reversion_blend','crypto_d7_aligned_add50')]:
        with np.load(ROOT/f'replay_lab/output/{folder}/20261014-v1/final_targets.npz',allow_pickle=False) as f:
            weights[name]=f[key].copy()
    baseline_checks=[]
    for phase,(begin,finish) in phases.items():
        for scenario in ['base','stress']:
            for name in ['kronos_paper','chronos2_paper','equal_weight','selected_default','selected_original']:
                if name in weights:
                    run=existing_simulate(market,{'target_weight':weights[name]},name,protocol,scenario,begin,finish,
                                          buffer=.2,sigmas=sigma)
                else:
                    equal=name=='equal_weight'
                    scale=calibrated['kronos_paper' if equal else name]['scale']
                    run=simulate(market,PaperAgent(scores.get(name,{}),scale,name=name,equal_weight=equal),
                                 protocol,scenario,begin,finish,sigma)
                row=save_run(run,phase,market);rows.append(row)
                if phase=='comparison' and name in weights:
                    folder='tail_calibration' if name=='selected_default' else 'reversion_blend'
                    key='tail_calibrated' if name=='selected_default' else 'crypto_d7_aligned_add50'
                    saved=json.loads((ROOT/f'replay_lab/output/{folder}/20261014-v1/test_result.json').read_text())
                    ref=next(r for r in saved['rows'] if r['policy']==key and r['scenario']==scenario)
                    if ref['metrics'] != row['metrics']:
                        raise ValueError('Existing baseline reproduction mismatch: '+name+'/'+scenario)
                    baseline_checks.append(dict(policy=name,scenario=scenario,all_metrics_exact=True))
                print(json.dumps(dict(phase=phase,policy=name,scenario=scenario,roi=row['metrics']['roi'],
                                      daily_vol=row['metrics']['daily_vol'])),flush=True)
    indexed={(r['policy'],r['scenario']):r for r in rows if r['phase']=='comparison'}
    k=indexed['kronos_paper','base']['metrics']; d=indexed['selected_default','base']['metrics']
    preserve=preservation()
    engineering=preserve['passed'] and all(r['audit']['passed'] and all(r['gates'].values()) for r in rows)
    gates=dict(higher_roi_than_selected_default=k['roi']>d['roi'],no_higher_daily_std=k['daily_vol']<=d['daily_vol'],
               positive_base=k['roi']>0,positive_stress=indexed['kronos_paper','stress']['metrics']['roi']>0,
               engineering=engineering)
    result=dict(created=datetime.now(timezone.utc).isoformat(),rows=rows,calibration=calibrated,
                improvement_gates=gates,improvement_passed=all(gates.values()),preservation=preserve,
                baseline_reproduction=baseline_checks,historical_reuse=True,independent_holdout=False,
                parameters_changed_after_comparison=False,live_orders_sent=0,
                official_competition_rank=None,
                notes=['Each phase starts with cash 1M and no positions.',
                       'Daily population STD uses complete UTC daily boundaries, not an established official snapshot time.',
                       'ROI includes terminal marked open positions; no forced terminal close.',
                       'Equal-weight control uses Kronos calibrated gross target and same execution/risk constraints.',
                       'Chronos control uses daily close only, Kronos uses OHLC; comparison includes feature-set differences.',
                       'Reference market replay, not recorded competition order-book fills.'])
    write(HERE/'results.json',result)
    with (HERE/'metrics.csv').open('w',newline='') as f:
        fields=['phase','policy','scenario','roi','daily_vol','max_drawdown','fees','funding','fills','complete_days']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:(r[k] if k in r else r['metrics'][k]) for k in fields} for r in rows)
    write(HERE/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
          packages={name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','huggingface-hub',
                     'safetensors','einops','chronos-forecasting','transformers','scipy']},device='cpu',threads_per_inference=4))
    print(json.dumps({'improvement':gates,'preservation':preserve}),flush=True)


if __name__=='__main__':
    main()
