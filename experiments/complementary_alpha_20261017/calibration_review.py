"""Reject unidentifiable risk calibration; preserve initial experiment evidence.

This correction was specified after the first comparison exposed an empty July
path. It changes no signal parameter. An unavailable expert receives cash, not
a fictitious risk estimate or a performance-selected position size.
"""
import gzip
import json
import time
import unittest

import numpy as np

from .research import ROOT,HERE,OUT,sha,save,protected,BlendController,path_stats,complement,selection_checks
from experiments.cost_horizon_20261016.research import data,ts
from experiments.performance_attribution_20261016.engine import simulate
from experiments.causal_repair_20261017.research import attribution
from replay_lab.ema_rebuild.engine import Controller


def choose_scale(probes):
    # A zero-exposure trajectory cannot identify a nonzero strategy risk budget.
    identifiable=[p for p in probes if p['mean_gross']>1e-12 and p['mean_expected_vol']>1e-12]
    admissible=[p['scale'] for p in identifiable if p['accepted']]
    return max(admissible) if admissible else None


class CalibrationTests(unittest.TestCase):
    def test_empty_path_cannot_choose_largest_position(self):
        probes=[dict(scale=s,mean_gross=0.,mean_expected_vol=0.,accepted=True) for s in [.01,.2]]
        self.assertIsNone(choose_scale(probes))
    def test_measurable_path_selects_only_admissible_risk(self):
        probes=[dict(scale=.01,mean_gross=.01,mean_expected_vol=.0001,accepted=True),
                dict(scale=.2,mean_gross=.2,mean_expected_vol=.002,accepted=False)]
        self.assertEqual(choose_scale(probes),.01)
    def test_no_performance_information_affects_risk_size(self):
        p=[dict(scale=.025,mean_gross=.02,mean_expected_vol=.0002,accepted=True,roi=-1)]
        first=choose_scale(p);p[0]['roi']=999
        self.assertEqual(choose_scale(p),first)


def main():
    dest=OUT/'reviewed';dest.mkdir(exist_ok=True)
    if (dest/'frozen.json').exists():raise FileExistsError('Correction already evaluated')
    initial=json.loads((OUT/'results.json').read_text());frozen=json.loads((OUT/'frozen.json').read_text())
    for path,h in frozen['sources'].items():assert sha(ROOT/path)==h,path
    protected()
    cfg=json.loads((HERE/'protocol.json').read_text());inp,a,p,m=data()
    resolved={name:choose_scale(c['probes']) for name,c in initial['calibration'].items()}
    unavailable=[n for n,v in resolved.items() if v is None]
    save(dest/'frozen.json',dict(at_unix=time.time(),source_sha256=sha(__file__),initial_result_sha256=sha(OUT/'results.json'),
         resolved_scales=resolved,unavailable_experts=unavailable,
         correction='Zero-activity risk calibration is unavailable; missing expert allocation stays in cash.',
         post_comparison_correction_disclosed=True,no_signal_tuning=True,historical_independence=False))
    with np.load(OUT/'targets.npz') as z:weights={k:z[k].copy() for k in z.files if k!='times'}
    second=weights['second_strategy'];affected=[]
    for name in unavailable:
        weights[name]=np.zeros_like(second);weights['blend_'+name]=.75*second;affected.extend([name,'blend_'+name])
    if unavailable:
        weights['blend_equal_experts']=.75*second+.25*np.mean([weights[k] for k in cfg['standalone']],axis=0)
        affected.append('blend_equal_experts')
    np.savez_compressed(dest/'targets.npz',times=m.times,**weights)
    (dest/'runs').mkdir(exist_ok=True);rows=[r for r in initial['rows'] if r['policy'] not in affected]
    comparisons=[r for r in initial['comparisons'] if r['policy'] not in affected];count=0
    for phase,(start,end) in cfg['periods'].items():
        for scenario in ('base','stress'):
            def old(name):
                with gzip.open(OUT/'runs'/f'{phase}-{name}-{scenario}.json.gz','rt') as f:return json.load(f)
            baseline=old('second_strategy');cash=old('second_75pct_cash')
            for name in affected:
                c=BlendController(m.times,a['retention'],second,weights[name]) if name.startswith('blend_') else Controller()
                r=simulate(m,weights[name],a['sigma'],p,m.exact(ts(start)),m.exact(ts(end)),scenario,c)
                cost=(p['execution']['fee_bps']+p['scenarios'][scenario]['impact_bps'])/10000
                terminal=sum(abs(np.array(r['metrics']['final_qty']))*m.close[m.exact(ts(end))])*cost
                r['metrics'].update(estimated_terminal_liquidation_cost=float(terminal),
                    roi_after_estimated_liquidation=r['metrics']['roi']-terminal/p['initial'],
                    one_leg_only_hours=0 if name in unavailable else None)
                r.update(phase=phase,policy=name,per_asset=attribution(r,m),risk_path=path_stats(r),
                         unavailable_expert_stays_in_cash=name in unavailable)
                if name.startswith('blend_') and name[len('blend_'):] in unavailable:
                    assert r['fills']==cash['fills']
                    np.testing.assert_array_equal([s['equity'] for s in r['snapshots']],[s['equity'] for s in cash['snapshots']])
                with gzip.open(dest/'runs'/f'{phase}-{name}-{scenario}.json.gz','wt') as f:json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
                rows.append({k:r[k] for k in ('phase','policy','scenario','metrics','gates','risk_path','per_asset')})
                comparisons.append(dict(phase=phase,scenario=scenario,policy=name,checks=selection_checks(r,baseline),
                    complement=complement(r,baseline),return_above_75pct_cash=r['metrics']['roi']-cash['metrics']['roi']))
                count+=1;print(phase,name,scenario,'ROI',r['metrics']['roi']*100,flush=True)
    assert sha(__file__)==json.loads((dest/'frozen.json').read_text())['source_sha256'];protected()
    save(dest/'results.json',dict(rows=rows,comparisons=comparisons,unavailable_experts=unavailable,
        resolved_scales=resolved,corrected_replays=count,original_economic_runs=initial['economic_runs'],
        original_calibration_runs=initial['calibration_runs'],baseline_reproductions=initial['baseline_reproductions'],
        historical_independence=False,live_replacement_approved=False,orders_sent=0,
        no_signal_parameters_changed=True,empty_expert_blend_equals_cash_control_exactly=True))


if __name__=='__main__':main()
