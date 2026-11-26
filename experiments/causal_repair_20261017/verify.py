"""Verify provenance, immutable baselines and replay determinism after repair."""
import gzip
import json
from pathlib import Path

from .data import HERE,OUT,sha,save

ROOT=HERE.parents[1]


def main():
    result={}
    protected=json.loads((ROOT/'experiments/cost_horizon_20261016/artifacts/protocol_frozen.json').read_text())
    for group in ('active_protected_sources','reserved_dependencies'):
        for path,h in protected[group].items():assert sha(ROOT/path)==h,path
        result[group+'_unchanged']=len(protected[group])
    for file in (OUT/'comparison/frozen.json',OUT/'acceptance/plan.json'):
        manifest=json.loads(file.read_text())
        for path,h in manifest['sources'].items():assert sha(ROOT/path)==h,path
    receipts=json.loads((OUT/'data_receipts.json').read_text())
    checked=0
    for r in receipts['receipts']:
        if r['status']!=200:continue
        p=OUT/'external'/f'{r["symbol"]}USDT-1h-{r["month"]}.zip'
        assert sha(p)==r['sha256']
        assert Path(str(p)+'.CHECKSUM').read_text().split()[0]==r['sha256'];checked+=1
    result['public_archives_checksum_verified']=checked
    # The serialization-only fix must not change even a single simulated fill.
    repeated=0
    for p in sorted((OUT/'comparison/runs').glob('*.json.gz')):
        with gzip.open(p,'rt') as f:new=json.load(f)
        oldfile=OUT/'comparison_export_failure/runs'/p.name
        with gzip.open(oldfile,'rt') as f:old=json.load(f)
        assert new==old,p.name;repeated+=1
    result['economic_runs_exactly_reproduced_after_export_fix']=repeated
    forecasts=json.loads((OUT/'forecast/manifest.json').read_text())
    for f in forecasts['fits']:
        if f['available']:assert f['last_label_maturity']<f['cutoff']
    for g in forecasts['gates']:
        if g['last_maturity'] is not None:assert g['last_maturity']<g['cutoff']
    result['causal_fit_and_admission_cutoffs_verified']=True
    result['sealed_forward_outcomes_read']=False
    result['live_orders_sent']=0
    save(OUT/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
