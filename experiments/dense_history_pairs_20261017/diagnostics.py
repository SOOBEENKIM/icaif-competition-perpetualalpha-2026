"""Post-evaluation cost attribution and pre-existing book coverage audit."""
import gzip,json
from pathlib import Path
import numpy as np
from experiments.risk_neutral_pairs_20261017.attribution import episode_attribution
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.par_direct_forecast_20261017.data import ROOT,OUT as INPUT,save,sha
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'


def main():
    results=json.loads((OUT/'results.json').read_text());frozen=json.loads((OUT/'frozen.json').read_text())
    for rel,digest in frozen['sources'].items():assert sha(ROOT/rel)==digest,rel
    cfgs={v:json.loads((HERE/f'{v}_protocol.json').read_text()) for v in ['reference','par']}
    rd,_,_=inputs(cfgs['reference'])
    with np.load(INPUT/'inputs.npz') as z:pd={k:z[k].copy() for k in z.files}
    data={'reference':rd,'par':pd};attributions=[]
    for row in results['rows']:
        if '_dense_' not in row['arm']:continue
        key='-'.join(row[k] for k in ['venue','phase','arm','scenario'])
        with gzip.open(OUT/'runs'/f'{key}.json.gz','rt') as f:run=json.load(f)
        audit=episode_attribution(run,data[row['venue']],cfgs[row['venue']]['scenarios'][row['scenario']])
        intents=[e for e in run['events'] if e['event']=='pair_intent']
        for episode,intent in zip(audit['episodes'],intents):
            episode.update(expected_net_cash_at_intent=intent['expected_net_cash'],source_price_alpha=intent['source_price_alpha'],
                forecast_source=intent['forecast_source'],forecast_maturity=intent['forecast_source']+intent['horizon']*3600)
        aggregate={k:sum(e[k] for e in audit['episodes']) for k in ['price_move','execution_cost','fees','funding','net_pnl','fills']}
        assert abs(aggregate['net_pnl']-run['metrics']['pair_owned_pnl'])<1e-6
        attributions.append(dict(**{k:row[k] for k in ['venue','phase','arm','scenario']},**audit,aggregate=aggregate))
    save(OUT/'attribution.json',dict(rows=attributions,not_used_for_selection=True,orders_sent=0))
    # Inspect only metadata for the five pre-existing, allowlisted tapes. These
    # intervals do not cover any of this family's one-hour decision windows.
    tape_protocol=ROOT/'experiments/basis_carry_20261017/protocol.json'
    old_results=ROOT/'experiments/basis_carry_20261017/artifacts/quote_probe/results.json'
    allowlist=json.loads(tape_protocol.read_text())['tapes']
    coverage=json.loads(old_results.read_text())['coverage'];assert {r['path'] for r in coverage}==set(allowlist)
    overlaps=[];cadence=6*3600*1000;lifetime=3600*1000
    for r in coverage:
        intervals=[];lo,hi=r['start_ms'],r['end_ms']
        for source in range(lo//cadence*cadence,hi+1,cadence):
            left,right=max(lo,source),min(hi,source+lifetime)
            if right>left:intervals.append(dict(start=left,end=right))
        overlaps.append(dict(path=r['path'],start_ms=lo,end_ms=hi,entry_window_overlaps=intervals,
            overlap_seconds=sum(i['end']-i['start'] for i in intervals)/1000))
    assert sum(x['overlap_seconds'] for x in overlaps)==0
    save(OUT/'book_coverage.json',dict(tapes=overlaps,total_entry_window_overlap_seconds=0,
        basis='Previously verified exact tape bounds; no new or reserved raw tape opened.',
        metadata_hashes={str(p.relative_to(ROOT)):sha(p) for p in [tape_protocol,old_results]},
        full_book_execution_or_exit_verified=False,not_account_pnl_or_Q=True,orders_sent=0))
    save(OUT/'diagnostic_status.json',dict(attribution_runs=len(attributions),
        maximum_episode_accounting_error=max(x['accounting_error'] for x in attributions),protected=protected(),orders_sent=0))
    for x in attributions:
        if x['venue']=='reference' and x['phase']=='comparison' and x['arm'] in ['first_dense_ridge_24','first_dense_ridge_72']:
            print(x['arm'],x['scenario'],x['aggregate'],flush=True)


if __name__=='__main__':main()
