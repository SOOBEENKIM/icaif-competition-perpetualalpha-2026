"""Verify completed outputs; seal sources without the runner's mutable stdout."""
import json
import time
from pathlib import Path
from html.parser import HTMLParser
from .research import ROOT,HERE,OUT,save,sha,verify,protected


def main():
    path=OUT/'conclusion_frozen.json'
    manifest=json.loads(path.read_text())['sources']
    changed=[rel for rel,digest in manifest.items() if sha(ROOT/rel)!=digest]
    # conclude.py sealed this redirected log before its final print. The
    # research code/data and all financial artifacts must still match exactly.
    mutable=str((HERE/'conclude_run.log').relative_to(ROOT))
    assert changed==[mutable],changed
    sources={rel:digest for rel,digest in manifest.items() if rel!=mutable}
    verify(sources)
    conclusion=json.loads((OUT/'conclusion.json').read_text())
    results=json.loads((OUT/'results.json').read_text())
    assert len(results['exact_controls'])==26
    assert len([r for r in results['rows'] if r['arm']==conclusion['new_candidate']])==14
    assert conclusion['tests_passed']==6 and conclusion['post_hoc_exact_restoration_replays']==2
    assert not conclusion['selection']['promote_graph']
    assert conclusion['selection']['selected']==conclusion['retained_improved_candidate']
    assert conclusion['all_reference_conditions_above_originals']
    assert conclusion['old_book_full_trade_pnl'] is None and conclusion['old_book_q'] is None
    report=ROOT/'reports/formation_graph_pairs_20261018/report.html'
    parser=HTMLParser();parser.feed(report.read_text());parser.close()
    assert report.read_text().endswith('</html>')
    prospective=json.loads((ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json').read_text())['sources']
    verify(prospective)
    verification=dict(at_unix=time.time(),passed=True,financial_artifacts_unchanged=True,
       exclusion=mutable,reason='The final console print occurs after its redirected log was hashed; exclude only mutable runner stdout, keep the original seal for audit.',
       prior_seal_preserved=True,source_mismatches_other_than_stdout=0,
       prospective_sources=len(prospective),protected=protected(),orders_sent=0)
    save(OUT/'verification.json',verification)
    for file in [Path(__file__),path,OUT/'verification.json']:
        sources[str(file.relative_to(ROOT))]=sha(file)
    verify(sources)
    save(OUT/'conclusion_frozen_v2.json',dict(at_unix=time.time(),sources=sources,
        excluded_mutable_stdout=[mutable],verification=verification))
    print('VERIFIED',len(sources),'immutable sources;',len(prospective),'prospective sources unchanged')


if __name__=='__main__':main()
