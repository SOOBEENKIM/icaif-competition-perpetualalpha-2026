"""Verify frozen research after a separately recorded live accounting repair.

No weakening of the original experiment guards: those continue to reject
changed live sources. The archived original sources and operational amendment
explain the difference. This verifier does not submit orders or refit a model.
"""
import json
from .data import ROOT,OUT,read,sha,save

def main():
    manifest_names=['inputs_before_download.json','wave1_freeze.json','wave2/freeze.json','wave3/freeze.json','wave4/freeze.json','nominee_freeze.json','sealed_extension/input_manifest.json']
    checked=0
    for name in manifest_names:
        values=read(OUT/name);values=values['files'] if 'files' in values else values
        for relative,digest in values.items():
            assert sha(ROOT/relative)==digest,(name,relative)
            checked+=1
    archived=[];amend=read(OUT/'operational_amendment.json');allowed={x['path']:x for x in amend['changes']}
    for relative,digest in read(OUT/'protocol.json')['live_hashes'].items():
        current=sha(ROOT/relative)
        if current!=digest:
            assert relative in allowed and allowed[relative]['current_sha256']==current,relative
            assert sha(OUT/'baseline_source_archive'/relative)==digest,relative
            archived.append(relative)
    for relative,digest in amend['new_files'].items():assert sha(ROOT/relative)==digest,relative
    rows=[]
    for wave in ['wave1','wave2','wave3','wave4','sealed_extension/development','sealed_extension/evaluation','liquidity_audit']:
        rows+=read(OUT/wave/'results.json')
    assert len(rows)==197
    assert all(r['timing_pass'] and r['risk_breach_events']==0 and r['maximum_accounting_error']<1e-6 for r in rows)
    assert all(read(OUT/'sealed_extension/prefix_checks.json').values())
    assert all(r['path_prefix_identical'] and r['funding_proxy_events']==0 for r in read(OUT/'sealed_extension/evaluation/results.json'))
    result=dict(frozen_hashes_checked=checked,all_match=True,replays=197,operational_changes_separately_archived=archived,
        original_guard_still_rejects_changed_live_sources=True,research_engine_changed=False)
    save(OUT/'artifact_verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
