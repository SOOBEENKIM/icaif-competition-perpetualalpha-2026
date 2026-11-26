"""Separate heavy computation, fixed model; no signer/client import or refit."""
import hashlib,json,time
import numpy as np
# Research modules must resolve before the deployment inserts its frozen path.
from pas_economic_allocation.model import features,predict
from pas_economic_allocation.anchored import RATIO
from competition_live.runtime import fingerprint,json_save
from universe_repair.core import REVISION,LEGACY,SYMBOLS,calculate,audit


def configured_revision(directory):
    path=directory/'strategy_revision.json'
    revision=json.loads(path.read_text())['revision'] if path.exists() else LEGACY
    if revision not in (LEGACY,REVISION):raise ValueError('Unsupported deployed core revision')
    contract_path=directory/'team_execution_contract.json'
    contract=json.loads(contract_path.read_text()) if contract_path.exists() else {}
    required=contract.get('required_core_revision')
    if required is not None:
        if (required!=REVISION or revision!=required or not path.exists()
                or tuple(contract.get('required_core_symbols',[]))!=SYMBOLS):
            raise ValueError('Required all-market core is missing; legacy fallback is forbidden')
    return revision


def validate(packet,profile,now,expected_revision=None):
    from competition_live.runner import load_publication
    from experiments.published_pair_lifecycle_20261017.signals import validate_envelope
    body=packet['body']
    revision=body.get('core_revision',LEGACY)
    if revision not in (LEGACY,REVISION) or (expected_revision is not None and revision!=expected_revision):
        raise ValueError('Unexpected core revision')
    if fingerprint(body)!=packet['sha256']:raise ValueError('Candidate signal digest mismatch')
    if body['candidate']!='budget_lifecycle' or body['profile_hash']!=fingerprint(profile):raise ValueError('Wrong candidate/profile')
    f=body['frame'];original=body['original_signal'];source=f['source_time']
    if revision==REVISION:
        rows=body.get('universe_audit',[])
        if (f.get('symbols')!=profile['trading']['symbols']
                or [r.get('symbol') for r in rows]!=profile['trading']['symbols']
                or not all(r.get('considered') is True for r in rows)):
            raise ValueError('All-market signal coverage is incomplete')
    old=validate_envelope(original['publication'])
    if (source!=old['source_time'] or f['source_hash']!=old['source_hash']
            or original['profile_hash']!=fingerprint(profile)
            or original['candidate']!=profile['candidate']):raise ValueError('Reference/retained-pair source mismatch')
    if not 0<=now-source<21600 or not source<=body['published_at']<=now:raise ValueError('Unavailable candidate signal')
    if not np.isfinite(f['risk_aversion']) or f['risk_aversion']!=5.:raise ValueError('Frozen risk aversion changed')
    for key in ('anchor_weights','price_forecast_168h','reference_funding_forecast_168h','volume','trades'):
        a=np.asarray(f[key],float)
        if a.shape!=(10,) or not np.isfinite(a).all():raise ValueError('Invalid candidate '+key)
    mask=np.asarray(f['protected']);s=np.asarray(f['sigma'],float)
    if mask.shape!=(10,) or mask.dtype!=bool or s.shape!=(10,10) or not np.isfinite(s).all():raise ValueError('Invalid risk state')
    if not np.allclose(s,s.T) or np.linalg.eigvalsh(s).min() < -1e-12:raise ValueError('Invalid covariance')
    for forecast in original['forecasts']:
        if forecast['forecast_source']!=source or forecast['pair'] not in profile['pairs']:raise ValueError('Pair source mismatch')
        if forecast['available_time']<original['published_at']:raise ValueError('Backdated pair forecast')
    return body


def generate(root,profile,revision=LEGACY):
    from competition_live.producer import read_reference
    data,_,digest=read_reference(root,profile)
    pointer=json.loads((root/'practice_live/latest.json').read_text())
    if hashlib.sha256((root/pointer['run']/'inputs.npz').read_bytes()).hexdigest()!=digest:
        raise ValueError('Reference publication changed during inference setup')
    saved=dict(np.load(root/pointer['run']/'targets.npz',allow_pickle=False))
    if not np.array_equal(saved['times'],data['market_times']):raise ValueError('Forecast/source timestamps differ')
    if revision not in (LEGACY,REVISION):raise ValueError('Unsupported deployed core revision')
    anchors,trace,parts=calculate(data,saved['chronos_forecast'],saved['frozen_regime_probability'],revision)
    old=dict(np.load(root/'reports/strategy_repairs_20261021/final_risk_targets.npz'))
    common=np.intersect1d(data['market_times'],old['times']);a=np.searchsorted(data['market_times'],common);b=np.searchsorted(old['times'],common)
    if revision==LEGACY:np.testing.assert_allclose(anchors[a],old['final_mix_risk'][b],atol=1e-10,rtol=0)
    fit=json.loads((root/'reports/pas_economic_allocation_20261022/fit.json').read_text())
    x,u,carry=features(data);forecast=predict(x,u,fit)*fit['attenuation']
    source=int(data['market_times'][-1])
    original=json.loads((root/'.runtime/pas_competition_live/signal.json').read_text())
    frame=dict(source_time=source,source_hash=digest,symbols=profile['trading']['symbols'],
        anchor_weights=(anchors[-1]*RATIO*2).tolist(),protected=trace['protected'][-1].tolist(),
        price_forecast_168h=forecast[-1].tolist(),reference_funding_forecast_168h=carry[-1].tolist(),
        sigma=parts['sigma'][-1].tolist(),volume=data['volume'][-1].tolist(),trades=data['trades'][-1].tolist(),
        risk_aversion=5.,forecast_venue='reference',execution_venue='pas',reference_overlap_rows=len(common),
        legacy_overlap_asserted=revision==LEGACY)
    body=dict(candidate='budget_lifecycle',profile_hash=fingerprint(profile),published_at=time.time(),
        frame=frame,original_signal=original,model_refit=False,orders_sent=0,
        core_revision=revision,universe_audit=audit(parts,trace,anchors),retained_pair_universe=profile['pairs'])
    packet=dict(body=body,sha256=fingerprint(body));validate(packet,profile,time.time());return packet


def main():
    import argparse,fcntl
    from pathlib import Path
    from competition_live.profile import ROOT,load_profile
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--once',action='store_true');args=p.parse_args()
    out=Path(args.directory);out.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (out/'producer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (out/'STOP').exists():
            try:
                current=int(time.time()//3600)*3600;path=out/'signal.json'
                existing=json.loads(path.read_text()) if path.exists() else None
                revision=configured_revision(out)
                if (existing is None or existing['body']['frame']['source_time']<current
                        or existing['body'].get('core_revision',LEGACY)!=revision):
                    json_save(path,generate(ROOT,load_profile(),revision))
                latest=json.loads(path.read_text())
                json_save(out/'producer_status.json',dict(at=time.time(),state='ready',
                    source=latest['body']['frame']['source_time'],signal_sha256=latest['sha256'],
                    core_revision=latest['body'].get('core_revision',LEGACY),orders_sent=0))
            except Exception as e:
                json_save(out/'producer_status.json',dict(at=time.time(),error=str(e),orders_sent=0))
                if args.once:raise
            if args.once:return
            time.sleep(30)


if __name__=='__main__':main()
