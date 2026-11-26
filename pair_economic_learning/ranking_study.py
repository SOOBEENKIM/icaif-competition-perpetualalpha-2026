import gzip,json,sys
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from pair_value_learning.study import ROOT,INPUT,read,save,sha
from .meta_study import verify as verify_parent,OUT as META
from .study import OUT as FEATURES
from .rank_reference import run as replay
from .ranker import build as build_models
OUT=ROOT/'reports/pair_economic_learning_20261023/wave8'
SOURCE=ROOT/'reports/pair_value_learning_20261023/scaled_coherence/coherent_scaled_72.npz'
ARMS=['ranked_pairs_72','ranked_core_72']


def freeze():
    verify_parent()
    if (OUT/'protocol.json').exists():raise FileExistsError('Already frozen')
    save(OUT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=sha(INPUT),arms=ARMS,primary='ranked_pairs_72',
        hypothesis='Scalar forecasts and expectedcash rank can select wrong oversizedalpha. Learn cross-pair opportunity ordering at eachsource, while retaining the priorcost/limit/physicalriskadmission and untouchedsourceforecasts.',
        label='Originalparentproposedside,sourceknowncapacity<=6000,normalizedcosted72hmarkout*capacity. Auxiliarynonjointfilllabelsare0, so they omit one-legrepairloss; fullreplay—notthese labels—must establish execution performance.',
        model='XGBoostrank:pairwise,depth2,64rounds,eta.04,lambda5,seedfixed.19features=prior17+logcapacity+side,noassetnames. Same90dpast,6horigins,strictmaturity<fit,30ddecay;groupweights atsourcelevel;no testearlystopping.',
        controls='Rankonlyalreadyadmissiblepositivecostedproposals;neverinterpret rankscoreas expectedreturn. Primarypaironly,secondaryoriginaldirectionalcore. Samecaps/risk/costs/renewal. No scoresmeans exactoldpathparity.',
        evaluation='All6reference/all6PASconditions. Reuseddevelopment;modelselectionoverfittingnotremovedbycausaltraining. Nolivechanges.',orders_sent=0,live_changed=False,independent_test=False,history_reused=True))
    paths=[Path(__file__),*[Path(__file__).with_name(n) for n in ['ranker.py','ranked_entry.py','ranked_replay.py','ranked_pas_replay.py','rank_reference.py','test_ranking.py']],SOURCE,FEATURES/'features.npz',META/'labels_predictions.npz',OUT/'protocol.json']
    save(OUT/'freeze.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})


def verify():
    verify_parent()
    for p,h in read(OUT/'freeze.json').items():assert sha(ROOT/p)==h,p


def build():
    verify();d=dict(np.load(INPUT));f=dict(np.load(FEATURES/'features.npz'));labels=dict(np.load(META/'labels_predictions.npz'))
    def progress(a):
        if a['source']//86400%20==0:print('FIT ranker',a,flush=True)
    scores,audit,models=build_models(d,f,labels['net'],labels['both'],1785542400,progress)
    s=dict(np.load(SOURCE));ii=np.searchsorted(f['times'],s['times']);assert np.array_equal(f['times'][ii],s['times']);s['ranking_score']=scores[ii]
    for arm in ARMS:np.savez_compressed(OUT/f'{arm}.npz',**s)
    save(OUT/'fit_audit.json',audit);folder=OUT/'models';folder.mkdir(exist_ok=True)
    for day,model in models:model.save_model(folder/f'{day}.json')


def parity():
    verify();replay(OUT,SOURCE,'parity','continuous','base',include_base=False)
    paths=[OUT/'reference/parity_continuous_base.json.gz',ROOT/'reports/pair_economic_learning_20261023/wave3/reference/unbounded_pairs_72_continuous_base.json.gz']
    aa,bb=[json.loads(gzip.open(p,'rt').read()) for p in paths]
    assert aa==bb
    (OUT/'rows/parity_continuous_base.json').unlink()
    save(OUT/'parity.json',dict(exact_full_path_match_without_scores=True));print('PASS rankadapter oldpath parity',flush=True)


def job(arm,phase,scenario):return replay(OUT,OUT/f'{arm}.npz',arm,phase,scenario,include_base=arm=='ranked_core_72')


def run():
    verify();assert read(OUT/'parity.json')['exact_full_path_match_without_scores']
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,a,p,s) for a in ARMS for p in ['validation','comparison','continuous'] for s in ['base','stress']]):
            r=f.result();print(json.dumps({k:r[k] for k in ['arm','phase','scenario','net_pnl','Q','completed_pairs','risk_breach_bars']}),flush=True)
    verify()


if __name__=='__main__':globals()[sys.argv[1]]()
