"""Build a small, offline reproduction bundle without frozen-source edits."""
import copy
import gzip
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .reproduce import digest, save, sha
from experiments.adaptive_mean_pairs_20261017.reference_research_v2 import inputs
from experiments.conditional_completion_pairs_20261017.research import verify
from experiments.complementary_alpha_20261017.research import protected

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
DEST=ROOT/'deliverables/sfinx_pairs_candidate_20261018_final'


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    if DEST.exists():raise FileExistsError('Do not overwrite a release bundle')
    gate=ROOT/'experiments/decision_gate_pairs_20261017/artifacts'
    adapter=ROOT/'experiments/decision_gate_publication_20261018/artifacts'
    seal=json.loads((adapter/'conclusion_frozen.json').read_text());verify(seal['sources'])
    configs={v:json.loads((ROOT/f'experiments/hourly_refresh_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']}
    reference,_,trading=inputs(configs['reference'])
    raw_paths={'par':ROOT/'experiments/par_direct_forecast_20261017/artifacts/inputs.npz',
               'par_extended':ROOT/'experiments/historical_par_extension_20261017/artifacts_v3/inputs.npz'}
    data={'reference':reference,**{v:load(p) for v,p in raw_paths.items()}}
    for name in ['experiments.decision_gate_pairs_20261017.portfolio','experiments.decision_gate_pairs_20261017.model',
                 'experiments.rolling_feature_pairs_20261017.model','experiments.adaptive_residual_pairs_20261017.model',
                 'experiments.decision_gate_publication_20261018.signals','experiments.decision_gate_publication_20261018.portfolio',
                 'experiments.shared_execution_pairs_20261017.portfolio','experiments.historical_par_extension_20261017.portfolio_v2',
                 'experiments.lifecycle_stability_pairs_20261018.portfolio']:
        importlib.import_module(name)
    source_paths=set()
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if not path.is_relative_to(ROOT):continue
        relative=path.relative_to(ROOT)
        if relative.parts[0] in ['experiments','replay_lab','strategy','scripts'] and path.suffix=='.py' and path.is_file():
            source_paths.add(path)
    DEST.mkdir(parents=True)
    for folder in ['source','data','models']: (DEST/folder).mkdir()
    provenance={}
    for path in sorted(source_paths):
        relative=path.relative_to(ROOT)
        # Builder/runner are the release interface, not runtime library inputs.
        if relative.parts[:2]==('experiments','candidate_release_20261018'):continue
        output=DEST/'source'/relative;output.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,output)
        provenance[str(output.relative_to(DEST))]=dict(source=str(relative),sha256=sha(path))
    shutil.copyfile(HERE/'reproduce.py',DEST/'reproduce.py')
    for venue,values in data.items():np.savez_compressed(DEST/'data'/f'{venue}.npz',**values)
    refs={'reference':ROOT/'experiments/risk_preserving_20261016/artifacts/inputs.npz',
          'par_reference':ROOT/'practice_live/updates/20261017T120009814481Z/inputs.npz'}
    for name,path in refs.items():
        values=load(path)
        np.savez_compressed(DEST/'data'/f'{name}_training.npz',**{k:values[k] for k in ['market_times','close','volume','trades']})
        provenance[f'data/{name}_training.npz']=dict(source=str(path.relative_to(ROOT)),sha256=sha(path),
            transform='Copy only market_times, close, volume, trades; no outcome filtering')
    for venue in data:shutil.copyfile(gate/f'{venue}_gate_state.npz',DEST/'models'/f'{venue}_gate_state.npz')
    for venue in ['reference','par']:
        shutil.copyfile(ROOT/f'experiments/persistent_intent_pairs_20261017/artifacts/{venue}_immediate_state.npz',
                        DEST/'models'/f'{venue}_legacy_state.npz')
    expert=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/model_protocol.json').read_text())
    for pair in expert['pairs']:
        filename='_'.join(pair)+'_24.json';path=ROOT/expert['model_directory']/filename
        shutil.copyfile(path,DEST/'models'/filename)
        provenance['models/'+filename]=dict(source=str(path.relative_to(ROOT)),sha256=sha(path))
    specification=dict(candidate='first_dense_hourlyhedge_decisiongate_24',venues=list(data),pairs=expert['pairs'],
        trading=trading,configs=configs,gate_spec=json.loads((gate.parent/'protocol.json').read_text()),
        archive_configs={v:json.loads((ROOT/f'experiments/shared_execution_pairs_20261017/{v}_protocol.json').read_text()) for v in ['reference','par']},
        baseline_target_rule='Replay original frozen first/second target arrays with the documented second risk-budget multiplier. This command refits paired models, not the original Chronos-based baseline from scratch.',
        limits=['All historical windows have been reused for research; not independent validation.',
                'Reference execution uses delayed candle-volume constraints and fee/impact assumptions, not historical par L2.',
                'Par candle simulations are separate and do not demonstrate additional completed profitable pairs.',
                'The separately registered received-book future study is pending and is not a submission prerequisite.',
                'This bundle has no live-order or deployment command; final jury format requirements remain external.'])
    save(DEST/'config.json',specification)
    result=json.loads((gate/'results.json').read_text())
    rows=[r for r in result['rows'] if r['arm'] in ['first_strategy','second_strategy',specification['candidate']]]
    expected=[]
    windows=[*result['windows'],['reference','continuous']]
    for venue,phase in windows:
        for arm in ['first_strategy','second_strategy',specification['candidate']]:
            for scenario in ['base','stress']:
                identity=f'{venue}-{phase}-{arm}-{scenario}'
                if phase=='continuous':path=ROOT/'experiments/lifecycle_stability_pairs_20261018/artifacts/runs'/(identity+'.json.gz')
                elif arm==specification['candidate']:path=gate/'runs'/(identity+'.json.gz')
                elif venue=='par_extended':path=ROOT/'experiments/historical_par_extension_20261017/artifacts_v3/runs'/f'{phase}-{arm}-{scenario}.json.gz'
                else:path=ROOT/'experiments/shared_execution_pairs_20261017/artifacts/runs'/(identity+'.json.gz')
                with gzip.open(path,'rt') as handle:run=json.load(handle)
                if phase!='continuous':
                    row=next(r for r in rows if (r['venue'],r['phase'],r['arm'],r['scenario'])==(venue,phase,arm,scenario))
                    assert run['metrics']==row['metrics'],identity
                expected.append(dict(id=identity,venue=venue,phase=phase,arm=arm,scenario=scenario,
                    engine=('lifecycle_stability_pairs_20261018.portfolio' if phase=='continuous' else
                            'decision_gate_pairs_20261017.portfolio' if arm==specification['candidate'] else
                            'historical_par_extension_20261017.portfolio_v2' if venue=='par_extended' else
                            'shared_execution_pairs_20261017.portfolio'),
                    start=run['start'],end=run['end'],canonical_run_sha256=digest(run),metrics=run['metrics'],
                    archived_source=str(path.relative_to(ROOT)),archived_source_sha256=sha(path)))
    assert len(expected)==42
    save(DEST/'expected.json',dict(runs=expected,not_independent_outcomes=True))
    save(DEST/'provenance.json',dict(files=provenance,input_freeze=str((adapter/'conclusion_frozen.json').relative_to(ROOT)),
        input_freeze_sha256=sha(adapter/'conclusion_frozen.json'),reference_construction='Same inputs() helper and fixed budget multiplier as the archived study.',
        par_inputs={v:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for v,p in raw_paths.items()},protected=protected()))
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','xgboost-cpu','torch','pandas','tqdm',
        'charset-normalizer','typing-extensions','python-dateutil','pytz','six']}
    (DEST/'requirements.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\n'+
        '\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n')
    (DEST/'README.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX 개선 후보 재현</title>
<h1>SFINX 페어 개선 후보 오프라인 재현</h1><p>Python 3.11.17에서 검증한다. requirements.txt의 버전을 설치한 환경에서 실행한다.</p>
<pre>python reproduce.py --output results</pre><p>기본 명령은 두 페어의 직접 예측과 진입 선별 모델을 과거 입력으로 다시 학습하고, 저장된 예측의 모든 필드 및 42개 계좌의 전체 기록과 비교한다. 각 원래 실행 버전의 기록을 정확히 재현한 뒤, 동일한 후보 실행 엔진에서도 실제 체결·계좌 평가·수수료·펀딩·손익 경로가 같은지 별도로 검사한다. 과거 실행 버전 사이에 달랐던 미체결 주문 의도와 판단 로그는 차이 여부를 표시한다. 결과 폴더는 새 이름이어야 한다. 빠른 계좌 재생만 필요하면 --cached-models를 추가한다.</p>
<p>기존 첫째·둘째 전략의 당시 목표 비중과 고정된 위험 규모를 비교 입력으로 사용한다. 이 명령이 Chronos 기준 모델 전체를 처음부터 재훈련하는 것은 아니다. 입력·소스·원래 결과의 출처와 해시는 manifest.json, provenance.json, expected.json에 기록돼 있다.</p>
<p>reference는 기준시장 과거 자료와 모의 비용·거래량 제약을 사용한다. par와 par_extended는 별도 대회 시장 캔들 자료다. 재사용 과거 구간을 독립 검증으로 표시하지 않는다. 종료 시 미청산 평가손익을 포함하며 원장·잔량은 결과에 남는다.</p>
<p>실제 수신 호가 운용 어댑터의 소스도 포함했다. 미래 비교 계획은 원래 작업공간에 별도로 고정돼 있으며 이 묶음은 예약 원자료를 포함하거나 읽지 않는다. 실거래·서명·배포 명령은 제공하지 않는다. 공식 최종 제출 형식에 대한 주최 측 안내를 대체하지 않는다.</p></html>''')
    manifest=dict(created_at_unix=time.time(),python=sys.version,versions=versions,
        files={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()},
        expected_accounts=42,live_orders=False,reserved_raw_included=False)
    save(DEST/'manifest.json',manifest);verify(seal['sources'])
    print('BUILT',DEST,'files',len(manifest['files']),'bytes',sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),flush=True)


if __name__=='__main__':main()
