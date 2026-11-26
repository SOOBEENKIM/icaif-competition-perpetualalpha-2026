"""Seal, verify, execute and publish a finite offline research comparison.

Run: .venv/bin/python -m replay_lab.common_comparison.experiment
The preserved strategy tree is read-only. No exchange or observer is started.
"""
import gzip
import hashlib
import html
import json
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from .models import load_inputs, features, fit_ridge, retained_models, timestamp
from .engine import Policy, simulate


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'replay_lab/output/retained_strategies'
OUT = ROOT / 'replay_lab/output/common_comparison'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported research artifact type: {type(value).__name__}')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_value) + '\n')


def protected_files():
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text())
    for group, base in [('files', ARCHIVE), ('source_hashes', ROOT/'replay_lab'), ('data_hashes', ROOT)]:
        for name, expected in manifest[group].items():
            if digest(base/name) != expected:
                raise AssertionError(f'Protected {group} mismatch: {name}')
    files = {str(p.relative_to(ROOT)): digest(p) for p in ARCHIVE.rglob('*') if p.is_file()}
    for name in manifest['source_hashes']:
        files['replay_lab/' + name] = digest(ROOT/'replay_lab'/name)
    files.update(manifest['data_hashes'])
    return files, manifest


def select_family(rows, kind):
    candidates = [r for r in rows if r['kind'] == kind and r['scenario'] == 'base']
    eligible = [r for r in candidates if all(r['gates'].values()) and r['metrics']['roi'] > 0]
    pool = eligible or candidates
    best_return = max(r['metrics']['roi'] for r in pool)
    tied = [r for r in pool if r['metrics']['roi'] >= best_return - .0001]
    chosen = min(tied, key=lambda r: (r['metrics']['daily_vol'], r['parameter'] if kind == 'momentum' else -r['parameter']))
    return dict(policy=chosen['policy'], eligible=bool(eligible), parameter=chosen['parameter'],
                metrics=chosen['metrics'], gates=chosen['gates'],
                role='family_winner' if eligible else 'rejected_diagnostic_reference')


def summary(run, file):
    return {k: run[k] for k in ('policy', 'kind', 'parameter', 'scenario', 'start', 'end', 'metrics', 'gates')} | {'file': file}


def audit_run(run):
    fills = run['fills']; m = run['metrics']
    if abs(sum(f['fee'] for f in fills) - m['fees']) > 1e-6:
        raise AssertionError('Fee total differs from fills')
    signed = sum(e['signed_amount'] for e in run['funding_events'])
    adverse = sum(e['adverse_amount'] for e in run['funding_events'])
    if abs(signed + adverse - m['funding']) > 1e-6:
        raise AssertionError('Funding total differs from settlements')
    for d in run['decisions']:
        if not (d['source_available'] <= d['decision_time'] < d['execution_time']):
            raise AssertionError('Information-time violation')
    for o in run['orders']:
        if o['filled_qty'] > min(o['requested_qty'], o['capacity']) + 1e-10:
            raise AssertionError('Exceeded IOC capacity')
    positions = np.zeros(len(m['final_qty']))
    for f in fills:
        positions[f['asset']] += f['signed_qty']
    np.testing.assert_allclose(positions, m['final_qty'], rtol=0, atol=1e-8)
    for s in run['snapshots']:
        if abs(s['equity'] - (1e6 + s['realized'] - s['fees'] + s['funding'] + s['unrealized'])) > 1e-6:
            raise AssertionError('Snapshot components do not reconcile')
    return dict(fills=len(fills), decisions=len(run['decisions']), orders=len(run['orders']),
                maximum_ledger_error=m['maximum_ledger_error'], accounting_pass=True,
                information_time_pass=True, fill_capacity_pass=True)


def write_html(dest, report, curves):
    labels = {
        'common-matched-control': '기존 규칙 · 공통 조건',
        'common-matched-lifecycle': '위험보정 수정판 · 공통 조건',
        'common-adaptive-horizons-lifecycle-d14-s1': '원래예산 수정판 · 공통 조건',
        'cash': '현금', 'basket-3pct-each': '종목별 3% 목표 · 공통 한도 적용'
    }
    def label(key): return labels.get(key, key)
    def pct(v): return f'{100*v:+.4f}%' if v is not None else '—'
    def table(rows):
        cells = []
        for r in rows:
            m=r['metrics']; status='통과' if all(r['gates'].values()) else '탈락: '+','.join(k for k,v in r['gates'].items() if not v)
            cells.append('<tr><td>'+html.escape(label(r['policy']))+'</td><td>'+html.escape(r['scenario'])+'</td>'
                         + ''.join('<td>'+v+'</td>' for v in [pct(m['roi']),pct(m['daily_vol']),pct(m['max_drawdown']),
                         f"{m['mean_gross']*100:.2f}%",f"{m['fees']:,.2f}",f"{m['funding']:,.2f}",str(m['trading_days']),
                         str(m['partial_or_unfilled_orders']),html.escape(status)])+'</tr>')
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in ['전략','조건','순수익률','일간 변동성','최대 낙폭','평균 명목/자본','수수료','Funding','거래일','미완전체결 주문','검증'])+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>'
    colors=['#2563eb','#0d9488','#7c3aed','#ea580c','#dc2626','#64748b','#db2777']
    items=[(key,values) for key,values in curves.items() if key.startswith('retro/') and key.endswith('/base')]
    all_y=[(s['equity']/1e6-1)*100 for _,v in items for s in v]
    lo,hi=min(all_y+[0]),max(all_y+[0]); width=max(hi-lo,.01)
    plot=[];legend=[]
    for index,(key,values) in enumerate(items):
        c=colors[index%len(colors)]; start=values[0]['time'];end=values[-1]['time']
        points=' '.join(f"{55+890*(s['time']-start)/(end-start):.1f},{255-210*((s['equity']/1e6-1)*100-lo)/width:.1f}" for s in values)
        plot.append(f'<polyline points="{points}" fill="none" stroke="{c}" stroke-width="1.6"/>')
        legend.append(f'<span style="color:{c}">● {html.escape(label(key.split("/")[1]))}</span>')
    svg='<svg viewBox="0 0 1000 285" role="img" aria-label="회고 기간 누적 순수익률">'+f'<text x="5" y="45">{hi:.3f}%</text><text x="5" y="255">{lo:.3f}%</text>'+''.join(plot)+'</svg>'
    selected=report['selection']; challenger=selected['challenger'] or '없음 — 선정 기준을 통과한 신규 후보 없음'
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>보존 전략과 A·B 공통 비교</title><style>body{{font:15px/1.65 system-ui,sans-serif;margin:32px auto;max-width:1250px;padding:0 22px;color:#172033;background:#fafbfc}}h1{{font-size:28px}}h2{{margin-top:34px}}.card{{background:white;border:1px solid #dce3ec;border-radius:12px;padding:20px;margin:18px 0}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#eff3f8}}svg{{width:100%}}.legend{{display:flex;gap:15px;flex-wrap:wrap}}code{{font-size:12px}}a{{color:#1d4ed8}}</style>
<h1>보존 전략 3개와 A·B의 공통 조건 비교</h1>
<p>BTC · ETH · HYPE | 초기자본 1,000,000 USDPR | 총 명목 상한 10% | 과거 자료의 재사용 비교</p>
<div class="card"><b>개발 구간에서 고정한 신규 후보: {html.escape(challenger)}</b><p>학습 3회 · 전략 경로 44개 · 원형 파일 보존 · 실제 주문 0건. 결과를 보고 설정을 추가하지 않았습니다.</p>
<p>A 대표: {selected['A']['policy']} ({selected['A']['role']})<br>B 대표: {selected['B']['policy']} ({selected['B']['role']})</p></div>
<h2>회고 비교 · 2026-08-26 10:00–10-05 16:00 UTC</h2><p>새로운 독립 holdout이 아닙니다. 아래 결과로 선택한 모델을 바꾸지 않았습니다.</p>
<div class="card">{svg}<div class="legend">{''.join(legend)}</div></div>{table(report['retro'])}
<h2>설정 선택 · 2026-07-27 06:00–08-26 10:00 UTC</h2>{table(report['select'])}
<h2>설명용 진단 · 선택 대상에서 제외</h2>{table(report['diagnostics'])}
<h2>해석과 검증 범위</h2><ul><li>모든 전략을 같은 체결·회계·기말평가로 다시 실행했습니다. 원형 수익률과 혼합하지 않았습니다.</li>
<li>기본: 편도 수수료 9bp + 불리한 가격 10bp, 거래량 참여 1%. 스트레스: 9bp + 20bp, 참여 0.25%, 양방향 명목에 시간당 1bp 추가 funding.</li>
<li>관측된 signed funding만 반영하고 나머지는 기본 조건에서 0으로 가정했습니다. 실제 과거 par 체결·oracle·전체 funding 재현이 아닙니다.</li>
<li>변동성은 완전한 UTC 날짜의 일간 자산 수익률 모집단 표준편차입니다. 기말 잔여 포지션은 참고 가격으로 평가하고 강제청산하지 않았습니다.</li>
<li>공통 상한이 같아도 실제 위험과 투자금액은 다릅니다. 위험 증가를 예측 우위로 해석하지 않습니다. 대회 점수·최소 거래대금 적합성은 확정하지 않았습니다.</li></ul>
<p><a href="report.json">전체 결과 JSON</a> · <a href="selection.json">고정 선택 기록</a> · <a href="manifest.json">실행 전 명세·해시</a> · <a href="verification.json">검증 기록</a></p>
<p><code>{report['identity']}</code> · 생성 {html.escape(report['finished_at'])}</p></html>'''
    (dest/'index.html').write_text(body)


def main():
    before, retained = protected_files()
    protocol = json.loads((SOURCE/'protocol.json').read_text())
    files = list(SOURCE.glob('*.py')) + [SOURCE/'protocol.json', ROOT/'tests/test_common_comparison.py']
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}
    identity = hashlib.sha256(json.dumps([protocol, sources, before],sort_keys=True).encode()).hexdigest()[:16]
    dest=OUT/identity
    if (dest/'report.json').exists():
        print('Existing immutable result:', dest, flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    verification = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_common_comparison', '-v'],
                                  cwd=ROOT, capture_output=True, text=True)
    (dest/'synthetic-tests.txt').write_text(verification.stdout+verification.stderr)
    if verification.returncode:
        raise RuntimeError('Synthetic tests failed; no research execution permitted')
    manifest=dict(identity=identity, sealed_at=stamp(), status='sealed_before_training_and_policy_runs',
                  protocol=protocol, sources=sources, protected_files=before,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__),
                  tests_passed_before_execution=True)
    if (dest/'manifest.json').exists():
        old=json.loads((dest/'manifest.json').read_text())
        assert old['identity']==identity and old['sources']==sources
    else:
        write(dest/'manifest.json', manifest)
    for name in sources:
        p=dest/'source'/name; p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    market, original, _ = load_inputs(ROOT)
    x=features(market)
    controls, states=retained_models(original, retained)
    for version in retained['versions']:
        if states[version['policy']] != json.loads((ARCHIVE/version['states_file']).read_text()):
            raise AssertionError('Retained deterministic states changed')
    periods={k:tuple(timestamp(v) for v in times) for k,times in protocol['periods'].items()}
    policies={}
    for key,model in controls.items():
        policies['common-'+key]=Policy('common-'+key,'control',state=model)
    for lookback in protocol['momentum_lookbacks']:
        key=f'A-L{lookback}';policies[key]=Policy(key,'momentum',lookback)
    training=[]; fitting_operations=0
    for alpha in protocol['ridge_alphas']:
        key=f'B-alpha{alpha:g}';path=dest/'models'/(key+'.json')
        reused_from=None
        if path.exists(): state=json.loads(path.read_text())
        else:
            # Recovery from artifact-writing failure must not retrain or retune.
            # Only reuse fits with identical fitting code, protocol, inputs and runtime.
            model_source='replay_lab/common_comparison/models.py'
            for prior in sorted(OUT.glob('*/models/'+key+'.json')):
                prior_manifest=json.loads((prior.parent.parent/'manifest.json').read_text())
                if (prior_manifest['protocol']==protocol and prior_manifest['protected_files']==before
                        and prior_manifest['runtime']==manifest['runtime']
                        and prior_manifest['sources'].get(model_source)==sources[model_source]):
                    reused_from=str(prior.relative_to(ROOT));state=json.loads(prior.read_text());break
            if reused_from is None:
                state=fit_ridge(market,x,*periods['fit'],alpha);fitting_operations+=1
            write(path,state)
        training.append(dict(policy=key,file=str(path.relative_to(dest)),sha256=digest(path),samples=state['samples'],
                             last_label_available=state['last_label_available'],reused_from=reused_from))
        policies[key]=Policy(key,'ridge',alpha,state)
    policies['cash']=Policy('cash','cash');policies['basket-3pct-each']=Policy('basket-3pct-each','basket')
    write(dest/'models/manifest.json',dict(distinct_fits=3,new_fitting_operations=fitting_operations,models=training))
    audits=[]; curves={}; all_rows=[]; routes=[]
    def run(phase,key,scenario,diagnostic=False):
        category='diagnostic' if diagnostic else 'main'
        relative=f'runs/{phase}-{category}-{key}-{scenario}.json.gz';path=dest/relative
        if path.exists():
            with gzip.open(path,'rt') as stream: result=json.load(stream)
        else:
            result=simulate(market,x,policies[key],protocol,scenario,*periods[phase])
            audit_run(result)
            path.parent.mkdir(exist_ok=True)
            payload=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'),default=json_value).encode()
            with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as stream:
                stream.write(payload)
        audits.append(dict(file=relative,sha256=digest(path),**audit_run(result)))
        row=summary(result,relative);row.update(phase=phase,diagnostic=diagnostic);all_rows.append(row)
        routes.append(dict(phase=phase,policy=key,scenario=scenario,diagnostic=diagnostic,file=relative))
        if not diagnostic:curves[f'{phase}/{key}/{scenario}']=result['snapshots']
        print(f"{phase:6} {key:54} {scenario:6} ROI {100*row['metrics']['roi']:+.5f}% vol {100*row['metrics']['daily_vol']:.5f}% gates {all(row['gates'].values())}",flush=True)
        return row
    select=[]
    for key in list(policies):
        for scenario in protocol['scenarios']:select.append(run('select',key,scenario))
    winners={name:select_family(select,kind) for name,kind in [('A','momentum'),('B','ridge')]}
    possible=[(family,w) for family,w in winners.items() if w['eligible']]
    challenger=None
    if possible:
        maximum=max(w['metrics']['roi'] for _,w in possible)
        tied=[(family,w) for family,w in possible if w['metrics']['roi']>=maximum-.0001]
        challenger=min(tied,key=lambda item:(item[1]['metrics']['daily_vol'],item[0]))[1]['policy']
    selection=dict(**winners,challenger=challenger,frozen_at=stamp(),retrospective_results_used=False,
                   selection_inputs=[dict(file=r['file'],sha256=digest(dest/r['file'])) for r in select],
                   protocol_sha256=digest(SOURCE/'protocol.json'))
    selection_path=dest/'selection.json'
    if selection_path.exists():
        previous=json.loads(selection_path.read_text())
        assert all(previous[k]==selection[k] for k in ('A','B','challenger','selection_inputs','protocol_sha256'))
        selection=previous
    else:write(selection_path,selection)
    selection_hash=digest(selection_path)
    frozen_keys=['common-'+v['policy'] for v in retained['versions']]+[winners['A']['policy'],winners['B']['policy'],'cash','basket-3pct-each']
    write(dest/'planned-routes.json',dict(select=[dict(policy=k,scenario=s) for k in policies for s in protocol['scenarios']],
        retro=[dict(policy=k,scenario=s) for k in frozen_keys for s in protocol['scenarios']],
        diagnostic_templates=protocol['diagnostics'],selection_sha256=selection_hash,total_policy_runs=44))
    retro=[]
    for key in frozen_keys:
        for scenario in protocol['scenarios']:retro.append(run('retro',key,scenario))
    a=policies[winners['A']['policy']];b=policies[winners['B']['policy']]
    policies['diag-A-equal-notional']=Policy('diag-A-equal-notional','momentum',a.parameter,equal=True)
    policies['diag-B-zero-buffer']=Policy('diag-B-zero-buffer','ridge',b.parameter,b.state,buffer_bps=0)
    diagnostics=[]
    for phase in ('select','retro'):
        for key in ('diag-A-equal-notional','diag-B-zero-buffer'):
            for scenario in protocol['scenarios']:diagnostics.append(run(phase,key,scenario,True))
    assert len(routes)==44 and len(training)==3
    assert digest(selection_path)==selection_hash
    after,_=protected_files()
    if after!=before:raise AssertionError('Protected originals changed during experiment')
    evidence=[]
    if challenger:
        for scenario in protocol['scenarios']:
            cand=next(r for r in retro if r['policy']==challenger and r['scenario']==scenario)
            for control in ['common-'+v['policy'] for v in retained['versions']]:
                base=next(r for r in retro if r['policy']==control and r['scenario']==scenario)
                cm,bm=cand['metrics'],base['metrics']
                evidence.append(dict(scenario=scenario,challenger=challenger,control=control,
                    roi_difference=cm['roi']-bm['roi'],daily_vol_difference=cm['daily_vol']-bm['daily_vol'],
                    candidate_pareto_dominates=cm['roi']>=bm['roi'] and cm['daily_vol']<=bm['daily_vol'] and (cm['roi']>bm['roi'] or cm['daily_vol']<bm['daily_vol']),
                    control_pareto_dominates=bm['roi']>=cm['roi'] and bm['daily_vol']<=cm['daily_vol'] and (bm['roi']>cm['roi'] or bm['daily_vol']<cm['daily_vol'])))
    report=dict(identity=identity,finished_at=stamp(),selection=selection,select=select,retro=retro,diagnostics=diagnostics,
                comparison=evidence,ridge_fits=3,new_fitting_operations=fitting_operations,policy_runs=44,original_policy_replays=0,
                retrospective_only=True,live_orders_sent=0,deployed=False,selection_sha256=selection_hash,
                original_archives_unchanged=True,protocol=protocol)
    write(dest/'verification.json',dict(synthetic_tests_passed=True,synthetic_test_log='synthetic-tests.txt',
        retained_states_reproduced=True,protected_files_unchanged=True,protected_file_count=len(before),
        source_files=sources,selection_sha256=selection_hash,selection_unchanged_after_retro=True,
        policy_runs=44,ridge_fits=3,live_orders_sent=0,run_audits=audits))
    write(dest/'routes.json',routes);write(dest/'report.json',report);write_html(dest,report,curves)
    write(OUT/'latest.json',dict(identity=identity,report=str((dest/'report.json').relative_to(ROOT)),html=str((dest/'index.html').relative_to(ROOT))))
    print('COMPLETE',dest,flush=True)


if __name__=='__main__':main()
