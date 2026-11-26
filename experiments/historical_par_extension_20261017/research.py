"""Fixed-account comparisons on appended venue history; no live deployment."""
import csv,gzip,html,json,time
import numpy as np
from experiments.complementary_alpha_20261017.research import protected
from experiments.hourly_refresh_pairs_20261017.policy import replay
from .collect import HERE,ROOT,OUT,ts,sha,save
from .prepare import prepare_data,prepare_states

def previous_path(arm,scenario):
    family=('dense_history' if arm.endswith('_strategy') else 'rolling_feature' if '_direct_' in arm else 'calibrated_daily_tree')
    return ROOT/f'experiments/{family}_pairs_20261017/artifacts/runs/par-comparison-{arm}-{scenario}.json.gz'

def verify_sources(sources):
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path

def render(result):
    report=ROOT/'reports/historical_par_extension_20261017';report.mkdir(exist_ok=True)
    fields=['window','arm','scenario','net_pnl','roi','financial_score','volatility_horizon','max_drawdown','fees','funding','completed_pairs','fills','residual_gross','pnl_after_liquidation_reserve']
    with (report/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in result['rows']:writer.writerow({k:row[k] if k in row else row['metrics'].get(k) for k in fields})
    tables=[]
    for window in result['windows']:
        lines=[]
        for row in result['rows']:
            if row['window']!=window:continue
            m=row['metrics'];q='산출 불가' if m['financial_score'] is None else f"{m['financial_score']:.8f}"
            vol='산출 불가' if m['volatility_horizon'] is None else f"{m['volatility_horizon']*100:.6f}%"
            lines.append(f"<tr><td>{html.escape(row['arm'])}</td><td>{row['scenario']}</td><td>{m['net_pnl']:+.6f}</td><td>{m['roi']*100:+.6f}%</td><td>{q}</td><td>{vol}</td><td>{m['max_drawdown']*100:.6f}%</td><td>{m['completed_pairs']}</td><td>{m['fees']:.6f}</td><td>{m['funding']:+.6f}</td><td>{m['residual_gross']:.4f}</td></tr>")
        tables.append(f'<h2>{html.escape(window)}</h2><table><tr><th>전략</th><th>비용</th><th>순손익 USDPR</th><th>수익률</th><th>Q</th><th>기간 변동성</th><th>MDD</th><th>완료 페어</th><th>수수료</th><th>펀딩</th><th>잔여 명목액</th></tr>'+''.join(lines)+'</table>')
    carry='<pre>'+html.escape(json.dumps(result['continuous_appended_day'],ensure_ascii=False,indent=2))+'</pre>'
    body='''<!doctype html><meta charset="utf-8"><title>추가 par 과거 24시간 비교</title><style>body{font:15px system-ui;margin:32px;line-height:1.65}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:7px;text-align:right}td:first-child{text-align:left}pre{background:#f3f4f5;padding:16px;overflow:auto}</style>
<h1>기존 페어 후보: 추가 par 과거 자료 비교</h1>
<p>추가 자료: 2026년 10월 16일 21:00 → 17일 21:00 KST. 기존 데이터와 예측값을 보존하고 설정을 바꾸지 않았다. 전체 연장 구간은 10월 8일 21:00 → 17일 21:00 KST. 각 계좌는 1M USDPR·빈 포지션에서 시작한다. 실제 지갑 성과가 아닌 과거 모의 계좌다.</p>
<p>4개 고정 전략 × 2개 비용 × 2개 기간 = 새 계좌 16개. 기존 기간 8개 재실행의 체결·이벤트·스냅샷·지표 일치와 연장 계좌의 기존 경로 일치를 확인했다. 저장된 회귀·트리 계수로 추가 7시간의 예측을 생성했다. 새 모델 학습·규칙 조정·후보 재선정은 없다.</p>
<p>base: 편도 수수료 9bp·불리한 가격 10bp·관측 거래량 참여 한도 10%. stress: 12bp·20bp·참여 5%·양의 펀딩 수취 절반. 양쪽 주문은 따로 체결된다. 실제 과거 호가를 재구성한 결과가 아니며, 종가·거래량 기반 체결 가정이다. 잔여 포지션은 평가손익에 포함되며 청산했다고 가정하지 않는다.</p>
<p>하루 단독 평가는 완성 일별 수익률 1개이므로 표본 표준편차와 Q를 산출할 수 없다. 연장 계좌는 9개 일별 수익률로 산출한다. 아래 연속 계좌 하루 변화는 10월 16일 포지션을 그대로 이어간 결과이며 빈 계좌 하루 실험과 다르다.</p>
<p>추가 수집한 거래소 자료라도 이미 연구에 사용한 기준시장 기간 및 이전 실행 평가 기간과 겹친다. 독립 미사용 검증이 아니다. 예약된 호가 원본·비교 결과를 열지 않았으며 새 예약 평가 시작 이후 가격을 쓰지 않았다. 실거래 전략 교체나 수익성 입증으로 해석하지 않는다.</p>'''
    (report/'report.html').write_text(body+''.join(tables)+'<h2>연속 계좌의 추가 하루 변화</h2>'+carry+'<p>원본: artifacts/results.json, inputs.npz, data_audit.json, model_audit.json, frozen.json. 주문 전송 0.</p>')
    return str((report/'report.html').relative_to(ROOT))

def main():
    if (OUT/'frozen.json').exists():raise FileExistsError('Evaluation already frozen; do not restart')
    spec=json.loads((HERE/'protocol.json').read_text())
    pre=json.loads((OUT/'precollection.json').read_text());verify_sources(pre['sources'])
    sources=json.loads((ROOT/'experiments/calibrated_daily_tree_pairs_20261017/artifacts/frozen.json').read_text())['sources'].copy()
    reserved=ROOT/'experiments/published_pair_lifecycle_20261017/artifacts/plan.json'
    reserved_sources=json.loads(reserved.read_text())['sources'];sources.update(reserved_sources)
    sources.update(pre['sources']);sources.update(json.loads((OUT/'collection.json').read_text())['sources'])
    paths=[reserved,OUT/'precollection.json',OUT/'collection.json']
    paths += [ROOT/spec['reference_run']/name for name in ['inputs.npz','targets.npz','manifest.json','verification.json']]
    paths += [previous_path(arm,sc) for arm in spec['arms'] for sc in ['base','stress']]
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in paths});verify_sources(sources)
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),protocol=spec,reserved_raw_read=False))
    d,ref,p=prepare_data();states=prepare_states(d,ref,p)
    cfg=json.loads((ROOT/'experiments/hourly_refresh_pairs_20261017/par_protocol.json').read_text())
    (OUT/'runs').mkdir(exist_ok=False);controls={};verified=[];rows=[];continuous=[]
    for arm in spec['arms']:
        state=states['cmix' if '_cmix_' in arm else 'direct']
        for sc in cfg['scenarios']:
            with gzip.open(previous_path(arm,sc),'rt') as f:old=json.load(f)
            run=replay(d,state,None,cfg,p,arm,sc,old['start'],old['end'])
            for key in ['fills','events','metrics','snapshots']:assert run[key]==old[key],(arm,sc,key)
            controls[arm,sc]=old;verified.append(dict(arm=arm,scenario=sc))
            print('EXACT old',arm,sc,flush=True)
    for window,(start,end) in spec['windows'].items():
        for arm in spec['arms']:
            state=states['cmix' if '_cmix_' in arm else 'direct']
            for sc in cfg['scenarios']:
                run=replay(d,state,None,cfg,p,arm,sc,ts(start),ts(end))
                if window=='extended':
                    old=controls[arm,sc];cut=old['end']
                    for key in ['fills','snapshots']:
                        assert [x for x in run[key] if x['time']<=cut]==old[key],(arm,sc,key,'continuous prefix')
                    assert [x for x in run['events'] if x['time']<cut]==[x for x in old['events'] if x['time']<cut]
                    begin=old['snapshots'][-1]['equity'];ending=run['snapshots'][-1]['equity']
                    continuous.append(dict(arm=arm,scenario=sc,start_equity=begin,end_equity=ending,marked_pnl=ending-begin,return_fraction=ending/begin-1))
                else:
                    assert run['metrics']['completed_days']==1
                    assert run['metrics']['financial_score'] is None and run['metrics']['volatility_horizon'] is None
                with gzip.open(OUT/'runs'/f'{window}-{arm}-{sc}.json.gz','wt') as f:
                    json.dump(run,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
                rows.append(dict(window=window,arm=arm,scenario=sc,metrics=run['metrics']))
                print(window,arm,sc,'PnL',round(run['metrics']['net_pnl'],6),'Q',run['metrics']['financial_score'],'pairs',run['metrics']['completed_pairs'],flush=True)
    verify_sources(sources)
    result=dict(rows=rows,windows=spec['windows'],new_account_replays=16,exact_controls=verified,continuous_appended_day=continuous,
        protected=protected(),reserved_sources_unchanged=len(reserved_sources),reserved_raw_read=False,orders_sent=0,live_strategy_changed=False,historical_independence=False)
    save(OUT/'results.json',result);report=render(result)
    save(HERE/'work_state.json',dict(status='progress_fixed_candidates_extra_historical_par_day_evaluated',report=report,
        new_account_replays=16,exact_control_replays=len(verified),orders_sent=0,live_strategy_changed=False,
        historical_independence=False,reserved_raw_read=False,goal_complete=False,
        findings=[dict(window=r['window'],arm=r['arm'],scenario=r['scenario'],pnl=r['metrics']['net_pnl'],q=r['metrics']['financial_score'],completed_pairs=r['metrics']['completed_pairs']) for r in rows]))
    print('DONE 16 new accounts +8 exact controls;',report,flush=True)

if __name__=='__main__':main()
