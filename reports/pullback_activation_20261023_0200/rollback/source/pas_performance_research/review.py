"""Audit every experiment and describe the modest, selected development result."""
import gzip,html,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .study import ROOT,OUT,BASE,save,sha,live_files
from .selected import NAME,controller

def read(path):return json.loads(path.read_text())

def audit():
    refs=[];venues=[];catalog=[]
    for wave,folder in [('wave1',OUT)]+[(f'wave{i}',OUT/f'wave{i}') for i in range(2,8)]:
        rr=read(folder/'reference_results.json');pp=read(folder/'pas_results.json')
        refs += [dict(wave=wave,**r) for r in rr];venues += [dict(wave=wave,**r) for r in pp]
        catalog.append(dict(wave=wave,arms=read(folder/'protocol.json')['arms'],reference_runs=len(rr),pas_runs=len(pp)))
    # Reproduction is checked for every duplicate baseline, not only a winner.
    original=read(ROOT/'reports/universe_repair_20261022/reference_results.json')
    for r in refs:
        assert r['timing_pass'] and r['accounting_error']<1e-6 and r['risk_breach_bars']==0
        if r['arm']==BASE:
            a=next(x for x in original if x['arm']==BASE and x['phase']==r['phase'] and x['scenario']==r['scenario'])
            assert abs(r['net_pnl']-a['net_pnl'])<1e-7
    for r in venues:
        assert r['timing_pass'] and r['maximum_accounting_error']<1e-6 and r['risk_breach_events']==0
    rr=read(OUT/'wave7/reference_results.json');pp=read(OUT/'wave7/pas_results.json')
    comparisons=[]
    for group,rows in [('reference',rr),('pas',pp)]:
        for c in [r for r in rows if r['arm']==NAME]:
            b=next(r for r in rows if r['arm']==BASE and r.get('phase')==c.get('phase') and r['scenario']==c['scenario'])
            comparisons.append(dict(group=group,phase=c.get('phase'),scenario=c['scenario'],baseline=b['net_pnl'],candidate=c['net_pnl'],
                pnl_improvement=c['net_pnl']-b['net_pnl'],Q_improvement=None if c['Q'] is None else c['Q']-b['Q']))
    assert all(r['pnl_improvement']>0 and (r['Q_improvement'] is None or r['Q_improvement']>0) for r in comparisons)
    assert live_files()==read(OUT/'protocol.json')['live_files']
    selected=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate=NAME,model_refit=False,
        live_changed=False,orders_sent=0,history_reused=True,independent_test=False,
        scope='Research candidate only. No change to active strategy, services, credentials or orders.',
        selection='Selected after seven declared development waves. All losses and prior candidates retained. Selection is not independent statistical validation.',
        mechanism='Inventory-aware3h/24h venue pullback gate on additions; preserve held thesis, exits and hard risk caps. Existing pair expert and trained forecasts unchanged.',
        comparison_pass=True,comparisons=comparisons,catalog=catalog,
        live_files=live_files(),sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).with_name('selected.py'),Path(__file__).with_name('soft_timing.py'),Path(__file__).with_name('wave7.py')]},
        limitations=['Modest gains; few completed pairs contribute strongly to reference improvement.',
          'PAS minute/tape conditional fills are not queue/L2/mark-level replay. Whole PAS history used is about71h.',
          'A formal daily Q is unavailable for the isolated23h interval. Continuous71h Q uses completed12UTC samples throughOct21, not the final intraday PnL.',
          'Need a current PAS price-history input adapter before considering live deployment; research deliberately does not install one.',
          'Earlier legacy3-core strategy had higher reference PnL than this candidate; target here is the currently deployed10-core strategy.'])
    save(OUT/'selected.json',selected);save(OUT/'all_reference_results.json',refs);save(OUT/'all_pas_results.json',venues)
    print('All comparisons passed',len(comparisons),'unique candidates',len({r['arm'] for r in refs if r['arm']!=BASE}),flush=True)

def reproduce():
    from competition_live.profile import load_profile,BUNDLE
    from pas_economic_allocation.replay import replay
    from pas_allocation_study.study import summarize
    import copy
    selected=read(OUT/'selected.json')
    for f,h in selected['sources'].items():assert sha(ROOT/f)==h
    p=load_profile();d=dict(np.load(BUNDLE/'data/reference.npz'));states=dict(np.load(BUNDLE/'models/reference_gate_state.npz'))
    z=dict(np.load(ROOT/'reports/universe_repair_20261022/targets.npz'));ix=np.searchsorted(z['times'],d['times'])
    f=dict(np.load(ROOT/'reports/pas_economic_allocation_20261022/forecasts.npz'));fi=np.searchsorted(f['times'],d['times'])
    source=read(ROOT/'reports/universe_repair_20261022/protocol.json')['input']['inputs'];raw=dict(np.load(ROOT/source))
    cfg=copy.deepcopy(p['configs']['reference']);cfg['conditional_completion']={'risk_aversion':0.}
    c=controller(d['times'],f['forecast'][fi],f['carry'][fi],z[BASE][ix],z[BASE+'_protected'][ix],price_times=raw['market_times'],prices=raw['close'])
    r=replay(d,states,None,cfg,p['trading'],p['candidate'],'base',1785585600,1792065600,base_policy=c)
    row=summarize(r,NAME,'continuous','base');expected=next(x for x in read(OUT/'wave7/reference_results.json') if x['arm']==NAME and x['phase']=='continuous' and x['scenario']=='base')
    for k in ['net_pnl','Q','fees','funding','mean_notional']:assert abs(row[k]-expected[k])<1e-10,k
    with gzip.open(OUT/'wave7/reference'/f'{NAME}_continuous_base.json.gz','rt') as fp:prior=json.load(fp)
    assert r['fills']==prior['fills']
    save(OUT/'reproduction.json',dict(exact_fills=True,metric_max_error=max(abs(row[k]-expected[k]) for k in ['net_pnl','Q','fees','funding','mean_notional']),model_refit=False,orders_sent=0))

def uncertainty():
    # Descriptive block bootstrap only; it does not correct model selection.
    pairs=[]
    for arm in [BASE,NAME]:
        with gzip.open(OUT/'wave7/reference'/f'{arm}_continuous_base.json.gz','rt') as fp:r=json.load(fp)
        daily=[s for s in r['snapshots'] if s['time']%86400==43200]
        pairs.append(np.diff([s['equity'] for s in daily]))
    delta=pairs[1]-pairs[0];n=len(delta);rng=np.random.default_rng(20261022);draws=[]
    for _ in range(2000):
        starts=rng.integers(0,n,size=int(np.ceil(n/7)))
        ix=np.concatenate([(s+np.arange(7))%n for s in starts])[:n];draws.append(float(delta[ix].sum()))
    ci=np.quantile(draws,[.025,.975])
    save(OUT/'uncertainty.json',dict(method='Paired circular7-day-block bootstrap,2000draws,seed20261022; descriptive,not selection-adjusted',days=n,
        observed_delta=float(delta.sum()),interval_95=ci.tolist(),includes_zero=bool(ci[0]<=0<=ci[1]),
        caution='Cannot infer statistically reliable improvement after repeated candidate selection from this interval.'))

def report():
    selected=read(OUT/'selected.json');rr=read(OUT/'wave7/reference_results.json');pp=read(OUT/'wave7/pas_results.json')
    controls=read(OUT/'wave2/controls/results.json');unc=read(OUT/'uncertainty.json')
    def num(v,d=2):return '—' if v is None else f'{v:,.{d}f}'
    def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    def pair(rows,phase,sc):return [next(r for r in rows if r['arm']==a and r['phase']==phase and r['scenario']==sc) for a in [BASE,NAME]]
    summary=[]
    for title,rows,phase,sc in [('기준시장8/1–10/15 기본',rr,'continuous','base'),('기준시장8/1–10/15 불리',rr,'continuous','stress'),('PAS10/19 21시–10/22 20시 기본',pp,'continuous71h','base_1pct'),('동일PAS71시간 불리',pp,'continuous71h','stress_025pct'),('PAS추가23시간 기본',pp,'later23h','base_1pct'),('기존PAS48시간 체결자료 기본',pp,'trade_tape48h','base_1pct')]:
        b,c=pair(rows,phase,sc);summary.append([title,num(b['net_pnl']),num(c['net_pnl']),num(c['net_pnl']-b['net_pnl']),num(b['Q'],6),num(c['Q'],6)])
    details=[]
    for r in rr:
        if r['arm'] in [BASE,NAME]:details.append([r['arm'],r['phase'],r['scenario'],num(r['net_pnl']),num(r['terminal_pnl']),num(r['Q'],6),num(r['volatility']*100,4),num(r['max_drawdown']*100,4),num(r['mean_notional']),r['fills'],r['short_entries']])
    pas=[]
    for r in pp:
        if r['arm'] in [BASE,NAME]:pas.append([r['arm'],r['phase'],r['scenario'],num(r['net_pnl']),num(r['pnl_after_liquidation_reserve']),num(r['Q'],6),num(r['mean_notional']),r['fills'],','.join(r['traded_symbols'])])
    leaderboard=read(OUT/'peers/official_summary.json');lb=[]
    for r in leaderboard['teams']:lb.append([r['name'],r['rank'],num(r['return']*100,5),num(r['volatility_horizon']*100,5),num(r['q'],6),num(r['score'],6),num(r['latest_chart_pnl']*100 if r['latest_chart_pnl'] is not None else None,5)])
    wallets=[]
    for f in sorted((OUT/'peers').glob('0x*.json')):
        w=read(f);pos=[r['position'] for r in w['responses']['clearinghouseState']['response']['assetPositions']];fills=[r for r in w['responses']['userFillsByTime']['response'] if r['coin'].startswith('pas:')]
        long=sum(float(r['positionValue']) for r in pos if float(r['szi'])>0);short=sum(float(r['positionValue']) for r in pos if float(r['szi'])<0)
        wallets.append([w['name'] or w['address'],num(long),num(short),','.join(str(x) for x in sorted({r['leverage']['value'] for r in pos})),len(fills),'; '.join(r['coin'][4:]+('롱' if float(r['szi'])>0 else '숏') for r in pos)])
    catalog=[]
    for wave in selected['catalog']:
        rows=[r for r in read(OUT/'all_reference_results.json') if r['wave']==wave['wave'] and r['phase']=='continuous']
        for arm in wave['arms']:
            if arm==BASE:continue
            vals=[next(r for r in rows if r['arm']==arm and r['scenario']==s)['net_pnl'] for s in ['base','stress']]
            catalog.append([wave['wave'],arm,num(vals[0]),num(vals[1])])
    b,c=pair(rr,'continuous','base');bs,cs=pair(rr,'continuous','stress')
    body='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX 독립 연구 결과</title><style>body{font:16px/1.65 system-ui;max-width:1350px;margin:40px auto;padding:24px;color:#172638;background:#f7f9fc}h1,h2{color:#173857}table{border-collapse:collapse;background:white;font-size:13px;margin:18px 0;width:100%;display:block;overflow:auto}th,td{padding:9px;border:1px solid #d8e0e8;text-align:left;white-space:nowrap}th{background:#e5edf5}.note{padding:15px;background:#fff4d7;border-left:5px solid #b67d12}a{color:#155d97}code{background:#e9eef3;padding:2px 5px}</style><h1>SFINX: 현재 제출판 대비 개발 성능 비교</h1>'''
    body+='<p>후보 <code>pullback_any3</code>. 실거래 전략·실행자·설정은 변경하지 않았고 연구 주문은0건입니다. 현재 배포판은 <code>all_market_signed_core_v1</code>입니다.</p>'
    body+='<p class="note">개발 구간을 반복 사용한 결과입니다.7개 실험 묶음의 후보를 보고 선택했으므로 독립 검증·과적합 부재·향후 수익을 입증하지 않습니다. PAS는 과거 호가 대기열이 없는 조건부 모의 체결입니다. 이 보고서는 교체나 주문 권한을 실행하지 않습니다.</p>'
    body+='<h2>핵심 결과</h2><p>모든 금액은 동일한50,000USDP 전략 자본과50,000USDP 비교용 현금을 적용했습니다. 수익률·Q의 분모는 전체100,000입니다. 기본은 수수료9bp+충격10bp, 불리한 조건은12bp+20bp입니다. PAS체결량 상한은 관측 거래량의1%/0.25%입니다.</p>'+table(['구간·조건','현재판 순PnL','후보 순PnL','차이','현재Q','후보Q'],summary)
    body+='<p>PAS71시간 Q는10/21 21시까지의 완료 일별 표본만 사용합니다.71시간 종료의 장중손익을 포함한Q로 읽으면 안 됩니다. 별도23시간 구간은 일별수익률2개가 없어Q를 산출하지 않습니다. 최초48시간 체결자료에서는 양쪽 모두 손실이며 후보가 손실을 줄였다는 의미입니다.</p>'
    body+='<h2>실제 변경점</h2><p>기존 모델·다종목 롱숏·페어 전략을 유지했습니다. 추가 매수는 현재 PAS가격이3시간 또는24시간 전 종가 이하일 때, 추가 숏은 그중 하나 이상일 때 허용합니다. 조건이 안 맞으면 같은 방향의 기존 수량을 유지하고 추가 진입만 보류합니다. 기존 청산·감축과 위험 상한은 계속 적용합니다.10개 종목 모두 같은 규칙이며 종목 이름으로 제외하지 않습니다. 판단 주기는 기존6시간을 유지합니다.</p>'
    body+='<p>이 후보는 새 예측 모델을 학습한 결과가 아닙니다. 기존 방향 판단을 주문으로 연결하는 진입 시점을 바꿨습니다. 실거래 연결 전에는 시점과 갱신 지연을 검증하는 PAS 가격 이력 입력기가 필요합니다. 이 작업에서는 운용 프로그램을 바꾸지 않았습니다.</p>'
    body+='<p>미해결 사항도 있습니다. 현재 외부시장 기반 가격·펀딩 전망이 PAS 가격·펀딩을 정확히 설명하는 것은 아닙니다. 실제 PAS펀딩을 투입한 별도 후보는 추가기간에서 뒤져 선정판에 합치지 않았습니다. 따라서 이번 결과가 기존 전략의 모든 데이터·예측 한계를 해결했다는 뜻은 아닙니다.</p>'
    body+='<h2>개선 원인과 한계</h2><p>기준시장 기본 조건에서 수수료는 '+num(b['fees'])+'→'+num(c['fees'])+', 평균 보유명목은 '+num(b['mean_notional'])+'→'+num(c['mean_notional'])+'입니다. 기본 포지션 소유분 손익은 '+num(b['base_pnl'])+'→'+num(c['base_pnl'])+'로 오히려 나빠졌고, 페어 소유분은 '+num(b['pair_pnl'])+'→'+num(c['pair_pnl'])+'로 개선됐습니다. 따라서 방향 예측력이 좋아졌다고 말할 수 없습니다. 진입 순서·공유 위험 한도·페어 체결의 상호작용이 주요 원인입니다.</p>'
    body+='<p>완료 페어는 기본5건, 불리한 조건3건으로 적습니다. 기본 조건의 기간 변동성은 '+num(b['volatility']*100,4)+'%→'+num(c['volatility']*100,4)+'%로 소폭 올랐지만 Q는 개선됐습니다. 불리한 기준시장 PnL개선은 '+num(cs['net_pnl']-bs['net_pnl'])+'에 불과합니다. 최대낙폭과 최종 청산비용 준비금을 함께 표시했습니다.</p>'
    body+='<p>참고용7일 블록 부트스트랩에서 기준시장 PnL차이의95%범위는 ['+num(unc['interval_95'][0])+', '+num(unc['interval_95'][1])+']입니다. 이는 후보 선택 편향을 보정하지 않은 기술통계이며 통계적으로 확실한 우월성의 증거가 아닙니다. 과거3종목 중심 legacy판의 기준시장 PnL(+1,884.23)은 이 후보보다 높았습니다. 이번 비교 대상은 현재 배포된10종목 판입니다.</p>'
    body+='<h2>기준시장 상세: 2026년8월1일–10월15일</h2>'+table(['전략','구간','조건','순PnL','청산비용 차감PnL','Q','기간변동성%','최대낙폭%','평균명목','체결조각','숏진입조각'],details)
    body+='<h2>PAS 상세: 10월19일21시–10월22일20시 KST</h2>'+table(['전략','구간','조건','순PnL','청산비용 차감PnL','Q','평균명목','체결조각','체결 종목'],pas)
    body+='<p>기준시장에서는10종목 모두 실제 모의 체결이 있었고 PAS71시간 후보에서는7종목이 체결됐습니다. 체결되지 않은 종목도 평가합니다. 체결조각 수는 독립적인 투자 판단이나 완결된 거래 횟수가 아닙니다. PAS캔들 평가는 매수시 다음 분 고가, 매도시 저가에 충격을 추가하며, 펀딩은 당시 알려진 거래가를 오라클의 대용으로 사용합니다. 긴 미체결·가격 정체와 오라클 차이에 대한 한계가 남습니다.</p>'
    body+='<h2>공식 순위와 공개 지갑 참고</h2><p>공식 자료 생성 시점 '+leaderboard['generated_at']+'. 순위와 장중차트를 구분했습니다. 공개공식피드는 팀명과 지갑주소를 연결해 주지 않으므로 SFINX 외 지갑을 특정 팀명으로 단정하지 않았습니다.</p>'+table(['팀','공식순위','공식R%','기간변동성%','Q','Score','차트장중R%'],lb)+table(['지갑/확인된팀','롱명목','숏명목','설정레버리지','조회기간 체결조각','현재 종목·방향'],wallets)
    body+='<p>공개 지갑 조회는4개 지갑의 포지션·최근체결8회로 제한했습니다. 타인의 현재 수익 순위로 과거의 추종 대상을 골라 백테스트하지 않았습니다. 설정 레버리지와 계좌 실제 노출은 다릅니다. 상위 지갑 중 하나는 다종목 롱·큰 노출을 유지했고, 다른 지갑은 롱숏을 혼합했습니다. 다른 팀의 성과 차이를 곧바로 방향예측력 차이라고 해석하지 않았습니다.</p>'
    body+='<h2>실패를 포함한 실험 목록</h2>'+table(['실험','후보','기준시장전체 기본PnL','동일 불리PnL'],catalog)
    body+='<p>펀딩후보는 처음48시간에서는 좋아졌지만 추가23시간에서 뒤져 제외했습니다. 단순 방향확인으로 보유분까지 청산한 후보는 비용과 손실을 늘렸습니다. 선정 후보의 이전·이후 구간과 모든 비교 비용 조건을 그대로 남겼습니다.</p>'
    body+='<h2>검증·재현·근거</h2><p>미래 펀딩·가격 변조, 접두 구간 일치, 학습/보정 종료 이후 정답 무시, 양방향·10종목 자격, 보유분 유지와 감축 보존 검사 통과. 기본 기준시장 재실행은 체결 배열까지 일치합니다. 전 실험의 시점·회계·위험 검사와 기존판 재현 검사를 수행했습니다.</p><p><a href="selected.json">선정 기록</a> · <a href="tests.txt">기초검사</a> · <a href="inventory_tests.txt">수량 인계 검사</a> · <a href="reproduction.json">재현</a> · <a href="uncertainty.json">불확실성</a> · <a href="all_reference_results.json">전체 기준시장 결과</a> · <a href="all_pas_results.json">전체 PAS결과</a> · <a href="extension/protocol.json">추가기간 프로토콜</a></p>'
    body+='<p>프로젝트 루트에서 다음 명령은 저장된 입력으로 선정 후보의 기준시장 체결 배열과 수치를 다시 검증합니다. 운용 소스·설정이 동시 변경됐다면 무시하지 않고 중단합니다.</p><pre>OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pas_performance_research.review reproduce</pre><p>후보 진입점: <code>pas_performance_research/selected.py</code>. 전체 실험: <code>pas_performance_research/wave7.py</code> 및 앞선wave별 소스. 근거 자료와 프로토콜을 삭제하거나 기존 실패 결과를 덮어쓰지 않았습니다.</p>'
    body+='<p><a href="https://perpetual-alpha26.github.io/leaderboard">공식 평가표</a> · <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding">펀딩 공식문서</a> · <a href="https://web.stanford.edu/~boyd/papers/cvx_portfolio.html">Boyd 외: 수익·위험·거래비용을 함께 고려하는 배분</a> · <a href="https://arxiv.org/html/2602.11708v1">2026년 AdaptiveTrend 검토 자료</a>. 마지막 논문의 성과를 우리 성과로 전용하지 않았으며, 고정70/30배분·시장총액 종목제한도 채택하지 않았습니다.</p></html>'
    (OUT/'report.html').write_text(body)
    save(OUT/'artifact_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'pas_performance_research').glob('*.py')})
    print(str(OUT/'report.html'))

if __name__=='__main__':globals()[sys.argv[1]]()
