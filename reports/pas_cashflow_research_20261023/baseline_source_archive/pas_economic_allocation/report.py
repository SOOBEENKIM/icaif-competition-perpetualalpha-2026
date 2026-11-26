"""Complete experiment ledger: improvement, failed checks and scope sensitivity."""
import csv,gzip,html,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from pas_adaptive_development.study import ROOT,save,sha

OUT=ROOT/'reports/strategy_replacement_review_20261022'
FAMILIES=['pas_economic_allocation','pas_economic_anchor','pas_economic_budget',
          'pas_economic_adaptive','pas_economic_slow','pas_economic_projected',
          'pas_economic_lifecycle','pas_economic_coexist','pas_unified_held']


def read(p):return json.loads(Path(p).read_text())
def gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def table(head,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def bootstrap(a,b,block=7):
    # Paired dollar-PnL changes, circular moving blocks. Descriptive ONLY:
    # adaptive reuse and selection invalidate independent-test p-value claims.
    x=np.diff([s['equity'] for s in a['snapshots'] if s['time']%86400==43200])-np.diff([s['equity'] for s in b['snapshots'] if s['time']%86400==43200])
    rng=np.random.default_rng(20261022);n=len(x)
    starts=rng.integers(0,n,(5000,math.ceil(n/block)))
    ids=(starts[:,:,None]+np.arange(block))%n
    means=x[ids.reshape(5000,-1)[:,:n]].mean(1)
    return dict(days=n,block=block,mean_daily_pnl_difference=float(x.mean()),
                interval95=np.quantile(means,[.025,.975]).tolist(),
                interpretation='Descriptive reused-history interval, not selection-adjusted or independent proof')


def main():
    from competition_live.profile import verify_deployment
    OUT.mkdir(exist_ok=True);allrows=[];families=[];violations=[]
    for name in FAMILIES:
        folder=ROOT/'reports'/f'{name}_20261022';p=read(folder/'protocol.json')
        rr=read(folder/'reference_results.json')
        for r in rr:allrows.append(dict(family=name,**r))
        for rel,h in read(folder/'source_freeze.json').items():
            if sha(ROOT/rel)!=h:violations.append(rel)
        nominee=p['nominee'];idx={(r['arm'],r['phase'],r['scenario']):r for r in rr};checks=[]
        # The first study saved controls in their original directory. Reuse
        # exactly those rows; never silently substitute another scope's control.
        for r in read(ROOT/'reports/pas_covariance_v2_20261021/reference_results.json'):
            if r['arm']=='incumbent':idx.setdefault((r['arm'],r['phase'],r['scenario']),r)
        for phase in ['continuous','comparison']:
            for scenario in ['base','stress']:
                a=idx[nominee,phase,scenario];b=idx['incumbent',phase,scenario]
                checks.append(dict(phase=phase,scenario=scenario,pnl_pass=a['net_pnl']>b['net_pnl'],
                    q_pass=a['Q']>b['Q'],engineering_pass=a['accounting_error']<1e-6 and a['timing_pass'] and not a['risk_breach_bars']))
        families.append(dict(name=name,nominee=nominee,rows=len(rr),checks=checks,
                             promotion_pass=all(all(c[k] for k in ['pnl_pass','q_pass','engineering_pass']) for c in checks)))
    assert not violations,violations
    allrows+= [dict(family='pas_economic_venue',**r) for r in read(ROOT/'reports/pas_economic_venue_20261022/results.json')]
    save(OUT/'all_results.json',allrows)
    fields=sorted(set().union(*(r.keys() for r in allrows)))
    with (OUT/'all_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrows)
    broad=read(ROOT/'reports/pas_economic_lifecycle_20261022/reference_results.json')
    scoped=read(ROOT/'reports/pas_economic_coexist_20261022/reference_results.json')
    selected=[r for r in broad if r['arm'] in ['incumbent','budget_lifecycle'] and r['phase'] in ['continuous','comparison']]
    broad_idx={(r['arm'],r['phase'],r['scenario']):r for r in broad}
    scoped_idx={(r['arm'],r['phase'],r['scenario']):r for r in scoped}
    def passes(idx,name):
        return all(idx[name,ph,sc]['net_pnl']>idx['incumbent',ph,sc]['net_pnl'] and idx[name,ph,sc]['Q']>idx['incumbent',ph,sc]['Q']
                   for ph in ['continuous','comparison'] for sc in ['base','stress'])
    uncertainty=[]
    for phase in ['continuous','comparison']:
        for scenario in ['base','stress']:
            a=gz(ROOT/f'reports/pas_economic_lifecycle_20261022/reference/budget_lifecycle_{phase}_{scenario}.json.gz')
            b=gz(ROOT/f'reports/pas_covariance_v2_20261021/reference/incumbent_{phase}_{scenario}.json.gz')
            for block in [3,7]:uncertainty.append(dict(phase=phase,scenario=scenario,**bootstrap(a,b,block)))
    save(OUT/'uncertainty.json',uncertainty)
    active=verify_deployment();adapter=read(ROOT/'shared_wallet_repair/manifest.json')
    assert all(sha(ROOT/k)==v for k,v in adapter['files'].items())
    tests=(ROOT/'reports/pas_economic_allocation_20261022/combined_tests.txt').read_text()
    joint=read(ROOT/'reports/joint_wallet_20261022/audit.json')
    preflight=read(ROOT/'reports/joint_wallet_20261022/public_preflight.json')
    staged=read(OUT/'candidate_frame.json')
    decision=dict(created_utc=datetime.now(timezone.utc).isoformat(),
       research_candidate='budget_lifecycle',selection='post-hoc diagnostic discovery, openly disclosed; not original primary winner',
       same_full_universe_pnl_Q_pass=passes(broad_idx,'budget_lifecycle'),
       actual_ETH_excluded_pnl_Q_pass=passes(scoped_idx,'candidate'),
       independent_test=False,history_reused=True,
       pas_execution_fidelity_pass=False,live_activation_allowed=False,
       research_nomination=True,joint_public_ledger_audit_pass=True,
       joint_public_preflight_pass=preflight['preflight_pass'],
       production_service_integrated=False,
       latest_candidate_frame_source=staged['source_time'],
       ownership_requirement='User confirmed joint same-market virtual ownership, 50000 USDP per strategy. ETH exclusion is a sensitivity test, not the required deployment scope.',
       reason='Full-universe historical improvement is real in the saved replay. Joint ownership needs a verified execution adapter. PAS candle replay does not reconstruct actual fills. Do not call this an overfit-free independent improvement.',
       source_hash_violations=violations,live_original_files=active['files'],operational_adapter_unchanged=True,
       current_turn_reference_families=len(FAMILIES),comparison_rows=len(allrows),rows_include_reused_controls=True,
       unique_reference_execution_files=sum(len(list((ROOT/'reports'/f'{name}_20261022'/'reference').glob('*.json.gz'))) for name in FAMILIES),
       tests_passed=tests.strip().endswith('OK'),orders_sent_by_research=0)
    save(OUT/'decision.json',decision);save(OUT/'family_checks.json',families)
    assert decision['same_full_universe_pnl_Q_pass'] and decision['tests_passed']
    series=[]
    for name,color,file in [('Incumbent','#64748b',ROOT/'reports/pas_covariance_v2_20261021/reference/incumbent_continuous_base.json.gz'),
        ('Budget + economic lifecycle (ETH allowed)','#0f766e',ROOT/'reports/pas_economic_lifecycle_20261022/reference/budget_lifecycle_continuous_base.json.gz')]:
        r=gz(file);eq=np.array([s['equity']+50000 for s in r['snapshots']])
        series.append(dict(name=name,color=color,time=[s['time'] for s in r['snapshots']],
            pnl=(eq-100000).tolist(),drawdown=((eq/np.maximum.accumulate(eq)-1)*100).tolist()))
    save(OUT/'plot_data.json',series)
    # Matplotlib is supplied by the system Python, not the trading virtualenv.
    import os,subprocess
    subprocess.run(['/usr/bin/python3',str(Path(__file__).with_name('plot.py')),str(OUT)],
        env={**os.environ,'MPLCONFIGDIR':'/tmp/icaif-matplotlib'},check=True)
    body=['<h1>전략 교체 검토 — 구현·개선 결과와 실제 적용 제약</h1>',
      '<p><b>ETH를 포함한 동일 조건에서는 PnL과 Q가 개선된 후보를 확인했다. 사용자가 동일 종목 공동 운용을 확인했으므로, ETH 제외는 필수 조건이 아니라 민감도 진단이다. 실거래 교체는 아직 하지 않았다.</b></p>',
      '<p>결과를 보고 발견한 연구 후보다. 이전13개 방향과 이번 비교를 포함해 같은 과거 기간을 반복 사용했다. 미래 정답을 모델 입력에 넣지는 않았지만, 이 결과를 독립 테스트·과적합 없음으로 표현할 수 없다.</p>',
      '<p>전체:2026-08-01 12:00~10-15 12:00 UTC. 최근:09-01 12:00~10-15 12:00 UTC. 전략50,000 + 동일 비교용 준비금50,000. 표의 수익률 분모100,000. σ는 일별 표본표준편차×√완료일수다.</p>',
      '<h2>동일 전체 종목 조건 — 발견한 개선 후보</h2>',
      table(['구간','비용','전략','순손익','수익률%','기간σ%','Q','0.8Q','MDD%','평균 명목금액'],[
        [r['phase'],r['scenario'],r['arm'],f"{r['net_pnl']:,.2f}",f"{100*r['return_fraction']:.6f}",f"{100*r['volatility']:.6f}",f"{r['Q']:.6f}",f"{r['score']:.6f}",f"{100*r['max_drawdown']:.4f}",f"{r['mean_notional']:,.2f}"] for r in selected]),
      '<img src="comparison.png" style="max-width:100%">',
      '<h2>팀원 ETH 제외 — 모든 비교 전략에 동일하게 적용</h2>',
      table(['구간','비용','전략','순손익','Q','기간σ%'],[
        [r['phase'],r['scenario'],r['arm'],f"{r['net_pnl']:,.2f}",f"{r['Q']:.6f}",f"{100*r['volatility']:.6f}"]
        for r in scoped if r['phase'] in ['continuous','comparison'] and r['arm'] in ['incumbent','candidate']]),
      '<h2>후보의 구현</h2><ol><li>기존 추세·7일 되돌림 및 페어를 유지한다. 위험 감축은 최종 주문 한도까지 보존한다.</li><li>원래0.1%였던 기본 일일 위험 목표를0.2%로 바꾸되, 실제 총노출·종목별·0.5%일일 위험 상한은 그대로 둔다. 수익 증가를 예측력 개선으로 포장하지 않는다.</li><li>전 종목의168시간 가격 변화와 펀딩을 같은 규칙으로 계산한다. 비용·리스크를 고려한 추가 배분은 보호된 위험 상한을 넘을 수 없다.</li><li>추가 포지션은 같은 방향의 순수익 근거가 유지되면 실제 취득 수량을 유지한다. 불필요한6시간 수량 재조정을 줄이고, 반대 방향으로 바꾸기 전 기존 수량을 먼저 청산한다.</li><li>독립 체결·미체결·편측 페어·유료 감축·수수료·펀딩과 장부 대사를 유지했다. 연구 코드는 실제 주문을 전송하지 않는다.</li></ol>',
      '<h2>학습·선택 및 한계</h2><p>학습:3/18~6/1 이전. 학습 정답 성숙 시각도6/1 이전. 보정:6월 중 성숙한 정답만 사용. 위험 회피 계수는7/1~7/29 비용 포함 결과로 선택. 이후8~10월 구간을 평가했다. 모델 학습은 비중첩 주간 시점6개×10종목=60행으로 표본이 작고, 교차 종목 행도 독립 표본이 아니다. 보정 계수는 약0.0563이며 새 가격 모델의 강한 예측력을 확인했다고 말할 수 없다.</p>',
      '<p>모형은 Stanford의 단일 기간 비용·위험 최적화 틀을 참고한 독립 구현이다. 논문의 실증 수익률을 재현한 것이 아니다. 장기 추세 진단도 AQR 연구의12개월 규칙을 그대로 복제하지 않았다.</p>',
      '<h2>전체 실험 공개</h2>',table(['방향','사전 지정 주후보','결과 행수','주후보 전조건 통과'],[[f['name'],f['nominee'],f['rows'],f['promotion_pass']] for f in families]),
      '<p>budget_lifecycle은 원래 주후보가 아니라 진단 대조군에서 발견했다. 이 사후 선택 사실을 유지하며, 반복한 후보 수를 독립 검증 횟수로 세지 않는다.</p>',
      '<p>동일 기간의 차이를 재표본화한 구간에도0이 포함된다. 선택 편향을 제거한 통계적 우월성이 증명된 것은 아니다. 실제 거래 종목은 BTC·ETH·HYPE·BRENT였다.10종목 양방향 가능성은 별도 구현 테스트로 확인했으며,10종목 모두 실제 거래했다는 뜻은 아니다.</p>',
      '<h2>동일 종목 공동 장부</h2><p>사용자가 동일 종목도 각 전략 장부로 공동 운용하도록 확인했다. 각50,000 USDP를 유지한다. 우리 목표 수량과 현재 우리 소유 수량의 차이만 주문하고, 거래소 순포지션으로 reduce-only 여부를 판단한다. 가상 청산이 거래소 순숏을 늘리는 경우에도 팀원의 가상 숏 수량은 바뀌지 않는다.</p>',
      '<p>joint_wallet/book.py에 실제 체결·수수료·펀딩 재구성, dispatch.py에 전송 전 소유자 기록·부분 체결·중복 방지·불명확한 결과 중단, pas_gateway.py에 PAS 공개 사전 검사를 구현했다. 서명자나 키를 읽지 않았다. 기존 실행 서비스에는 아직 연결하지 않았다. 팀원 프로그램도 가상 소유 수량을 기준으로 주문해야 하며, 서로 거래소 절대 순포지션을 덮어쓰는 독립 봇 두 개를 안전하다고 가정하지 않는다.</p>',
      '<p>실제 공개 체결 '+str(joint['fill_count'])+'건과 펀딩 '+str(joint['funding_count'])+'건을 재구성했다. 수량 오차0, 장부 내부 평가액 오차 '+str(joint['conservation_errors']['equity'])+' USDP. 비동시 거래소 잔고 조회와 평가액 차이 '+f"{abs(joint['equity']-joint['ledger_equity']):.6f}"+' USDP. 예시 ETH매수의 공개 사전 검사도 통과했다. 예시는 모델 매매 신호가 아니며 주문은0건이다.</p>',
      '<p>상쇄되어 실제 순수량0인 시장은 실제 지급액0만으로 전략별 펀딩을 알아낼 수 없다. 당시 관측한 오라클과 확정 펀딩률이 없으면 임의로0을 배분하지 않고 오류로 처리한다. 전체 계좌 단위의 레버리지 설정도 한쪽 전략이 임의로 변경하지 않는다.</p>',
      '<p>최신 기준시장 입력으로 고정 후보의 신호를 다시 생성했다. 재학습0회, 기존5,184시점 위험 보정값의 재현 오차0. 다만 이 신호의 기준시장 펀딩을 PAS 펀딩으로 간주하면 안 된다. 현재 후보의 PAS 가격·펀딩 연결과 서비스 교체는 완료하지 않았으므로 본대회에서 개선 성과가 확인됐다고 표현하지 않는다.</p>',
      '<h2>PAS 검증</h2><p>10/19 시작~10/21 13:00 UTC 분봉·실제 펀딩으로 별도 비교했다. 거래소 과거 호가·오라클·마크가 없는 분봉 모형은 실제 첫 체결을 약12시간 늦게 재현했다. 조건부 PnL/Q를 공식 PAS 성과나 실거래 적합성 증거로 사용할 수 없다. 새 후보가 PAS에서 더 잘 벌 것이라는 결론은 아직 내리지 않았다.</p>',
      '<p>현재 상위 지갑을 사후 선정해 과거에 따라 샀다고 계산한 결과는 포함하지 않았다. 관측 시각 이후의 가격·지연·체결량을 적용할 수 있는 사전 선정 기록이 필요하다.</p>',
      '<h2>검증 및 파일</h2><pre>'+html.escape(tests)+'</pre>',
      '<p><a href="all_results.csv">모든 결과CSV</a> · <a href="decision.json">활성화 판단</a> · <a href="uncertainty.json">선택 편향을 제거하지 않은 블록 재표본화</a></p>',
      '<p><a href="https://web.stanford.edu/~boyd/papers/cvx_portfolio.html">최적화 원문</a> · <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding">펀딩 공식 문서</a> · <a href="https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum">추세 연구</a></p>']
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font-family:sans-serif;max-width:1250px;margin:40px auto;line-height:1.65;color:#182230}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border:1px solid #d1d5db;text-align:right}th{background:#eff6ff}pre{white-space:pre-wrap}</style>'+''.join(body))
    save(OUT/'artifact_hashes.json',{str(f.relative_to(ROOT)):sha(f) for f in OUT.iterdir() if f.is_file() and f.name!='artifact_hashes.json'})
    print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
