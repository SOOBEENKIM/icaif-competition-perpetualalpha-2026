"""Post-test accounting/report only. Never fit, replay, or select a strategy."""
import csv
import gzip
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fresh_test_2022.data import OUT, ROOT, read, save, sha
from fresh_test_2022.study import verify
from auto_pair_selection.study import verify as verify_live


def stamp(t):
    return datetime.fromtimestamp(int(t), timezone.utc).isoformat()


def table(headers, rows):
    def cell(v): return html.escape(str(v))
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+cell(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+cell(c)+'</td>' for c in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def main():
    verify(); verify_live()
    protocol = read(OUT/'protocol.json')
    rows = sorted(read(OUT/'results.json'), key=lambda r: (r['scenario'], r['arm']))
    inputs = dict(np.load(OUT/'raw_inputs.npz'))
    times = inputs['market_times']
    audit = {'at_utc': datetime.now(timezone.utc).isoformat(), 'frozen_sources_unchanged': True,
             'live_manifest_unchanged': True, 'orders_sent': 0, 'post_result_strategy_changes': 0,
             'full_ten_asset_validation': False, 'prospective_validation': False,
             'scope': 'Newly acquired, previously unexposed in retained project records, retrospective BTC/ETH external component test; NOT a full-strategy or chronological post-development holdout.'}
    receipts = read(OUT/'receipts.json')
    assert len(receipts) == 64
    for row in receipts:
        path = OUT/'raw'/Path(row['url']).name
        assert sha(path) == row['sha256'] == path.with_suffix('.zip.CHECKSUM').read_text().split()[0]
    audit['official_archives_reverified'] = len(receipts)
    for name, digest in read(OUT/'evaluation_inputs.json').items():
        assert sha(OUT/name) == digest
    download = read(OUT/'download_freeze.json')
    assert sha(OUT/'collector_before_timestamp_normalization.txt') == download['collector_sha256']
    assert sha(OUT/'protocol.json') == download['protocol_sha256']
    assert protocol['created_utc'] < download['at_utc'] < min(r['received_utc'] for r in receipts)
    assert max(r['received_utc'] for r in receipts) < read(OUT/'evaluation_started.json')['time']
    audit['freeze_download_evaluation_order_pass'] = True
    audit['timestamp_normalization'] = read(OUT/'schema_adjustment.json')
    audit['adapter_parity'] = read(OUT/'adapter_parity.json')
    log = (OUT/'tests.log').read_text()
    assert 'Ran 3 tests' in log and log.strip().endswith('OK')
    audit['regression_tests'] = {'passed': 3, 'log': 'tests.log'}
    fit_summary = {}
    for arm in ['candidate72', 'incumbent24']:
        fits = read(OUT/f'{arm}_fit_audit.json')
        for rec in fits:
            if arm == 'candidate72':
                if rec['model'] is not None:
                    assert times[rec['model']['last_price_index']] <= rec['source']
            else:
                for key in ['direct', 'gate']:
                    if rec[key] is not None:
                        assert rec[key]['last_maturity'] < rec['source']
                        assert rec[key]['fit_time'] == rec['source']
        fit_summary[arm] = {'scheduled_fits': len(fits),
                           'no_admissible_model_days': sum(x['model'] is None for x in fits) if arm == 'candidate72' else 0,
                           'gate_solver_failures': sum(x['failure'] is not None for x in fits) if arm == 'incumbent24' else 0,
                           'past_only_inputs_pass': True}
    audit['model_fit_audit'] = fit_summary
    curves, quarterly, decompositions, checks = {}, [], [], []
    bounds = [protocol['test_start']] + [int(datetime(2022,m,1,12,tzinfo=timezone.utc).timestamp()) for m in [4,7,10]] + [protocol['test_end']]
    for row in rows:
        name = f"{row['arm']}_{row['scenario']}"
        run = json.load(gzip.open(OUT/'runs'/f'{name}.json.gz', 'rt'))
        s = run['snapshots']; t = np.array([x['time'] for x in s]); e = np.array([x['equity']+50000 for x in s])
        assert t[0] == protocol['test_start'] and t[-1] == protocol['test_end']
        d = e[t%86400 == 43200]; ret = d[1:]/d[:-1]-1
        assert len(ret) == 364 and abs(d[0]-100000) < 1e-9
        R = float(np.prod(1+ret)-1); sigma = float(np.std(ret, ddof=1)*np.sqrt(len(ret)))
        Q = (max(R,0)+.005)/(abs(R)+1.5*sigma+.05)
        for actual, expected in [(R,row['return_fraction']),(sigma,row['volatility']),(Q,row['Q']),(.8*Q,row['score']),(e[-1]-e[0],row['net_pnl'])]:
            assert abs(actual-expected) < 1e-8
        # Independently sum settlements using the position held immediately
        # before each funding timestamp, with the predeclared stress haircut.
        fund = 0.
        for previous, current in zip(s,s[1:]):
            i = int(np.searchsorted(times,current['time']))
            cash = -np.array(previous['qty'])*inputs['close'][i]*inputs['funding'][i]
            if row['scenario'] == 'stress': cash = np.where(cash>0,cash*.5,cash)
            fund += float(cash.sum())
        assert abs(fund-row['funding']) < 1e-8
        m = run['metrics']
        assert abs(m['price_move_pnl']-m['execution_cost']-m['fees']+fund-row['net_pnl']) < 1e-7
        assert all(f['source_available'] <= f['decision_time'] < f['time'] for f in run['fills'])
        curves[name] = {'time':t,'equity':e,'run':run}
        decompositions.append(dict(arm=row['arm'],scenario=row['scenario'],price_move=m['price_move_pnl'],
                                   fees=m['fees'],execution_cost=m['execution_cost'],funding=fund,
                                   net_pnl=row['net_pnl'],terminal_liquidation_reserve=m['liquidation_cost_estimate']))
        checks.append(dict(arm=row['arm'],scenario=row['scenario'],independent_score_pass=True,
                           independent_funding_pass=True,cash_identity_pass=True,timing_pass=True,
                           risk_breaches=run['risk_breaches'],max_accounting_error=row['accounting_error']))
        for q,(lo,hi) in enumerate(zip(bounds,bounds[1:]),1):
            mask=(t>=lo)&(t<=hi)&(t%86400==43200); ev=e[mask]; rr=ev[1:]/ev[:-1]-1
            sig=float(np.std(rr,ddof=1)*np.sqrt(len(rr))); r=float(ev[-1]/ev[0]-1)
            quarterly.append(dict(arm=row['arm'],scenario=row['scenario'],quarter=q,start=stamp(lo),end=stamp(hi),
                                  pnl=float(ev[-1]-ev[0]),return_fraction=r,volatility=sig,
                                  Q=(max(r,0)+.005)/(abs(r)+1.5*sig+.05),days=len(rr)))
        assert abs(sum(x['pnl'] for x in quarterly if x['arm']==row['arm'] and x['scenario']==row['scenario'])-row['net_pnl']) < 1e-8
    audit['runs'] = checks
    audit['improvement_pass'] = all(
        (a:=next(r for r in rows if r['arm']=='candidate72' and r['scenario']==sc))['net_pnl'] > 0
        and a['net_pnl'] > (b:=next(r for r in rows if r['arm']=='incumbent24' and r['scenario']==sc))['net_pnl']
        and a['Q'] > b['Q'] and a['risk_breach_bars']==0 for sc in ['base','stress'])
    save(OUT/'final_audit.json',audit);save(OUT/'quarterly.json',quarterly);save(OUT/'pnl_decomposition.json',decompositions)
    with (OUT/'comparison.csv').open('w') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'daily_equity.csv').open('w') as fp:
        writer=csv.writer(fp);names=sorted(curves);writer.writerow(['time_utc']+names)
        first=curves[names[0]]
        for i,t in enumerate(first['time']):
            if t%86400==43200:writer.writerow([stamp(t)]+[curves[n]['equity'][i] for n in names])

    # The system plotting runtime already contains matplotlib. Keep the live
    # trading venv unchanged; this subprocess reads the exported CSV only.
    import subprocess
    subprocess.run(['python3','-c','''
import csv,sys
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1])
rows=list(csv.DictReader((p/'daily_equity.csv').open()))
dates=[datetime.fromisoformat(r['time_utc']) for r in rows]
fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,constrained_layout=True)
for ax,scenario in zip(axes,['base','stress']):
    for arm,label,color in [('incumbent24','Incumbent pair component','#4265a8'),('candidate72','Candidate pair component','#c65038')]:
        ax.plot(dates,[float(r[arm+'_'+scenario])-100000 for r in rows],label=label,color=color,linewidth=1)
    ax.axhline(0,color='#aaa',linewidth=.7);ax.set_title(scenario.capitalize());ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
axes[0].set_ylabel('Cumulative net PnL (quote units), daily 12:00 UTC');axes[1].legend(fontsize=8)
fig.suptitle('Unused 2022 historical test | BTC/ETH component only | 100k total / 50k sleeve')
fig.savefig(p/'equity.png',dpi=150);plt.close(fig)
''',str(OUT)],check=True)
    names={'candidate72':'개선 후보 · 72시간','incumbent24':'기존 페어 모듈 · 24시간'}
    conditions={'base':'기본 비용','stress':'불리한 비용·체결'}
    main_table=table(['조건','전략','순손익','수익률 R','기간 변동성 σ','Q','Score (J=0)','완료 페어','최대낙폭'],[
        [conditions[r['scenario']],names[r['arm']],f"{r['net_pnl']:+,.2f}",f"{r['return_fraction']*100:+.4f}%",f"{r['volatility']*100:.4f}%",f"{r['Q']:.6f}",f"{r['score']:.6f}",r['completed_pairs'],f"{r['max_drawdown']*100:.4f}%"] for r in rows])
    cost_table=table(['조건','전략','가격 변화 손익','수수료 차감','체결 비용 차감','펀딩 가감','순손익'],[
        [conditions[r['scenario']],names[r['arm']]]+[f"{r[k]:+,.2f}" for k in ['price_move','fees','execution_cost','funding','net_pnl']] for r in decompositions])
    quarters_table=table(['조건','전략','분기 구간','일수','구간 순손익','구간 수익률','구간 Q'],[
        [conditions[r['scenario']],names[r['arm']],f"Q{r['quarter']}",r['days'],f"{r['pnl']:+,.2f}",f"{r['return_fraction']*100:+.4f}%",f"{r['Q']:.6f}"] for r in quarterly])
    report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SFINX · 새 2022 외부 테스트</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:35px auto;padding:0 22px;line-height:1.7;color:#1c2738;background:#f9fafc}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}.verdict{{padding:18px;background:#fff0ed;border-left:5px solid #bf4935}}.scope{{padding:16px;background:#edf2fa}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}td,th{{border:1px solid #dce2eb;padding:8px;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}th{{background:#edf2fa}}.scroll{{overflow:auto}}img{{width:100%}}code{{overflow-wrap:anywhere}}a{{color:#275b9a}}</style>
<h1>새로 확보한 미사용 2022년 자료의 검증 결과</h1>
<p class="verdict"><strong>판정: 개선 재현 실패.</strong> 기본 조건과 불리한 조건 모두 개선 후보의 순손익과 Q가 기존 페어 모듈보다 낮았습니다. 이 결과를 근거로 후보를 실거래에 교체할 수 없습니다. 테스트 이후 전략 규칙·파라미터를 바꾸거나 유리한 결과를 다시 고르지 않았습니다.</p>
<p class="scope"><strong>범위: BTC·ETH 페어 모듈의 외부 과거 구간 시험.</strong> 현재 실거래 전략 전체, 10개 종목·45개 페어 전체, PAS 시장 성과를 검증한 결과가 아닙니다. 2026년 개발 후 2022년으로 돌아간 회고적 검증으로, 개발 시점 이후의 전향적 테스트라고 부르지 않습니다. 보관된 프로젝트 기록에서 2022년 사용 흔적을 찾지 못했으며, 그 이상의 완전한 미노출을 보증하지 않습니다.</p>
<h2>비교 결과</h2><p>고정 테스트: <strong>2022-01-01 12:00 UTC ~ 2022-12-31 12:00 UTC</strong> (한국시간 각 21:00, 완료 일별 수익률 364개). 가상 총자본 100,000 중 전략에 50,000, 나머지 50,000은 현금으로 유지했습니다. 동일 페어 진입 명목 한도 6,000과 위험 한도를 적용했습니다. 수익률·σ·Q는 총자본 기준입니다. 손익 단위는 외부 선물 USDT 가격에 기초한 모의 달러 환산 단위이며, 실제 USDP 수익이 아닙니다.</p>
{main_table}<p>σ는 일별 수익률의 표본 표준편차 × √364입니다. Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), Score=0.8Q (J=0). 계좌 원금이나 다른 참가자의 수익을 섞지 않았습니다. 연간 σ와 Q를 기간이 짧은 현재 대회 리더보드 수치에 직접 비교해서는 안 됩니다.</p>
<img src="equity.png" alt="2022년 새 테스트의 기본·불리한 조건 누적 손익 곡선">
<p>기본 후보는 테스트 종료 시 명목 5,981.65의 잔여 포지션이 있습니다. 종료가 평가손익 −2,044.98이며, 청산 비용 추정 11.37을 추가 차감하면 −2,056.35입니다. 추정 청산 비용은 실제 청산 체결로 표시하지 않습니다. 나머지 세 실행은 종료 시 포지션이 없습니다.</p>
<h2>학습과 테스트를 분리한 방법</h2><ol>
<li>기존 사용 기록을 먼저 확인했습니다. 이전 연구에서 사용한 2018~2021·2023~2026 구간을 새로운 테스트라고 재사용하지 않았습니다. 이번 테스트는 기록상 미사용인 2022년으로, 시세 다운로드 전에 기간·후보·비교군·비용·판정 기준을 고정했습니다.</li>
<li>초기 상태 준비에는 2021년 9~12월 자료를 사용했습니다. 기존 페어 모듈은 직전 90일의 만기가 끝난 24시간 정답만 6시간 간격으로 사용해 릿지 회귀와 기존 의사결정 게이트를 재학습했습니다. 후보는 직전 60일 가격으로 헤지 비율·AR(1) 회귀를 추정하고 72시간 변화를 계산했습니다.</li>
<li>두 알고리즘 모두 사전에 정해진 하루 한 번의 갱신을 유지했습니다. 테스트가 진행되면 앞선 테스트 날짜의 이미 끝난 관측은 이후 날짜의 학습에 들어갑니다. 이는 고정된 온라인 갱신 방식의 순차 시험이며, 2022년 전체 정답을 한번에 학습한 정적 테스트가 아닙니다. 결과를 본 뒤의 파라미터 선택·후보 교체는 하지 않았습니다.</li>
<li>공개 사전학습 모델이 2022년을 보았을 수 있으므로 Chronos 등 외부 사전학습 가중치는 사용하지 않았습니다. 비교 대상은 기존 전략의 BTC·ETH 페어 모듈입니다. 추세·되돌림 기본 포지션이나 나머지 종목을 합친 기존 실거래 전략 전체를 재현한 것이 아닙니다.</li>
<li>2022년에는 이번 후보의 전체 10개 종목 데이터를 동일 시장에서 확보할 수 없습니다. 없는 종목을 합성하거나 가격을 채우지 않았습니다. 페어가 하나일 때 여러 페어 간 일관성 보정은 항등 연산이므로, 이번 시험으로 45개 페어 선별·분산 효과의 좋고 나쁨까지 결론내릴 수 없습니다.</li></ol>
<h2>시세·펀딩·체결 검증</h2><p><a href="https://github.com/binance/binance-public-data">Binance 공식 공개 데이터</a>에서 2021년 9월~2022년 12월 BTCUSDT·ETHUSDT 시간봉과 펀딩 ZIP 64개를 새로 받았습니다. 모두 공식 SHA-256 체크섬과 일치합니다. 종목별 시간봉 11,688개이며, 내부 누락 시간봉·예정 펀딩 결제 누락은 0입니다. 원본과 수신 시각은 <a href="receipts.json">수집 영수증</a>에 남겼습니다.</p>
<p>펀딩은 실제 8시간 결제 시각에만 반영했습니다. 당시 보유 수량과 종료가를 결제 평가가격 대용으로 사용했습니다. 향후 펀딩 예상에는 직전까지 확인된 펀딩만 사용했습니다. 공식 calc_time의 0~47ms 지연은 정각 결제 시각으로 정규화했고 원래 밀리초 시각을 보관했습니다. 이 파서 수정은 성과 계산 전에 수행했으며 <a href="schema_adjustment.json">수정 이유</a>와 이전 코드 해시가 보존돼 있습니다.</p>
<p>기본 조건은 편도 수수료 9bp·체결 비용 10bp·참여율 1%, 불리한 조건은 12bp·20bp·참여율 0.25%·받는 펀딩 50%입니다. 이는 사전 고정한 보수적 비교 비용이며 Binance 실제 계정 수수료라고 주장하지 않습니다. 판단 후 다음 시간봉에서 각 다리를 따로 체결하고, 미체결·한쪽 체결 후 보완 및 취소·유료 청산을 유지했습니다. 과거 L2 호가와 실제 주문 대기열이 없어 체결 가격·수량은 조건부 시뮬레이션입니다.</p>
<h2>손익 원인과 위험 기록</h2>{cost_table}
<p>기본 후보는 비용 차감 전 가격 변화 손익도 −693.90입니다. 수수료 660.20과 체결 비용 750.19, 펀딩 수취 59.31을 합쳐 −2,044.98이 됩니다. 따라서 수수료만 줄이면 이 구간의 수익성이 검증된다는 결론은 성립하지 않습니다. 후보는 기본 조건 평균 명목 4,380.80, 기존 모듈은 1,337.12로 실제 노출도 다릅니다. 같은 상한을 적용했지만 동일 실현 위험으로 맞춘 비교는 아닙니다.</p>
<p>후보의 기본 실행에서 2022-11-10 14:00 UTC 추정 일별 위험이 설정 한도 0.5000%를 넘어 0.5033%가 된 시간봉이 1개 있었습니다. 다음 시간봉인 15:00 UTC 두 포지션을 청산했고, 16:00에는 줄인 수량으로 재진입했습니다. 위험 한도 무위반이라고 표시하지 않습니다. 나머지 실행의 기록된 한도 위반은 0입니다. 이는 시간봉 평가이며 봉 중간의 위험까지 검증하지 않습니다.</p>
<p>불리한 체결 조건에서는 거래 자체가 거절되거나 줄어듭니다. 그래서 기존 모듈의 불리한 조건 손실이 기본 조건보다 작을 수 있습니다. 같은 거래에 비용만 추가한 고정 거래 목록 비교가 아니라, 비용에 반응하는 동일 알고리즘을 다시 실행한 것입니다.</p>
<h2>모든 분기 구간</h2><p>위 연간 실행의 연속 계좌 곡선을 나눈 표입니다. 분기마다 포지션·자본을 초기화하지 않았고 유리한 분기를 선별하지 않았습니다. 경계는 1월 1일·4월 1일·7월 1일·10월 1일·12월 31일의 12:00 UTC입니다.</p>{quarters_table}
<h2>재현 자료와 최종 판정</h2><p>기존에 공개된 2026년 입력에서 두 종목용 실행 어댑터가 원본 10차원 실행자의 해당 페어 예측·체결·손익 경로와 일치함을 테스트 전에 확인했습니다. 미래 가격을 바꿔도 과거 예측·학습 계수가 변하지 않는 검사 등 3개 테스트가 통과했습니다. 본 실행 4개 모두 시간 순서·손익 항등식·펀딩 재계산·점수 재계산을 통과했고 최대 회계 오차는 3.06×10⁻¹⁰입니다. 이 검사 통과는 수익성 보증이 아닙니다.</p>
<p><strong>사전 판정 기준은 통과하지 못했습니다.</strong> 이전 재사용 구간의 개선은 이번 미사용 과거 구간에서 일반화되지 않았습니다. 또한 후보 기본 조건에서 위험 초과 기록도 있습니다. 전체 PAS 실거래 전략의 우열은 이 두 종목 결과만으로 단정할 수 없지만, 새 데이터에서 개선이 입증됐다고 말할 근거는 없습니다. 현재 실거래 설정·보유 포지션은 이 실험에서 변경하지 않았습니다.</p>
<p>이제 2022년 결과도 관찰했으므로 이후 이 결과를 보며 수정한다면 2022년은 개발·진단 구간으로 분류해야 합니다. 다음 수정판의 독립 성과라고 이 구간을 다시 제시하지 않습니다.</p>
<ul><li><a href="protocol.json">다운로드 전 고정한 시험 계획</a> · <a href="frozen_strategy.json">원본 전략 해시</a> · <a href="adapter_freeze.json">시험 어댑터 해시</a></li>
<li><a href="comparison.csv">전체 비교 CSV</a> · <a href="results.json">실행 결과 JSON</a> · <a href="daily_equity.csv">매일 12:00 UTC 계좌 가치</a> · <a href="quarterly.json">모든 분기 결과</a></li>
<li><a href="final_audit.json">최종 검증</a> · <a href="tests.log">테스트 로그</a> · <a href="data_audit.json">데이터 검증</a> · <a href="adapter_parity.json">원본 실행자 동등성</a></li>
<li><code>fresh_test_2022/</code>: 고정 어댑터·학습·실행 코드. <code>runs/*.json.gz</code>: 네 실행의 모든 주문·체결·계좌·위험 기록. <code>*_fit_audit.json</code>: 각 날짜 학습 시점·계수·사용 정답 만기.</li></ul>
</html>'''
    (OUT/'report.html').write_text(report)
    assert not audit['improvement_pass']
    verify();verify_live()
    save(OUT/'report_manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
         'role':'Post-test report only; no strategy edits or reruns',
         'files':{str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='report_manifest.json'}})
    print(json.dumps({'report':str(OUT/'report.html'),'improvement_pass':False,'rows':len(rows),'independent_cash_score_audit':'passed','live_unchanged':True}))


if __name__=='__main__':main()
