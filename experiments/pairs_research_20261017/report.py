"""Read completed research artifacts only; never accesses reserved forward tapes."""
import collections,gzip,html,json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from .research import HERE,ROOT,save,sha
from experiments.cost_horizon_20261016.research import data
from experiments.complementary_alpha_20261017.research import protected

DEST=ROOT/'reports/pairs_research_20261017'

def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)

def attribution(run,m):
    impact=sum(f['signed_qty']*(f['price']-m.close[m.exact(f['time']),f['asset']]) for f in run['fills'])
    metric=run['metrics'];events=run.get('events',[])
    result=dict(net_pnl=metric['net_pnl'],fee=metric['fees'],funding=metric['funding'],
        execution_price_penalty=float(impact),fixed_fills_before_fees_impact_funding=float(metric['net_pnl']+metric['fees']-metric['funding']+impact),
        entry_intents=sum(e['event']=='entry_intent' for e in events),
        partial_entry_aborts=sum(e['event']=='partial_entry_abort' for e in events),
        exit_reasons=dict(collections.Counter(e.get('reason') for e in events if e['event']=='exit_intent')),
        final_flat=not any(metric['final_qty']))
    pairs=[]
    for e in events:
        if e['event']!='entry_intent':continue
        end=next((x['time'] for x in events if x['event']=='flat_confirmed' and x['key']==e['key'] and x['time']>e['time']),run['end'])
        symbols=e['key'].removeprefix('ref:').split('/')[-2:]
        fills=[f for f in run['fills'] if f['symbol'] in symbols and e['time']<f['time']<=end]
        funding=0.
        ids=[list(m.symbols).index(s) for s in symbols]
        for f in run['funding_events']:
            if e['time']<f['time']<=end:
                q=np.array(f['qty'])[ids];p=np.array(f['prices'])[ids];rates=np.array(f['rates'])[ids]
                signed=-float(q*p@rates)
                total_gross=float(abs(np.array(f['qty']))@np.array(f['prices']))
                adverse=f['adverse_amount']*float(abs(q)@p)/total_gross if total_gross else 0.
                funding+=signed+adverse
        pnl=sum(f['realized']-f['fee'] for f in fills)+funding
        pairs.append(dict(pair=e['key'],entry=e['time'],exit=end,z=e.get('z'),net_pnl=pnl,
            partial_abort=any(x['event']=='partial_entry_abort' and x['key']==e['key'] and e['time']<x['time']<=end for x in events)))
    result['pair_trades']=pairs
    if result['final_flat'] and pairs:
        result['pair_attribution_error']=sum(x['net_pnl'] for x in pairs)-metric['net_pnl']
        assert abs(result['pair_attribution_error'])<1e-6,result
    return result

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    primary=json.loads((HERE/'artifacts/results.json').read_text());partial=json.loads((HERE/'partial_artifacts/results.json').read_text())
    literature=json.loads((HERE/'literature.json').read_text());probe=json.loads((HERE/'public_probe/summary.json').read_text())
    _,_,_,m=data();selected=partial['selection']['selected']
    baseline=read(HERE/'artifacts/runs/comparison-second-base.json.gz')
    old=read(ROOT/'experiments/q_execution_20261017/artifacts/runs/later-point_x1-base.json.gz')
    assert baseline['fills']==old['fills']
    np.testing.assert_array_equal([s['equity'] for s in baseline['snapshots']],[s['equity'] for s in old['snapshots']])
    runs={'기존 1차':read(HERE/'artifacts/runs/comparison-first-base.json.gz'),'현재 2차':baseline,
          '공적분 OU':read(HERE/'artifacts/runs/comparison-ou_1440-base.json.gz'),
          '영구·일시 변화 분해 + 최적정지':read(HERE/f'partial_artifacts/runs/comparison-{selected}-base.json.gz'),
          '현재 75% + 공적분 pair 25%':read(HERE/'artifacts/runs/comparison-second_plus_pairs_same_gross_budget-base.json.gz')}
    audits={name:attribution(run,m) for name,run in runs.items()}
    checks=protected();plan=json.loads((ROOT/'experiments/q_execution_20261017/artifacts/prospective/plan.json').read_text())
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
    for group in ('artifacts','partial_artifacts'):
        frozen=json.loads((HERE/group/'frozen.json').read_text())
        for p,h in frozen.get('sources',frozen.get('source',{})).items():assert sha(ROOT/p)==h,p
    tests=(HERE/'tests.log').read_text();assert 'Ran 8 tests' in tests and '\nOK' in tests
    verification=dict(protected=checks,reserved_q_dependencies_unchanged=len(plan['sources']),tests_passed=8,
        completed_backtests=len(primary['rows'])+len(partial['rows']),baseline_fills_and_equity_exactly_reproduced=True,
        all_accounting_audits_pass=True,all_time_and_risk_gates=all(all(r['gates'].values()) for r in primary['rows']+partial['rows']),
        excluded_failed_attempts=['attempt_00_reporting_key_error','attempt_01_baseline_scale_and_endpoint_errors'],
        fixes='Reporting field name, baseline-only scale scope, available evaluation endpoint. Previous attempts preserved. No signal thresholds modified in repair.',
        partial_followup_registered_after_primary_failure=True,historical_independence=False,live_orders_sent=0)
    save(DEST/'verification.json',verification);save(DEST/'attribution.json',audits)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,5));names=['First','Current second','Cointegration OU','Partial CI + stopping','75/25 signal blend']
        for label,run in zip(names,runs.values()):
            ax.plot([datetime.fromtimestamp(s['time'],timezone.utc) for s in run['snapshots']],
                    [s['equity']-1e6 for s in run['snapshots']],label=label,lw=1.5)
        ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Net PnL (reference backtest USD)');ax.legend(fontsize=8,ncol=2)
        ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout();fig.savefig(DEST/'pnl.png',dpi=160);plt.close(fig)
        chart='<img src="pnl.png" alt="Reference backtest net PnL curves">'
    except ImportError:chart=''
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    def pct(v):return f'{v*100:.6f}%'
    comparison=table(['전략','순손익','수익률','일별 STD','기간 σ','Q','Score(J=0)','평균 총노출'],
        [[name,f"{r['metrics']['net_pnl']:+,.2f}",pct(r['metrics']['roi']),pct(r['official']['daily_sample_std']),pct(r['official']['volatility_horizon']),f"{r['official']['financial_score']:.6f}",f"{r['official']['score']:.6f}",pct(r['metrics']['mean_gross'])] for name,r in runs.items()])
    selections=table(['후보','8월 순손익','실행 스트레스 순손익','통과'],
        [[x['name'],f"{x['pnl']:+,.2f}",f"{x['stress_pnl']:+,.2f}",x['eligible']] for x in primary['selection']['ranking']])
    selection2=table(['분해 모델 후보','8월 순손익','선정 조건 통과'],[[x['name'],f"{x['pnl']:+,.2f}",x['eligible']] for x in partial['selection']['ranking']])
    costs=table(['후보','체결 가격 변동 손익¹','실행가격 비용','수수료','펀딩 수입','최종 순손익','진입/부분체결 중단'],
        [[k,f"{a['fixed_fills_before_fees_impact_funding']:+,.2f}",f"{a['execution_price_penalty']:,.2f}",f"{a['fee']:,.2f}",f"{a['funding']:+,.2f}",f"{a['net_pnl']:+,.2f}",f"{a['entry_intents']}/{a['partial_entry_aborts']}"] for k,a in audits.items()])
    quotes=table(['par 종목','현재 호가/오라클 괴리','500 명목 왕복비용','10,000 명목 왕복비용','60,000 명목 왕복비용'],
        [[r['coin'],f"{r['mid_basis_bps']:.2f}bp",*[f"{s['instantaneous_roundtrip_cost_bps']:.2f}bp" if s['instantaneous_roundtrip_cost_bps'] is not None else '수량 부족' for s in r['sizes']]] for r in probe['rows'] if r['tradable']])
    sources=''.join(f'<li><a href="{html.escape(p["url"])}">{html.escape(p["title"])}</a> ({p["year"]})<br><b>검토 범위:</b> {html.escape(p["review"])}<br>{html.escape(p["finding"])}<br><b>적용:</b> {html.escape(p["application"])}</li>' for p in literature['papers'])
    at=datetime.fromtimestamp(probe['at'],timezone(timedelta(hours=9))).isoformat(timespec='seconds')
    allrows=primary['rows']+partial['rows']
    full=table(['단계','후보','비용 가정','순손익','수익률','Q','체결 조각 수'],[[r['phase'],r['policy'],r['scenario'],f"{r['metrics']['net_pnl']:+,.2f}",pct(r['metrics']['roi']),f"{r['official']['financial_score']:.6f}",r['metrics']['fills']] for r in allrows])
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Pairs trading 연구 및 비교</title>
    <style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;color:#17212e}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{border:1px solid #d5dde5;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f8}}.finding{{background:#fff1df;padding:20px;border-left:5px solid #d48b26}}li{{margin:18px 0}}img{{max-width:100%}}code{{background:#f1f4f7;padding:2px 4px}}details{{margin:20px 0}}</style>
    <h1>Pairs trading 연구·구현·비용 차감 비교</h1>
    <p class="finding"><b>이번 실험에서는 현재 2차 전략보다 PnL이 개선된 교체 후보를 확인하지 못했습니다.</b> 8개 공적분 기반 후보는 8월에 진입할 쌍이 없었고, 추가 4개 분해 모델 중 선정된 후보는 이후 기간에서 손실이 났습니다. 변동성이 낮아졌다는 이유로 채택하지 않았습니다. 실제 계좌와 기존 미래 실험은 변경하지 않았습니다.</p>
    <p>2024–2026 관련 문헌 16편과 기반 논문 1편을 조사했습니다. 본문에서 확인한 내용과 초록만 접근된 내용을 아래에서 구분합니다. 모든 논문을 망라한 체계적 문헌고찰이나 논문 전체의 정확한 재현이라고 주장하지 않습니다.</p>
    <h2>같은 기간 비교</h2><p><b>2026-09-01 21:00 ~ 10-13 21:00 KST, 42일, 초기자본 1,000,000.</b> 아래 값은 실제 지갑 실적이 아니라 기준시장 시간봉 백테스트입니다. 기존에 개발에 사용한 과거 자료이므로 독립 검증이 아닙니다.</p>{comparison}{chart}
    <p>일별 수익률은 12:00 UTC(21:00 KST) 자산가치 기준, STD는 표본 표준편차(ddof=1), σ=STD×√42입니다. Q와 Score는 현재 Practice 리더보드 공식을 적용한 참고치입니다. 본대회 확정 점수나 실제 순위가 아닙니다. 표의 평균 총노출 차이를 확인해야 하며, 동일 위험 상한이 동일 실현 위험을 뜻하지 않습니다.</p>
    <h2>구현과 실험 순서</h2><p>암호화폐 3개와 주가지수 3개 안에서 경제적으로 관련된 6개 쌍을 사전에 정했습니다. 21/60일 형성창, 주간 재추정, 시점 이전 관측만 사용했습니다. 공적분형은 Engle–Granger·BH 다중검정, 헤지 비율·반감기·안정성 필터를 사용합니다. OU, 유한시간 최적정지, signature 필터, reference-asset copula 네 계열을 구현했습니다.</p>
    <p>첫 연구의 무거래 원인은 8월의 다중검정 통과 쌍 부재였습니다. 원시 p≤0.1은 21일창에서 24회 중 4회, 60일창에서 24회 중 3회였지만 BH 보정 후는 각각 0회였습니다. 기준을 나중에 낮추지 않았습니다. 대신 별도 실험에서 random walk와 AR1을 나누는 부분 공적분 기반 모델을 구현했습니다. 이 추가 실험은 첫 실패를 본 뒤 만들었으며, 완전히 새로운 검증으로 포장하지 않습니다.</p>
    <p>분해 모델은 Kalman 최대우도 추정, 현재까지의 필터 상태, 평균회귀 분산 비중 및 random-walk AIC 대조를 사용합니다. 원문의 통계적 LRT 검정을 재현하지 않았으므로 ‘부분 공적분 유의성 확인’으로 해석할 수 없습니다. 최적정지는 72시간 단일 거래의 이산 근사이며 최신 논문의 연속시간 해법이나 PPO 구현 자체가 아닙니다.</p>
    <h2>체결 및 비용</h2><p>두 주문은 독립적으로 한 시간 뒤 체결합니다. 양쪽 종목의 실제 기준시장 거래량으로 수량을 제한하고, 한쪽만 체결되면 다음 시간부터 실제 비용을 지불하며 청산을 재시도합니다. 보유 중 수량·헤지·모형을 고정하고 실제 잔고가 0이 된 뒤 청산 완료로 기록합니다. 신호를 새로 계산했다고 가상의 체결이나 이익을 만들지 않습니다.</p>
    <p>기본 비용은 체결마다 taker 9bp + 가격 불리함 10bp, 실행 스트레스는 20bp와 참여율 0.25%, 펀딩 스트레스는 추가 1bp/시간입니다. 기준시장의 실제 부호 있는 펀딩을 사용했으나 par 과거 펀딩 재현은 아닙니다. 각 100을 롱/숏한 총노출 200의 왕복 수수료는 0.36, 즉 총노출의 18bp입니다.</p>
    <h2>손익 원인</h2>{costs}<p>¹ 동일 체결 시각·수량을 유지한 비용 분해입니다. 비용을 없애고 전략을 다시 실행한 성과가 아닙니다. 실제 비용을 제거해도 분해 모델의 가격 판단 손익이 음수이면 수수료만이 실패 원인은 아닙니다. 완전 청산된 pair별 손익 합계를 계좌 손익과 대조했습니다.</p>
    <h2>실제 par 호가 확인</h2><p>조회 시각: {at}. 아래는 공개 호가 수량을 소진하며 계산한 즉시 왕복 비용으로, 미래 청산 가격이나 수익 보장이 아닙니다. 두 종목의 원자적 체결을 가정하지 않았습니다. 두 leg의 비용은 해당 명목 비중으로 가중해야 합니다.</p>{quotes}
    <h2>사전에 고정한 후보 선정</h2><p>8월에 기본·실행 스트레스 모두 순이익이고 회계·시점·위험 검사를 통과한 후보 중 순손익 최대를 선택했습니다. 이후 9월~10월 비교를 수행했습니다. 공적분 후보에는 적격자가 없어 OU60일은 진단용이며, 영구/일시 분해 후보의 적격 선정판은 {html.escape(selected)}입니다.</p>{selections}{selection2}
    <p>공적분 후보의 선정 기간 노출이 0이어서 동일 노출 대조의 배율도 0입니다. 따라서 해당 현금 대조를 동등한 전략 비교나 성능 개선으로 인정하지 않습니다. ‘현재 75% + pair 25%’는 별도 가상 시그널을 합친 보조 비교이며 실제 pair 체결 검증을 대신하지 않습니다. 내부 이름의 same_gross_budget은 동일 평균 노출을 보장하지 않습니다.</p>
    <details><summary>64개 완료 백테스트 전체 수치</summary>{full}</details>
    <h2>검증과 한계</h2><p>의미 있는 테스트 8개 통과: 공적분 합성자료, 코퓰라 조건부 확률, 경로 면적, 만기 청산비용, 부분체결 청산, 양쪽 leg 수수료, 잠재상태 추정, 미래 가격을 바꾸어도 과거 신호가 불변임을 검사했습니다. 기존 2차 전략의 체결과 전체 자산가치 경로는 이전 검증된 실행과 정확히 일치했습니다.</p>
    <p>실행 중인 전략 관련 {checks['active_protected_sources']}개, 기존 보존 자료 {checks['reserved_dependencies']}개, 미래 Q 실험 관련 {len(plan['sources'])}개 파일의 해시가 유지됩니다. 기존 봉인 자료의 결과는 읽지 않았고 실제 주문은 0개입니다. 실패한 중간 실행 2건은 보존하고 필드명·비교배율 범위·자료 종료시점 문제를 수정한 뒤 다시 평가했습니다.</p>
    <p>최근 참고 기간은 10월13일21시~15일21시 KST 2일입니다. 기존 입력의 마지막 시점이 10월16일16시 KST여서 16일21시 일별 경계를 만들어내지 않았습니다. 짧은 최근 구간의 양수만으로 42일 손실을 뒤집어 채택하지 않습니다.</p>
    <p><b>판정: 교체 보류.</b> 이번에 구현한 후보는 장기 순이익의 개선 근거가 없습니다. 논문·모형 이름, 낮은 변동성, 8월 성과, 최근 이틀 일부 양수 중 어느 것도 이를 대신하지 않습니다.</p>
    <h2>문헌과 실제 반영 범위</h2><ol>{sources}</ol>
    <p><a href="{HERE/'protocol.json'}">연구 규칙</a> · <a href="{HERE/'artifacts/results.json'}">공적분 실험 결과</a> · <a href="{HERE/'partial_artifacts/results.json'}">분해 모델 결과</a> · <a href="verification.json">검증</a> · <a href="attribution.json">pair별 손익 분해</a></p></html>'''
    (DEST/'report.html').write_text(body)
    print(json.dumps(dict(report=str(DEST/'report.html'),verification=verification,attribution=audits),ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))

if __name__=='__main__':main()
