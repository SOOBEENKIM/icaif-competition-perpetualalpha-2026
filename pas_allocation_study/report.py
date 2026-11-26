"""Report every fixed arm, including failed and ineligible results."""
import collections
import csv
import gzip
import html
import json
import math
from datetime import datetime,timezone,timedelta
import numpy as np
from .protocol import ROOT,OUT,save,sha
from .study import check

NAMES={'incumbent':'기존 제출판','size_only':'규모 1.441배 복원판',
       'directional':'새 방향·배분 후보 + 기존 페어','venue_aware':'PAS 조건 반영 후보',
       'exposure_control':'7월 노출 기준 대조군'}


def read_run(arm,phase,scenario):
    with gzip.open(OUT/'reference'/f'{arm}_{phase}_{scenario}.json.gz','rt') as f:return json.load(f)


def daily_cash(r):
    a=[s['equity']+50000 for s in r['snapshots'] if s['time']%86400==43200]
    return np.diff(a)


def paired_bootstrap(delta,length):
    rng=np.random.default_rng(26);n=len(delta);sums=[]
    for _ in range(5000):
        starts=rng.integers(0,n-length+1,size=math.ceil(n/length))
        sample=np.concatenate([delta[a:a+length] for a in starts])[:n]
        sums.append(sample.mean())
    return dict(days=n,block_days=length,repetitions=5000,seed=26,
        actual_mean=float(delta.mean()),lower95=float(np.quantile(sums,.025)),
        upper95=float(np.quantile(sums,.975)),
        interpretation='Dependent-day resampling uncertainty only; does not undo prior research selection or model misspecification')


def kst(t):
    return datetime.fromtimestamp(t,timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S')


def main():
    check()
    p=json.loads((OUT/'protocol.json').read_text())
    reference=json.loads((OUT/'reference_results.json').read_text())
    pas=json.loads((OUT/'pas_results.json').read_text())
    models=json.loads((OUT/'models.json').read_text())
    trace=json.loads((OUT/'pas_decision_trace.json').read_text())
    lookup={(r['arm'],r['phase'],r['scenario']):r for r in reference}
    tests=[]
    for phase in ['validation','comparison','continuous']:
        for scenario in ['base','stress']:
            c=lookup['directional',phase,scenario]
            for control in ['incumbent','size_only']:
                b=lookup[control,phase,scenario]
                tests.append(dict(phase=phase,scenario=scenario,control=control,
                    pnl_delta=c['net_pnl']-b['net_pnl'],Q_delta=c['Q']-b['Q'],
                    passed=c['net_pnl']>max(b['net_pnl'],0) and c['Q']>b['Q']))
    bootstrap=[]
    for scenario in ['base','stress']:
        delta=daily_cash(read_run('directional','comparison',scenario))-daily_cash(read_run('incumbent','comparison',scenario))
        for length in [7,3]:bootstrap.append(dict(scenario=scenario,**paired_bootstrap(delta,length)))
    # Check actual original bot fills, excluding the unattributed ETH order.
    with (ROOT/'reports/team_comparison_20261021/fills.csv').open(encoding='utf-8-sig') as f:
        actual=[r for r in csv.DictReader(f) if r['address'].lower()=='0x125778065a578d598d88bddaa3fdd635280d0e47'
            and r['coin'] in ['pas:BTC','pas:HYPE']]
    first_actual=min(int(r['time']) for r in actual)
    with gzip.open(OUT/'pas/incumbent_base.json.gz','rt') as f:conditional=json.load(f)
    first_proxy=min(r['time'] for r in conditional['fills'])
    by_symbol=[]
    for s in ['BTC','HYPE']:
        initial=[r for r in actual if r['coin']=='pas:'+s and r['dir']=='Open Long']
        first_time=min(int(r['time']) for r in initial)
        initial_qty=sum(float(r['sz']) for r in initial)
        at=[r for r in conditional['snapshots'] if r['time']<=first_time+900000]
        proxy_qty=at[-1]['qty'][p['symbols'].index(s)]
        by_symbol.append(dict(symbol=s,actual_initial_qty=initial_qty,
            proxy_qty_within_15m=proxy_qty,actual_first_fill_kst=kst(first_time/1000)))
    fidelity=dict(actual_first_fill_kst=kst(first_actual/1000),
        conditional_first_fill_kst=kst(first_proxy/1000),
        delay_hours=(first_proxy-first_actual)/3600000,initial_positions=by_symbol,
        matched=False,conclusion='Minute-volume participation proxy fails to reproduce known taker fills. It is neither a validated fill model nor a guaranteed lower PnL bound.',
        hypothesis='Actual IOC takes resting depth; historical trade volume is not the same as available depth. No historical book can be reconstructed from these candles.',
        no_after_result_participation_tuning=True)
    save(OUT/'execution_fidelity.json',fidelity)
    # Eligibility must not be confused with the algebraic Q=0.1 of no trade.
    from competition_live.scoring import evaluate
    for r in pas:
        with gzip.open(OUT/'pas'/f"{r['arm']}_{r['scenario']}.json.gz",'rt') as f:run=json.load(f)
        score=evaluate(run['daily'],trading_days=len({f['time']//86400000 for f in run['fills']}))
        r['proxy_eligible']=score['eligible'];r['proxy_score']=score['score']
        r['score_cutoff_kst']=kst(run['daily'][-1]['time'])
        r['net_pnl_cutoff_kst']=kst(p['pas_end'])
        r['official_score']=None
        r['score_note']='Trade-close proxy; no official mark history. No-trade arms are ineligible even when formula Q=0.1.'
    save(OUT/'pas_display_results.json',pas)
    diagnostic=dict(decision_asset_checks=len(trace),
        fresh_candle_checks=sum(r['net_forecast'] is not None for r in trace),
        basis_fit_checks=sum(r['basis_model']['reason']=='estimated' for r in trace),
        basis_pair_max=max(r['basis_model']['pairs'] for r in trace),
        entry_cost_bps=43.,
        max_abs_available_net_bps=max(abs(r['net_forecast'])*10000 for r in trace if r['net_forecast'] is not None),
        directional_calibration_scale=models['calibration']['scale'],
        exposure_control_scale=models['exposure_control_scale'],
        exposure_control_note='Calibration made July base target gross zero; matching scalar degenerates to zero. This is an ablation, not evidence of superior risk-matched alpha.')
    # Reproduce both previously reported comparator curves, not just one PnL.
    old=json.loads((ROOT/'reports/submission_candidate_review_20261021/review.json').read_text())['candidate']['reproduced_windows']
    errors=[]
    for r in old:
        for arm,prefix in [('incumbent','current'),('size_only','candidate')]:
            a=lookup[arm,r['phase'],r['scenario']]
            errors.append(abs(a['net_pnl']-r[prefix+'_pnl']))
    assert max(errors)<1e-7
    summary=dict(recommendation='DO_NOT_REPLACE',reason='Nominee failed predictive calibration and retrospective PnL/Q comparisons. PAS basis module lacked its preregistered minimum history and no candidate PAS fills occurred.',
        reference_tests=tests,all_reference_pass=all(r['passed'] for r in tests),
        paired_bootstrap=bootstrap,pas_diagnostics=diagnostic,execution_fidelity=fidelity,
        comparator_max_pnl_reproduction_error=max(errors),
        max_accounting_error=max(r['accounting_error'] for r in reference),
        risk_breach_bars=sum(r['risk_breach_bars'] for r in reference+pas),
        live_orders_sent=0,live_deployment_changed=False,historical_independence=False,
        live_status=json.loads((ROOT/'.runtime/pas_competition_live/status.json').read_text()))
    save(OUT/'decision.json',summary)
    for filename,rows in [('reference_results.csv',reference),('pas_results.csv',pas)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def table(rows,columns):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for key,label,fmt in columns)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(fmt(r.get(key)))+'</td>' for key,label,fmt in columns)+'</tr>' for r in rows)+'</tbody></table>'
    num=lambda v:'—' if v is None else f'{v:,.2f}'
    pct=lambda v:'—' if v is None else f'{v*100:.6f}%'
    cols=[('arm','전략',lambda v:NAMES[v]),('scenario','비용 조건',str),('net_pnl','순손익',num),
          ('return_fraction','총자본 수익률',pct),('volatility','기간 σ',pct),('Q','Q',lambda v:f'{v:.6f}'),
          ('score','0.8Q, J=0',lambda v:f'{v:.6f}'),('max_drawdown','최대낙폭',pct),('mean_notional','평균 보유 명목금액',num)]
    pieces=['''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX PAS 다종목 후보 검증</title>
<style>body{font-family:system-ui,sans-serif;max-width:1300px;margin:40px auto;padding:0 24px;color:#172334;line-height:1.6}h1{font-size:28px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{padding:9px;border-bottom:1px solid #dde3ea;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef2f7}.notice{background:#fff1df;padding:18px;border-left:5px solid #b65f00}a{color:#095baf}code{background:#eef2f7}li{margin:7px 0}</style>
<h1>PAS 다종목 양방향 판단·자본 배분 후보 검증</h1>
<p class="notice"><b>판정: 이번 후보로 교체하지 않음.</b> 구현·검증·비교는 완료했으나 개선을 입증하지 못했다. 기존판이 최적이라는 뜻은 아니다. 이번 후보가 교체 기준을 통과하지 못했다는 뜻이다. 실제 주문 0건, 운용 코드·설정 변경 없음.</p>''']
    pieces.append(f'<p>총자본 100,000 USDP, 사용자 전략 50,000, 팀원 몫 50,000 현금 대조. 실제 팀원 성과는 제외했다. 기준시장 평가 2026-08-01 21:00~10-15 21:00 KST; PAS 조건부 평가 {kst(p["pas_start"])}~{kst(p["pas_end"])} KST.</p>')
    pieces.append('''<h2>무엇을 구현했나</h2><p>7개 과거 가격·변동성 특징으로 24시간 후 기준시장 가격 변화를 예측하는 부분 풀링 릿지 회귀다. 공통 계수와 암호화폐·주식·원자재·통화 그룹별 계수의 차이를 규제한다. 모든 10개 종목에 같은 롱·숏·대기 규칙을 적용한다. 새 방향 예측은 현재 거래가격부터 미래 목표가격까지의 이익, 왕복 비용과 펀딩을 고려하고, 기존 진입 근거가 사라지면 목표를 0으로 줄인다. 전체 위험 한도를 적용한 뒤 위험 감소를 다시 확대하지 않는다.</p>
<p>PAS에서는 최근 실제 체결가와 기준시장 가격의 로그 괴리에 과거 AR(1) 추정을 적용하고, 과거 24시간 펀딩 평균의 절반을 다음 24시간에 적용하는 가정을 사용한다. 기준시장 가격이나 오라클 가격으로 체결됐다고 처리하지 않는다. 과거 호가가 없으므로 본 구간의 체결 및 평가 가격은 캔들 기반 조건부 가정이다. 별도의 bid/ask 함수와 방향·펀딩·누출 방지 검사는 구현했지만 전체 구간의 실제 L2 수익성을 검증한 것은 아니다.</p>
<h2>시간 분리와 사전 고정</h2><ul><li>학습: 4월 1일~6월 말. 학습 정답이 7월로 넘어가는 행은 제외.</li><li>보정: 7월. 예측 크기를 0~1로만 줄이는 계수 하나를 정답으로 보정. PnL 최적화 없음.</li><li>평가: 8월, 9월~10월 15일, 전체 연결 구간. PAS는 별도 10월 19일~21일.</li><li>훈련 모델·전처리·보정 계수는 이후 평가 정답으로 재학습하지 않음. 펀딩과 가격 괴리 추정은 매 판단 당시까지 관측된 자료만 사용.</li><li>모든 구간은 기존 프로젝트에서 이미 노출된 과거 자료다. 이번 모델의 미래 정답 누출은 검사했지만, 반복 연구에 의한 선택 과적합이 없다는 보장은 할 수 없다. 독립 최종 테스트라고 부르지 않는다.</li></ul>
<h2>교체 기준</h2><p>후보를 평가 전에 하나로 고정했다. 기존판·규모 복원판보다 순이익과 Q가 모든 선언 기간의 기본·불리한 비용 조건에서 함께 높아야 한다. 투자 규모에 따른 효과를 별도 확인하고, 시간 블록 재표집의 개선 하한도 확인한다. 일별 변동성을 무조건 줄이게 하지 않으며 Q와 위험 한도를 함께 본다. PAS 실행 검증과 공유 지갑 소유권 대조가 해결돼야 실전 준비가 완료된다.</p>''')
    pieces.append('<h2>기준시장 백테스트: 전체 75일</h2>'+table([r for r in reference if r['phase']=='continuous'],cols))
    c=lookup['directional','continuous','base']
    pieces.append(f'<p><b>새 후보의 +{c["net_pnl"]:,.2f}는 새 방향 모델의 성공이 아니다.</b> 기존 페어 부분 {c["pair_pnl"]:+,.2f}, 새 기본 배분 부분 {c["base_pnl"]:+,.2f}다. 동일한 페어 알고리즘을 유지해도 공유 계좌의 주문 상계·반올림 때문에 다른 기본 포지션과 결합했을 때 페어 손익이 완전히 같지는 않다.</p>')
    for phase,title in [('validation','8월'),('comparison','9월 1일~10월 15일')]:
        pieces.append('<h2>'+title+'</h2>'+table([r for r in reference if r['phase']==phase],cols))
    pieces.append('''<h2>실패 원인과 해석</h2><ol><li><b>방향 예측의 검증 실패:</b> 880개 종목·일 학습 행, 300개 보정 행을 사용했다. 7월 원시 예측 방향 정확도 45.67%, MAE 140.20bp로 변화 없음 139.34bp보다 나빴다. 보정 계수는 0이 됐다. 결과를 보고 부호를 뒤집거나 보정 규칙을 해제하지 않았다.</li>
<li><b>PAS 괴리 모델을 충분히 가동할 과거 자료 부족:</b> 연속적인 실제 거래가 있는 시간쌍 48개를 요구했으나 평가 판단 시점에서 최대 46개였다. 따라서 괴리 예측은 전부 비활성화됐다. 이 결과로 괴리 전략 자체가 불가능하다고 결론 낼 수 없다.</li>
<li><b>체결 모델의 재현성 부족:</b> 거래량 1% 한도는 과거 거래량을 이용한 가정이다. 실제 IOC가 가져간 대기 호가 수량을 재현하지 못했다. 이 가정은 수익의 보장된 하한도 아니다. 좋은 거래와 나쁜 거래가 서로 다르게 누락될 수 있다.</li></ol>''')
    pieces.append('<h2>PAS 조건부 비교 — 공식 수익·점수 아님</h2>'+table(pas,[('arm','전략',lambda v:NAMES[v]),('scenario','비용',str),('net_pnl','가정상 순손익',num),('fills','체결 조각',str),('mean_notional','평균 명목금액',num),('proxy_eligible','가정상 채점 자격',str),('proxy_score','가정상 0.8Q',lambda v:'미충족' if v is None else f'{v:.6f}')]))
    pieces.append(f'<p>새 PAS 후보는 체결 0건이다. 무거래의 대수식 Q=0.1을 순위 점수로 쓰지 않았다. 90개 판단·종목 중 신선한 체결가가 있는 경우 {diagnostic["fresh_candle_checks"]}개, 활성 괴리 적합 0개였다. 순손익 종료는 {kst(p["pas_end"])} KST, 일별 Q 계산 종료는 10월 21일 21시다. 실제 일별 mark 평가 이력이 없으므로 공식 Q를 재현한 결과가 아니다.</p>')
    pieces.append('<h2>실제 최초 체결과의 대조</h2>'+table(by_symbol,[('symbol','종목',str),('actual_first_fill_kst','실제 최초 체결 KST',str),('actual_initial_qty','실제 최초 진입 수량',str),('proxy_qty_within_15m','이후 15분 내 모의 보유',str)]))
    pieces.append(f'<p>실제 최초 체결 {fidelity["actual_first_fill_kst"]}, 모의 최초 체결 {fidelity["conditional_first_fill_kst"]}. 약 {fidelity["delay_hours"]:.2f}시간 차이다. 현재 PAS 모의 체결표로 실전 수익 우위를 단정하지 않았다. 실제 ETH 주문은 현 봇 원장 밖 주문이므로 이 대조에서 제외했다.</p>')
    pieces.append('<h2>불확실성·검증</h2>'+table(bootstrap,[('scenario','비용',str),('block_days','블록 일수',str),('actual_mean','일 평균 손익 차이',num),('lower95','95% 하한',num),('upper95','95% 상한',num)]))
    pieces.append(f'<p>기준시장 24건·PAS 10건 실행 완료. 기능 검사 6건 통과: 전 종목 롱/숏/대기, 펀딩 부호, 미래 괴리 오염, bid/ask 비용, 정답 만기, 실제 공유 계좌·컨트롤러를 통과하는 10종목 숏→청산→롱 모의 체결. 마지막 검사는 수수료 지불·수량 부호·미래 유동성 삭제 시 과거 체결 불변도 확인한다.</p><p>기존판·규모 복원판 12개 결과 재현 최대 오차 {max(errors):.3g}. 회계 최대 오차 {summary["max_accounting_error"]:.3g}. 위험 한도 위반 {summary["risk_breach_bars"]}건. 미래 데이터 오염·절단 검사 통과. 기존 실행자는 <code>{html.escape(summary["live_status"]["state"])}</code> 상태로 확인됐고, 별도 ETH 포지션과 원장 불일치 문제는 이번 연구에서 변경하지 않았다.</p>')
    pieces.append('''<h2>재현 자료</h2><p><a href="protocol.json">사전 프로토콜</a> · <a href="models.json">모델·보정</a> · <a href="decision.json">판정 근거 JSON</a> · <a href="reference_results.csv">24개 기준시장 결과 CSV</a> · <a href="pas_results.csv">10개 PAS 결과 CSV</a> · <a href="causality.json">시간·누출 검사</a> · <a href="execution_fidelity.json">체결 재현성 대조</a> · <a href="pas_decision_trace.json">종목별 판단 기록</a></p>
<p>평가 기준: <a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a>. 데이터 범위: <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint">Hyperliquid 공개 Info API</a>. 구현: 저장소의 <code>pas_allocation_study/</code>. 이 연구는 실전 실행자에 연결되지 않았다.</p></html>''')
    (OUT/'report.html').write_text('\n'.join(pieces))
    save(OUT/'evidence_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in
        [*(ROOT/'pas_allocation_study').glob('*.py'),*(OUT).glob('*.json'),*(OUT/'pas/raw').glob('*.json'),OUT/'report.html'] if f.name!='evidence_manifest.json'})
    print(json.dumps(dict(recommendation=summary['recommendation'],bootstrap=bootstrap,diagnostics=diagnostic,fidelity=fidelity),indent=2))


if __name__=='__main__':main()
