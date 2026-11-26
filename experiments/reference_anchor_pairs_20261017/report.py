import csv
import gzip
import json
import numpy as np
from .research import HERE, OUT, ROOT
from experiments.quote_anchored_pairs_20261017.report import table
from experiments.par_direct_forecast_20261017.data import save, sha
from experiments.complementary_alpha_20261017.research import protected


def main():
    result=json.loads((OUT/'results.json').read_text())
    quotes=json.loads((OUT/'quotes/results.json').read_text())
    sensitivity=json.loads((OUT/'exit_sensitivity.json').read_text())
    rs=json.loads((OUT/'universe_audit_v2/results.json').read_text())
    variance=json.loads((OUT/'estimator_audit/results.json').read_text())
    official=json.loads((HERE.parent/'quote_anchored_pairs_20261017/artifacts/official_leaderboard.json').read_text())
    for key,value in {'lambda':1.5,'c':.05,'epsilon':.005,'w_financial':.8,'w_jury':.2}.items():
        assert official['scoring'][key]==value
    manifests=['frozen.json','quotes/frozen.json','universe_audit_v2/frozen.json','estimator_audit/frozen.json']
    for name in manifests:
        for path,digest in json.loads((OUT/name).read_text())['sources'].items():
            assert sha(ROOT/path)==digest,path
    max_error=0.;runs={}
    for path in sorted((OUT/'runs').glob('*.json.gz')):
        with gzip.open(path,'rt') as f:r=json.load(f)
        runs[path.stem]=r
        m=r['metrics'];returns=np.asarray([x['return_'] for x in r['daily']])
        vol=float(np.std(returns,ddof=1)*np.sqrt(len(returns)))
        q=(max(m['roi'],0)+.005)/(abs(m['roi'])+1.5*vol+.05)
        assert abs(vol-m['volatility_horizon'])<1e-12
        assert abs(q-m['financial_score'])<1e-12
        assert abs(np.prod(1+returns)-1-m['roi'])<1e-10
        assert all(f['source_available']<=f['decision_time']<f['time'] for f in r['fills'])
        for event in r['events']:
            if event['event']=='pair_intent':
                assert event['minimum_expected_cash_at_limits']>=event['buffer_cash']-1e-8
        max_error=max(max_error,m['maximum_accounting_error'])
    assert len(runs)==40
    same_fills=all(runs[f'{phase}-first_strategy-{scenario}.json']['fills']==runs[f'{phase}-second_strategy-{scenario}.json']['fills']
        for phase in ('validation','comparison') for scenario in ('base','stress'))
    guards=protected();target=ROOT/'reports/reference_anchor_pairs_20261017';target.mkdir(exist_ok=True)
    body='<h1>기준시장 가격 예측과 실제 페어 주문 연결: 검증 결과</h1>'
    body+='<p><strong>현재 교체 조건을 통과한 후보는 없습니다.</strong> 긴 기준시장 자료의 가격 예측, 현재 par 가격으로의 환산, 펀딩 중복 방지를 구현했습니다. 과거 비교 40회와 실제 호가 기반 판단 11,256건을 완료했습니다. 연구 주문은 0건이며 실거래 전략을 변경하지 않았습니다.</p>'
    body+='<h2>구현과 비교 조건</h2><p>6·24시간 뒤 두 종목의 상대 가격 변화를 예측합니다. 90일 자료 중 마지막 14일은 보정에만 사용하고 그보다 앞서 결과가 확정된 표본으로 릿지와 얕은 트리를 훈련했습니다. 원본 두 모델, 보정한 혼합 모델, 가격 예측을 0으로 둔 대조군을 비교했습니다. 기준시장 종가에서 출발한 예측을 현재 par 가격과 동일한 보유 수량의 현금 가치로 환산하며, 펀딩은 가격 학습 목표에서 제외하고 별도로 한 번 반영합니다. 미래 par 가격이 기준시장 가격으로 수렴한다는 가정 자체는 아직 검증된 사실이 아닙니다.</p>'
    body+='<p>선택 구간은 10월 1일 21시~8일 21시(KST), 이후 비교 구간은 8일 21시~16일 21시입니다. 모든 기간은 이미 다른 연구에서 사용했습니다. 선택 파일은 비교 실행 전에 고정했지만 새로운 독립 검증이라고 부르지 않습니다. 초기 자본은 1,000,000 USDPR이고 기존 두 전략에도 같은 체결·위험 한도를 적용했습니다.</p>'
    for phase,title in [('validation','선택 구간'),('comparison','이후 비교 구간')]:
        rows=[]
        for r in result['rows']:
            if r['phase']!=phase or r['scenario']!='base':continue
            m=r['metrics'];stress=next(x['metrics'] for x in result['rows'] if x['phase']==phase and x['arm']==r['arm'] and x['scenario']=='stress')
            rows.append([r['arm'],f"{m['net_pnl']:+.4f}",f"{stress['net_pnl']:+.4f}",f"{100*m['roi']:+.8f}%",
                f"{100*m['volatility_horizon']:.8f}%",f"{m['financial_score']:.9f}" if m['eligible'] else '— (요건 미충족)',
                f"{m['completed_pairs']} / {m['aborted_pairs']}",f"{100*m['mean_gross']:.6f}%",f"{m['residual_gross']:.2f}"])
        body+='<h3>'+title+'</h3>'+table(['전략','순손익 USDPR','스트레스 순손익','R','기간 σ','Q','페어 완료/중단','평균 투자 비중','잔여 명목액'],rows)
    body+='<p>기본 편도 수수료 9bp·불리한 가격 10bp, 스트레스는 12bp·20bp 및 절반의 거래량 한도와 양수 펀딩 수취액입니다. 다음 5분봉 종가와 관측 거래량으로 체결을 근사하고, 두 종목은 독립적으로 부분 체결됩니다. 한쪽 체결·불균형은 비용을 내고 감축하며 관측 종료 시 남은 포지션을 공짜로 청산하지 않습니다. 전체 과거 호가가 없으므로 실제 계좌 수익을 재현한 결과는 아닙니다. 평가는 체결 종가이며 공식 마크가격과 다를 수 있습니다.</p>'
    body+=f'<p>두 기존 전략의 체결 기록이 같은지 직접 대조한 결과는 {same_fills}입니다. 목표 신호가 같다는 뜻은 아닙니다. 후보의 작은 손실과 Q≈0.1은 매우 적은 투자·체결과 함께 봐야 합니다. 진단 선택 후보 calibrated_24는 비교 구간 −0.1238 USDPR, 기존 2번과 합친 결과는 −88.1131 USDPR로 기존 −87.9329보다 나빴습니다. 완료된 페어 거래가 없는 작은 손실을 수익성 개선으로 인정하지 않았습니다.</p>'
    body+='<h2>실제 호가 기록의 조건부 진단</h2><p>기록된 지정가 대기열에서 체결 결과를 보기 전에 주문의 수익 조건을 평가했습니다. 알려진 체결 조각에 대해 후속 헤지·부분 체결 감축·유료 청산을 계산했습니다. 겹치는 가정상 주문이므로 계좌 수익률이나 Q로 합산하지 않습니다. 미래 청산 호가가 없는 긴 보유 경로는 거래량·비용을 적용한 캔들 모형을 사용했습니다. 누락·관측 종료된 결과는 0 손익으로 처리하지 않았습니다.</p>'
    body+=table(['모델','비용','선택 판단','체결 조각','고유 체결 주문','조건부 순손익 bp','헤지 불명'],[
        [x['model'],x['scenario'],x['selected'],x['known_fill_probes'],x['unique_filled_quotes'],
         '—' if x['conditional_net_bps'] is None else f"{x['conditional_net_bps']:+.3f}",x['unknown_hedge_outcomes']]
        for x in quotes['summaries']])
    body+='<p>tree_24의 기본 +6.817bp와 스트레스 +4.435bp는 양수였지만 스트레스 관측은 주문 한 건의 체결 조각 두 개입니다. 추가 청산 오차 5bp만 적용해도 스트레스 손익이 음수가 됩니다. 수익 개선을 입증하는 독립 표본으로 볼 수 없습니다.</p>'
    body+=table(['비용 조건','청산 추가 불리함 bp','조건부 순손익 bp'],[[x['scenario'],x['extra_modeled_exit_bps'],f"{x['conditional_net_bps']:+.6f}"] for x in sensitivity['rows'] if x['model']=='tree_24'])
    body+='<h2>페어 선정 방식 점검과 수정</h2><p>논문의 평균회귀 선정 조건을 다시 확인했습니다. 처음 사용한 장기 R/S 추정과 공개 구현의 짧은 시차 분산 추정은 다른 측정입니다. 후자를 같은 수학적 정의로 별도 구현하여 45개 페어를 재검사했습니다. 앞선 R/S 검사에서 10월 통과 페어가 없었다는 결과를 논문 방식의 결과로 해석하면 안 됩니다.</p>'
    body+=table(['선정일','R/S 구조 조건 통과','분산 추정 구조 조건 통과','분산 추정 + BH q≤0.1','분산 추정 상위 2개'],[
        [a['date'],a['structural_count'],b['structural_count'],b['fdr_count'],' / '.join('-'.join(p) for p in b['top_two'])]
        for a,b in zip(rs['formations'],variance['formations'])])
    body+='<p>10월 상위 후보는 BTC/HYPE와 BTC/ETH입니다. 구조 조건을 통과해도 45개 중 다중검정을 보정한 q≤0.1 조건까지 통과하지는 못했습니다. 이것은 수익성 판정이 아닙니다. 다음 비교에서 선정 규칙을 고정하고 보유 중 목표 평균만 갱신하는 전략의 현금 손익을 시험할 근거입니다.</p>'
    body+='<p>참고 연구는 대규모 암호화폐 유니버스에서 페어를 고르고, 진입 시 헤지 비율·위험 척도를 고정한 채 평균을 갱신하는 기초 전략에 강화학습을 추가합니다. 우리 시장의 종목 수·거래 비용·체결 자료는 다르므로 결과를 그대로 가져올 수 없습니다. <a href="https://arxiv.org/html/2606.04574v1">Lebiedź &amp; Ślepaczuk (2026)</a>. 추정법은 <a href="https://github.com/damianlebiedz/pair-trading-with-rl/blob/main/modules/core/indicators.py">저자의 공개 지표 정의</a>와 대조했습니다. 외부 매매 코드는 복사하거나 운용에 넣지 않았습니다.</p>'
    body+='<h2>검증 및 판단</h2>'
    body+=f'<p>전략 관련 검증 10개, 최초 선정 검사 3개, 수정 추정법 검사 3개를 통과했습니다. 모든 저장 실행의 복리 수익률·기간 변동성·Q와 시점 순서를 다시 계산했습니다. 원장 최대 오차는 {max_error:.3g} USDPR입니다. 기존 운용 소스 {guards["active_protected_sources"]}개와 예약 의존 파일 {guards["reserved_dependencies"]}개의 해시가 유지되었습니다.</p>'
    body+=f'<p>Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05), σ는 완료 일별 수익률의 표본 표준편차×√n입니다. 계좌별 심사 점수 J는 이 비교에 없습니다. 공식 연습 리더보드 공개 자료 시각 {official["generated_at"]}, 평가 종료 {official["to"]}. <a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 기준</a>.</p>'
    body+='<p><strong>현재 판단: 교체 보류, 페어 선정·보유 중 청산 목표를 수정한 후보의 개발 계속.</strong> 이번 모델의 양수 조건부 사례를 전체 계좌 개선으로 확대 해석하지 않습니다.</p>'
    css='body{font-family:system-ui,sans-serif;max-width:1360px;margin:32px auto;padding:0 24px;line-height:1.7;color:#25364a}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border:1px solid #ccd9e4;text-align:left}th{background:#edf3f8}a{color:#1762a0}'
    (target/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>기준시장 페어 예측 검증</title><style>'+css+'</style><body>'+body+'</body></html>')
    flat=[dict(phase=r['phase'],arm=r['arm'],scenario=r['scenario'],**r['metrics']) for r in result['rows']]
    with (target/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    save(target/'summary.json',dict(selection=result['selection'],comparisons=result['comparisons'],tests=16,replays=len(runs),
        quote_decisions=len(quotes['decisions']),conditional_fill_probes=len(quotes['trials']),maximum_accounting_error=max_error,
        protected=guards,baseline_fills_identical=same_fills,performance_improvement_established=False,orders_sent=0,historical_independence=False))
    save(HERE/'work_state.json',dict(status='progress',goal_complete=False,performance_improvement_established=False,
        this_turn='40 historical replays; 11256 pre-fill quote decisions, 37 conditional fill probes; 16 tests; exact cash anchor conversion and single-count funding; formation Hurst estimator discrepancy found and tested separately.',
        report=str(target/'report.html'),
        next_action='Use original implementation of fixed-entry hedge and risk with adaptive held mean. Freeze the variance-scaling selection of BTC/HYPE and BTC/ETH before its trading replay. Compare fixed versus adaptive mean, full lifecycle, costs and own incumbents. Existing histories are reused, not independent. Do not copy restricted external trading code, inspect sealed forward studies or change live strategies.',
        orders_sent=0,protected_sources_unchanged=True,data_wait_required=False))
    print(target/'report.html',flush=True)


if __name__=='__main__':main()
