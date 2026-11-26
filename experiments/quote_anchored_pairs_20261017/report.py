import csv
import gzip
import html
import json
import numpy as np
from .research import HERE,OUT,ROOT
from experiments.par_direct_forecast_20261017.data import save,sha
from experiments.complementary_alpha_20261017.research import protected


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    data=json.loads((OUT/'results.json').read_text());quotes=json.loads((OUT/'quotes/results.json').read_text())
    diag=json.loads((OUT/'diagnostics.json').read_text());official=json.loads((OUT/'official_leaderboard.json').read_text())
    scoring=official['scoring']
    for key,expected in {'lambda':1.5,'c':.05,'epsilon':.005,'w_financial':.8,'w_jury':.2}.items():assert scoring[key]==expected
    assert scoring['std']=='sample' and scoring['horizon']=='sqrt_n'
    for manifest in (OUT/'frozen.json',OUT/'quotes/frozen.json'):
        for path,digest in json.loads(manifest.read_text())['sources'].items():assert sha(ROOT/path)==digest,path
    maximum_error=0.;boundary_errors=0
    for path in (OUT/'runs').glob('*.json.gz'):
        with gzip.open(path,'rt') as f:r=json.load(f)
        m=r['metrics'];returns=np.array([x['return_'] for x in r['daily']])
        vol=float(np.std(returns,ddof=1)*np.sqrt(len(returns)))
        Q=(max(m['roi'],0)+.005)/(abs(m['roi'])+1.5*vol+.05)
        assert abs(vol-m['volatility_horizon'])<1e-12 and abs(Q-m['financial_score'])<1e-12
        assert abs(np.prod(1+returns)-1-m['roi'])<1e-10
        maximum_error=max(maximum_error,m['maximum_accounting_error'])
        for event in r['events']:
            if event['event']=='pair_intent':assert event['minimum_expected_cash_at_limits']>=event['buffer_cash']-1e-8
        assert all(f['source_available']<=f['decision_time']<f['time'] for f in r['fills'])
    protected_state=protected()
    target=ROOT/'reports/quote_anchored_pairs_20261017';target.mkdir(exist_ok=True)
    selected=data['selection'];arms=['first_strategy','second_strategy','ridge_6','ridge_24','tree_6','tree_24','blend_6','blend_24','second_plus_'+selected['selected']]
    labels={'first_strategy':'기존 1번','second_strategy':'기존 2번','second_plus_'+selected['selected']:'기존 2번 + 진단 후보'}
    body='<h1>현재 주문가격을 반영한 페어 전략 비교</h1>'
    body+='<p><strong>교체 기준을 통과한 후보가 없습니다.</strong> 예측 목표와 주문가격을 연결하는 구현, 과거 비교 36회, 실제 호가의 지정가 진입 판단 9,648건을 실행했습니다. 현재 실거래 전략은 변경하지 않았고 연구 주문은 0건입니다.</p>'
    body+='<h2>수정한 판단</h2><p>미래의 가상 체결가격을 예측의 출발점으로 삼던 구조를 바꿔, 판단 시점에 이미 아는 가격을 기준으로 6·24시간 뒤 페어 가치를 예측합니다. 현재 두 종목의 진입 가격, 각 수수료, 펀딩과 청산 비용을 차감한 후 5bp의 여유가 남는 범위에서 종목별 제한가를 고정합니다. 주문 이후 유리한 두 종목 가격을 함께 보고 체결 여부를 소급 선택하지 않습니다. 한쪽만 체결되면 별도 주문으로 비용을 내고 감축합니다.</p>'
    body+='<p>모델은 릿지·얕은 트리·동일 비중 혼합입니다. 10월 8·10·12·14일에 직전 7일 중 결과가 이미 확정된 자료만으로 훈련합니다. 거래가 없던 봉도 관측값으로 사용하되 마지막 거래 후 경과시간을 입력에 추가했습니다. 첫 6시간 모델의 학습 행은 3,090개이지만, 겹치는 시계열이므로 3,090개의 독립 사례가 아닙니다.</p>'
    body+='<h2>같은 자본과 체결 모형으로 비교</h2><p>자본 1,000,000 USDPR. 선택 구간은 10월 8일 21시~12일 21시(KST), 이후 비교는 12일 21시~16일 21시입니다. 후보 선택 파일은 이후 비교 실행 전에 고정했습니다. 이 기간은 이전 연구에 사용했으므로 독립 검증 기간이라고 주장하지 않습니다.</p>'
    for phase,title in [('validation','선택 구간'),('comparison','이후 비교 구간')]:
        rows=[]
        for arm in arms:
            a=next(r['metrics'] for r in data['rows'] if r['phase']==phase and r['arm']==arm and r['scenario']=='base')
            b=next(r['metrics'] for r in data['rows'] if r['phase']==phase and r['arm']==arm and r['scenario']=='stress')
            q=f"{a['financial_score']:.9f}" if a['eligible'] else '— (점수 요건 미충족)'
            rows.append([labels.get(arm,arm),f"{a['net_pnl']:+.4f}",f"{b['net_pnl']:+.4f}",
                f"{100*a['roi']:+.8f}%",f"{100*a['volatility_horizon']:.8f}%",q,
                f"{a['completed_pairs']} / {a['aborted_pairs']}",f"{a['mean_gross']*100:.6f}%",f"{a['residual_gross']:.2f}"])
        body+='<h3>'+title+'</h3>'+table(['전략','기본 순손익','스트레스 순손익','기본 R','기간 σ','기본 Q','페어 완료/중단','평균 투자 비중','잔여 포지션 명목액'],rows)
    body+='<p>순손익 단위는 USDPR입니다. 기본 비용은 편도 수수료 9bp와 불리한 체결가격 10bp, 스트레스는 12bp·20bp와 절반의 거래량 한도·펀딩 수취액입니다. 기존 두 전략도 동일한 제한가·거래량 한도·지연·위험 한도를 적용했습니다. 두 기준 전략의 결과가 같은 것은 이 표본에서 체결량 제한에 걸려 실현된 포지션이 같아졌기 때문입니다. 목표 비중이 같은 전략이라는 뜻은 아닙니다.</p>'
    body+='<p><strong>이 표는 캔들 기반 계좌 모형의 결과입니다.</strong> 실제 과거 par 체결 봉과 펀딩률을 사용하지만 전체 기간의 호가가 없어 다음 5분봉 종가와 제한된 거래량을 체결 근사로 사용했습니다. 거래량이 0인 구간의 실제 호가 수량을 재현하지 못합니다. 평가가격은 체결 종가이며 공식 마크가격과도 다를 수 있습니다. 평균 투자 비중 차이를 고려하면 작은 손실·변동성만으로 전략의 우수성을 주장할 수 없습니다.</p>'
    body+=f'<p>선택 구간에서 최고 진단 Q를 기록한 {selected["selected"]}도 적격 후보가 아닙니다. 선택 구간의 소폭 양수는 완료된 페어 거래 0건에서 발생했으며, 이후 구간에서는 음수였습니다. 기존 2번과 합친 진단은 추가 페어 거래가 없어 기존 2번과 동일했습니다.</p>'
    body+='<h2>실제 호가와 체결 기록을 사용한 별도 검사</h2><p>미리 기록된 보수적 지정가 대기열 탐침에서, 각 주문 시점의 정보로만 진입 여부를 먼저 정했습니다. 이후 알려진 체결 조각을 열어 제한가 헤지, 실패 시 축소, 지연 청산을 계산했습니다. 중복 탐침이므로 계좌 순손익·Q로 합산하지 않습니다.</p>'
    qr=[]
    for r in quotes['summaries']:
        v='—' if r['conditional_net_bps'] is None else f"{r['conditional_net_bps']:+.2f}"
        qr.append([r['model'],r['scenario'],r['selected'],r['known_fill_probes'],r['unique_filled_quotes'],v,
            r['unknown_hedge_outcomes'],r['selected_queue_statuses'].get('data_gap',0)+r['selected_queue_statuses'].get('right_censored',0)])
    body+=table(['모델','비용','선택된 주문·페어','체결 조각 경로','고유 체결 주문','조건부 순손익(bp)','헤지 관측 불명','주문 관측 미완료'],qr)
    body+='<p>실제 제출 주문이 아니라 과거 호가·공개 체결로 재생한 가정상 결과입니다. 장기 청산에는 실제 호가가 없어 거래량 제한과 비용을 적용한 캔들 모형을 사용했습니다. 헤지 도착 자료가 없으면 미체결 가정으로 계산하되 불명 사례를 표시했습니다. 데이터 단절이나 관측 종료 이후의 주문 결과를 0 손익으로 처리하지 않았습니다.</p>'
    body+='<h2>손실 원인 분해</h2>'
    dr=[]
    for r in diag['summaries']:
        if r['scenario']!='base' or r['model'] not in ('ridge_24','tree_24','blend_24') or r['subset'] not in ('all_selected','known_filled'):continue
        w=r['weighted'];dr.append([r['model'],r['subset'],r['quote_pair_decisions'],
            f"{w['predicted_source_price_bps']:+.2f}",f"{w['observed_source_price_bps']:+.2f}",f"{w['forecast_error_bps']:+.2f}",r['future_active_marks']])
    body+=table(['모델','부분집합','주문·페어 수','예측 가격 변화(bp)','관측 가격 변화(bp)','예측 과대평가(bp)','미래 양쪽 거래 봉 존재'],dr)
    body+='<p>현재 표본에서는 지정가 체결 여부를 보기 전 선택된 주문 전체에서도 예측과 관측 방향이 달랐습니다. 따라서 손실을 수수료나 지정가의 불리한 체결 선택만으로 설명할 수 없습니다. 모든 관측은 이틀에 걸친 반복 기록이고, 미래 가격에도 무거래 봉이 포함되므로 장기 일반화나 통계적 인과효과의 증거로 쓰지 않습니다. 관측 미체결 주문의 가상 가격 변화는 실현 가능한 수익이 아닙니다.</p>'
    body+='<h2>검증과 현재 판단</h2>'
    body+=f'<p>미래 자료를 잘라냈을 때 특징·훈련·예측의 동일성, 제출 전 제한가 고정, 다른 종목의 미래 가격을 이용한 소급 취소 금지, 부분 체결 감축, 보유 만기 일치, 수수료·펀딩·계좌 원장 등 14개 검증을 통과했습니다. 모든 저장 실행에서 Q와 복리수익률을 별도 재계산했고 계좌 원장 최대 오차는 {maximum_error:.3g} USDPR였습니다. 보호된 기존 소스 {protected_state["active_protected_sources"]}개와 예약 의존 파일 {protected_state["reserved_dependencies"]}개의 변경이 없음을 확인했습니다.</p>'
    body+=f'<p>공식 리더보드의 {official["generated_at"]} 공개 자료에서 λ=1.5, c=0.05, ε=0.005, 재무 80%·심사 20%, 표본 표준편차×√n을 확인했습니다. 공개 연습 순위의 종료 시각은 {official["to"]}이며 이 실험의 시뮬레이션 표와 구분해야 합니다. <a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 평가 설명</a></p>'
    body+='<p><strong>결정: 실거래 교체 보류.</strong> 다음 연구는 이번 결과를 뒤집기 위한 임계값 탐색보다, 오래 축적된 기준시장 자료로 상대가격 움직임을 학습하는 부분과 par의 현재 가격 괴리를 분리하는 가설을 검증하는 것입니다. 기존의 기준시장 예측에 포함된 펀딩이나 미래 진입가격을 그대로 이식하면 중복 계산과 시점 불일치가 생기므로, 먼저 가격·펀딩·진입 기준을 맞춰야 합니다.</p>'
    css='body{font-family:system-ui,sans-serif;max-width:1280px;margin:36px auto;padding:0 24px;line-height:1.7;color:#263647}h1,h2,h3{color:#14283d}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border:1px solid #d8e2ea;text-align:left}th{background:#eef4f8}a{color:#1762a0}'
    (target/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>페어 가격 기준 개선 비교</title><style>'+css+'</style><body>'+body+'</body></html>')
    flat=[dict(phase=r['phase'],arm=r['arm'],scenario=r['scenario'],**r['metrics']) for r in data['rows']]
    with (target/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    save(target/'summary.json',dict(selection=selected,comparisons=data['comparisons'],tests=14,replays=36,
        quote_decisions=len(quotes['decisions']),conditional_fill_probes=len(quotes['trials']),
        official_scoring=scoring,official_snapshot_time=official['generated_at'],maximum_accounting_error=maximum_error,
        protected=protected_state,performance_improvement_established=False,orders_sent=0,independent_validation=False))
    save(HERE/'work_state.json',dict(status='progress',goal_complete=False,performance_improvement_established=False,
        previous_turn_classification='progress: actual-book execution calibration and all-path accounting completed',
        this_turn='Source-anchored future-pair-value targets, stale-trade age features, fixed per-leg economic limits and quote-level prefill selection implemented. 14 tests, 36 historical replays, 9648 quote decisions and 131 conditional fill probes completed; no eligible replacement.',
        evidence='Prediction error persists before fill conditioning: selected 24h forecasts predicted roughly +57 to +62bp but source-to-future candle movements averaged -38 to -40bp on two reused observation days. All paid 24h maker/hedge diagnostics negative in base and stress.',
        report=str(target/'report.html'),
        next_action='Test a separately declared reference-price plus venue-basis decomposition using longer pre-October reference history and current executable par anchors. Existing reference pair targets include reference funding and next-hour entry prices, so do not transplant their output or double-count carry. Use price-only source-anchored targets, fixed fitting dates, the same comparison risk/costs, and explicit model uncertainty; retain all negative outcomes. Do not inspect sealed forward evidence or alter live/reserved strategies.',
        orders_sent=0,protected_sources_unchanged=True,historical_independence=False,data_wait_required=False))
    print(target/'report.html')


if __name__=='__main__':main()
