import gzip
import html
import json
import time
import numpy as np
from strategy.agent import TAIL
from experiments.complementary_alpha_20261017.research import protected
from experiments.q_execution_20261017.metrics import paired_uncertainty
from .data import HERE,ROOT,OUT,ts,sha,save
from .model import build
from .engine import run


def main():
    if (OUT/'frozen.json').exists():raise FileExistsError('Do not overwrite an evaluated historical family')
    cfg=json.loads((HERE/'protocol.json').read_text());p=json.loads((TAIL/'baseline.json').read_text())['protocol']
    with np.load(OUT/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    sources={str(q.relative_to(ROOT)):sha(q) for q in [*HERE.glob('*.py'),HERE/'protocol.json',OUT/'inputs.npz']}
    save(OUT/'frozen.json',dict(at_unix=time.time(),sources=sources,protected=protected(),historical_independence=False,orders_sent=0))
    signals,models=build(d,cfg,p['symbols']);save(OUT/'models.json',models)
    np.savez_compressed(OUT/'signals.npz',**signals)
    (OUT/'runs').mkdir(exist_ok=True);rows=[];runs={}
    def evaluate(phase,start,end):
        for arm in cfg['variants']:
            for scenario in cfg['scenarios']:
                r=run(d,signals,cfg,p,arm,scenario,ts(start),ts(end));r['phase']=phase
                key=f'{phase}-{arm}-{scenario}';runs[key]=r
                with gzip.open(OUT/'runs'/f'{key}.json.gz','wt') as f:
                    json.dump(r,f,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
                rows.append(dict(phase=phase,arm=arm,scenario=scenario,metrics=r['metrics']))
                m=r['metrics'];print(key,'PnL',round(m['net_pnl'],4),'Q',round(m['financial_score'],8),
                                    'fills',m['fills'],'pairs',m['completed_pairs'],'aborts',m['aborted_pairs'],flush=True)
    evaluate('selection',cfg['start_utc'],cfg['split_utc'])
    rank=[]
    for name in cfg['variants'][2:]:
        a=runs[f'selection-{name}-base']['metrics'];b=runs[f'selection-{name}-stress']['metrics']
        eligible=a['net_pnl']>0 and b['net_pnl']>0 and a['fills']>0 and a['risk_breach_hours']==0 and b['risk_breach_hours']==0
        rank.append(dict(arm=name,eligible=bool(eligible),Q=a['financial_score']))
    rank.sort(key=lambda r:(-r['eligible'],-r['Q'],r['arm']))
    selection=dict(selected=rank[0]['arm'],eligible=rank[0]['eligible'],ranking=rank,
                   made_before_later_replays=True,historical_independence=False)
    save(OUT/'selection.json',selection);selection_hash=sha(OUT/'selection.json')
    evaluate('later',cfg['split_utc'],cfg['end_utc'])
    evaluate('full',cfg['start_utc'],cfg['end_utc'])
    comparisons=[]
    for phase in ('later','full'):
        for scenario in cfg['scenarios']:
            for name in cfg['variants'][2:]:
                c=runs[f'{phase}-{name}-{scenario}'];b=runs[f'{phase}-second_strategy-{scenario}']
                cm,bm=c['metrics'],b['metrics']
                comparisons.append(dict(phase=phase,scenario=scenario,arm=name,
                    pnl_difference=cm['net_pnl']-bm['net_pnl'],Q_difference=cm['financial_score']-bm['financial_score'],
                    volatility_difference=cm['volatility_horizon']-bm['volatility_horizon'],
                    paired_uncertainty=paired_uncertainty(c,b),
                    positive_and_better=bool(cm['net_pnl']>0 and cm['net_pnl']>bm['net_pnl'] and cm['financial_score']>bm['financial_score'])))
    assert sha(OUT/'selection.json')==selection_hash
    for path,h in sources.items():assert sha(ROOT/path)==h,path
    protected()
    result=dict(rows=rows,selection=selection,comparisons=comparisons,model_fits=len(models),
        executed_historical_replays=len(rows),orders_sent=0,historical_independence=False,
        actual_par_candles_and_funding=True,actual_historical_orderbook=False,
        protected_sources_unchanged=True,automatic_live_replacement=False)
    save(OUT/'results.json',result)
    report(result,cfg)


def report(result,cfg):
    dest=ROOT/'reports/historical_par_20261017';dest.mkdir(exist_ok=True)
    def table(phase,scenario):
        rows=[r for r in result['rows'] if r['phase']==phase and r['scenario']==scenario]
        rows.sort(key=lambda r:-r['metrics']['financial_score'])
        names={'first_strategy':'첫 번째 전략','second_strategy':'두 번째 전략','pairs_carry':'페어 펀딩',
               'pairs_basis_carry':'페어 가격 괴리＋펀딩','pairs_ridge_basis_carry':'릿지＋가격 괴리＋펀딩','second_plus_basis_carry':'두 번째＋페어 가격 괴리·펀딩'}
        out='<table><tr><th>전략</th><th>순손익 USDPR</th><th>수익률 %</th><th>일별 표준편차 %</th><th>기간 변동성 σ %</th><th>Q</th><th>0.8Q</th><th>MDD %</th><th>가격 변화 손익</th><th>펀딩</th><th>수수료</th><th>슬리피지 추정</th><th>체결</th><th>완료 페어 / 불균형 중단</th><th>잔여 명목금액</th></tr>'
        for r in rows:
            m=r['metrics'];values=[names[r['arm']],f"{m['net_pnl']:+.4f}",f"{m['roi']*100:+.6f}",f"{m['daily_sample_std']*100:.6f}",f"{m['volatility_horizon']*100:.6f}",f"{m['financial_score']:.8f}",f"{.8*m['financial_score']:.8f}",f"{m['max_drawdown']*100:.6f}",f"{m['price_move_pnl']:+.4f}",f"{m['funding']:+.4f}",f"{m['fees']:.4f}",f"{m['execution_cost']:.4f}",str(m['fills']),f"{m['completed_pairs']} / {m['aborted_pairs']}",f"{m['residual_gross']:.4f}"]
            out+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
        return out+'</table>'
    s=result['selection']
    body='''<!doctype html><meta charset="utf-8"><title>실제 par 과거 자료 백테스트</title><style>body{font:16px/1.65 system-ui;margin:36px;color:#17212b}table{border-collapse:collapse;font-size:13px;margin:20px 0}td,th{border:1px solid #cad2db;padding:8px;text-align:right;white-space:nowrap}th{background:#e8eef4}td:first-child{text-align:left}p{max-width:1100px}.note{background:#fff3cf;padding:16px}</style><h1>실제 par 과거 자료로 실행한 15일 백테스트</h1>
<p>측정: 2026년 10월 1일 21시~16일 21시 한국시간. 모든 계좌는 1,000,000 USDPR 현금에서 시작합니다. 실제 주문을 제출하지 않았습니다.</p>
<p class="note">실제 par 시간봉·거래량·확정 펀딩률을 사용한 역사적 시뮬레이션입니다. 체결가는 다음 시간봉 종가에 불리한 가격을 더한 추정이며, 역사적 호가를 복원한 결과가 아닙니다. 평가가격은 par 체결 종가, 펀딩 기준 오라클은 기준시장 가격으로 대용했습니다. 따라서 거래소의 실제 PnL·공식 점수와 일치한다고 주장할 수 없습니다.</p>
<h2>시점과 학습</h2><p>기존 두 전략의 고정 신호 경로를 기존 저장 결과와 정확히 대조했습니다. 신규 릿지는 8월 1일 이후 자료 중 72시간 예측 결과가 학습 시점 전에 확정된 행만 사용합니다. 10월 1·8·15일 00:00 UTC에 재학습하며, 표준화도 해당 학습 행으로만 계산합니다. 펀딩 예측은 직전 시간까지의 마지막 값과 24시간 평균을 반씩 혼합합니다. 미래를 잘라낸 입력으로 다시 계산해 이전 신호가 바뀌지 않는지 검사했습니다.</p>
<p>이 프로젝트에서 이미 반복해서 본 과거이므로 미사용 독립 검증이라고 부르지 않습니다. 10월 1~8일 결과로 후보를 고정하고 그 뒤 8~16일 결과를 별도 계산했습니다. 기존 전략의 설계·선택도 과거 구간을 재사용했습니다.</p>'''
    body+=f'<p>앞 구간 선정 후보: <b>{html.escape(s["selected"])}</b>. 양쪽 비용 조건에서 양수인 선정 요건 충족: <b>{s["eligible"]}</b>. 모든 후보 결과를 공개합니다.</p>'
    for phase,title in [('full','전체 15일'),('selection','앞 7일: 후보 선정'),('later','뒤 8일: 선정 이후 비교')]:
        for scenario,label in [('base','기본 비용'),('stress','불리한 비용·유동성·펀딩')]:
            body+=f'<h2>{title} — {label}</h2>'+table(phase,scenario)
    body+='''<h2>체결 및 회계 가정</h2><p>기본: 편도 수수료 9bp＋불리한 가격 10bp, 직전·다음 시간 실제 거래량 중 작은 값의 1% 이내 체결. 스트레스: 수수료 12bp＋가격 20bp, 거래량 0.25%, 받는 펀딩 50%만 인정하고 내는 펀딩은 전액 유지. 두 경우 모두 0 거래량에는 체결하지 않습니다. 이는 민감도 가정이며 실제 IOC 체결량 측정값은 아닙니다.</p>
<p>페어는 168시간 기준시장 수익률로 헤지를 추정하고, 72시간 보유 예상수익이 왕복 비용＋5bp를 넘을 때만 진입합니다. 양쪽 주문은 각각 체결하며 체결 비율 차이가 5%를 넘으면 다음 시간부터 축소합니다. 종료 24시간 전부터 청산을 시도하고, 남은 포지션은 실제 청산 완료로 꾸미지 않습니다. 종료 시점 잔여량과 추가 청산 비용 추정은 JSON에 별도 저장합니다. 일별 점수는 UTC 12시, 한국시간 21시 스냅샷으로 산출합니다.</p>
<p>순손익 = 가격 변화 손익 − 불리한 체결 가격 비용 − 수수료 ＋ 펀딩. 이 식으로 각 시간의 계좌가치를 독립 재계산했습니다. 변동성은 일별 수익률 표본 표준편차 × √일수, Q=(max(R,0)+0.005)/(|R|+1.5σ+0.05). J는 가정하지 않았습니다.</p>
<p>참고: <a href="https://perpetual-alpha26.github.io/docs/">공식 시장·수수료·펀딩 문서</a>, <a href="https://perpetual-alpha26.github.io/leaderboard#practice">공식 점수 정의</a>. 본 구현은 페어 헤지·가격 괴리·펀딩과 단순 릿지 비교이며 특정 논문의 전체 재현을 주장하지 않습니다.</p>'''
    (dest/'report.html').write_text(body)
    save(dest/'results.json',result)
    print('REPORT',dest/'report.html',flush=True)


if __name__=='__main__':main()
