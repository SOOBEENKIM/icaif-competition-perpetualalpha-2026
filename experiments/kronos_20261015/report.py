"""Generate a self-contained research report from saved, audited results."""
import gzip
import html
import json
import os
import sys
from datetime import datetime, timezone
import numpy as np
from .data import HERE, write

NAMES={'kronos_paper':'Kronos 신규판','chronos2_paper':'Chronos-2 · 동일 매매 규칙',
       'selected_default':'기존 하방위험 보정판','selected_original':'기존 추세＋7일 되돌림 원래 규모',
       'equal_weight':'단순 10종목 동일 비중'}


def pct(v):return f'{v*100:+.4f}%'
def vol(v):return f'{v*100:.4f}%'
def number(v):return '—' if v is None else f'{v:.4f}'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    results=json.loads((HERE/'results.json').read_text())
    predictions=json.loads((HERE/'prediction_metrics.json').read_text())['results']
    parity=json.loads((HERE/'model_parity.json').read_text())
    verification=json.loads((HERE/'verification.json').read_text())
    rows=results['rows']
    index={(r['phase'],r['policy'],r['scenario']):r for r in rows}
    order=sorted(NAMES,key=lambda n:-index['comparison',n,'base']['metrics']['roi'])
    comparison=[]
    for n in order:
        b=index['comparison',n,'base']['metrics'];s=index['comparison',n,'stress']['metrics']
        comparison.append([NAMES[n],pct(b['roi']),vol(b['daily_vol']),pct(b['max_drawdown']),pct(s['roi']),b['fills'],f"{b['fees']:,.2f}",f"{b['funding']:+,.2f}"])
    table_main=table(['전략','순수익률','일별 변동성','최대 낙폭','스트레스 수익률','체결 수','수수료','펀딩 손익'],comparison)
    phase_rows=[]
    for phase in ['calibration','validation','comparison']:
        for n in NAMES:
            b=index[phase,n,'base']['metrics'];s=index[phase,n,'stress']['metrics']
            phase_rows.append([{'calibration':'규모 보정 구간','validation':'8월 점검','comparison':'9–10월 비교'}[phase],NAMES[n],pct(b['roi']),vol(b['daily_vol']),pct(s['roi']),b['complete_days']])
    diagnostic_rows=[]
    for kind,label in [('daily','Kronos 일봉 90→10'),('chronos_daily','Chronos-2 일봉 90→10'),('hourly','Kronos 시간봉 80→12')]:
        r=predictions[kind]['comparison'];s=r['pooled_return']
        diagnostic_rows.append([label,r['origins'],number(s['pearson']),number(r['mean_cross_section_rank_ic']),vol(s['directional_accuracy']),vol(s['positive_target_fraction']),number(s['mse']/s['zero_return_mse'])])
    # Standard plotting artifact, also embedded as SVG in the report.
    # Optional report-only dependencies are isolated from the trading venv.
    sys.path.insert(0,str(HERE/'report_dependencies'))
    os.environ.setdefault('MPLCONFIGDIR',str(HERE/'report_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    english={'kronos_paper':'Kronos','chronos2_paper':'Chronos-2 / same policy','selected_default':'Existing default',
             'selected_original':'Existing original','equal_weight':'Equal weight'}
    for n in NAMES:
        r=index['comparison',n,'base']
        with gzip.open(HERE/r['file'],'rt') as f: run=json.load(f)
        x=[datetime.fromtimestamp(s['time'],timezone.utc) for s in run['snapshots']]
        y=[100*(s['equity']/1e6-1) for s in run['snapshots']]
        ax.plot(x,y,label=english[n],linewidth=2 if n=='kronos_paper' else 1.2)
    ax.axhline(0,color='#7b8492',linewidth=.7);ax.grid(alpha=.15)
    ax.set_ylabel('Net account return (%)');ax.set_xlabel('UTC · historical reference replay')
    ax.legend(fontsize=8,ncol=2);fig.savefig(HERE/'comparison.svg');fig.savefig(HERE/'comparison.png',dpi=150);plt.close(fig)
    svg=(HERE/'comparison.svg').read_text();svg=svg[svg.index('<svg'):]
    k=index['comparison','kronos_paper','base'];d=index['comparison','selected_default','base']
    status='사전 지정 개선 기준 통과' if results['improvement_passed'] else '사전 지정 개선 기준 미통과'
    gates=table(['확인 항목','결과'],[[{'higher_roi_than_selected_default':'기본판보다 높은 순수익률',
            'no_higher_daily_std':'기본판 이하의 일별 변동성','positive_base':'기본 비용에서 양의 수익',
            'positive_stress':'스트레스 비용에서 양의 수익','engineering':'회계·정보시점·위험·보존 검사'}[name],
            '통과' if yes else '미통과'] for name,yes in results['improvement_gates'].items()])
    scales=table(['모델','개발 구간 고정 배율','보정 후 목표 총 노출'],[[NAMES[n],f"{v['scale']:.8f}",vol(.27*v['scale'])] for n,v in results['calibration'].items()])
    attribution=k['audit']['fixed_trajectory_decomposition']
    attribution_table=table(['Kronos 비교 구간 손익 구성','금액'],[[name,f'{attribution[key]:+,.2f}'] for name,key in [
        ('가격 변화 손익','price_move_pnl'),('체결 불리함: 차감액','execution_adversity'),
        ('수수료: 차감액','fees'),('펀딩: 부호 포함','funding'),('최종 순손익','net_pnl')]])
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kronos 신규 재현 실험 · 2026-10-15</title><style>
body{{font:16px/1.7 system-ui,sans-serif;color:#233047;background:#f4f6fa;margin:0}}main{{max-width:1120px;margin:auto;background:white;padding:38px}}
h1{{font-size:30px;line-height:1.35}}h2{{font-size:23px;margin-top:38px}}h3{{font-size:18px}}p{{max-width:1000px}}a{{color:#1355bb}}.tag{{color:#53647e;font-size:14px}}
.callout{{padding:18px 24px;background:#edf3ff;border-left:4px solid #4267bc}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}th,td{{padding:11px 12px;border-bottom:1px solid #dce2eb;text-align:left;white-space:nowrap}}th{{background:#f0f3f8}}pre{{white-space:pre-wrap;background:#f4f6fa;padding:18px;font-size:13px}}svg{{width:100%;height:auto}}li{{margin:8px 0}}code{{background:#f2f4f8;padding:2px 4px}}.small{{font-size:14px;color:#53647e}}</style>
<main><p class="tag">고정 설정 실험 v1 · 기존 전략 보존 · 과거 자료 재사용 · 실제 주문 0건</p>
<h1>Kronos를 우리 백테스트에 옮긴 결과</h1>
<div class="callout"><b>{status}.</b> Kronos 순수익률 {pct(k['metrics']['roi'])}, 일별 변동성 {vol(k['metrics']['daily_vol'])}.
기존 기본판은 각각 {pct(d['metrics']['roi'])}, {vol(d['metrics']['daily_vol'])}입니다.
기존 두 전략과 설정은 그대로 유지했습니다. 새 결과는 독립된 실험 폴더에 저장했습니다.</div>
<p>비교 기간은 <b>2026-09-01 00:00–10-13 03:00 UTC</b>입니다. 한국시간으로 9월 1일 09:00–10월 13일 12:00입니다.
초기 자본은 각 계좌 1,000,000이며 완전한 일별 수익률은 42개입니다. 표는 기본 비용 순수익률 순서입니다.
공식 대회 순위가 아닙니다. 심사위원 점수와 공식 정규화 점수를 계산하지 않았습니다.</p>
{table_main}{svg}
<p class="small">수익률 = 최종 평가자산/초기 자본−1. 미청산 포지션 평가손익 포함. 일별 변동성 = UTC 일별 계좌 수익률의 모집단 표준편차.
공식 대회 실현손익 점수와 동일하다고 간주할 수 없습니다. 기말 강제 청산 및 그 비용은 포함하지 않았습니다.</p>
<h2>무엇을 재현했고 무엇을 바꿨나</h2>
<p><a href="https://arxiv.org/html/2508.02739v1">Kronos 논문</a>의 공개 사전학습 모델을 사용하는 zero-shot 실험을 기준으로 삼았습니다.
<a href="https://github.com/shiyu-coder/Kronos/tree/67b630e67f6a18c9e9be918d9b4337c960db1e9a">공식 코드의 고정 커밋</a>과 공개 base 모델·토크나이저를 가져왔습니다.
새 사전학습이나 미세조정을 수행한 것은 아닙니다. 중국 주식 원 데이터 및 논문의 전체 비교모델 실험도 재현한 것은 아닙니다.</p>
{table(['구성','적용 내용'],[
 ['모델','공식 Kronos-base 약 102M + Kronos-Tokenizer-base, 원본 safetensors 및 SHA-256 검증'],
 ['전처리','공식 컨텍스트별 정규화·clip 5·토큰화·역정규화 유지'],
 ['확률 추론','논문 Table 6: temperature 0.6, top-p 0.9, 10개 경로 평균; 날짜별 고정 seed'],
 ['주 매매 실험','논문 투자 실험: 일봉 90개 입력 → 10개 예측'],
 ['순위 점수','논문 Eq.13: 예측 10일 종가 평균 / 현재 종가 − 1'],
 ['매매 규칙','상위 종목 장기 보유, 하루 최대 1개 순위 교체, 최소 5 UTC 일 보유'],
 ['시장 교체','CSI300/800 → 보존된 10개 기준시장; top-50/drop-5 → top-2/drop-1'],
 ['OHLC 입력','논문 crypto/FX의 거래량 없는 입력 규칙을 10개 혼합 시장 전체에 적용'],
 ['위험·비용 교체','기존 공용 위험한도·회전 제한·1시간 후 모의 체결·9bp 수수료·펀딩 사용'],
 ['보조 진단','논문 예측 설정: 시간봉 80개 → 12개; 하루 1회 진단만, 거래 선택에 미사용']])}
<p>공식 미세조정 데모의 <code>finetune/qlib_test.py</code>는 정규화 가격 차이를 순위 점수로 사용합니다.
이번 실험은 가격 단위를 복원해 수익률로 나누는 논문 Eq.13을 따릅니다. 데모의 5경로 기본값 대신 논문 표의 10경로를 사용했습니다.
예측이 모두 음수여도 순위 상위 종목을 보유하는 long-only 규칙입니다. 임의의 양수 진입 필터를 추가하지 않았습니다.</p>
<h2>동결 및 평가 순서</h2>
<p>6월 17일–7월 31일 자료에서 모델별 기본 규모의 일별 표준편차를 계산해 <code>min(1, 0.0009 / 표준편차)</code>를 한 번 적용했습니다.
기본 목표 총 노출은 27%이며 종목별 위험한도 15%와 90% 안전 여유에 맞춘 두 종목 배분입니다.
이후 8월과 9–10월 결과를 보기 전에 배율을 고정했습니다. 결과를 보고 재학습·규칙 수정·모델 선택을 하지 않았습니다.</p>
{scales}
<p>각 구간은 현금 1M에서 독립적으로 시작합니다. 기존 두 전략의 목표 포지션은 보존 파일에서 그대로 읽었습니다.
기존 전략의 9–10월 모든 지표가 원본 기록과 정확히 일치합니다.
Chronos-2 비교군은 같은 90→10일 매매 규칙과 비용을 쓰되 종가만 입력하므로, 모델 외에 입력 변수 차이도 포함합니다.
동일 비중 비교군은 Kronos와 같은 보정 후 목표 총 노출을 10종목에 나눴습니다.</p>
{table(['구간','전략','기본 수익률','일별 변동성','스트레스 수익률','완전한 일수'],phase_rows)}
<h2>예측력과 손익을 별도로 확인</h2>
<p>아래는 9–10월 중 예측의 전체 미래 구간이 종료된 사례만 계산했습니다. 일봉은 미래 10일 평균 수익률, 시간봉은 12시간 뒤 수익률입니다.
예측 오차는 현재 가격이 유지된다는 기준과 비교했습니다. MSE 비율이 1보다 작으면 이 기준보다 오차가 작습니다.
일봉 정답 구간은 겹치므로 330개 자산별 관측을 330개의 독립 표본으로 해석할 수 없습니다.</p>
{table(['예측','예측 날짜 수','수익률 상관','평균 종목 순위 상관','방향 정확도','정답 상승 비율','MSE / 무변화 기준'],diagnostic_rows)}
<p class="small">양수 수익률 비율을 함께 표시해 항상 상승을 예측하는 기준과 방향 정확도를 비교할 수 있습니다.
원시 가격 수준을 자산 간에 섞어 상관계수를 부풀리지 않습니다. 논문의 원본 데이터·정규화와 다른 시장 진단이므로 논문 수치와 직접 비교하지 않습니다.</p>
{attribution_table}
<p>위 손익 분해는 동일한 체결 경로에서 계산했습니다. 비용을 제거했을 때의 새 전략 경로를 재실행한 결과는 아닙니다.</p>
<h2>비용과 실제 대회 적용의 범위</h2>
<p>기본 비용은 편도 수수료 9bp와 모의 체결 불리함 10bp입니다. 스트레스에서는 체결 불리함 20bp,
거래량 참여 한도 1%→0.25%, 매시간 총 노출에 1bp 추가 펀딩 비용을 사용합니다. 두 경우 모두 기준시장 실제 펀딩의 부호를 반영했습니다.
체결가는 의사결정 한 시간 뒤 기준 종가이고, 현재와 체결 시점의 거래량 중 작은 값으로 체결량을 제한했습니다.</p>
<p><b>이 자료는 대회 DEX의 과거 호가·대기열·체결 원장이 아닙니다.</b> 보존된 mainnet 기준가격·거래량·펀딩을 사용한 선형 계약 모의 회계입니다.
대회 오라클 차이, 주문 대기열, 실제 지연·슬리피지, 역사적 증거금·강제청산을 재구성하지 않습니다.
기간은 이미 다른 전략 연구에 노출되었습니다. 새 모델을 넣었다고 독립 검증 기간이 되지는 않습니다.</p>
<h2>검증과 기존 파일 보존</h2>
{gates}
<ul><li>기존 보호 대상 {results['preservation']['protected_files']}개 파일의 해시 일치. 공식 소스 {results['preservation']['official_files']}개 원본도 변경 없음.</li>
<li>신규 테스트 {verification['new_tests']}개, 기존 테스트 {verification['existing_tests']}개 통과. 미래 값 변경·미래 구간 제거, 보유 기간, 재시작, 무효 입력, 공통 회계 엔진 일치 검증 포함.</li>
<li>독립 계좌 회계 검증: 체결 수수료, 펀딩 의무, 매시간 자산, UTC 일별 수익률을 별도 계산해 대조.</li>
<li>CPU 속도 개선용 KV cache는 별도 어댑터입니다. 원본 sampler·tokenizer·가중치는 유지했습니다. 세 사례에서 모든 샘플 토큰과 최종 예측값이 정확히 일치했습니다. 모든 가능한 입력에 대한 수학적 동일성 증명은 아닙니다.</li>
<li>초기 개발 구간 기록 저장 중 NumPy 값의 JSON 직렬화 문제를 수정했습니다. 평가 이전 수정 이력과 기존 코드 해시를 보존했습니다. 거래 규칙·배율 산식 변경은 없었습니다.</li>
<li>실제 주문은 0건입니다. 새 에이전트는 입력된 계좌 상태로 주문 의도를 계산하며 거래소 서명·전송은 연결하지 않았습니다.</li></ul>
<h2>재실행 및 파일</h2>
<p>프로젝트 루트에서 실행합니다. 모델·예측 캐시가 이미 로컬에 있으므로 공개 다운로드 없이 재계산할 수 있습니다.</p>
<pre>OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest experiments.kronos_20261015.test_experiment -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m experiments.kronos_20261015.forecast --kind daily --cached
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m experiments.kronos_20261015.forecast --kind hourly --cached
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m experiments.kronos_20261015.chronos_control
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.kronos_20261015.run
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.kronos_20261015.diagnostics
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.kronos_20261015.report</pre>
<p><a href="plan.json">사전 설정</a> · <a href="calibration.json">규모 보정</a> · <a href="results.json">전체 결과</a> ·
<a href="metrics.csv">성과 CSV</a> · <a href="prediction_metrics.json">자산별 예측 진단</a> ·
<a href="model_provenance.json">가중치 출처</a> · <a href="model_parity.json">공식 출력 대조</a> ·
<a href="verification.json">검증 기록</a> · <a href="environment.json">실행 환경</a> · <a href="policy.py">에이전트 코드</a> ·
<a href="comparison.svg">그래프 SVG</a></p>
<p class="small">생성 시각: {html.escape(results['created'])}. 모든 손익은 모의 계좌 기준입니다.</p></main></html>'''
    (HERE/'report.html').write_text(body,encoding='utf-8')
    print(HERE/'report.html')


if __name__=='__main__':main()
