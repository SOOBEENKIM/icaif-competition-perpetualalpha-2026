"""Render the evidence-backed diagnosis; does not import an exchange client."""
import hashlib
import html
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
f=json.loads((HERE/'findings.json').read_text())
p=json.loads((HERE/'probes.json').read_text())
r=f['runs']


def table(head, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def num(x):return f'{x:,.2f}'
def pct(x):return f'{100*x:.6f}%'
def link(path,label):return f'<a href="{ROOT/path}">{html.escape(label)}</a>'


doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>전략 성능 실패 원인 점검 · 2026-10-17</title>
<style>body{font:16px/1.75 system-ui,sans-serif;max-width:1140px;margin:32px auto;padding:0 22px;color:#192638;background:#f6f8fb}section{background:white;padding:24px;margin:20px 0;border:1px solid #dfe5ed;border-radius:12px}h1,h2{line-height:1.4}h2{font-size:21px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce3eb;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f8}.scroll{overflow-x:auto}a{color:#185cab}.lead{border-left:5px solid #c75738}code{background:#eef2f6;padding:2px 5px}.note{color:#536173}</style>
<h1>왜 논문을 참고해도 성능이 개선되지 않았는가</h1>
<p>2026-10-17 작성 · 과거 실험 28개 재정산 및 6개 경로 재생 · 실거래 전략 변경 0</p>
<section class="lead"><p><strong>확인된 핵심은 논문의 핵심 실험과 다른 구현, 종목별로 무너진 예측, 실제 보유 경로를 반영하지 못한 규모 보정, 수익 기회보다 큰 거래비용입니다.</strong> 8월에는 가격 방향을 잘못 잡았고, 9월~10월에는 작은 총이익이 비용에 소진됐습니다. 알고리즘 이름이나 코드 테스트 통과만으로 수익 기회가 입증되지 않습니다.</p>
<p>아래는 초기자본 1,000,000의 <strong>기준시장 과거자료 백테스트</strong>입니다. 지갑의 실제 손익이 아닙니다. 비교 기간은 UTC 2026-09-01 00:00~10-13 12:00, 검증 기간은 UTC 08-01 00:00~09-01 00:00입니다. 종가 평가이며 마지막 보유분의 강제 청산비용은 차감되지 않았습니다.</p></section>
<section><h2>1. 논문의 핵심 조건을 재현하지 않은 적용</h2>
<p>참고한 <a href="https://arxiv.org/html/2606.00060v1">Bysik &amp; Ślepaczuk 연구</a>는 약 7만 시간봉, 12개월 학습·3개월 검증·3개월 평가의 27회 순차 평가를 보고합니다. 1시간 수익률 예측과 거래비용 문턱을 사용합니다. 최고 예의 연환산 수익 65.40%에는 연환산 변동성 56.22%, 최대낙폭 67.42%가 함께 따릅니다. 매수 후 보유 대비 샤프비율의 통계적 우위는 확립하지 못했습니다. 공개 코드는 저자 요청 방식입니다. 논문 표의 수익률만 낮은 위험의 대회 전략으로 옮길 근거는 없습니다.</p>
'''
doc+=table(['항목','이번 구현의 실제 내용'],[
    ['학습','4~7월 한 번 학습. 24시간 모델 1,452행, 72시간 모델 1,428행; 세 종목을 합한 수치.'],
    ['표본 의존','72시간 미래 구간을 6시간마다 생성하여 인접 정답이 66시간 겹침. 1,428개의 독립 사례가 아님.'],
    ['평가','8월 한 구간 선택, 이전 연구에서 이미 본 9~10월 비교. 새로운 독립 시험 아님.'],
    ['투자 대상','새 예측형은 BTC·ETH·HYPE만. 기존판은 원유 등 10종목까지 포함.'],
    ['예측·의사결정','24/72시간 예측 중 하나 선택, 6시간마다 비중 조정. 여러 시간 예측의 통합 제어는 아님.'],
    ['모델 선택','고정된 얕은 XGBoost·Ridge 설정. 논문의 반복 학습·조정·선택 절차 재현 아님.'],
    ['비용','이번 기본 모형은 편도 수수료 9bp+체결가격 불리함 10bp와 가격단위 반올림.'],
    ['자료 출처','mainnet_reference_settled. par 시장의 과거 호가·체결·펀딩을 재현한 자료가 아님.']])
doc+='''<p><a href="https://arxiv.org/html/2502.04284v1">Ma &amp; Smith</a>는 단일 자산의 장기 의사결정 모형에서 거래하지 않는 영역을 연구합니다. 이번 다자산 평균·분산 목적함수는 그 정책의 재현이 아닙니다.
<a href="https://arxiv.org/html/2605.01176v4">Wang &amp; Hasuike</a>의 실험은 주식·ETF, 월별 재조정, SPO+ 학습과 10% 부분 조정입니다. 우리는 SPO+를 구현하지 않았고, 6시간마다 50% 조정 개념만 차용했습니다. 참고 문헌의 이론적 보장을 이번 코드에 그대로 적용할 수 없습니다.</p>
<p class="note">확인한 링크는 arXiv 원문입니다. 게재 사실과 특정 시장에서의 재현 가능한 수익성은 별도 확인 대상입니다.</p></section>
<section><h2>2. 8월에는 예측 방향이 실패했다</h2>'''
doc+=table(['8월 기본 조건','가격 변화 손익','펀딩','체결비용+수수료','순손익'],[
    [label,num(r[key]['price_move_pnl']),num(r[key]['funding']),num(r[key]['execution_cost']+r[key]['fees']),num(r[key]['net_pnl'])]
    for label,key in [('첫 전략','validation-first_strategy-base'),('두 번째 전략','validation-second_strategy-base'),('새 XGBoost 72h','validation-tree_72-base'),('새 Ridge 24h','validation-ridge_24-base')]])
aug=r['validation-tree_72-base']
doc+=table(['새 XGBoost의 8월 보유','기준 자산 상승률','롱 보유시간','숏 보유시간','순손익'],[
    [x['symbol'],pct(x['underlying_return']),x['long_hours'],x['short_hours'],num(x['net_pnl'])] for x in aug['per_asset'][:3]])
doc+='''<p>가격 변화 손익부터 마이너스여서 수수료만 낮춰 해결할 수 없습니다. 72시간 예측의 평균 방향과 이후 상승장의 방향이 어긋났습니다. 학습·검증 차이는 확인됐지만, 자료 부족·정권 변화·모델 과적합 각각의 책임 비율은 이 진단만으로 식별되지 않습니다. 8월 여섯 예측형 모두 탈락했으며 XGBoost 72h는 설명용 비교 대상으로만 남았습니다.</p></section>
<section><h2>3. 전체 예측 지표가 종목별 실패를 가렸다</h2>'''
ff=[x for x in f['forecast_diagnostics'] if x['period']=='comparison' and x['model']=='tree' and x['horizon']==72 and x['sampling']==72]
doc+=table(['72h 예측 · 9~10월','시간 구간 수','예측/정답 상관계수','0 예측 대비 MSE 변화','평균 예측','평균 실제 정답'],[
    [x['symbol'],x['time_blocks'],f"{x['correlation']:.4f}",pct(x['mse']/x['zero_mse']-1),pct(x['mean_prediction']),pct(x['mean_actual'])] for x in ff])
doc+='''<p>세 종목 합산 상관계수는 +0.2395지만 BTC는 −0.6867입니다. BTC는 6시간 간격 전체 표본에서도 −0.4283으로 같은 방향입니다. 겹치지 않는 표본도 시간 구간 14개뿐이며, 종목 사이의 의존성이 남습니다. 이를 유의한 예측력이나 BTC 숏 전략의 근거로 간주하면 안 됩니다.</p>'''
doc+=table(['9~10월 종목별 순손익','두 번째 전략','새 XGBoost','변화'],[
    [b['symbol'],num(b['net_pnl']),num(a['net_pnl']),num(a['net_pnl']-b['net_pnl'])]
    for a,b in zip(r['comparison-selected_third-base']['per_asset'],r['comparison-second_strategy-base']['per_asset']) if a['fills'] or b['fills']])
doc+='''<p>새 모델은 HYPE에서 개선했지만 BTC에서 더 크게 악화됐습니다. 또한 기존판의 수익 기여가 큰 BRENT를 새 예측형에서는 제외했습니다. 모델·종목군·예측 기간·주문 방식이 함께 바뀌어 전체 성과를 모델 하나의 개선 효과로 해석할 수 없습니다. 이 표를 보고 사후에 HYPE만 고르는 것은 새로운 전략 탐색이며, 검증 완료가 아닙니다.</p></section>
<section><h2>4. 9~10월에는 거래비용이 총이익보다 컸다</h2>'''
doc+=table(['기본 조건','가격 변화 손익','펀딩','체결비용','수수료','순손익','거래대금'],[
    [label,num(r[key]['price_move_pnl']),num(r[key]['funding']),num(r[key]['execution_cost']),num(r[key]['fees']),num(r[key]['net_pnl']),num(r[key]['metrics']['turnover'])]
    for label,key in [('첫 전략','comparison-first_strategy-base'),('두 번째 전략','comparison-second_strategy-base'),('새 XGBoost','comparison-selected_third-base'),('새 분산형','comparison-diversified-base'),('안쪽 위험 상한 조정','comparison-inward_band-base')]])
doc+='''<p>새 XGBoost의 가격손익+펀딩은 +1,769.01, 수수료+체결비용은 3,254.91입니다. 기록된 거래대금 대비 감당할 수 있는 비용은 약 10.39bp였지만 적용된 비용은 약 19.11bp였습니다. 수수료는 거래 횟수보다 거래대금에 비례하므로 체결 건수 감소만으로 비용 개선을 판단해서는 안 됩니다.</p>
<p>347/397 체결은 동일 방향 포지션의 증감이었습니다. 다만 부분 조정 자체가 원인이라는 가설은 아래 비교에서 지지되지 않았습니다. 같은 예측·비용·위험 계수를 유지하고 부분 조정만 바꾼 사후 진단입니다.</p>'''
half=r['comparison-selected_third-base']['metrics']
full=next(x for x in p['results'] if x['period']=='comparison' and x['partial']==1)['metrics']
doc+=table(['목표 비중 반영','수익률','일별 STD','체결 수','거래대금','수수료'],[
    [name,pct(z['roi']),pct(z['daily_std_sample']),z['fills'],num(z['turnover']),num(z['fees'])] for name,z in [('50%씩 · 기존 후보',half),('100% 즉시 · 진단',full)]])
doc+='''<p>100% 반영은 체결 수가 줄어도 거래대금과 손실이 증가했습니다. 비용 최적화 제거 역시 기존 기록에서 수익률을 −0.148590%에서 −0.402430%로 악화시켰습니다. 비용 제어는 일부 도움이 됐으며, 검증되지 않은 예측의 수익성을 만들어 주지는 못했습니다.</p></section>
<section><h2>5. 빈 계좌를 가정한 규모 보정은 실제 보유 위험을 맞추지 못했다</h2>
<p>기존 보정 함수는 7월 매일 UTC 00시, 보유 수량 0인 상황에서 목표 비중을 구해 평균 예상 일별 변동성을 0.055403%로 맞춥니다. 실제 제어기는 기존 보유량·누적 조정·비용 때문에 다른 비중을 유지합니다. 동일한 7월 자료와 고정된 모델로 순차 재생하여 확인했습니다.</p>'''
july=[x for x in p['results'] if x['period']=='july']
doc+=table(['7월 재생','보유 경로의 평균 예상 일별 변동성','실현 일별 STD'],[
    [x['policy'],pct(x['mean_path_expected_daily_volatility']),pct(x['metrics']['daily_std_sample'])] for x in july])
doc+='''<p>Ridge 24h의 보유 경로 예상 위험은 기준판의 약 2.16배였습니다. 빈 계좌 계산을 모든 6시간 판단 시각으로 확장해도 0.057400%에 그쳐 실제 경로 0.120007%를 설명하지 못합니다. 시각 표본뿐 아니라 보유 상태를 생략한 보정에 문제가 있음을 보여 줍니다. XGBoost에서는 차이가 작았으므로 모든 후보가 같은 크기의 결함을 보인다고 일반화하지 않습니다. 8월 Ridge의 실현 일별 STD는 기준판의 3.95배였습니다.</p>
<p>이는 공통의 최대 위험 한도를 넘었다는 뜻이 아닙니다. 같은 한도와 같은 실현 위험은 다릅니다. 또한 7월 예측은 학습 내 예측이므로 이 재생의 수익률은 독립 성과가 아닙니다.</p>'''
doc+=f'<p>근거: {link(Path("experiments/cost_horizon_20261016/research.py"),"calibrate 함수")}, {link(Path("experiments/cost_horizon_20261016/policy.py"),"실제 보유량을 쓰는 CostController")}</p></section>'
doc+='''<section><h2>6. 평가 조건과 엔지니어링 검증의 의미</h2>
<p>스트레스 조건은 호가 영향 확대 외에 매시간 총보유금액의 1bp를 추가 펀딩비로 차감합니다. 이는 관측된 par 펀딩이 아니라 가정입니다. 두 번째 전략의 기본/스트레스 손익 차이 중 약 85.3%는 펀딩 항목 차이입니다. 스트레스 손실을 현재 지갑에서 발생한 손실로 설명하면 안 됩니다.</p>
<p><a href="https://perpetual-alpha26.github.io/about">공식 평가</a>는 oracle 기준 순수익, 일별 수익률 표준편차, 보고서 심사입니다. 우리가 사용한 ‘두 기준판보다 수익↑·STD↓·MDD↓를 기본/스트레스 모두 충족’은 내부 교체 기준입니다. ‘모두 통과하지 못함’과 ‘어떤 지표도 개선되지 않음’을 구분해야 합니다.</p>
<p>28개 기록의 손익을 수량×가격변화, 펀딩, 수수료, 체결비용으로 재구성했고 원장과 일치했습니다. 새 후보의 원래 조건을 다시 실행했을 때 모든 체결과 시간별 자산값이 저장 결과와 정확히 일치했습니다. 특징의 시간상 인과성, 학습 정답 종료 시각, 다음 종가부터의 목표 구간을 점검했습니다. 점검한 범위에서 손익 부호 오류·수수료 이중 차감·정답 정렬 오류는 발견하지 못했습니다. 모든 잠재적 누수를 배제했다는 뜻은 아닙니다.</p>
<p>기존의 34개 테스트 통과는 주문·원장·인과성 구현의 검사입니다. 경제적 예측력이나 실거래 우위를 검증하는 시험이 아닙니다. 104회 백테스트도 여러 시장에서 얻은 104개의 독립 증거가 아닙니다.</p></section>
<section><h2>이 진단으로 바뀌어야 할 개발 기준</h2>
<ol><li><strong>재현과 변형을 분리:</strong> 한 논문의 핵심 예측·포지션·비용 규칙을 먼저 재현한 기준선을 만들고, 종목·기간·제어기를 한 번에 하나씩 바꿉니다.</li>
<li><strong>예측 검증이 주문 최적화보다 먼저:</strong> 종목별 순차 외부 구간의 방향·크기 보정과 비용 전 성과를 확인합니다. 잘못된 BTC 신호를 전체 평균이 가리지 않게 합니다.</li>
<li><strong>규모는 순차 운용 경로로 보정:</strong> 학습 외 예측을 사용하고 실제 보유량·거래비용·체결을 포함합니다. 같은 위험 한도만으로 동일 위험이라고 보고하지 않습니다.</li>
<li><strong>기준시장→대회 시장 연결을 직접 검증:</strong> 기준 종가 예측이 아니라 현재 par 체결 가능 가격에서 시작한 비용 차감 수익을 확인해야 합니다. 대회 호가로 하는 별도 시험이 필요합니다.</li>
<li><strong>변경 효과를 단계별로 보고:</strong> 가격손익·펀딩·거래대금·체결비용·종목 기여를 함께 기록합니다. 같은 자료 재사용에 따른 사후 선택을 공개하고, 고정 후 새 기간에서 평가합니다.</li></ol>
<p>이번 작업은 원인 점검입니다. 검증되지 않은 수정의 수익 개선을 주장하지 않으며, 현재 실거래와 봉인한 후속 실험의 소스·결과를 변경하거나 중간 성과를 열람하지 않았습니다.</p>
<p><a href="findings.json">전체 수치</a> · <a href="probes.json">6개 재생 진단</a> · <a href="audit.py">재정산 스크립트</a> · <a href="probes.py">재생 스크립트</a> · <a href="manifest.json">근거 해시</a></p></section></html>'''
(HERE/'report.html').write_text(doc)
manifest=json.loads((HERE/'manifest.json').read_text())
for name in ('findings.json','probe_plan.json','probes.py','probes.json','report.py','report.html'):
    q=HERE/name;manifest[str(q.relative_to(ROOT))]=hashlib.sha256(q.read_bytes()).hexdigest()
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(HERE/'report.html')
