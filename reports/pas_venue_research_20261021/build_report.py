import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from practice_live.infer import protected_hashes
from competition_live.profile import verify_bundle,verify_deployment


def read(name):
    return json.loads((OUT/name).read_text())


def kst(at):
    return dt.datetime.fromtimestamp(at,dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def link(path,label):
    return '<a href="'+html.escape(str(path))+'">'+html.escape(label)+'</a>'


structure=read('market_structure.json')
aligned=read('synchronized_analysis.json')
short=read('replay_corrected/shadow_results.json')
plan=read('forward_72h/plan.json')
health=read('forward_72h/health.json')
returns=read('reference_returns.json')
markets={row['symbol']:row for row in structure['markets']}
status=json.loads((ROOT/'.runtime/pas_competition_live/status.json').read_text())
producer=json.loads((ROOT/'.runtime/pas_competition_live/producer_status.json').read_text())
tests=(OUT/'policy_tests.txt').read_text()
integration=json.loads((OUT/'replay_tests.txt').read_text())
assert 'Ran 11 tests' in tests and tests.rstrip().endswith('OK')
assert integration['synthetic_partial_exit_check']=='PASS'
assert status['state']=='running' and not status['fatal'] and status['signal_error'] is None
assert time.time()-status['at']<90 and status['reserved_cash']==50000
assert health['state']=='capturing' and time.time()-health['at']<30
assert health['counts'].get('pas:l2Book',0)>0 and health['counts'].get('main:l2Book',0)>0
for name,digest in plan['code_hashes'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
deployment=verify_deployment()
verification=dict(at=time.time(),tests_pass=11,synthetic_partial_exit_check='PASS',
    preserved_baseline_files=len(protected_hashes()),bundle_hash=verify_bundle(),
    deployment_file_count=len(deployment['files']),research_orders_sent=0,
    alpha_validated=False,live_strategy_replaced=False,capital_increased=False,
    frozen_forward_sources_verified=len(plan['code_hashes']),health=health,status=status,producer=producer)
(OUT/'final_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))

subprocess.run(['python3',str(OUT/'plot_market.py')],check=True,
    env=dict(os.environ,MPLCONFIGDIR='/tmp/pas-venue-plot-cache'))

fee=structure['total_participant_fees']/structure['participant_notional']*10000
content=f"""
<header>
<p class="eyebrow">SFINX / PAS MARKET &amp; PROFITABILITY REVIEW</p>
<h1>가격 차이는 확인됐습니다.<br>수익성 개선은 아직 입증되지 않았습니다.</h1>
<p class="lead">대회 시장 구조와 47개 팀 계좌의 공개 체결을 확인하고, PAS 호가·오라클·실제 거래 빈도를 반영하는 후보와 검증기를 구현했습니다.
기존 전략을 대체할 근거는 아직 없습니다. 고정한 후보의 주문 없는 72시간 관측·평가를 진행합니다.</p>
<p class="meta">작성 {kst(verification['at'])} · 통화 USDP · 1bp = 0.01% · 본 보고서는 검증 결과와 미해결 항목을 구분합니다.</p>
</header>
<div class="cards">
<article><strong>47개 팀 계좌</strong><span>공식 명단·10만 USDP 지급 내역 일치</span></article>
<article><strong>29개 거래 계좌</strong><span>공개 체결 조회 시점 기준 · 18개 미관측</span></article>
<article><strong>95.80%</strong><span>관측 거래 중 상대방이 47개 명단 밖인 비율</span></article>
<article><strong>실전 신규 적용 없음</strong><span>연구 주문 0 · 기존 서비스 정상</span></article>
</div>

<h2>1. “46명끼리 거래하는 시장”이라는 전제부터 다릅니다</h2>
<p>공식 명단은 47팀입니다. 주최 측 공개 지급 계정에서 100,000 USDP를 받은 주소도 47개였고,
{kst(structure['wallet_cutoff_ms']/1000)}까지 체결을 조회한 결과 29개 주소에서 거래를 관측했습니다.
공식 리더보드의 거래·순위 대상 28개와는 조회 시각이 다릅니다. 팀 수는 개인 수가 아닙니다.</p>
<p>이 팀 주소가 적어도 한쪽에 포함된 고유 거래 {structure['unique_trades']:,}건 중 양쪽 모두 명단에 있는 거래는
{structure['both_sides_registered']:,}건, 한쪽만 있는 거래는 {structure['one_side_registered']:,}건입니다.
따라서 이 47개 계좌만으로 닫힌 거래 상대 집합을 가정할 수 없습니다.
공식 안내도 주최 측 유동성 공급 봇의 참여를 명시합니다.
다만 명단 밖 개별 주소를 곧바로 주최 측 봇으로 식별하지는 않았습니다.
<a href="https://perpetual-alpha26.github.io/">대회 공식 안내</a></p>
<p>참가자 쪽 명목 거래액은 {structure['participant_notional']:,.2f} USDP, 그중 maker 비중은
{structure['participant_maker_notional_fraction']*100:.2f}%였습니다.
참가자 부담 수수료는 {structure['total_participant_fees']:,.2f} USDP로, 가중 평균 {fee:.3f}bp입니다.
양쪽 모두 참가자인 거래는 참가자 명목액에 두 번 포함되므로 거래소 전체 거래량과 같은 수치가 아닙니다.
기간 중 거래한 팀은 종목별 10~23개이고, 최대 팀의 해당 종목 참가자 거래액 비중은 약 28~70%로 편중돼 있습니다.</p>
<p class="note">참가자별 포지션은 순차 조회한 스냅샷입니다. 시장 미결제약정과 비교한 보유 비율은 근사치이며,
시장 전체 활동이나 유동성 공급자의 신원을 완전히 관측한 자료는 아닙니다.</p>

<h2>2. 거래 가격·평가 가격·외부 참조 가격을 분리했습니다</h2>
{table(['가격','무엇을 뜻하나','전략에서의 용도'],[
('PAS bid/ask·깊이','우리 계좌가 거래할 독립 주문장의 호가','진입·청산 가격과 실행 가능한 수량'),
('PAS oracle','외부 가격 공급 체계가 제출하는 참조값','괴리 가설·펀딩 기준의 입력; 체결가 아님'),
('PAS mark','외부 참조값과 현지 호가 등을 반영하는 평가 가격','계좌 평가·증거금; 청산 대금과 다를 수 있음'),
('mainnet reference','BTC/ETH/HYPE 또는 xyz의 해당 무기한 선물','외부 방향·변동성 정보; PAS와 동일 상품·주문장 아님')])}
<p>대회는 Hyperliquid testnet의 전용 HIP-3 주문장을 사용합니다. PAS 마크 가격에는 오라클,
현지 가격과 EMA·변동 제한이 개입하므로 일반적인 mainnet 설명을 그대로 적용하면 안 됩니다.
USDEUR도 문서상 EUR/USD 방향이며, 이름만 보고 역환율로 처리하면 안 됩니다.
<a href="https://perpetual-alpha26.github.io/docs/">대회 거래·가격 문서</a>,
<a href="https://hyperliquid.gitbook.io/hyperliquid-docs/hyperliquid-improvement-proposals-hips/hip-3-builder-deployed-perpetuals">Hyperliquid HIP-3</a></p>
<p>따라서 PAS와 mainnet 가격 차이를 두 시장에서 동시에 잠글 수 있는 무위험 차익으로 계산하지 않았습니다.
mainnet 참조도 현물 가격 자체가 아니며, USDP testnet 잔고를 같은 가치의 실제 달러로 바꿀 수 있다는 전제도 두지 않았습니다.</p>
<img src="market_structure.png" alt="PAS와 mainnet 및 오라클 간 괴리, 종목별 호가 스프레드"/>
{table(['종목','PAS−mainnet (bp)','PAS−oracle (bp)','스프레드 (bp)','새 체결 수','정합 초 표본'],[
(r['symbol'],f"{r['main_basis_bps']['median']:+.2f}",f"{r['oracle_basis_bps']['median']:+.2f}",
 f"{r['spread_bps']['median']:.2f}",r['actual_new_trades'],r['aligned_seconds']) for r in aligned['markets']])}
<p class="meta">표는 중앙값입니다. 02:03:15~02:18:17 KST 수신 기록 중 양 주문장 서버 시각 차이가 2초 이내인 자료를 초당 한 번 정렬했습니다.
오라클 컨텍스트는 3초, 주문장은 15초 이내 자료만 사용했습니다. 02:14:15 PAS 연결 단절 이후의 결측을 보간하지 않았습니다.
거래 수는 최초 구독 때 받은 과거 체결 묶음을 제외한 고유 신규 체결 22건입니다. 연결 단절 뒤 거래가 없었다는 뜻은 아닙니다.</p>
<p>HYPE는 PAS와 mainnet 중간호가가 약 1.12bp 차이였지만, PAS 자체 오라클과는 약 10.48bp 차이였습니다.
참조를 무엇으로 고르는지가 신호를 바꿉니다. SP500은 약 23.74bp 할인됐지만 관측된 신규 체결은 0건이어서,
괴리 크기로 체결 확률이나 수렴 시간을 대신할 수 없습니다.</p>

<h2>3. 수수료·체결 빈도가 현재 후보의 수익을 제한합니다</h2>
<p>우리 계정의 현재 수수료는 maker 3bp, taker 9bp입니다. taker 진입·청산은 수수료만 18bp,
maker 진입 후 taker 청산은 12bp입니다. 두 번 모두 maker로 체결돼도 6bp가 필요합니다.
실제 수수료 응답과 위 참가자 체결 집계가 이 수준과 일치합니다.</p>
<p>외부 가격으로 완전히 수렴하고 스프레드가 그대로라는 단순 가정에서도,
왕복 taker의 괴리 수익에서 현 스프레드와 18bp 수수료를 빼면 BTC·ETH·HYPE는 음수입니다.
SP500도 약 −0.83bp입니다. 이는 실제 예측치가 아니라 비용의 크기를 보여주는 근사 계산이며,
추가 충격 비용과 방향 위험을 넣기 전입니다.</p>
<p>maker 전환도 자동 해법은 아닙니다. BTC·GOLD·USDJPY의 관측 중앙 스프레드는 6bp에 못 미칩니다.
다른 종목의 작은 차익 여지도 대기 순서·한쪽만 체결되는 재고 위험·체결 직후 불리한 가격 이동을 반영하면 달라집니다.
당시 양쪽 최우선 호가 중 작은 쪽의 종목별 중앙 명목액은 약 54~500 USDP 수준으로, 화면에 호가가 있다는 사실과
원하는 수량을 원하는 시점에 거래할 수 있다는 사실을 구분해야 합니다.</p>
{table(['종목','대회 기간 관측 5분 봉 중 거래 있는 봉','연속 활성 봉 쌍','평균 펀딩 (bp/시간)'],[
(s,f"{m['scored_active_5m_bars']}/{m['scored_completed_5m_bars']}",
 m['consecutive_active_basis_pairs'],f"{m['realized_funding_mean_bps_hour']:+.3f}") for s,m in markets.items()])}
<p class="note">봉 분모는 반환된 완료 봉 범위 안의 개수입니다. 종목마다 마지막 체결 시점이 달라 범위가 같지 않습니다.
빈 봉이나 낡은 종가를 새 거래·새 학습 표본으로 세지 않았습니다. 연속 활성 봉 쌍은 별도의 전체 가용 정합 구간 진단입니다.
10일을 요청했지만 PAS 가격 이력은 약 2일뿐이었습니다. 긴 독립 실전 검증 데이터가 생긴 것은 아닙니다.</p>
<p>음의 펀딩은 롱에 도움이 될 수 있으나 지속·가격 위험이 보장되지 않습니다.
예를 들어 BTC의 위 평균 수취율 약 0.82bp/시간이 유지된다는 가정에서 왕복 taker 수수료 18bp만 회수하는 데도 약 22시간이 필요합니다.
스프레드·가격 하락·펀딩 반전은 별도입니다. 연구 후보는 유리한 미래 펀딩을 선반영해 알파로 만들지 않습니다.
<a href="https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding">Hyperliquid 펀딩 설명</a></p>

<h2>4. 하락장 여부도 기간을 고정해야 합니다</h2>
{table(['mainnet 참조','최근 24시간','최근 72시간','최근 7일'],[
(r['symbol'],*[f"{r['returns_percent'][h]:+.2f}%" for h in ['24','72','168']]) for r in returns])}
<p>마지막 완료 5분 봉 기준 시각은 {kst(returns[0]['as_of']/1000)}입니다.
주간 약세와 단기 반등이 동시에 관측됩니다. 따라서 하락 상태를 분류한 모델이 있다는 이유만으로
현재가에서 숏 진입의 비용 차감 기대수익이 양수라고 추론할 수 없습니다.
PAS가 오라클보다 할인되고 음의 펀딩인 상황에서 숏은 그 비용도 함께 부담합니다.</p>

<h2>5. 구현한 개선과 현재 검증 결과</h2>
{table(['구현','검증한 동작 / 범위'],[
('PAS 기준 후보 정책','오래된 외부 종가 대신 수신 시각이 확인된 PAS 오라클과 호가를 사용. 방향 변동성·수수료·깊이·청산 비용을 합산'),
('두 가지 고정 후보','오라클 괴리의 일부 수렴 가설을 taker와 maker 방식으로 각각 평가. 수렴율 50%, 보유 60초는 검증할 가정이며 학습으로 입증된 계수가 아님'),
('maker 대기 순서','주문 도착 뒤 실제 반대방향 거래량으로만 가상 체결. 앞 대기 물량을 먼저 소진. 단순 호가 접촉·취소만으로 체결하지 않음'),
('IOC·부분 청산','가격 제한과 표시 깊이 적용. 이미 사용한 호가 수량을 반복 사용하지 않음. 관측된 수량 증가만 보충. 남은 포지션을 공짜로 청산 처리하지 않음'),
('데이터 품질','수신·서버 시각, 연결 결측, 중복 거래, 과거 구독 스냅샷, 미래 참조를 구분'),
('검증 무결성','코드·원시 입력 해시와 시작·종료·비용 시나리오 고정. 중간 결과를 보고 후보를 다시 고르지 않음'),
('보호 범위','공개 읽기 전용 API. 서명키·거래 전송 경로 없음. 연구 결과로 운용 전략·자금·레버리지를 자동 변경하지 않음')])}
<p>수정본의 회귀 테스트 <strong>11개가 통과</strong>했습니다. 추가 통합 시나리오에서는 일부 청산 뒤 동일 호가와 타 시장 이벤트가 반복돼도
잔여 물량이 체결되지 않고, 실제 표시 수량 증가가 관측된 뒤에만 다음 가상 청산이 발생함을 확인했습니다.</p>
{table(['짧은 기록 재생','판단 횟수','후보 허용','체결','손익','채택'],[
(r['mode'],r['checks'],r['accepted_proposals'],r['completed_trades'],r['completed_net_pnl'],'불가') for r in short['models']])}
<p><strong>0손익은 수익성 개선 증거가 아닙니다.</strong> 비용·위험 조건에서 거래할 후보가 없었다는 뜻입니다.
이 자료는 연결 단절을 포함하며, 수정한 검증기로 이미 관측한 자료를 재생한 사후 진단입니다.
독립적인 미래 실험으로 부르지 않습니다. 처음 등록했던 02:25 관측은 검증기 구현 점검 중 중단했고,
결과를 선택에 사용하지 않은 채 원래 계획·소스·중단 사유를 보존했습니다.</p>
<p>앞서 만든 최종 위험 필터 후보도 재사용 역사에서 원안보다 손익·Q가 낮았습니다.
기존 페어 전략의 희소 체결과 작은 기본 포트폴리오 기여, mainnet 신호와 PAS 비용의 불일치가 해결됐다고 볼 근거는 아직 없습니다.
{link(ROOT/'reports/strategy_repairs_20261021/report.html','앞선 실행 장애 수정·위험 후보 결과')},
{link(ROOT/'reports/strategy_full_audit_20261021/report.html','전체 전략 감사')}</p>

<h2>6. 새 72시간 평가 — 진행 중, 채택 결정 아님</h2>
<p><strong>{kst(plan['start'])} → {kst(plan['end'])}</strong><br>
수정된 코드·기준을 고정하고 PAS/mainnet 공개 호가·체결과 기존 전략의 실제 발행 신호·계좌 관측을 기록합니다.
기준 시나리오와 깊이 감소·지연 증가·청산 충격 증가 시나리오를 종료 후 자동 재생해 결과 파일을 저장합니다.</p>
{table(['항목','고정한 내용'],[
('가상 주문 크기','종목당 제안 250 USDP; 종목 한도 500 / 모델 총한도 2,500'),
('기준 비용','maker 3bp / taker 9bp, 표시 깊이 25%, 지연 1초, 추가 청산 충격 10bp'),
('악화 시나리오','표시 깊이 10%, 지연 3초, 추가 청산 충격 20bp'),
('펀딩','정시 결제 전에 대기·보유·청산 유예가 끝나도록 새 진입 차단. 결제 시각을 넘긴 잔여 물량이 있으면 비용 불명으로 판정'),
('최소 증거 조건','기준·악화 모두 완료 순손익 양수, 거래 있는 비중첩 시간 블록 30개 이상, 블록 평균 손익의 95% 부트스트랩 하한 양수'),
('불완전 자료','연결 결측·미해결 상태·잔여 포지션·미상 펀딩이 있으면 전체 구간의 수익성 통과로 인정하지 않음'),
('실전 채택','항상 자동 비활성. 구성요소 손익 조건 통과만으로 전체 포트폴리오 Q 개선이나 실전 적격성을 주장하지 않음')])}
<p>이는 최소한의 사전 점검입니다. 72시간·서로 겹치지 않는 시간 블록도 독립 표본을 보장하지 않으며,
국면 변화·장기 자기상관·새 유동성 공급 행동을 충분히 검증하지 못합니다.
현재 방식의 부트스트랩 신뢰구간은 블록 사이 종속성에 민감합니다.
짧은 시험에서 보이는 이익만으로 배정액 확대나 장기 수익률을 정당화하지 않습니다.</p>
<p>기존 전략 신호를 함께 기록하지만, 후보와 기존 전략을 합쳤을 때의 전체 계좌 Q,
한쪽 체결에 따른 꼬리 손실, 자본 규모별 시장 충격까지 검증한 것은 아닙니다.
새 모델이 기본 포트폴리오를 대체할지, 소규모 추가 전략이 될지 역시 미결정입니다.
이번 후보는 60초 괴리 수렴 가설을 시험하므로, 기존 장기 방향·페어 전략의 모든 한계를 해결한 모델이 아닙니다.
공식 리더보드도 수집 시점에는 모든 팀이 일별 수익률 표본 부족으로 Q·점수 산정 적격이 아니었습니다.
<a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a>
과거 중단된 별도 시험과 겹치는 날짜를 두 개의 독립 실험으로 세지 않습니다.</p>

<h2>7. 실행 상태와 재현 자료</h2>
<p>작성 시점 기존 운용·신호 서비스 정상, 포지션 BTC {status['qty'][0]:.5f} / HYPE {status['qty'][2]:.2f},
누적 체결 {status['fills']}건입니다. 이 연구가 낸 주문은 0건입니다.
원래 모델·데이터·기준 파일 {verification['preserved_baseline_files']}개, 실행 배포 파일
{verification['deployment_file_count']}개와 제출 묶음 해시를 검증했습니다.</p>
<p>지갑 로그인은 이번 공개 데이터 조사와 주문 없는 검증에 필요하지 않았습니다.
연구 서비스는 <code>icaif-pas-venue-research-forward-v2.service</code>이며, 예정 종료 뒤 결과를 로컬에 저장합니다.
별도 알림은 설정하지 않았습니다. 현재 상태는 아래 health 파일의 최신 시각으로 확인할 수 있습니다.</p>
<ul>
<li>{link(OUT/'market_structure.json','공개 계좌·시장 구조 집계')} · {link(OUT/'synchronized_analysis.json','동시 호가 정합 결과')} · {link(OUT/'reference_returns.json','참조 시장 수익률')}</li>
<li>{link(ROOT/'pas_venue_research/policy.py','후보 정책 코드')} · {link(ROOT/'pas_venue_research/shadow.py','체결·비용 재생 코드')} · {link(ROOT/'pas_venue_research/forward.py','72시간 관측 코드')}</li>
<li>{link(OUT/'policy_tests.txt','회귀 테스트')} · {link(OUT/'replay_tests.txt','부분 청산 통합 검증·과거 재생')} · {link(OUT/'final_verification.json','운용·해시 검증')}</li>
<li>{link(OUT/'forward_72h/plan.json','고정한 평가 계획')} · {link(OUT/'forward_72h/health.json','진행 상태')} · {link(OUT/'forward_72h_aborted_20261021_0225/aborted.json','이전 시도 중단 기록')}</li>
<li>종료 후 생성 경로: <code>forward_72h/evaluation_status.json</code>, <code>shadow_results.json</code>, <code>shadow_stress_results.json</code></li>
</ul>
<p class="note">현재 판정: 시장 구조 조사·후보/검증기 구현 완료, 미래 검증 진행 중, 수익성 개선 미입증.
실전 전략·규모의 변경은 이 보고서의 성과가 아닙니다.</p>
"""

css="""
*{box-sizing:border-box}body{margin:0;background:#f6f8fc;color:#172638;font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;line-height:1.8}
main{max-width:1120px;margin:auto;padding:48px 28px 80px}h1{font-size:38px;line-height:1.35;letter-spacing:-1px;margin:12px 0 24px}h2{margin-top:46px;font-size:25px}
.eyebrow{letter-spacing:2px;font-size:12px;color:#315dba;font-weight:750}.lead{font-size:19px;max-width:980px}.meta{font-size:13px;color:#617187}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:30px 0}.cards article{background:white;border:1px solid #d9e2ef;border-radius:12px;padding:20px}.cards strong{display:block;font-size:21px}.cards span{display:block;font-size:13px;color:#617187}
table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:11px 14px;border-bottom:1px solid #e2e9f1;text-align:left;vertical-align:top}th{background:#e8eef7;white-space:nowrap}
.scroll{overflow:auto}.note{padding:15px 20px;border-left:4px solid #d88924;background:#fff9ed}a{color:#315dba;text-underline-offset:3px}img{width:100%;border-radius:12px}
code{font-size:13px;overflow-wrap:anywhere}li{margin:8px 0}@media(max-width:720px){main{padding:22px 16px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}h2{font-size:22px}}
"""
(OUT/'report.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SFINX PAS 시장 구조·수익성 검증</title><style>'+css+'</style></head><body><main>'+content+'</main></body></html>')
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='evidence_manifest.json'}
(OUT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(report=str(OUT/'report.html'),verification={
    k:v for k,v in verification.items() if k not in ['status','producer','health']},
    forward_state=health['state'],forward_errors=health['total_connection_errors'],
    forward_deadline=kst(plan['end'])),ensure_ascii=False))
