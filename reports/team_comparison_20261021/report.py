"""Offline, searchable evidence report with CSV exports."""
import csv
import datetime as dt
import html
import json
from pathlib import Path

O=Path(__file__).resolve().parent;ROOT=O.parents[1]
USER='0x125778065a578d598d88bddaa3fdd635280d0e47'
E=lambda x:html.escape(str(x))
def n(x,d=2):return '—' if x is None else f'{x:,.{d}f}'
def pct(x,d=4):return '—' if x is None else n(x,d)+'%'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+E(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def addr(a):return a[:10]+'…'+a[-6:]
def link(r):return '<a href="#w'+r['address']+'">'+('SFINX ' if r['name'] else '')+addr(r['address'])+'</a>'

def main():
    d=json.loads((O/'analysis.json').read_text());s=d['summary'];own=s['sfinx'];off=s['sfinx_official'];teams=d['official']
    wallets=sorted(d['wallets'],key=lambda r:(not r['funded_team'],-(r['return_pct'] if r['return_pct'] is not None else -1e10)))
    active=[r for r in wallets if r['funded_team'] and r['fills']]
    old=json.loads((ROOT/'reports/sizing_review_20261021/raw/leaderboard.json').read_text())['response'];old={x['name']:x for x in old['teams']}
    daily=[]
    for t in teams:
        if t['eligible'] and t['name'] in old:
            r1=old[t['name']]['return'];r2=(1+t['return'])/(1+r1)-1
            daily.append(dict(name=t['name'],day1_return=r1,day2_return=r2,derived_sigma=abs(r1-r2),official_sigma=t['volatility_horizon']))
    (O/'daily_returns_from_official.json').write_text(json.dumps(daily,ensure_ascii=False,indent=2))
    body=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PAS 참가 지갑 비교 · SFINX</title><style>body{font:15px/1.65 system-ui;color:#182538;background:#fafbfd;max-width:1450px;margin:32px auto;padding:0 22px}h1{font-size:28px}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{border:1px solid #dbe2ea;padding:8px;white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3f8;position:sticky;top:0}.scroll{overflow:auto;max-height:680px;margin:12px 0 24px}.box{border-left:5px solid #206878;background:#e9f4f5;padding:18px}.warn{border-color:#bc6b21;background:#fff4e8}a{color:#086ea0}code{word-break:break-all}input{width:min(650px,85%);padding:12px;font:inherit;border:1px solid #aab9c7;border-radius:5px}details{background:#fff;border:1px solid #dbe2ea;padding:14px;margin:16px 0}summary{cursor:pointer;font-weight:700}.muted{color:#637484}.nav{display:flex;gap:18px;flex-wrap:wrap}</style>']
    times=[r['snapshot_kst'] for r in wallets if r['snapshot_kst']]
    body += ['<h1>PAS 참가 지갑·공식 성과 비교</h1>',f'<p>본대회 시작: 2026-10-19 21:00 KST. 계좌 수집: {E(min(times))} ∼ {E(max(times))}. 여러 계좌를 순차 조회했으므로 동일 초의 동시 순위가 아니다. 공식 지표 생성 시각: {E(s["official_generated_at"])}.</p>', '<div class="nav"><a href="#official">공식 팀 순위</a><a href="#wallets">지갑별 비교</a><a href="#own">SFINX 상세</a><a href="#markets">종목별 롱·숏</a><a href="#details">모든 지갑 상세</a><a href="#methods">계산 방법·한계</a></div>']
    body += [f'<div class="box"><b>SFINX 공식 Score 순위 {off["rank"]} / {s["official_ranked_teams"]}팀.</b> 누적 수익률 {pct(off["return"]*100,7)}, 일별 표본표준편차 s {pct(off["volatility"]*100,7)}, 점수에 쓰는 기간 변동성 σ {pct(off["volatility_horizon"]*100,7)}, Q {n(off["q"],9)}, Score {n(off["score"],9)}. J는 현재 0이다.<br>공식 지급 기록에서 참가 지갑 {s["funded_wallets"]}개를 확인했고, PAS 체결을 확인한 지갑은 {s["funded_trading_wallets"]}개다. 참가 지급 기록 밖의 체결 상대 {s["collected_wallets"]-s["funded_wallets"]}개도 별도 조사했다.</div>']
    body += ['<p><b>확인 가능한 것은 거래 행동과 계좌 결과다.</b> 다른 팀의 소스 코드·학습 데이터·모델·진입 이유는 공개 체결만으로 알 수 없다. 팀 이름과 지갑 주소의 공식 대응표도 공개 피드에 없으며, SFINX 이외에는 이름을 임의 연결하지 않았다. 아래 공식 팀 표와 지갑별 실시간 표는 이 차이를 명시한다.</p>']
    body += ['<h2 id="official">공식 팀별 성과</h2><p>공식 순위는 매일 21:00 KST 시작 라운드의 첫 계좌 스냅샷(4분 이내, 누락 시 직전 값)으로 계산한다. 현재는 완료 일별 수익률 2개뿐이다. 일중 Top10 차트와 다르며, 심사점수는 현재 모두 0이다.</p>']
    body += [table(['Score 순위','팀','PnL R','일별 표준편차 s','기간 σ','Q','Score','평가 자격'],[[n(t['rank'],0),E(t['name']),pct(t['return']*100,6),pct(t['volatility']*100,6) if t['volatility'] is not None else '—',pct(t['volatility_horizon']*100,6) if t['volatility_horizon'] is not None else '—',n(t['q'],8),n(t['score'],8),'평가 대상' if t['eligible'] else E(', '.join(t.get('reasons') or []))] for t in teams])]
    body += ['<p>수익률만으로 순위가 정해지지 않는다. 예를 들어 Rentech2는 KKTY보다 R이 크지만 σ도 커서 Score 순위가 낮다. FINX는 SFINX와 다른 팀이며, R이 양수여도 변동성 때문에 27위다.</p>']
    body += [table(['공식 지표로 복원한 일별 수익률','10/19→10/20','10/20→10/21','기간 σ'],[[E(r['name']),pct(r['day1_return']*100,6),pct(r['day2_return']*100,6),pct(r['official_sigma']*100,6)] for r in daily if r['name'] in ('SFINX','Algo Hedge','KKTY','FINX')])]
    body += ['<h2 id="wallets">참가 지갑별 현재 거래·노출</h2><p>이 표의 순손익은 각 조회 시점 USDP 계좌가치에서 확인된 순입금을 뺀 값이다. 공식 하루 마감 Score 순위가 아니다. 모든 참가 지갑의 입출금과 체결 이력을 별도로 확인했다. 설정 배수와 실효 레버리지(총 포지션 명목액 / 계좌가치)를 구분한다.</p>']
    body += [table(['지갑','현재 순손익 USDP','현재 수익률','현재 방향','총 노출/자본','설정 배수','체결 주문/조각','거래 종목 수','수수료','펀딩 수령(+)','메이커 거래액 비율','미체결'],[[link(r),n(r['balance_net_pnl']),pct(r['return_pct']),E(r['current']),pct(r['gross_equity_pct'],2),E(', '.join(str(x)+'×' for x in r['configured_leverages']) or '—'),str(r['filled_orders'])+'/'+str(r['fills']),len(r['markets_traded']),n(r['fees']),n(r['funding']),pct(r['maker_notional_pct'],1),r['open_orders']] for r in wallets if r['funded_team']])]
    body += ['<h2 id="own">SFINX 계좌 전체와 기존 실행자의 구분</h2>',f'<p>계좌 조회 시각 {E(own["snapshot_kst"])}. 계좌가치 {n(own["equity"],6)} USDP, 순손익 {n(own["balance_net_pnl"],6)}, 수익률 {pct(own["return_pct"],6)}. 총 포지션 {n(own["gross_notional"])} USDP = 계좌의 {pct(own["gross_equity_pct"],4)}, 실효 총 레버리지 {n(own["effective_gross_leverage"],6)}배. 설정 레버리지는 1배다. 팀원 예산 5만을 분모에서 임의 제외하지 않고 팀 자본 전체와 비교했다.</p>']
    ownp=[p for p in d['positions'] if p['address']==USER]
    body += [table(['종목','방향','수량','평균 진입','마크','명목금액','미실현손익','설정 배수','증거'],[[E(p['coin']),p['side'],n(p['qty'],5),n(p['entry'],5),n(p['mark'],5),n(p['notional']),n(p['unrealized']),str(p['leverage'])+'×','기존 실행자 기록 있음' if p['coin']=='pas:BTC' else '기존 실행자 기록 밖; 주문 주체 미확정'] for p in ownp])]
    owna=[a for a in d['assets'] if a['address']==USER]
    body += [table(['종목','청산 실현손익','수수료','펀딩','미실현손익','합산 순손익'],[[E(a['coin']),n(a['closed_pnl'],6),n(a['fees'],6),n(a['funding'],6),n(a['unrealized'],6),n(a['net_pnl'],6)] for a in owna])]
    body += ['<div class="box warn">ETH 숏 0.4064개는 10/21 18:13:52.905에 체결됐다. 현재 봇의 주문 기록에 없는 거래다. 팀원 또는 다른 프로그램인지 미확인이다. BTC와 HYPE 진입·HYPE 청산은 기존 봇 기록과 일치한다. 따라서 계좌 전체 수익률을 기존 전략 하나의 성과라고 단정할 수 없다. 직전 점검에서 18:14:08부터 계좌와 원장 불일치로 실행자가 재시작을 반복했다. 이번 조사는 주문·원장·서비스를 변경하지 않았다.</div>']
    body += ['<h2 id="markets">참가 지갑의 종목별 롱·숏</h2>',table(['종목','체결한 참가 지갑','현재 롱 지갑','현재 숏 지갑','롱 명목액','숏 명목액'],[[E(r['coin']),r['teams_traded'],r['long_holders'],r['short_holders'],n(r['long_notional']),n(r['short_notional'])] for r in d['market_counts']])]
    body += [f'<p>현재 방향 분포: {E(s["current_direction_counts"])}. 본대회 누적 진입 방향 분포: {E(s["historical_direction_counts"])}. 참가 거래 지갑의 현재 총 노출/자본 중앙값은 {pct(s["median_gross_equity_pct"],3)}다. 같은 지갑이 여러 종목에 포함되므로 종목별 지갑 수를 더하면 팀 수와 다르다.</p>']
    body += ['<h2>참가 지급 기록 밖의 거래 상대</h2><p>관측된 두 지갑은 10만 USDP 지급 명단에 없어 참가팀 비교 집계에서 제외했다. 시장조성 행동과 유사해도 Keyrock 소유라는 공개 증거는 확인하지 못했다.</p>',table(['지갑','체결 조각','현재 방향','명목액','메이커 비율','데이터 한계'],[[link(r),r['fills'],E(r['current']),n(r['gross_notional']),pct(r['maker_notional_pct'],2),E('; '.join(r['history_issues']))] for r in wallets if not r['funded_team']])]
    body += ['<h2 id="details">모든 지갑 상세</h2><p>지갑 주소·종목으로 검색. 각 지갑을 펼치면 포지션, 종목별 손익, 최근 20개 체결을 볼 수 있다. 전체 체결은 CSV에 저장했다.</p><input id="search" placeholder="주소 또는 종목 검색: BTC, EWJ, 0x125…">']
    fills=list(csv.DictReader((O/'fills.csv').open(encoding='utf-8-sig')))
    for r in wallets:
        a=r['address'];ps=[x for x in d['positions'] if x['address']==a];ss=[x for x in d['assets'] if x['address']==a];fs=[x for x in fills if x['address']==a][-20:][::-1]
        body += [f'<details class="wallet" id="w{a}" data-search="{E(a+" "+" ".join(r["markets_traded"]))}"><summary>{E(r["name"] or addr(a))} · {n(r["balance_net_pnl"])} USDP · {E(r["current"])} · {r["filled_orders"]}개 체결 주문 / {r["fills"]}개 조각</summary><p><code>{a}</code> · 관측 {E(r["snapshot_kst"])} · 최초 {E(r["first_fill"])} / 최근 {E(r["last_fill"])}</p>']
        body += [f'<p>관측 행동: {E(r["historical"])}; 메이커 체결액 비율 {pct(r["maker_notional_pct"],2)}; 거래량 {n(r["volume"])} USDP. 현재 총 노출 {pct(r["gross_equity_pct"],3)}, 순노출 {pct(r["net_equity_pct"],3)}. 주문 유형 {E(r["order_types"])}; 주문 상태 {E(r["order_statuses"])}. 주문 이력은 API의 최근 최대 2,000개 한도가 적용된다.</p>']
        body += [table(['종목','방향','수량','진입가','마크','명목액','미실현손익','레버리지','증거금','청산가'],[[E(p['coin']),p['side'],n(p['qty'],5),n(p['entry'],5),n(p['mark'],5),n(p['notional']),n(p['unrealized']),str(p['leverage'])+'× '+E(p['margin_type']),n(p['margin_used']),E(p['liquidation_price'] or 'N/A')] for p in ps])]
        body += [table(['종목','실현손익','수수료','펀딩','미실현','순손익','롱 진입액','숏 진입액','완료 보유시간 중앙값(h)'],[[E(x['coin']),n(x['closed_pnl']),n(x['fees']),n(x['funding']),n(x['unrealized']),n(x['net_pnl']),n(x['open_long_notional']),n(x['open_short_notional']),n(x['median_completed_position_hours'])] for x in ss])]
        body += [table(['시각 KST','종목','진입/청산','매수/매도','가격','수량','실현손익','수수료','메이커/테이커','주문번호','상대'],[[E(f['time_kst']),E(f['coin']),E(f['dir']),'매수' if f['side']=='B' else '매도',E(f['px']),E(f['sz']),E(f['closedPnl']),E(f['fee']),'테이커' if f['crossed']=='True' else '메이커',E(f['oid']),E(addr(f['counterparty']) if f['counterparty'] else '확인 불가')] for f in fs])]
        body += [f'<p class="muted">회계 점검: 계좌 순손익 − 체결/펀딩/미실현 합산 = {n(r["accounting_snapshot_difference"],6)} USDP. 조회 간 가격·주문 변화와 진입가 반올림 때문에 차이가 생길 수 있다. 경고: {E("; ".join(r["history_issues"]) or "없음")}. 계좌 모드 {E(r["mode"])}.</p></details>']
    body += ['<h2 id="methods">계산·해석 범위</h2><ul><li>공식 R·s·σ·Q·Score는 주최 측 공개 JSON을 사용했다. Q 공식 재계산 오차를 검사했다. 지갑별 일중 순손익은 공식 일별 지표와 별개다.</li><li>거래 순손익 = 청산 실현손익 − 수수료 + 실제 펀딩 입출금 + 현재 마크 기준 미실현손익. 미실현손익은 실현된 수익이 아니다. 청산 시 추가 수수료·슬리피지가 발생할 수 있다.</li><li>unifiedAccount의 USDP total은 미실현손익을 이미 포함하므로 이를 다시 더하지 않았다. 일반 계좌는 USDP 현물 잔고와 PAS 파생 계좌가치를 합산했다. 자금 이동은 손익에서 제외했다.</li><li>실효 총 레버리지 = 롱·숏 명목액의 합 / 계좌가치. 설정 20배는 계좌 전체를 20배 투자했다는 뜻이 아니다. 설정 배수는 현재 포지션 기준이며 과거 변경 이력은 이 조회로 알 수 없다.</li><li>Open Long/Short와 Close Long/Short를 startPosition 전이로 구분했다. 매도 체결을 모두 숏 진입으로 세지 않았다. 보유시간은 0포지션에서 진입해 다시 0이 되거나 방향이 바뀐 구간의 길이이며 개별 부분청산의 FIFO 보유시간은 아니다.</li><li>API는 체결 응답당 최대 2,000개, 계좌별 최근 10,000개만 제공한다. 펀딩·체결은 시간 페이지를 겹쳐 중복 제거했고 한도 가능성을 공개했다. 최근 주문 이력은 최대 2,000개라 누적 주문 취소·거절 건수를 완전 집계한 것으로 볼 수 없다.</li><li>시장조성 지갑도 조회했지만 참가 지급 명단과 별도로 구분했다. 공개 데이터는 거래 알고리즘·팀원별 주문 주체·모델 성능을 직접 보여주지 않는다. 수익이 비슷하다는 이유로 이름과 주소를 연결하지 않았다.</li><li>현재 노출은 순간 상태다. 현재 현금 비중이 높아도 이전에 크게 거래했을 수 있으므로 누적 거래량·실현손익·체결 수를 함께 봐야 한다.</li></ul>']
    body += ['<p>출처: <a href="https://perpetual-alpha26.github.io/leaderboard">공식 리더보드</a> · <a href="https://d23m2l9bjnbgul.cloudfront.net/public/leaderboard.json">공개 공식 지표 JSON</a> · <a href="https://app.hyperliquid-testnet.xyz/trade/pas:BTC">Hyperliquid PAS</a> · <a href="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint">Hyperliquid 공개 API 문서</a>. 원문 응답과 조회 시각은 raw/에 저장했다.</p>']
    body += ['<h2>전체 자료 다운로드</h2><p>'+ ' · '.join(f'<a href="{x}.csv">{label}</a>' for x,label in [('official_teams','공식 47팀 지표'),('wallets','지갑별 요약'),('positions','전체 포지션'),('fills','전체 체결·거래 상대'),('by_market','종목별 손익'),('funding','실제 펀딩'),('open_orders','미체결 주문'),('historical_orders','최근 주문 이력'),('market_counts','종목별 방향 집계')])+'</p><script>document.getElementById("search").addEventListener("input",function(){let q=this.value.toLowerCase();document.querySelectorAll(".wallet").forEach(e=>e.hidden=!e.dataset.search.toLowerCase().includes(q));});window.addEventListener("hashchange",()=>{let e=document.getElementById(location.hash.slice(1));if(e&&e.tagName==="DETAILS")e.open=true;});</script></html>']
    (O/'report.html').write_text(''.join(body))
    print(O/'report.html')

if __name__=='__main__':main()
