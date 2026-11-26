import datetime as dt
import hashlib
import html
import json
from pathlib import Path
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
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def link(path,label):
    return '<a href="'+html.escape(str(path))+'">'+html.escape(label)+'</a>'


deployment=read('deployment_result.json')
risk=read('risk_history_results.json')
construction=read('risk_construction.json')
economics=read('economics_snapshot.json')
status=json.loads((ROOT/'.runtime/pas_competition_live/status.json').read_text())
producer=json.loads((ROOT/'.runtime/pas_competition_live/producer_status.json').read_text())
tests=(OUT/'final_tests.txt').read_text()
assert 'Ran 38 tests' in tests and tests.rstrip().endswith('OK')
assert status['state']=='running' and not status['fatal'] and status['signal_error'] is None
assert time.time()-status['at']<90 and status['reserved_cash']==50000
assert status['qty']==deployment['after']['qty']
assert status['fills']==deployment['after']['fills']
verification=dict(at=time.time(),test_count=38,tests_pass=True,
    preserved_baseline_files=len(protected_hashes()),bundle_hash=verify_bundle(),
    deployment_file_count=len(verify_deployment()['files']),status=status,producer=producer,
    executed_revision='20261021-recovery-freshness-v1',
    risk_candidate_live=False,alpha_model_validated=False,capital_increased=False,
    code_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((ROOT/'strategy_final_risk').glob('*.py'))})
(OUT/'final_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))

content=f"""
<h1>SFINX 수정 및 검증 결과</h1>
<p class="meta">{kst(verification['at'])} · PAS testnet · 운용 배정액 50,000 USDP / 예비자금 50,000 USDP</p>
<p class="lead">재현된 운영 장애 3건은 실행 서비스에 반영했습니다. 최종 위험 필터 후보와 PAS 비용·규모 검증기는 별도로 구현했습니다.
위험 후보는 과거 비교에서 채택 기준을 통과하지 못해 실전에 반영하지 않았습니다. PAS 순수익 알파가 확보됐다는 결론도 내리지 않았습니다.</p>
<h2>1. 실제 적용 범위</h2>
{table(['항목','수정 전','수정 후 / 상태'],[
('재시작 복구','신호가 오래되면 보유 계좌 복구 전에 대기','기존 체크포인트부터 복구하고 위험 감축 수행. 신규 평탄 계좌만 신호 대기. 적용 완료'),
('순차 주문 호가','앞 주문의 체결 확인 중 뒤 종목 호가 만료','필요한 종목마다 펀딩·체결 확인 이후 호가 재조회. 오래된 조회 결과에는 주문하지 않음. 적용 완료'),
('미보유 종목 결측','USDJPY 가격 누락 하나로 BTC 감축도 중단','물리·소유자 장부 모두 0인 종목만 0 명목가치로 처리. 신규 위험 차단, 확인된 보유 종목 감축 허용. 적용 완료'),
('신호 무결성','실행 벡터와 프레임 해시의 일치 검사가 없음','실행 목표를 검증 프레임과 비교. 손상된 신호는 신규 진입에 쓰지 않고 기존 위험관리로 전달. 적용 완료'),
('확정되지 않은 체결','복구 시 중단','기존 중단·중복 주문 방지 동작 보존. 실제 체결 기록만 반영'),
('최종 위험 규칙','JM/Chronos 뒤의 정규화·되돌림이 위험 감축을 상쇄','후보 수정 및 검증 완료. 실전 미적용'),
('PAS 알파·규모','원시 비중만으로 비용 차감 기대수익을 알 수 없음','비용·깊이·펀딩·검증 알파 조건을 별도 계산. 자본 확대 적격성 없음; 실전 규모 유지')
])}
<p>가격이 없는 실제 보유 종목이나 상쇄된 소유자 포지션에는 가짜 가격을 넣지 않습니다. 이 경우 평가가 중단되는 제약은 남습니다.
0 명목가치 표식은 체결 가격이 아니며, 실제 주문에는 해당 종목의 새 호가가 필요합니다.</p>
<h2>2. 재현과 운영 검증</h2>
<p>수정 전 새 테스트 8개 중 5개 실패·1개 오류로 결함을 고정했습니다. 수정 후 운영 관련 31개,
최종 위험 계약 3개, PAS 비용·규모 4개로 총 <strong>38개 테스트가 통과</strong>했습니다.
4초 체결 확인 지연 뒤의 두 번째 종목 체결, 오래된 호가 차단, 신호 없는 복구, 결측 종목이 있는 실제 감축,
부분 체결·잘못된 수수료 통화·미확정 체결·예비자금 보존을 포함합니다.</p>
<p>실계좌 체크포인트 사본을 공개 API와 대조하는 주문 없는 점검도 통과했습니다. 서명키·주문 클라이언트는 만들지 않았습니다.
이후 {kst(deployment['restart_at'])} 기존 실행 서비스를 재시작했습니다.
확인 시각 {kst(status['at'])}: state={status['state']}, fatal=[], BTC={status['qty'][0]}, HYPE={status['qty'][2]},
확정 체결 {status['fills']}건, 예비자금 {status['reserved_cash']:,.0f} USDP.
신호 서비스는 {producer['state']}입니다. 적용 후 점검 구간에서 보유 수량과 체결 수는 일치했습니다.</p>
<p>고정 모델·데이터·기준 코드 {verification['preserved_baseline_files']}개와 제출 묶음 해시는 유지됐습니다.
수정 실행본의 {verification['deployment_file_count']}개 배포 파일 해시도 일치합니다.
새 코드가 시작 실패하면 코드만 되돌리며, 과거 체크포인트로 계좌 기록을 덮어쓰지 않도록 배포 스크립트를 작성했습니다.</p>
<h2>3. 최종 위험 후보: 구현과 채택은 별도</h2>
<p>순서는 ‘위험 필터 전 추세 규모 계산 → 기존 되돌림 절반 합산 → 기존 한도·고정 배율 → 합산 방향에 JM 및 sticky Chronos 감축’입니다.
두 필터의 기존 평균 결합을 유지하며 그 뒤 재정규화하지 않습니다.
새 숏 조건·종목 확대·총배율 상향은 추가하지 않았습니다.
기본 전략 소유자의 감축에는 20% 거래 생략 및 일반 회전율 제한이 감축을 상쇄하지 않게 했습니다.
동일 발행 신호에서 수량 상한을 고정해 매 가격 변화마다 다시 감축하는 것을 막았습니다.
페어 소유자는 별도이며 전체 계좌 한도는 유지됩니다.</p>
<p>검증 입력 끝 {kst(construction['last_source'])}: 예시 합산 비중 BTC {construction['last_before'][0]*100:.4f}% → {construction['last_after'][0]*100:.4f}%,
HYPE {construction['last_before'][2]*100:.4f}% → {construction['last_after'][2]*100:.4f}%.
이는 고정 입력 후보 예시이며 현재 실전 목표가 아닙니다. 48시간 뒤 데이터를 제거해도 앞선 후보 비중의 차이는 0이었습니다.</p>
{table(['구간','비용 시나리오','기존 순손익','후보 순손익','기존 Q','후보 Q'],[
(phase,sc,
 f"{next(r for r in risk if r['phase']==phase and r['scenario']==sc and r['variant']=='incumbent')['net_pnl']:.2f}",
 f"{next(r for r in risk if r['phase']==phase and r['scenario']==sc and r['variant']=='final_mix_risk')['net_pnl']:.2f}",
 f"{next(r for r in risk if r['phase']==phase and r['scenario']==sc and r['variant']=='incumbent')['Q']:.5f}",
 f"{next(r for r in risk if r['phase']==phase and r['scenario']==sc and r['variant']=='final_mix_risk')['Q']:.5f}")
for phase in ('validation','comparison','continuous') for sc in ('base','stress')])}
<p>validation=8월, comparison=9월~10월 15일, continuous=8월~10월 15일.
기준 전략 재구성 일치, 회계 오차 &lt; 10⁻⁶, 체결 시점 검사 통과, 전역 위험 위반 0건입니다.
하지만 재사용한 reference 가격과 조건부 체결 모형이며 독립 검증·PAS 실현 수익이 아닙니다.
역사 재생은 기존 6시간 의사결정의 감축 보호를 검사하고, 실제 후보의 창 밖 감축은 모의 체결 테스트로 검사했습니다.
<strong>연속 구간 순손익과 Q가 악화됐으므로 후보 자동 적용을 중단했습니다.</strong></p>
<h2>4. PAS 비용과 규모: 관측과 미검증 부분</h2>
<p>관측 시각 {kst(economics['at'])}. 아래는 매수 1,000 USDP를 현재 호가로 왕복하는 가정입니다.
미래 청산 호가는 보장되지 않으며, 24시간 펀딩은 예측이 아닌 불리한 비용 시나리오입니다.
유리한 펀딩을 미래 수익으로 미리 인정하지 않습니다.</p>
{table(['종목','현재 펀딩 bp/시간','호가+왕복 수수료 bp','추가 청산 충격+진입 여유 bp','요구 가격 알파 bp'],[
(m['symbol'],f"{m['funding_bps_hour']:.4f}",f"{r['roundtrip_book_and_fees_bps']:.2f}",
 f"{r['exit_impact_reserve_bps']+5:.2f}",f"{r['required_price_edge_bps']:.2f}")
for m in economics['markets'][:3] for r in m.get('sizes',[]) if r.get('notional')==1000 and r['side']==1])}
<p>현재 양 페어·양 방향 4개 모두 gate 거절이며, gate를 우회한 기존 planner도 4개 모두 edge_below_full_cost로 거절했습니다.
500/1,000/2,500/5,000/7,500 USDP 규모와 양 방향을 검사하도록 비용 검증기를 구현했습니다.
검증된 PAS 기대 가격수익이 없으면 최대 적격 확대 규모는 0입니다.
이는 기존 포지션 전량 청산 지시나 최적 규모가 0이라는 통계적 결론이 아니라 <em>확대 근거가 없다는 판정</em>입니다.</p>
<p><strong>남은 과제:</strong> 실제 공개 시점의 신호·PAS 비용을 고정하고 미래에 만기가 지난 결과로 비용 차감 수익을 검증해야 합니다.
현재 기본 전략 비중은 기대수익 값이 아니며, 이 부족분은 비용 계산 코드만으로 해결되지 않습니다.
기존 실전 위험 의미도 후보를 채택하기 전까지 유지됩니다. “모델 감축률이 최종 노출 감축률”이라는 설명은 현재 실행본에 적용할 수 없습니다.</p>
<h2>5. 재현 가능한 파일</h2>
<ul>
<li>{link(OUT/'operational.patch','실행본 변경 diff')} · {link(OUT/'deployment_result.json','실제 배포·복구 결과')}</li>
<li>{link(OUT/'operational_before.txt','수정 전 실패 증거')} · {link(OUT/'final_tests.txt','38개 테스트 결과')} · {link(OUT/'checkpoint_shadow.json','주문 없는 체크포인트 복구')}</li>
<li>{link(ROOT/'strategy_final_risk/model.py','최종 위험 후보')} · {link(ROOT/'strategy_final_risk/candidate.py','검증된 발행 계약과 감축')} · {link(OUT/'risk_protocol.json','평가 전 고정 규칙')}</li>
<li>{link(OUT/'risk_history_results.json','12개 역사 비교 결과')} · {link(OUT/'risk_decision.json','미채택 결정')}</li>
<li>{link(ROOT/'strategy_final_risk/economics.py','PAS 비용·규모 검증기')} · {link(OUT/'economics_snapshot.json','현재 비용 및 페어 결과')} · {link(OUT/'economics_receipts.json','공개 호가 원자료')}</li>
<li>{link(OUT/'final_verification.json','배포·기준 파일 해시 및 최종 상태')}</li>
</ul>
<p class="meta">검증 결과는 관측 시점과 재현한 조건의 결과입니다. 모든 미래 장애나 손익을 보증하지 않습니다. 지갑 재로그인은 필요하지 않았습니다.</p>
"""
style="""body{max-width:1120px;margin:40px auto;padding:0 24px;font-family:system-ui,sans-serif;color:#172738;line-height:1.8;background:#f8fafc}
h1{font-size:32px}h2{font-size:23px;margin-top:44px;border-top:1px solid #ccd5df;padding-top:20px}.lead{font-size:19px;background:#e6eff8;padding:20px;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border:1px solid #d3dbe3;padding:10px;text-align:left}th{background:#edf2f7}.meta{font-size:13px;color:#536170}a{color:#1656a0}"""
(OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>SFINX 수정·검증 결과</title><style>'+style+'</style><body>'+content+'</body></html>')
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='evidence_manifest.json'}
(OUT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(report=str(OUT/'report.html'),tests=38,live_state=status['state'],
    state_time=kst(status['at']),qty=status['qty'],fills=status['fills'],candidate_activated=False)))
