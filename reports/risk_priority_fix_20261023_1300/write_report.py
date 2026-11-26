import html,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
from competition_live.profile import ROOT
out=Path(__file__).parent
stage=json.loads((out/'stage_check.json').read_text())
replay=json.loads((out/'replay_09.json').read_text())
verified=json.loads((out/'activation_receipt.json').read_text()) if (out/'activation_receipt.json').exists() else None
kst=lambda t:datetime.fromtimestamp(t,timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
status=f"실행 확인: {kst(verified['verified_at'])}" if verified else '배포 후 첫 정상 실행 확인 중'
text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>PAS 실행 오류 수정 · 2026-10-23</title>
<style>body{{font:16px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#18303d}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd8dc;padding:10px;text-align:left}}code{{background:#edf4f6;padding:2px 6px}}small{{color:#586b75}}</style>
<h1>PAS 본대회 실행 오류 수정</h1><p><b>{status}</b></p>
<p>전략 <code>all_market_signed_core_v1 + pullback_any3</code>를 유지하고 실행 조정 계층을 <code>risk_priority_resume_v1</code>로 수정했습니다. 모델 재학습·기간 선택·포지션 확대를 통한 성능 조정은 없습니다.</p>
<table><tr><th>문제</th><th>수정</th><th>검증</th></tr>
<tr><td>모델 위험 감축 때 같은 회차의 정상 목표까지 삭제</td><td>목표와 원래 만료시간을 보존. 실제 부분 체결 잔량과 현재 위험 한도를 확인한 뒤 이어서 처리.</td><td>실제 09시 수량 재생, 부분 체결·중복·미체결·위험 한도 재확대 방지</td></tr>
<tr><td>10 USDP 미만 부분 감축이 계속 판단을 선점</td><td>소액 잔여 감축을 별도 기록하고 다른 종목 및 기존 페어 관리를 계속. 잔여 수량을 체결했다고 처리하지 않음.</td><td>공동 지갑의 실제 순포지션 기준 전량 청산 예외, 혼합 감축, 빈 주문 반복 방지</td></tr>
<tr><td>주문 준비 중 목표 만료 가능</td><td>정상 리밸런싱 주문은 사전 점검 전후와 실제 전송 직전에 유효시간 검사. 미전송 거절로 처리.</td><td>미완료 주문을 남기지 않고 전송 차단</td></tr></table>
<p>소액 유예는 모델 목표와 실제 보유 간 차이를 허용하는 실행상 처리입니다. 전 계좌 위험 한도·낙폭 중단·오래된 신호 청산·종료 시 청산은 기존 강제 경로를 유지합니다. 거래소 최소 주문 제약 자체가 없어지는 것은 아닙니다.</p>
<h2>검증 결과</h2><p><b>최종 77개 테스트 통과</b> (신규 16개 + 기존 61개). 기존 선택 전략과 저장된 PAS 판단 결과의 일치 검증도 포함합니다.</p>
<table><tr><th>09시 관측 목표 재생</th><th>기존</th><th>수정 후</th></tr><tr><td>먼저 필요한 감축</td><td>BTC·USDEUR</td><td>BTC·USDEUR</td></tr><tr><td>다음 처리의 나머지 목표</td><td>삭제됨</td><td>ETH·HYPE·EWY·GOLD 유지</td></tr></table>
<p><b>재생은 당시 관측한 보유·목표 수량과 재구성한 가격, 고정 공분산·가상 호가·가상 체결로 실행 순서를 검증한 것입니다. 해당 추가 거래가 당시 실제로 체결되거나 수익을 냈다는 증거가 아닙니다.</b></p>
<h2>거래소 인계 검증</h2><p>{kst(stage['at'])}: 공개 체결·펀딩·계좌를 대조해 우리 기존 체결 {stage['own_fills']}건 및 보유 수량, 팀원 장부를 유지했습니다. 서명 없는 현재 시점 실행 점검에서 예정 주문은 {html.escape(str(stage['first_scheduled_order']))}였습니다.</p>
<p>우리 보유: {html.escape(str(stage['own_qty']))}<br>팀원 보유: {html.escape(str(stage['teammate_qty']))}<br>종목 순서: BTC, ETH, HYPE, SP500, EWY, EWJ, BRENT, GOLD, USDEUR, USDJPY.</p>
<p>정상 판단은 기존대로 03·09·15·21시(KST), 유효시간은 경계부터 15분입니다. 만료된 09시 목표를 현재 주문으로 복원하지 않았습니다. 위험 감시와 기존 페어 관리는 각 실행 회차에 진행합니다.</p>
<h2>보존 및 복귀</h2><p>기존 소스·설정은 <code>rollback/</code>, 교체 직전 장부는 <code>before_switch/</code>에 보관했습니다. 과거 장부를 현재 장부에 덮어쓰면 이후 체결이 사라지므로, 복귀할 때도 최신 장부와 거래소를 먼저 대조해야 합니다. 이전 전략으로 돌아갈 경우 <code>baseline_rollback_policy.json</code>을 사용하면 이번 실행 오류 수정은 유지할 수 있습니다.</p>
<h2>근거 파일</h2><ul>
<li><a href="tests_all_final.txt">77개 최종 검증</a></li><li><a href="tests_before.txt">수정 전 실패 재현</a></li><li><a href="replay_09.json">09시 재생 비교</a></li><li><a href="stage_check.json">서명 없는 계좌·실행 점검</a></li><li><a href="activation_receipt.json">배포 후 실제 실행 확인</a></li>
<li><a href="{ROOT/'candidate_handoff/risk_priority.py'}">실행 수정 코드</a></li></ul>
<p><b>판정:</b> 확인된 실행 오류를 고치는 배포입니다. 예측력이나 향후 PnL 개선이 검증되었다는 판정은 아닙니다. 실제 추가 진입·청산과 손익은 배포 이후 기록으로 평가해야 합니다.</p></html>'''
(out/'report.html').write_text(text)
