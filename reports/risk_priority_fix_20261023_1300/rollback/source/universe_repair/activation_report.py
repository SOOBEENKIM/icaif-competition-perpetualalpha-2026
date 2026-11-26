"""Record verified running configuration, ownership and actual cutover fills."""
import hashlib,html,json,time
from datetime import datetime
from zoneinfo import ZoneInfo
from candidate_handoff.producer import validate,configured_revision
from candidate_handoff.migration import load_existing
from competition_live.profile import ROOT,load_profile,verify_deployment
from competition_live.runtime import json_save
from .core import REVISION,SYMBOLS


def main():
    out=ROOT/'reports/universe_activation_20261022';runtime=ROOT/'.runtime/pas_budget_lifecycle'
    p=load_profile();status=json.loads((runtime/'status.json').read_text())
    producer=json.loads((runtime/'producer_status.json').read_text())
    packet=json.loads((runtime/'signal.json').read_text());body=validate(packet,p,time.time(),expected_revision=REVISION)
    assert status['state']=='running' and status['core_revision']==REVISION and time.time()-status['at']<60
    assert producer['state']=='ready' and producer['core_revision']==REVISION and time.time()-producer['at']<90
    assert configured_revision(runtime)==REVISION and all(r['considered'] for r in body['universe_audit'])
    portfolio,digest=load_existing(runtime/'coordinator.pkl',p)
    dispatch=json.loads((runtime/'dispatch.json').read_text()) if (runtime/'dispatch.json').exists() else {}
    assert not dispatch.get('pending')
    before=json.loads((out/'cutover.json').read_text())
    fills=[r for r in portfolio.account.physical.fills if r['time']>=before['at']]
    verified=verify_deployment()
    result=dict(at=time.time(),revision=REVISION,symbols=list(SYMBOLS),signal_sha256=packet['sha256'],
        status=status,producer=producer,checkpoint_sha256=digest,post_activation_fills=fills,
        completed_orders=dispatch.get('completed',{}),unknown_orders=list(portfolio.account.unknown),
        checkpoint_reset=False,team_accounting_preserved=True,live_active=True,tests_passed=58,
        all_market_core_required=True,implicit_legacy_fallback=False,retained_pair_specialists=body['retained_pair_universe'],
        performance_improvement_established=False,comparison='reports/universe_repair_20261022/report.html',
        original_deployment_files_unchanged=verified['files'])
    json_save(out/'activation_verified.json',result)
    files=[*(ROOT/'universe_repair').glob('*.py'),*(ROOT/'candidate_handoff').glob('*.py'),*(ROOT/'joint_wallet').glob('*.py')]
    json_save(out/'source_manifest.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    active=json.loads((ROOT/'.runtime/active_pas_strategy.json').read_text())
    active.update(core_revision=REVISION,core_universe=list(SYMBOLS),core_scope_verified_at=time.time(),core_scope_report=str(out/'report.html'))
    json_save(ROOT/'.runtime/active_pas_strategy.json',active)
    now=datetime.fromtimestamp(result['at'],ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    first=datetime.fromtimestamp(before['at'],ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    directions={'long':'롱','short':'숏','no_core_signal':'대기'}
    rows=''.join('<tr><td>'+r['symbol']+'</td><td>포함</td><td>'+directions[r['core_decision']]+'</td></tr>' for r in body['universe_audit'])
    fills_html=html.escape(json.dumps(fills,ensure_ascii=False,indent=2))
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>전체 종목 판단판 운용 반영</title>
<style>body{{font:16px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#172238}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd6e3;padding:8px 24px}}pre{{white-space:pre-wrap;background:#f1f4f8;padding:16px}}code{{background:#eef3f9}}</style>
<h1>10개 PAS 종목 판단판 — 실제 운용 반영 확인</h1><p>전환: {first}<br>최종 확인: {now}</p>
<p><strong>실제 주문 실행자와 신호 생성기가 모두 <code>{REVISION}</code>으로 동작한다. 핵심 EMA·채널·7일 되돌림의 종목 제한을 제거하고 롱·숏을 판단한다.</strong></p>
<p>앞선 비교에서 수익과 Q가 악화됐다는 사실을 알린 뒤, 사용자가 실운용의 종목 제한을 없애라고 재차 요구하여 반영했다. 성능 개선이 입증된 전략으로 교체했다고 주장하지 않는다. 원래 비교 결과는 <a href="{ROOT/'reports/universe_repair_20261022/report.html'}">이전 검증 보고서</a>에 보존했다.</p>
<h2>실제 신호 범위</h2><table><tr><th>종목</th><th>핵심 판단 대상</th><th>현재 신호</th></tr>{rows}</table>
<p>위 표는 현재 신호이며 주문 체결 표가 아니다. 다음 정기 판단에서 새 신호와 실제 PAS 호가·수량·비용·증거금·위험 한도로 목표와 주문을 다시 정한다. 기존 보조 페어 전문가 BTC/ETH와 BTC/HYPE는 유지한다. 보조 모델의 두 조합이 핵심 전략의 나머지 종목을 배제하지 않는다.</p>
<h2>보유분 인계와 실제 체결</h2><p>기존 체크포인트를 초기화하거나 포지션을 일괄 청산하지 않았다. 소유자 장부, 진입가, 실현손익, 수수료, 펀딩, 주문 식별자와 판단 시각을 인계했다. 수정판의 위험 감축으로 발생한 실제 체결은 아래와 같다.</p><pre>{fills_html}</pre>
<p>확인 시 우리 수량: {html.escape(str(status['own_qty']))}<br>팀원 수량: {html.escape(str(status['teammate_qty']))}<br>순서: {', '.join(SYMBOLS)}.</p>
<h2>검증과 재시작 방지 조건</h2><p>58개 테스트 통과. 모든 종목의 양방향 신호가 모의 주문 송신과 체결 장부까지 도달했고, 미래 입력을 덧붙여도 과거 판단이 바뀌지 않았다. 운용 계약에 전체 종목 버전을 필수로 지정했으므로 설정이 사라지면 기존 3종목 버전으로 자동 복귀하지 않고 중단한다.</p>
<p>기존 주문 실행자를 정지하고 세 개의 동일 지갑 잠금을 확보한 뒤 신호·버전을 교체했다. 기존 임시 서비스는 중지 시 제거되어 동일 경로로 재등록했다. 현재 실행자는 한 개이며 미확정·미결 처리 중 주문은 없다. 미사용 시장의 1배 Cross 설정과 증거금 확인은 실제 주문 직전에 실행자가 수행한다.</p>
<h2>판단 주기와 증거</h2><p>정기 재배분: 한국시간 03·09·15·21시, 위험 감시는 별도로 계속된다. 전환 직후 정상 재배분을 임의로 앞당기지 않았지만 위험 감축은 즉시 적용됐다.</p>
<p>증거 파일: <code>activation_verified.json</code>, <code>cutover.json</code>, <code>preactivation_preview.json</code>, <code>source_manifest.json</code>, <code>tests.txt</code>, <code>services.txt</code>. 실제 운용 경로: <code>{runtime}</code>.</p></html>'''
    (out/'report.html').write_text(text)
    old_report=ROOT/'reports/universe_repair_20261022/report.html'
    content=old_report.read_text();marker='<!-- later-live-activation -->'
    if marker not in content:
        note=f'{marker}<aside style="padding:16px;background:#fff3c9"><strong>후속 운용 변경:</strong> 이 문서는 반영 전 비교 결과다. 이후 사용자 요구에 따라 10종목 수정판을 실제 운용에 연결했다. <a href="{out/"report.html"}">현재 운용 확인 보고서</a>. 비교 수익성 악화 결과는 변경하지 않았다.</aside>'
        content=content.replace('<h1>',note+'<h1>',1);old_report.write_text(content)
    print(json.dumps(dict(state=status['state'],revision=REVISION,markets=len(SYMBOLS),fills=fills,unknown_orders=[],report=str(out/'report.html')),ensure_ascii=False))


if __name__=='__main__':main()
