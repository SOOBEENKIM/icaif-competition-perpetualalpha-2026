"""Summarize an already completed cleanup and mark the old inventory historical."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "cleanup_records/20261015_selected_strategy"
INV = ROOT / "reports/internal_strategy_inventory_20261015"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    manifest, deleted, after = [read(REC / name) for name in ("manifest.json", "deleted.json", "after_verification.json")]
    assert not deleted["preserved_mismatches"]
    assert all(r["exact_targets"] and r["rows"] == 5006 for r in after["profiles"])
    assert after["same_signal_fixed_scale_exact"]
    assert all("\nOK\n" in (REC / name).read_text() for name in ("tests.log", "registry_tests.log"))
    old_report = INV / "report.html"
    text = old_report.read_text()
    original = REC / "inventory_report.before.html"
    if not original.exists():
        original.write_text(text)
    # Prevent dead links from being mistaken for currently retained result files.
    retired = []
    def replace(match):
        href, label = html.unescape(match.group(1)), match.group(2)
        if href.startswith(("https:", "http:", "#")) or (INV / href).exists():
            return match.group(0)
        retired.append(href)
        return f'<span title="2026-10-15 선택 전략 정리로 삭제됨">{label} <small>[정리됨]</small></span>'
    text = re.sub(r'<a href="([^"]+)">(.*?)</a>', replace, text, flags=re.S)
    marker = '<!-- selected-strategy-cleanup -->'
    if marker not in text:
        notice = marker + '<div class="notice"><b>이 문서는 정리 전 성과 스냅샷입니다.</b> 2026-10-15 사용자의 선택에 따라 추세＋되돌림 기본 규모와 원래 규모 비교본을 남기고 불필요한 실험 파일을 삭제했습니다. 아래 과거 용량·권고는 정리 전 값입니다. 현재 상태는 <a href="../../cleanup_records/20261015_selected_strategy/report.html">정리 완료 보고서</a>를 확인하세요. 삭제된 원본 링크는 [정리됨]으로 표시합니다.</div>'
        text = text.replace('<h1>내 전략 버전·성과·정리 목록</h1>', '<h1>내 전략 버전·성과·정리 목록</h1>' + notice)
    old_report.write_text(text)
    (INV / "CLEANUP_NOTICE.txt").write_text("This inventory is a pre-cleanup historical snapshot. See ../../cleanup_records/20261015_selected_strategy/report.html for retained files and validation. Deleted source references do not denote currently available files.\n")
    expected_edits = {"replay_lab/README.md", "reports/internal_strategy_inventory_20261015/report.html"}
    mismatches = []
    checked = 0
    for row in manifest["keep"]:
        if row["path"] in expected_edits:
            continue
        p = ROOT / row["path"]
        if not p.exists() or ("sha256" in row and sha(p) != row["sha256"]):
            mismatches.append(row["path"])
        checked += 1
    assert not mismatches, mismatches
    assert not any((ROOT / row["path"]).exists() for row in manifest["delete"] if "__pycache__" not in Path(row["path"]).parts)
    with (REC / "retained_files.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "reason"])
        w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in manifest["keep"])
    audit = {"preserved_files_rechecked": checked, "unexpected_changes": mismatches,
             "intentional_document_updates": sorted(expected_edits | {"README.md"}),
             "source_profiles_unchanged": True, "whole_history_rows": 5006,
             "profiles_exact": [r["profile"] for r in after["profiles"]],
             "core_tests_passed": 21, "registry_tests_passed": 13,
             "deleted_files": deleted["deleted_files"], "deleted_bytes": deleted["deleted_bytes"],
             "live_orders_sent": 0, "new_profitability_evaluation": False,
             "historical_inventory_links_retired": len(retired)}
    (REC / "final_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    def link(path, label):
        return f'<a href="{html.escape(os.path.relpath(ROOT/path, REC), quote=True)}">{html.escape(label)}</a>'
    entries = [
        ("strategy/profiles.json", "기본/비교 프로필 설정"), ("strategy/agent.py", "공통 진입점"),
        ("replay_lab/reversion_blend/model.py", "추세＋되돌림 공통 신호"),
        ("replay_lab/tail_calibration/agent.py", "기본 고정 배율"),
        ("replay_lab/output/tail_calibration/20261014-v1/test_result.json", "기본 규모 원본 성과"),
        ("replay_lab/output/reversion_blend/20261014-v1/test_result.json", "원래 규모 원본 성과"),
        ("reports/wallet_leaderboard_20261015/report.html", "타 지갑 비교"),
        ("reports/internal_strategy_inventory_20261015/report.html", "정리 전 성과 목록"),
    ]
    rows = ''.join(f'<tr><td>{html.escape(k)}</td><td>{v/1e6:,.2f}</td></tr>' for k,v in list(manifest['delete_by_group_bytes'].items())[:15])
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>선택 전략 정리 완료</title><style>body{{font:16px/1.7 system-ui;max-width:1050px;margin:35px auto;padding:20px;color:#1d2a3a}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;text-align:left;padding:9px}}.notice{{padding:16px;background:#eff5fc}}a{{color:#135999}}</style>
<h1>추세＋되돌림 전략으로 정리 완료</h1>
<p>기본 규모 보정판과 원래 규모 비교본을 하나의 진입점으로 관리합니다. 신호·모델 가중치·고정 배율은 변경하지 않았습니다.</p>
<table><tr><th>프로필</th><th>공통 신호</th><th>원래 비교본 대비 규모</th></tr><tr><td>default (기본값)</td><td>crypto_d7_aligned_add50</td><td>69.38788714042263%</td></tr><tr><td>original</td><td>같음</td><td>100%</td></tr></table>
<p class="notice"><b>{deleted['deleted_files']:,}개 파일 · {deleted['deleted_bytes']/1e9:.3f} GB 삭제.</b> 선택 전략과 관련 없는 실험 코드·파생 결과·UI·중복 모델 패키지·불필요한 논문 실행 환경을 정리했습니다.</p>
<p>선택 전략의 모델·입력·원본 결과·동결 실행 패키지, 공용 원본 시장자료, 지갑 비교 자료, 평가 노출 이력, 계좌 설정과 비밀키, Python 환경을 유지했습니다. 일부 과거 이름의 모듈은 선택 전략의 import 의존성으로 남아 있습니다.</p>
<p>삭제 전과 후 모두 <b>두 프로필 × 5,006개 시간 관측</b>의 목표 포지션이 저장된 원본과 정확히 일치했습니다. 재시작 상태 보존과 프로필 혼용 거부도 확인했습니다. 계좌·주문 처리·지갑 회계 21개 테스트, 평가 Registry 13개 테스트를 통과했습니다. 보존 파일 {checked:,}개의 해시/존재를 재확인했습니다.</p>
<p>선택 전략의 동결 당시 출처 소스 182개는 <a href="selected_frozen_sources.zip">약 0.38MB 소스 스냅샷</a>으로 보존했습니다. 전체 삭제 대상의 백업을 만든 것은 아닙니다. 삭제 전후 증거와 경로 목록을 아래에 제공합니다.</p>
<ul>{''.join('<li>'+link(p,l)+'</li>' for p,l in entries)}</ul>
<p><a href="manifest.json">파일별 보존/삭제 명세</a> · <a href="retained_files.csv">보존 파일 CSV</a> · <a href="deleted.json">삭제 실행 기록</a> · <a href="before_verification.json">삭제 전 재현</a> · <a href="after_verification.json">삭제 후 재현</a> · <a href="final_audit.json">최종 감사</a></p>
<h2>주요 정리 용량</h2><table><tr><th>경로</th><th>삭제 MB</th></tr>{rows}</table>
<p>현재 어댑터는 오프라인 목표 포지션/주문 의도 계산용입니다. 이번 작업은 수익성 개선 실험이나 실제 거래 시작이 아닙니다. 실제 주문 0건.</p></html>'''
    (REC / "report.html").write_text(page)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
