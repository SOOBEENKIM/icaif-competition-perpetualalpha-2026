"""Explicit file-level cleanup for the user's selected trend/pullback family.

Default writes a reviewable manifest. --apply consumes that exact manifest.
Never follows directory symlinks or touches credentials, environments, or Git.
"""
import argparse
import ast
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "cleanup_records/20261015_selected_strategy"
TAIL = "replay_lab/output/tail_calibration/20261014-v1"
PARENT = "replay_lab/output/reversion_blend/20261014-v1"
SCOPES = ("replay_lab", "reports", "bot", "scripts", "tests")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(name, obj):
    RECORD.mkdir(parents=True, exist_ok=True)
    (RECORD / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def local_module(parts):
    p = ROOT.joinpath(*parts)
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    if (p / "__init__.py").is_file():
        return p / "__init__.py"


def dependencies(seeds):
    kept, queue = set(), [ROOT / s for s in seeds]
    while queue:
        p = queue.pop()
        if not p.is_file() or p in kept:
            continue
        kept.add(p)
        for parent in p.parents:
            if parent == ROOT:
                break
            if (parent / "__init__.py").is_file():
                queue.append(parent / "__init__.py")
        tree = ast.parse(p.read_text(), filename=str(p))
        package = list(p.parent.relative_to(ROOT).parts)
        for node in ast.walk(tree):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [x.name.split(".") for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                base += node.module.split(".") if node.module else []
                candidates = [base] + [base + x.name.split(".") for x in node.names if x.name != "*"]
            for parts in candidates:
                dep = local_module(parts)
                if dep:
                    queue.append(dep)
    return {str(p.relative_to(ROOT)) for p in kept}


SEEDS = [
    "strategy/agent.py", "strategy/verify.py",
    "bot/connection.py", "bot/check.py", "bot/setup.py", "bot/execution.py", "bot/smoke.py", "bot/verify_execution.py",
    "bot/market_data.py", "bot/public_data.py", "bot/portfolio.py", "bot/ledger_audit.py",
    "scripts/check_account.py", "scripts/check_api_key.py", "scripts/check_environment.py",
    "scripts/build_public_wallet_leaderboard.py", "scripts/refresh_public_wallets.py", "scripts/validate_public_wallet_snapshot.py",
    "scripts/prune_to_selected_strategy.py", "replay_lab/evaluation_control/registry.py",
    "replay_lab/evaluation_control/test_registry.py",
    "tests/test_bot_connection.py", "tests/test_execution.py", "tests/test_public_wallet_accounting.py",
]
PREFIX_KEEP = {
    TAIL: "선택한 기본 규모 버전의 코드·모델·데이터·평가·동결 패키지",
    PARENT: "원래 규모 비교본의 학습/평가 자료",
    "replay_lab/output/v10_revision": "공통 기준선과 개발 입력",
    "replay_lab/output/retained_strategies": "선택한 전략의 Donchian 기준선 및 근거",
    "replay_lab/output/evaluation_control": "평가 노출 이력; 과거 자료를 새 검증으로 오인하지 않도록 보존",
    "replay_lab/output/rules_review": "대회 규칙 근거",
    "replay_lab/output/competition_rules_update": "대회 규칙 변경 근거",
    "replay_lab/output/competition_public_funding": "대회 펀딩 원본",
    "replay_lab/output/competition_long_history": "공용 가격 원본",
    "replay_lab/output/data_availability_20261007": "시장 자료 확보 범위",
    "replay_lab/output/full_work_audit": "과거 연구 노출 감사",
    "reports/wallet_leaderboard_20261015": "타 지갑 비교 기준과 원본 체결",
    "reports/combined_leaderboard_20261014": "기존 타 지갑 비교 기준",
    "reports/wallet_review_20261014": "기존 계좌 조회 원본",
    "reports/internal_strategy_inventory_20261015": "정리 전 전략 성과 목록",
    "reports/market_opportunities_20261014": "공용 시장 조사·논문 근거",
    "reports/strategy_recovery_20261015/public_capture": "향후 선택 전략 검증에 사용할 원본 호가",
    "reports/maker_validation_20261014/prospective": "원본 호가·체결·펀딩",
    "reports/strategy_revision_v2_20261014/prospective": "원본 호가·체결·펀딩",
    "reports/directional_ridge_20261015/raw": "과거 가격·펀딩 원본",
    "reports/directional_ridge_20261015/reference_observer": "외부 가격 원본",
    "reports/directional_quotes_20261015/fresh_capture": "대회 호가 원본",
    "reports/directional_quotes_expanded_20261015/fresh_capture": "대회 호가 원본",
    "reports/directional_quotes_expanded_20261015/next_24h_capture": "대회 호가 원본",
    "replay_lab/output/frozen_forward/20261013-v8/inputs": "연속 시간봉 입력",
    "replay_lab/output/frozen_forward/20261013-v8/receipts": "가격·펀딩 수신 원본",
}


def reason(rel, closure):
    p = Path(rel)
    if "__pycache__" in p.parts or p.suffix == ".pyc":
        return None
    if rel in closure:
        return "실행·검증·계좌·공용 데이터 도구의 전이 import 의존성"
    for prefix, why in PREFIX_KEEP.items():
        if rel == prefix or rel.startswith(prefix + "/"):
            return why
    # Preserve non-code market observations in their original locations.
    if "observations" in p.parts and rel.startswith("replay_lab/output/"):
        return "기존 공개시장 관측 원본"
    if p.name == "initial-market.json" or (p.name.startswith(("market-", "reference-")) and p.suffix == ".jsonl"):
        return "공개시장·외부 기준가격 원본"
    if rel.startswith("reports/directional_ridge_20261015/forward/") and (p.name == "events.jsonl" or p.name.startswith("funding_") or p.name == "manifest.json"):
        return "대회 호가·펀딩 원본"
    if rel.startswith("replay_lab/output/") and ("receipts" in p.parts or "venue_probe" in p.parts):
        return "원본 API 가격·펀딩 응답"
    if rel.startswith("replay_lab/output/macro_development/20261013-v6/long_history/") and "funding" in p.name and "partial" not in p.name:
        return "공통 장기 펀딩 이력"
    if rel.startswith("reports/strategy_recovery_20261015/") and p.name in {"development_funding.json", "ledger_audit.json"}:
        return "공용 펀딩·원장 감사"
    if rel.startswith("replay_lab/competition_rules_update/") and p.suffix in {".json", ".txt", ".md"}:
        return "현행 대회 규칙"
    if rel.startswith("replay_lab/paper_reproductions/ema_wf/") and not any(x in p.parts for x in ("r-env", "upstream", "results")):
        if p.suffix == ".py":
            return None  # Only Python modules already in the active import closure survive.
        return "선택 전략에 사용된 EMA 논문 재현·출처 자료"
    if rel.startswith("replay_lab/output/portfolio_rebuild/20261013-v10/") and len(p.parts) == 5 and p.name in {"candidate.json", "summary.json", "protocol.json", "literature_to_implementation.json", "covariance.npy"}:
        return "선택 전략의 V10 선행 모델 근거"
    if rel.startswith("replay_lab/output/competition_rebuild/20261013-v9/") and len(p.parts) == 5 and p.name in {"candidate.json", "summary.json", "plan.json", "literature_crosswalk.json", "frozen_inputs.npz", "discord_source.txt"}:
        return "선택 전략의 V9 선행 모델 근거·입력"
    if rel in {"replay_lab/pairs_rl/protocol.json", "replay_lab/research_protocol.json", "replay_lab/experiment_protocol.json", "replay_lab/README.md"}:
        return "공용 실행 모듈 설정·설명"
    if p.parent == Path("replay_lab") and p.name.startswith("literature_review_"):
        return "논문 조사 출처"
    return None


def scan():
    for scope in SCOPES:
        for folder, dirs, files in os.walk(ROOT / scope, followlinks=False):
            # Symlinks are single entries; never enumerate external targets.
            links = [d for d in dirs if (Path(folder) / d).is_symlink()]
            dirs[:] = [d for d in dirs if d not in links]
            for name in files + links:
                yield Path(folder) / name


def plan():
    closure = dependencies(SEEDS)
    before = RECORD / "before_verification.json"
    if before.exists():
        observed = json.loads(before.read_text())["loaded_project_sources"]
        closure |= {p for p in observed if p.endswith(".py") and (ROOT / p).is_file()
                    and Path(p).parts[0] in {*SCOPES, "strategy"}}
    # The original freeze covered whole research directories, including unused
    # experiments. Preserve those exact small sources as provenance, not active code.
    snapshots = {}
    for prefix in (TAIL, PARENT):
        snapshots.update(json.loads((ROOT / prefix / "selected.json").read_text())["sources"])
    for p, h in snapshots.items():
        if sha(ROOT / p) != h:
            raise ValueError("Frozen source changed before cleanup: " + p)
    RECORD.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RECORD / "selected_frozen_sources.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(snapshots):
            archive.write(ROOT / p, p)
    with zipfile.ZipFile(RECORD / "selected_frozen_sources.zip") as archive:
        assert archive.testzip() is None
        for p, h in snapshots.items():
            assert hashlib.sha256(archive.read(p)).hexdigest() == h
    write("selected_frozen_sources.json", snapshots)
    keep, delete = [], []
    for p in scan():
        rel = str(p.relative_to(ROOT))
        why = reason(rel, closure)
        st = p.lstat()
        record = {"path": rel, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "symlink": p.is_symlink()}
        if why:
            record.update(reason=why)
            if not p.is_symlink():
                record["sha256"] = sha(p)
            keep.append(record)
        else:
            record["reason"] = "선택 전략 의존성·원본 데이터·비교/감사 근거에 포함되지 않는 실험/파생 산출물"
            if p.suffix == ".py" and not p.is_symlink():
                record["sha256"] = sha(p)
            delete.append(record)
    # Only the unused diffusion dependency list is removed outside enumerated scopes.
    p = ROOT / "requirements-diffusion.txt"
    if p.exists():
        st = p.stat()
        delete.append({"path": p.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "symlink": False, "reason": "삭제 대상 확산 모델 전용 의존성", "sha256": sha(p)})
    totals = Counter()
    for row in delete:
        parts = Path(row["path"]).parts
        key = "/".join(parts[:3] if parts[0] == "replay_lab" and len(parts) > 2 and parts[1] in {"output", "paper_reproductions"} else parts[:2])
        totals[key] += row["bytes"]
    manifest = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "root": str(ROOT),
                "keep": keep, "delete": delete, "active_sources": sorted(closure),
                "delete_bytes": sum(r["bytes"] for r in delete), "keep_bytes": sum(r["bytes"] for r in keep),
                "delete_by_group_bytes": dict(totals.most_common()),
                "protected_outside_scope": [".git", ".agents", ".codex", ".secrets", "config.local.json", ".venv", ".tools", ".cache", "history", "strategy", "cleanup_records"]}
    write("manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"keep", "delete", "active_sources"}}, ensure_ascii=False, indent=2))
    print("KEEP", len(keep), "DELETE", len(delete), "ACTIVE_SOURCES", len(closure))


def apply():
    if (RECORD / "deleted.json").exists():
        raise RuntimeError("Cleanup already applied; do not silently repeat")
    manifest = json.loads((RECORD / "manifest.json").read_text())
    assert manifest["root"] == str(ROOT)
    verified = json.loads((RECORD / "before_verification.json").read_text())
    assert len(verified["profiles"]) == 2 and verified["same_signal_fixed_scale_exact"]
    assert all(r["exact_targets"] and r["rows"] == 5006 for r in verified["profiles"])
    # Validate every intended deletion before deleting the first file.
    for row in manifest["delete"]:
        rel = Path(row["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        assert rel.parts[0] in SCOPES or str(rel) == "requirements-diffusion.txt"
        p = ROOT / rel
        assert p.parent.resolve().is_relative_to(ROOT)
        st = p.lstat()
        if st.st_size != row["bytes"] or st.st_mtime_ns != row["mtime_ns"]:
            raise RuntimeError("File changed after plan: " + str(rel))
    deleted = []
    for row in manifest["delete"]:
        (ROOT / row["path"]).unlink()
        deleted.append(row["path"])
    for scope in SCOPES:
        for folder, dirs, files in os.walk(ROOT / scope, topdown=False, followlinks=False):
            p = Path(folder)
            if not p.is_symlink() and not any(p.iterdir()):
                p.rmdir()
    mismatches = [r["path"] for r in manifest["keep"] if not (ROOT / r["path"]).exists()
                  or ("sha256" in r and sha(ROOT / r["path"]) != r["sha256"])]
    result = {"completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "deleted_files": len(deleted),
              "deleted_bytes": manifest["delete_bytes"], "preserved_hashes_checked": len(manifest["keep"]),
              "preserved_mismatches": mismatches, "deleted_paths": deleted}
    write("deleted.json", result)
    assert not mismatches, mismatches
    print(json.dumps({k: v for k, v in result.items() if k != "deleted_paths"}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    apply() if args.apply else plan()
