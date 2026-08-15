"""build_submission.py — results.jsonl → submission.json + data/ → validate → submission.zip.

Chạy:
  python scripts/build_submission.py
  python scripts/build_submission.py --out data/out/submission_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.config import Config  # noqa: E402
from vifinqa.submission import build, pack, validate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "api.yaml"))
    ap.add_argument("--results", default=None, help="results.jsonl (mặc định data/out/results.jsonl)")
    ap.add_argument("--out", default=None, help="thư mục out (mặc định data/out/submission)")
    ap.add_argument("--tol", type=float, default=None, help="override answer_abs_tol")
    ap.add_argument("--no-strict", action="store_true", help="không fail khi validate < 100%")
    args = ap.parse_args()

    cfg = Config.load(Path(args.config))
    out_dir_root = cfg.resolved_out_dir()
    results_path = Path(args.results) if args.results else out_dir_root / "results.jsonl"
    out_dir = Path(args.out) if args.out else out_dir_root / "submission"
    questions_path = cfg.resolved_data_dir() / "questions" / "questions.jsonl"
    derived_dir = cfg.resolved_derived_dir()
    tol = args.tol if args.tol is not None else cfg.answer_abs_tol

    if not results_path.exists():
        print(f"❌ Không thấy results: {results_path}")
        return 1

    print(f"[1/3] Build submission.json + data/ → {out_dir}")
    summ = build(results_path, out_dir, derived_dir, questions_path)
    print(f"      n={summ['n']} materialized={summ['materialized']} tidy_regen={summ.get('tidy_regen', 0)} missing_ids={len(summ['missing_ids'])}")
    print(f"      evidence_filtered={summ.get('evidence_filtered', 0)} relevant_tables_rewritten={summ.get('relevant_tables_rewritten', 0)}")
    if summ["missing_ids"]:
        print(f"      ⚠️ {len(summ['missing_ids'])} câu thiếu → fallback 0.0 (vd: {summ['missing_ids'][:5]})")

    print(f"[2/3] Validate (re-exec, tol={tol})")
    val_path = out_dir / "validation.jsonl"
    vsumm = validate(
        out_dir / "submission.json", out_dir / "data", val_path,
        abs_tol=tol, timeout=cfg.sandbox.timeout,
        max_code_len=cfg.sandbox.max_code_len, max_ast_nodes=cfg.sandbox.max_ast_nodes,
    )
    print(f"      {json.dumps(vsumm)}")

    print(f"[3/3] Pack ZIP")
    zip_path = pack(out_dir)
    print(f"      → {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")

    total = vsumm["total"]
    if total == 0:
        print("❌ submission rỗng")
        return 1
    ok_rate = vsumm["ok"] / total
    print(f"\nValidate ok: {vsumm['ok']}/{total} = {ok_rate:.1%}")
    if ok_rate < 1.0 and not args.no_strict:
        print("⚠️  Validate chưa 100% — xem validation.jsonl. Dùng --no-strict để vẫn pack (đã pack rồi).")
    if vsumm["crash"] or vsumm["mismatch"]:
        print(f"   crash={vsumm['crash']} mismatch={vsumm['mismatch']} bad_path={vsumm['bad_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())