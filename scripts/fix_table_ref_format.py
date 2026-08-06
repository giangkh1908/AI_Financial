"""fix_table_ref_format.py — Convert relevant_tables từ `report_id|N` → `report_id|table_N`.

Bug: `search.py.relevant_tables_key()` cũ trả về `report_id|{position}` (thiếu `table_`),
trong khi format chuẩn benchmark (DSKT-NOWJ/ViFinQA `common/schemas/table_ref.py`:
`make_table_ref` → `f"{doc_name}|table_{table_id}"`, `parse_table_ref` tách bằng "|table_")
là `report_id|table_N`. → F2 retrieval = 0 dù trúng bảng.

Script này sửa format trên results.jsonl đã có (idempotent — chỉ thêm `table_` khi phần
sau `|` là số nguyên thuần), rồi chạy lại build_submission để đóng gói zip mới.

Chạy:
  python scripts/fix_table_ref_format.py            # sửa + build + pack
  python scripts/fix_table_ref_format.py --no-build  # chỉ sửa results.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "revalidate"):
    pass
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
INT_RE = re.compile(r"^-?\d+$")


def convert_ref(ref: str) -> str:
    """`report_id|N` → `report_id|table_N`. Bỏ qua nếu đã có `table_` hoặc N không phải số."""
    if not ref or "|" not in ref:
        return ref
    doc, _, suffix = ref.rpartition("|")
    if not doc or not suffix:
        return ref
    if suffix.startswith("table_"):
        return ref  # đã đúng format
    if not INT_RE.match(suffix):
        return ref  # không phải số thuần → để nguyên (tránh hỏng)
    return f"{doc}|table_{suffix}"


def fix_file(path: Path) -> tuple[int, int]:
    """Sửa results.jsonl in-place (backup `.bak`). Trả (n_records, n_changed_refs)."""
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    n_rec = 0
    n_changed = 0
    lines_out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rec += 1
            rec = json.loads(line)
            refs = rec.get("relevant_tables") or []
            new_refs: list[str] = []
            for r in refs:
                nr = convert_ref(r)
                if nr != r:
                    n_changed += 1
                new_refs.append(nr)
            rec["relevant_tables"] = new_refs
            lines_out.append(json.dumps(rec, ensure_ascii=False))
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for l in lines_out:
            f.write(l + "\n")
    return n_rec, n_changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="results.jsonl (mặc định data/out/results.jsonl)")
    ap.add_argument("--no-build", action="store_true", help="không chạy build_submission sau khi sửa")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from vifinqa.config import Config  # noqa: E402

    cfg = Config.load(ROOT / "configs" / "api.yaml")
    out_dir = cfg.resolved_out_dir()
    results_path = Path(args.results) if args.results else out_dir / "results.jsonl"

    if not results_path.exists():
        print(f"❌ Không thấy results: {results_path}")
        return 1

    n_rec, n_changed = fix_file(results_path)
    print(f"✅ Đã sửa {results_path}")
    print(f"   records={n_rec}  refs đã convert={n_changed}  (backup: {results_path}.bak)")

    if args.no_build:
        print("--no-build → dừng. Chạy `python scripts/build_submission.py` để đóng gói zip mới.")
        return 0

    print("\n[build] Chạy build_submission.py ...")
    import subprocess
    cmd = [sys.executable, str(ROOT / "scripts" / "build_submission.py")]
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())