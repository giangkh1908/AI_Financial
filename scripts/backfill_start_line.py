"""backfill_start_line.py — thêm cột `start_line` vào catalog_tables.csv từ OCR.

Không re-write 146K wide CSV. Đọc OCR mỗi report, tính dòng vật lý của mỗi
`<table>` (khớp `grep -n "<table"`), join vào catalog hiện có. Idempotent.

Sau khi chạy: `submission/builder.py` dùng `start_line` để xuất `relevant_tables`
theo format đặc tả BTC: `report_id|<start_line>` (vị trí dòng bắt đầu bảng trong OCR).

Chạy:
  python scripts/backfill_start_line.py [--config configs/api.yaml]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.config import Config  # noqa: E402
from vifinqa.etl.parser import TABLE_RE  # noqa: E402
from vifinqa.loader import iter_reports  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _start_lines_for_report(text: str) -> dict[int, int]:
    """table_idx (1-based) → số dòng vật lý của `<table>` trong full OCR text."""
    out: dict[int, int] = {}
    for i, m in enumerate(TABLE_RE.finditer(text), start=1):
        out[i] = text.count("\n", 0, m.start()) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "api.yaml"))
    args = ap.parse_args()

    cfg = Config.load(Path(args.config))
    derived = cfg.resolved_derived_dir()
    catalog_path = derived / "catalog_tables.csv"
    if not catalog_path.exists():
        print(f"Không tìm thấy {catalog_path}", file=sys.stderr)
        return 1

    # 1. Đọc catalog hiện có (multi-line rows → csv.DictReader an toàn)
    with catalog_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Catalog rows: {len(rows)}", flush=True)

    # 2. Build report_id → {table_idx: start_line} từ OCR
    data_dir = cfg.resolved_data_dir()
    reports = iter_reports(data_dir)
    print(f"Reports OCR: {len(reports)}", flush=True)
    rid_map: dict[str, dict[int, int]] = {}
    for r in reports:
        try:
            text = r.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rid_map[r.report_id] = _start_lines_for_report(text)

    # 3. Header: thêm start_line nếu thiếu
    header = list(rows[0].keys()) if rows else []
    if "start_line" not in header:
        header = header + ["start_line"]

    # 4. Fill start_line cho mỗi row
    set_count = 0
    miss_report = 0
    for row in rows:
        rid = row["report_id"]
        tid = row["table_id"]  # "table_N"
        try:
            idx = int(tid.split("_", 1)[1])
        except (ValueError, IndexError):
            idx = 0
        sl_map = rid_map.get(rid)
        if sl_map is None:
            miss_report += 1
            row["start_line"] = "0"
            continue
        sl = sl_map.get(idx, 0)
        row["start_line"] = str(sl)
        if sl:
            set_count += 1

    # 5. Ghi lại
    with catalog_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print(
        f"Xong: {set_count}/{len(rows)} row có start_line ≠ 0 | "
        f"report thiếu OCR: {miss_report}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())