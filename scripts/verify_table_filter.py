"""verify_table_filter.py — Test giả thuyết numbering BTC.

BTC (DSKT-NOWJ/ViFinQA): table_N = số thứ tự dày đặc trên các bảng vượt filter
`is_table_eligible` (table_filters.py + constants.py):
  - n_rows >= MIN_ROWS (3)
  - count_numeric_cells >= MIN_NUMERIC_CELLS (6)
  - numeric regex: ^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?%?$  (VN format: . nghin, , thập phân)

Nếu áp filter lên 146,246 wide table của ta mà survivor ~143,815 (số HF card công bố)
→ xác nhận table_N của BTC = rank dày đặc trong reading order trên survivor.

Chạy: python scripts/verify_table_filter.py
"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "data" / "derived" / "tables"

MIN_ROWS = 3
MIN_NUMERIC_CELLS = 6
NUMERIC_RE = re.compile(r"^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?%?$")


def is_numeric_cell(cell: str) -> bool:
    cell = (cell or "").strip()
    if not cell or cell == "-":
        return False
    return bool(NUMERIC_RE.match(cell))


def check_csv(path: Path) -> tuple[int, int, int, bool]:
    """Trả (n_rows, n_numeric, n_cols_total, eligible)."""
    import csv
    n_rows = 0
    n_numeric = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            n_rows += 1
            for cell in row:
                if is_numeric_cell(cell):
                    n_numeric += 1
    eligible = (n_rows >= MIN_ROWS) and (n_numeric >= MIN_NUMERIC_CELLS)
    return n_rows, n_numeric, 0, eligible


def iter_csvs():
    for report_dir in TABLES_DIR.iterdir():
        if not report_dir.is_dir():
            continue
        yield from report_dir.glob("table_*.csv")


def main() -> int:
    csvs = list(iter_csvs())
    print(f"Total wide CSVs: {len(csvs)}")

    total = 0
    eligible = 0
    dropped_rows = 0      # n_rows < 3
    dropped_numeric = 0   # numeric < 6 (nhưng rows OK)
    both_drop = 0
    # phân phối để debug
    per_report: dict[str, tuple[int, int]] = {}  # report_id -> (total, eligible)

    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(check_csv, p): p for p in csvs}
        done = 0
        for fut in as_completed(futs):
            p = futs[fut]
            n_rows, n_num, _, elig = fut.result()
            total += 1
            rid = p.parent.name
            t, e = per_report.get(rid, (0, 0))
            per_report[rid] = (t + 1, e + (1 if elig else 0))
            if elig:
                eligible += 1
            else:
                if n_rows < MIN_ROWS:
                    dropped_rows += 1
                else:
                    dropped_numeric += 1
            done += 1
            if done % 20000 == 0:
                print(f"  ...{done}/{len(csvs)}  eligible so far={eligible}")

    print()
    print(f"=== KẾT QUẢ ===")
    print(f"Tổng wide CSV        : {total}")
    print(f"Eligible (qua filter): {eligible}")
    print(f"Dropped (n_rows < 3) : {dropped_rows}")
    print(f"Dropped (numeric < 6): {dropped_numeric}")
    print(f"Tỷ lệ survivor       : {eligible/total:.4%}")
    print()
    print(f"Mục tiêu HF card     : 143,815 normalized tables")
    print(f"Hiệu số              : {eligible - 143815:+d} (so với 143,815)")
    print()
    # kiểm tra per-report: survivor có dày đặc không (để reuse numbering)
    # in 5 report đầu
    for rid in list(per_report)[:5]:
        t, e = per_report[rid]
        print(f"  {rid}: total={t} eligible={e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())