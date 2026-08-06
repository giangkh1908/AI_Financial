"""nrows_distribution.py — Phân phối n_rows trên 146,246 wide CSV để tìm ngưỡng
mà BTC dùng (sao cho còn lại 143,815 = bỏ 2,431)."""
from __future__ import annotations
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "data" / "derived" / "tables"


def count_rows(path: Path) -> int:
    import csv
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return sum(1 for _ in csv.reader(f))


def main() -> int:
    csvs = list(p for d in TABLES_DIR.iterdir() if d.is_dir() for p in d.glob("table_*.csv"))
    print(f"Total: {len(csvs)}")
    dist = Counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(count_rows, p) for p in csvs]
        for i, fut in enumerate(as_completed(futs)):
            dist[fut.result()] += 1
            if (i + 1) % 30000 == 0:
                print(f"  ...{i+1}/{len(csvs)}")
    print("\n=== Phân phối n_rows ===")
    cumulative = 0
    for n in sorted(dist):
        cumulative += dist[n]
        kept = len(csvs) - cumulative
        marker = ""
        if kept == 143815:
            marker = "  <<< KẾT CẤU 143,815 (bỏ tất cả n_rows <= %d)" % n
        print(f"  n_rows={n:4d}: count={dist[n]:6d}  cum_drop={cumulative:6d}  kept={kept:6d}{marker}")
    # thử target
    print("\n=== Tìm ngưỡng giữ 143,815 ===")
    total = len(csvs)
    drop_needed = total - 143815
    cum = 0
    for n in sorted(dist):
        cum += dist[n]
        if cum >= drop_needed:
            print(f"  Bỏ n_rows <= {n}: drop={cum} kept={total-cum} (target 143815, diff {total-cum-143815:+d})")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())