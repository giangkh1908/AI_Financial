"""rebuild_evidence.py — sinh tidy evidence `evidence/{rid}__{tid}.csv` TOÀN BỘ mọi bảng.

Mọi bảng (statement + notes/segment) → tidy schema cố định [chi_tieu, Mãsố, ky, value]
dùng layout từ `layouts/{rid}.json` (nguồn sự thật) — không tự đoán cột.

- Parallel worker theo report (thread-safe: mỗi report 1 file layout đọc độc lập).
- Bảng không có layout / không có cột kỳ → ghi file header-only (query → 0.0, không crash).
- Idempotent (overwrite). Chạy ~2-4 phút.

Chạy:  .venv\\Scripts\\python scripts/rebuild_evidence.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from vifinqa.config import Config  # noqa: E402
from vifinqa.etl.tidy import write_tidy_csv  # noqa: E402

_TIDY_COLS = ["chi_tieu", "Mãsố", "ky", "value"]


def _process_report(report_id: str, tables_dir: Path, layouts_dir: Path, ev_dir: Path) -> int:
    tdir = tables_dir / report_id
    if not tdir.is_dir():
        return 0
    lpath = layouts_dir / f"{report_id}.json"
    layout_data: dict = {}
    if lpath.exists():
        try:
            layout_data = json.loads(lpath.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            layout_data = {}
    n = 0
    for csvf in sorted(tdir.glob("table_*.csv"), key=lambda p: int(p.stem.split("_")[1])):
        tid = csvf.stem
        dst = ev_dir / f"{report_id}__{tid}.csv"
        try:
            grid = pd.read_csv(csvf, dtype=str, header=None, keep_default_na=False).values.tolist()
            from vifinqa.etl.tidy import grid_to_tidy

            tidy = grid_to_tidy(grid, _report_year(report_id), layout=layout_data.get(tid))
            write_tidy_csv(tidy, dst)
            n += 1
        except Exception:  # noqa: BLE001
            # header-only fallback — không brick report
            write_tidy_csv(pd.DataFrame(columns=_TIDY_COLS), dst)
    return n


def _report_year(report_id: str) -> int:
    import re

    m = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", report_id)
    return int(m.group(0)) if m else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = Config.load(ROOT / "configs" / "api.yaml")
    derived = cfg.resolved_derived_dir()
    tables_dir = derived / "tables"
    layouts_dir = derived / "layouts"
    ev_dir = derived / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)

    report_ids = sorted(p.name for p in tables_dir.iterdir() if p.is_dir())
    print(f"Report: {len(report_ids)}")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_report, rid, tables_dir, layouts_dir, ev_dir): rid for rid in report_ids}
        for i, fut in enumerate(as_completed(futs), 1):
            done += fut.result()
            if i % 200 == 0 or i == len(report_ids):
                print(f"  ...{i}/{len(report_ids)} report, {done} bảng, {time.time()-t0:.0f}s")
    n_files = len(list(ev_dir.glob("*.csv")))
    print(f"Xong: {done} bảng tidy → {n_files} file evidence | {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
