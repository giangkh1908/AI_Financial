"""run_etl.py — chạy ETL toàn corpus (M1): wide tables + catalog + documents.

- Checkpoint theo ticker: `data/derived/etl_state.json` — chạy lại bỏ ticker đã xong.
- Catalog ghi theo phần mỗi ticker (`catalog_parts/catalog_{ticker}.csv`), cuối cùng
  gộp thành `catalog_tables.csv` — KHÔNG re-process toàn bộ khi resume.
- Lỗi report → ghi `data/derived/etl_errors.tsv`, ticker vẫn đánh dấu xong (không brick).
- Parallel 4–8 worker (ThreadPool — lxml giải phóng GIL).

Chạy:  .venv\\Scripts\\python scripts/run_etl.py [--workers 6] [--tickers HPG,VCB]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Console Windows (cp1252) không in được tiếng Việt → ép UTF-8
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.config import Config  # noqa: E402
from vifinqa.etl.catalog_builder import (  # noqa: E402
    CatalogRow,
    merge_catalog_parts,
    process_report,
    write_catalog_csv,
    write_documents_csv,
)
from vifinqa.loader import iter_reports, load_stocks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--config", type=str, default="api.yaml")
    ap.add_argument("--tickers", type=str, default="", help="chỉ chạy các ticker, phân cách ','")
    args = ap.parse_args()

    cfg = Config.load(ROOT / "configs" / args.config)
    data_dir = cfg.resolved_data_dir()
    derived_dir = cfg.resolved_derived_dir()
    derived_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = derived_dir / "catalog_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    stocks = load_stocks(data_dir / "code_stock.csv")
    reports = iter_reports(data_dir)
    if args.tickers:
        wanted = set(args.tickers.split(","))
        reports = [r for r in reports if r.ticker in wanted]

    # Group theo ticker để checkpoint
    by_ticker: dict[str, list] = {}
    for r in reports:
        by_ticker.setdefault(r.ticker, []).append(r)

    state_path = derived_dir / "etl_state.json"
    state: dict[str, int] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    errors_path = derived_dir / "etl_errors.tsv"

    todo = [t for t in by_ticker if t not in state]
    print(f"Ticker tổng: {len(by_ticker)}, đã xong: {len(by_ticker) - len(todo)}, sẽ chạy: {len(todo)}")

    if todo:
        t0 = time.time()
        # rows tích luỹ theo ticker (chỉ ticker đang chạy)
        ticker_rows: dict[str, list[CatalogRow]] = defaultdict(list)
        total_reports = sum(len(by_ticker[t]) for t in todo)
        completed_per_ticker = {t: 0 for t in todo}

        def _finalize_ticker(ticker: str) -> None:
            """Ghi catalog part + checkpoint khi xong hết report của 1 ticker."""
            write_catalog_csv(ticker_rows.get(ticker, []), parts_dir / f"catalog_{ticker}.csv")
            state[ticker] = 1
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_report, r, derived_dir): r for t in todo for r in by_ticker[t]}
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                try:
                    rows = fut.result()
                    ticker_rows[r.ticker].extend(rows)
                except Exception as e:  # noqa: BLE001
                    with open(errors_path, "a", encoding="utf-8", newline="") as f:
                        f.write(f"{r.report_id}\t{type(e).__name__}\t{e}\n")
                    print(f"  [err] {r.report_id}: {e}", file=sys.stderr)
                completed_per_ticker[r.ticker] += 1
                if completed_per_ticker[r.ticker] == len(by_ticker[r.ticker]):
                    _finalize_ticker(r.ticker)
                if i % 200 == 0 or i == total_reports:
                    print(f"  ...{i}/{total_reports} report, {time.time()-t0:.0f}s")
        print(f"Hoàn tất {len(todo)} ticker trong {time.time()-t0:.0f}s.")

    # Gộp catalog từ các phần đã có (không re-process)
    n_rows = merge_catalog_parts(parts_dir, derived_dir / "catalog_tables.csv")
    write_documents_csv(reports, stocks, derived_dir / "documents.csv")
    print(f"Catalog: {n_rows} bảng -> {derived_dir / 'catalog_tables.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
