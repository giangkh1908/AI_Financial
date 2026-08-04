"""run_facts.py — chạy facts_builder (M2) toàn corpus: facts/{report_id}_facts.csv + facts_all.csv.

- Checkpoint theo ticker: `data/derived/facts_state.json` — chạy lại bỏ ticker đã xong.
- Facts ghi theo phần mỗi ticker (`facts_parts/facts_{ticker}.csv`), cuối cùng gộp
  thành `facts_all.csv` — KHÔNG re-process khi resume.
- Lỗi report → ghi `data/derived/facts_errors.tsv`, ticker vẫn đánh dấu xong (không brick).
- Parallel worker (ThreadPool — lxml/BeautifulSoup giải phóng GIL).

Chạy:  .venv\\Scripts\\python scripts/run_facts.py [--workers 6] [--tickers HPG,VCB]
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
from vifinqa.etl.facts_builder import (  # noqa: E402
    FACTS_HEADER,
    build_report_facts,
    merge_facts_parts,
    write_facts_csv,
)
from vifinqa.loader import iter_reports  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--config", type=str, default="api.yaml")
    ap.add_argument("--tickers", type=str, default="", help="chỉ chạy các ticker, phân cách ','")
    ap.add_argument("--no-merge", action="store_true", help="bỏ bước gộp facts_all.csv")
    args = ap.parse_args()

    cfg = Config.load(ROOT / "configs" / args.config)
    data_dir = cfg.resolved_data_dir()
    derived_dir = cfg.resolved_derived_dir()
    facts_dir = derived_dir / "facts"
    parts_dir = derived_dir / "facts_parts"
    facts_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    reports = iter_reports(data_dir)
    if args.tickers:
        wanted = set(args.tickers.split(","))
        reports = [r for r in reports if r.ticker in wanted]

    by_ticker: dict[str, list] = {}
    for r in reports:
        by_ticker.setdefault(r.ticker, []).append(r)

    state_path = derived_dir / "facts_state.json"
    state: dict[str, int] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    errors_path = derived_dir / "facts_errors.tsv"

    todo = [t for t in by_ticker if t not in state]
    print(f"Ticker tổng: {len(by_ticker)}, đã xong: {len(by_ticker) - len(todo)}, sẽ chạy: {len(todo)}")

    if todo:
        t0 = time.time()
        ticker_facts: dict[str, list[dict]] = defaultdict(list)
        total_reports = sum(len(by_ticker[t]) for t in todo)
        completed_per_ticker = {t: 0 for t in todo}

        def _finalize_ticker(ticker: str) -> None:
            """Ghi part facts + checkpoint khi xong hết report của 1 ticker.

            Per-report `facts/{report_id}_facts.csv` đã ghi ngay khi mỗi report
            hoàn thành (trong loop), nên đây chỉ gộp part + checkpoint.
            """
            write_facts_csv(ticker_facts.get(ticker, []), parts_dir / f"facts_{ticker}.csv")
            state[ticker] = 1
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(build_report_facts, r): r for t in todo for r in by_ticker[t]}
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                try:
                    facts = fut.result()
                    # per-report facts CSV (mọi fact thuộc đúng 1 report)
                    if facts:
                        write_facts_csv(facts, facts_dir / f"{r.report_id}_facts.csv")
                    ticker_facts[r.ticker].extend(facts)
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

    if not args.no_merge:
        n_rows = merge_facts_parts(parts_dir, derived_dir / "facts_all.csv")
        print(f"Facts: {n_rows} dòng -> {derived_dir / 'facts_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())