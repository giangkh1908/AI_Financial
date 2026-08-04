"""Smoke test M0 — kiểm tra pipeline dữ liệu đọc được (KHÔNG gọi API).

Chạy:  .venv\\Scripts\\python scripts/smoke_test.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Console Windows (cp1252) không in được tiếng Việt → ép UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.config import Config  # noqa: E402
from vifinqa.loader import iter_reports, load_questions, load_stocks  # noqa: E402

TABLE_RE = re.compile(r"<table>.*?</table>", re.S)


def count_tables_in_sample(reports: list, n: int = 5) -> int:
    """Đếm sơ bộ số <table> trong n file đầu tiên."""
    total = 0
    for r in reports[:n]:
        text = r.path.read_text(encoding="utf-8", errors="replace")
        total += len(TABLE_RE.findall(text))
    return total


def main() -> int:
    cfg = Config.load(ROOT / "configs" / "api.yaml")
    data_dir = cfg.resolved_data_dir()

    print("=== Config ===")
    print("  llm.provider :", cfg.llm.provider)
    print("  llm.base_url :", cfg.llm.base_url)
    print("  llm.model_id :", cfg.llm.model_id)
    print("  has api key  :", bool(cfg.llm.effective_api_key()))
    print("  data_dir     :", data_dir)

    print("\n=== Questions ===")
    questions = load_questions(data_dir / "questions" / "questions.jsonl")
    print(f"  total questions: {len(questions)}")
    for q in questions[:3]:
        print(f"  [{q.get('id')}] {q.get('question')}")

    print("\n=== Stocks ===")
    stocks = load_stocks(data_dir / "code_stock.csv")
    print(f"  total tickers: {len(stocks)}")
    sample = list(stocks.items())[:3]
    for t, n in sample:
        print(f"  {t} -> {n}")

    print("\n=== Reports ===")
    reports = iter_reports(data_dir)
    from collections import Counter
    by_type = Counter(r.report_type for r in reports)
    print(f"  total reports: {len(reports)}")
    print(f"  by type      : {dict(by_type)}")
    if reports:
        r0 = reports[0]
        print(f"  sample report: {r0.report_id} | ticker={r0.ticker} year={r0.year} type={r0.report_type}")
        print(f"  path         : {r0.path}")

    print("\n=== Tables (sơ bộ) ===")
    n_tables = count_tables_in_sample(reports, n=5)
    print(f"  <table> trong 5 file đầu: {n_tables}")
    if reports:
        avg = n_tables / min(5, len(reports))
        print(f"  ước lượng toàn corpus (~{len(reports)} file): ~{int(avg * len(reports))} bảng")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
