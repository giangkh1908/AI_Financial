"""run_merged_evidence.py — gộp facts → evidence bảng statement toàn corpus.

facts/{report_id}_facts.csv → data/derived/evidence_merged/{rid}__{statement}.csv
(schema [chi_tieu, Mãsố, ky, value]) + data/derived/statement_meta.csv:
  report_id, statement, src_table_ids (JSON)

Fix: fragment-split (CF table_9+table_10), label mojibake OCR → label sạch ASCII,
bỏ số dư đầu kỳ (year_start). Idempotent (overwrite). Chạy ~1-2 phút.

Chạy:  .venv\\Scripts\\python scripts/run_merged_evidence.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from vifinqa.config import Config  # noqa: E402
from vifinqa.etl.merged_evidence import merge_statement_facts  # noqa: E402
from vifinqa.etl.tidy import report_year  # noqa: E402


def main() -> int:
    cfg = Config.load(ROOT / "configs" / "api.yaml")
    derived_dir = cfg.resolved_derived_dir()
    facts_dir = derived_dir / "facts"
    out_dir = derived_dir / "evidence_merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(facts_dir.glob("*_facts.csv"))
    t0 = time.time()
    meta_rows: list[dict] = []
    n_merged = 0
    n_reports = 0
    errors = 0
    for fp in files:
        report_id = fp.name[: -len("_facts.csv")]
        try:
            facts = pd.read_csv(fp, dtype=str, low_memory=False)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ read lỗi {fp.name}: {e}")
            errors += 1
            continue
        try:
            merged = merge_statement_facts(report_year(report_id), facts)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ merge lỗi {fp.name}: {e}")
            errors += 1
            continue
        for stmt, (df, table_ids) in merged.items():
            dst = out_dir / f"{report_id}__{stmt}.csv"
            df.to_csv(dst, index=False, float_format="%.6f")
            meta_rows.append(
                {"report_id": report_id, "statement": stmt, "src_table_ids": json.dumps(table_ids)}
            )
            n_merged += 1
        n_reports += 1

    meta = pd.DataFrame(meta_rows, columns=["report_id", "statement", "src_table_ids"])
    meta.to_csv(derived_dir / "statement_meta.csv", index=False)

    print(f"Xong: {n_reports} report / {n_merged} bảng gộp / {errors} lỗi → {out_dir}")
    print(f"statement_meta.csv: {len(meta)} dòng | {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
