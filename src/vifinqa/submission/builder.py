"""builder.py — results.jsonl → submission.json + data/ (evidence CSVs materialized).

- Assert đủ câu (map theo questions.jsonl; thiếu → record fallback answer=0.0).
- Materialize **tidy** evidence CSV (`data/derived/evidence/{rid}__{tid}.csv`, schema
  cố định 4 cột [chi_tieu, Mãsố, ky, value]) → `out_dir/data/{rid}__{tid}.csv` (flat,
  dedupe theo path). Tidy CSV là đúng schema mà `pandas_query` (sinh bởi codegen) expect;
  KHÔNG package wide raw (query sẽ KeyError trên tên cột).
- Nếu tidy CSV thiếu (stale `evidence/` do codegen cũ/interrupt) → regenerate từ wide
  raw `tables/{rid}/{tid}.csv` qua `wide_csv_to_tidy` với `unit_factor` từ catalog.
- Ghi `out_dir/submission.json` đúng schema §3.1 (bỏ field nội bộ `_ok/_error`).
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from vifinqa.config import ROOT
from vifinqa.etl.tidy import wide_csv_to_tidy, write_tidy_csv
from vifinqa.loader import load_questions

_INTERNAL_FIELDS = {"_ok", "_error"}

# vn_num self-contained — grader BTC KHÔNG inject helper này (đã thấy 54 NameError).
# Nhúng def vào đầu mỗi query để query chạy độc lập ở môi trường bất kỳ.
# Không dùng re.compile (ast_check của ta chặn `compile`) → thuần str/float.
# Giữ đồng bộ logic với sandbox/runner.py vn_num: . nghìn, , thập phân, (x) âm, - rỗng.
_VN_NUM_DEF = '''def vn_num(s):
    """Parse số tiền VN: 1.234,56 / (1.234) / -1234 / 1 234 / - = None."""
    if s is None:
        return None
    s = str(s).strip().replace(" ", "").replace(" ", "")
    if not s or s in {"-", "–", "--", "/", "x", "X", "n/a", "N/A"}:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    elif s[:1] in "-–":
        neg = True
        s = s[1:]
    s = s.rstrip("%")
    if not s:
        return None
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(".", "")
    try:
        return -float(s) if neg else float(s)
    except ValueError:
        return None'''


def _with_vn_num(query: str) -> str:
    """Nhúng vn_num vào query không-trivial để grader chạy được (bỏ fallback)."""
    q = (query or "").strip()
    if not q or q in {"result = 0.0", "result=0.0"}:
        return query or ""
    return _VN_NUM_DEF + "\n\n" + q


def _source_tidy_path(report_id: str, table_id: str, derived_dir: Path) -> Path:
    """Tidy evidence CSV (schema 4 cột) — đúng schema `pandas_query` expect."""
    return derived_dir / "evidence" / f"{report_id}__{table_id}.csv"


def _source_wide_path(report_id: str, table_id: str, derived_dir: Path) -> Path:
    """Wide raw ETL `tables/{report_id}/{table_id}.csv` (fallback regenerate tidy)."""
    return derived_dir / "tables" / report_id / f"{table_id}.csv"


def _flat_name(report_id: str, table_id: str) -> str:
    return f"{report_id}__{table_id}.csv"


def _load_unit_factors(derived_dir: Path) -> dict[tuple[str, str], float]:
    """catalog_tables.csv → {(report_id, table_id): unit_factor}."""
    catalog = derived_dir / "catalog_tables.csv"
    out: dict[tuple[str, str], float] = {}
    if not catalog.exists():
        return out
    with catalog.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[(row["report_id"], row["table_id"])] = float(row["unit_factor"] or 1.0)
            except (KeyError, ValueError):
                continue
    return out


def _parse_evidence_var_and_src(ev: dict) -> tuple[str, str, str]:
    """`csv_path` = `data/{report_id}__{table_id}.csv` → (var, report_id, table_id)."""
    var = ev["variable"]
    csv_path = ev["csv_path"]
    name = csv_path.split("/", 1)[-1]  # bỏ prefix "data/"
    if not name.endswith(".csv"):
        raise ValueError(f"csv_path không hợp lệ: {csv_path}")
    stem = name[:-len(".csv")]
    # {report_id}__{table_id}; table_id = "table_N"; report_id có thể chứa "__"? không.
    # Tách theo "__table_" để an toàn (table_id luôn bắt đầu "table_").
    marker = "__table_"
    idx = stem.find(marker)
    if idx < 0:
        raise ValueError(f"không tách được table_id: {csv_path}")
    report_id = stem[:idx]
    table_id = "table_" + stem[idx + len(marker):]
    return var, report_id, table_id


def build(results_jsonl: Path, out_dir: Path, derived_dir: Path, questions_path: Path) -> dict:
    """Xây submission. Trả summary {n, materialized, missing_ids}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Map id → record
    records: dict[int, dict] = {}
    with open(results_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec

    questions = load_questions(questions_path)
    missing_ids = [q["id"] for q in questions if q["id"] not in records]

    # Materialize tidy evidence CSVs (dedupe). Regenerate từ wide nếu tidy thiếu.
    unit_factors = _load_unit_factors(derived_dir)
    materialized: set[str] = set()
    regen_count = 0
    for qid, rec in records.items():
        new_evidence = []
        for ev in rec.get("evidence", []):
            var, report_id, table_id = _parse_evidence_var_and_src(ev)
            flat = _flat_name(report_id, table_id)
            dst = data_dir / flat
            if flat not in materialized:
                tidy = _source_tidy_path(report_id, table_id, derived_dir)
                if not tidy.exists():
                    # Regenerate tidy từ wide raw (evidence/ stale do codegen cũ).
                    wide = _source_wide_path(report_id, table_id, derived_dir)
                    if not wide.exists():
                        raise FileNotFoundError(f"thiếu wide raw để regenerate tidy: {wide}")
                    uf = unit_factors.get((report_id, table_id), 1.0)
                    tidy_df = wide_csv_to_tidy(wide, uf)
                    tidy.parent.mkdir(parents=True, exist_ok=True)
                    write_tidy_csv(tidy_df, tidy)  # rỗng → ghi header-only (query → 0.0)
                    regen_count += 1
                    if tidy_df.empty:
                        print(f"   ⚠️ wide→tidy rỗng (header-only): {wide.name}")
                shutil.copyfile(tidy, dst)
                materialized.add(flat)
            new_evidence.append({"variable": var, "csv_path": f"data/{flat}"})
        rec["evidence"] = new_evidence

    # Bổ sung fallback record cho câu thiếu
    for qid in missing_ids:
        q = next(q for q in questions if q["id"] == qid)
        records[qid] = {
            "id": qid,
            "question": q["question"],
            "answer": 0.0,
            "relevant_docs": [],
            "relevant_tables": [],
            "evidence": [],
            "pandas_query": "result = 0.0",
        }

    # Sắp xếp theo id + bỏ field nội bộ + nhúng vn_num self-contained
    out_list = []
    for q in sorted(records.values(), key=lambda r: r["id"]):
        clean = {k: v for k, v in q.items() if k not in _INTERNAL_FIELDS}
        if "pandas_query" in clean:
            clean["pandas_query"] = _with_vn_num(clean["pandas_query"])
        out_list.append(clean)

    (out_dir / "submission.json").write_text(
        json.dumps(out_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "n": len(out_list),
        "materialized": len(materialized),
        "tidy_regen": regen_count,
        "missing_ids": missing_ids,
    }