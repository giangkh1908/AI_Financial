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
import re
import shutil
from pathlib import Path

from vifinqa.etl.merged_evidence import STATEMENTS
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


def _source_merged_path(report_id: str, statement: str, derived_dir: Path) -> Path:
    """Bảng gộp statement (evidence_merged) — đúng schema 4 cột, value VND."""
    return derived_dir / "evidence_merged" / f"{report_id}__{statement}.csv"


def _source_wide_path(report_id: str, table_id: str, derived_dir: Path) -> Path:
    """Wide raw ETL `tables/{report_id}/{table_id}.csv` (fallback regenerate tidy)."""
    return derived_dir / "tables" / report_id / f"{table_id}.csv"


def _flat_name(report_id: str, item: str) -> str:
    return f"{report_id}__{item}.csv"


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


def _load_start_lines(derived_dir: Path) -> dict[tuple[str, str], int]:
    """catalog_tables.csv → {(report_id, table_id): start_line}.

    `start_line` = vị trí dòng bắt đầu bảng trong file OCR (dòng vật lý của `<table>`).
    Dùng để xuất `relevant_tables` theo đặc tả BTC: `report_id|<start_line>`.
    """
    catalog = derived_dir / "catalog_tables.csv"
    out: dict[tuple[str, str], int] = {}
    if not catalog.exists():
        return out
    with catalog.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[(row["report_id"], row["table_id"])] = int(row.get("start_line") or 0)
            except (KeyError, ValueError):
                continue
    return out


def _table_ref_to_line(key: str, start_lines: dict[tuple[str, str], int]) -> str:
    """`report_id|table_N` → `report_id|<start_line>` (format BTC).

    Nếu không tìm được start_line (catalog thiếu) → giữ nguyên `report_id|table_N`.
    """
    marker = "|table_"
    idx = key.find(marker)
    if idx < 0:
        return key
    report_id = key[:idx]
    table_id = "table_" + key[idx + len(marker):]
    sl = start_lines.get((report_id, table_id), 0)
    if sl:
        return f"{report_id}|{sl}"
    return key


_DF_VAR_RE = re.compile(r"\b(df\d+)\b")


def _strip_vn_num_def(query: str) -> str:
    """Bỏ phần `def vn_num(s): ...` ở đầu query (builder inject) để parse dfN."""
    if not query or not query.startswith("def vn_num"):
        return query
    lines = query.split("\n")
    end_def = 0
    in_def = False
    for i, line in enumerate(lines):
        if line.startswith("def vn_num"):
            in_def = True
            continue
        if in_def:
            if line.strip() == "" or not line.startswith(" "):
                end_def = i
                break
    if end_def:
        return "\n".join(lines[end_def:]).strip()
    return query


def _used_df_vars(query: str) -> set[str]:
    """Trả set các dfN được nhắc đến trong pandas_query (bỏ vn_num def)."""
    if not query:
        return set()
    q = _strip_vn_num_def(query)
    return set(_DF_VAR_RE.findall(q))


def _renumber_query(query: str, var_map: dict[str, str]) -> str:
    """Replace dfN trong query theo var_map. Chỉ thay code body (sau vn_num def).

    Single-pass regex sub + dict lookup → KHÔNG chain (df14→df2 rồi df2→df1
    sẽ KHÔNG biến df14 thành df1 — mỗi token chỉ replace 1 lần).
    """
    if not query or not var_map:
        return query
    if query.startswith("def vn_num"):
        # Split: phần def + phần code body
        body_start = len(query) - len(_strip_vn_num_def(query))
        vn_part = query[:body_start]
        body = query[body_start:]
    else:
        vn_part = ""
        body = query

    def _repl(m: re.Match) -> str:
        return var_map.get(m.group(0), m.group(0))

    new_body = _DF_VAR_RE.sub(_repl, body)
    return vn_part + new_body


def _filter_evidence_and_tables(
    evidence: list[dict],
    relevant_tables: list[str],
    pandas_query: str,
    *,
    filter_rt: bool = True,
) -> tuple[list[dict], list[str], str]:
    """Lọc evidence + relevant_tables chỉ giữ bảng thực sự dùng trong pandas_query.

    Trả (filtered_evidence, filtered_relevant_tables, updated_pandas_query).
    - Phân tích pandas_query tìm các dfN được nhắc đến.
    - Giữ chỉ evidence có variable trong set used.
    - Renumber dfN (df1, df2, ...) theo thứ tự giữ lại + cập nhật pandas_query.
    - relevant_tables: giữ chỉ table thuộc evidence còn lại (nếu filter_rt=True;
      khi query thất bại → giữ nguyên để không mất recall).
    """
    if not evidence or not pandas_query:
        return evidence, relevant_tables, pandas_query

    used = _used_df_vars(pandas_query)
    if not used:
        # Query không nhắc dfN nào (vd `result = 0.0`) → giữ tất cả
        return evidence, relevant_tables, pandas_query

    # Lọc evidence: giữ chỉ ev có variable trong used
    kept = [ev for ev in evidence if ev.get("variable") in used]
    if not kept:
        # LLM dùng biến không có trong evidence → giữ tất cả (safety)
        return evidence, relevant_tables, pandas_query

    # Renumber: df1, df2, ... theo thứ tự kept
    var_map: dict[str, str] = {}
    new_evidence: list[dict] = []
    for i, ev in enumerate(kept):
        old_var = ev["variable"]
        new_var = f"df{i+1}"
        if old_var != new_var:
            var_map[old_var] = new_var
        new_ev = dict(ev)
        new_ev["variable"] = new_var
        new_evidence.append(new_ev)

    # Cập nhật pandas_query: replace dfN theo var_map
    new_query = _renumber_query(pandas_query, var_map) if var_map else pandas_query

    if not filter_rt:
        # Query thất bại → giữ nguyên relevant_tables (không mất recall)
        return new_evidence, relevant_tables, new_query

    # Lọc relevant_tables: giữ chỉ table thuộc evidence còn lại
    kept_rids: set[str] = set()
    kept_tids: set[tuple[str, str]] = set()
    for ev in new_evidence:
        csv_path = ev.get("csv_path", "")
        name = csv_path.split("/", 1)[-1]
        if not name.endswith(".csv"):
            continue
        stem = name[:-4]
        is_merged = False
        for stmt in STATEMENTS:
            m = f"__{stmt}"
            if m in stem:
                rid = stem[:stem.find(m)]
                kept_rids.add(rid)
                is_merged = True
                break
        if not is_merged:
            marker = "__table_"
            idx = stem.find(marker)
            if idx >= 0:
                rid = stem[:idx]
                tid = "table_" + stem[idx + len(marker):]
                kept_tids.add((rid, tid))

    def _keep_table(key: str) -> bool:
        if "|" not in key:
            return True
        rid = key.split("|")[0]
        # Bảng gộp → giữ tất cả table cùng report_id
        if rid in kept_rids:
            return True
        # Bảng đơn → match cụ thể (rid, tid)
        if "table_" in key:
            tid = "table_" + key.split("table_")[-1]
            return (rid, tid) in kept_tids
        # Line format (đã rewrite) → giữ nếu rid match (safety)
        return any(rid == r for r in kept_rids) or any(rid == t[0] for t in kept_tids)

    filtered_rt = [k for k in relevant_tables if _keep_table(k)]
    return new_evidence, filtered_rt, new_query


def _parse_evidence_var_and_src(ev: dict) -> tuple[str, str, str]:
    """`csv_path` = `data/{report_id}__{item}.csv` → (var, report_id, item).

    `item` = `table_N` (bảng đơn) hoặc `balance_sheet|income|cash_flow` (bảng gộp).
    """
    var = ev["variable"]
    csv_path = ev["csv_path"]
    name = csv_path.split("/", 1)[-1]  # bỏ prefix "data/"
    if not name.endswith(".csv"):
        raise ValueError(f"csv_path không hợp lệ: {csv_path}")
    stem = name[:-len(".csv")]
    # {report_id}__{item}; tách theo marker để an toàn.
    marker = "__table_"
    idx = stem.find(marker)
    if idx >= 0:
        return var, stem[:idx], "table_" + stem[idx + len(marker):]
    for stmt in STATEMENTS:
        m = f"__{stmt}"
        idx = stem.find(m)
        if idx >= 0:
            return var, stem[:idx], stmt
    raise ValueError(f"không tách được item (table/statement): {csv_path}")


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
    start_lines = _load_start_lines(derived_dir)  # → relevant_tables `rid|<start_line>`
    materialized: set[str] = set()
    regen_count = 0
    for qid, rec in records.items():
        new_evidence = []
        for ev in rec.get("evidence", []):
            var, report_id, item = _parse_evidence_var_and_src(ev)
            flat = _flat_name(report_id, item)
            dst = data_dir / flat
            if flat not in materialized:
                if item in STATEMENTS:
                    # Bảng gộp statement — chỉ có ở evidence_merged (ETL), không regen.
                    src = _source_merged_path(report_id, item, derived_dir)
                    if not src.exists():
                        raise FileNotFoundError(
                            f"thiếu bảng gộp {src.name} — chạy scripts/run_merged_evidence.py"
                        )
                else:
                    src = _source_tidy_path(report_id, item, derived_dir)
                    if not src.exists():
                        # Regenerate tidy từ wide raw (evidence/ stale do codegen cũ).
                        wide = _source_wide_path(report_id, item, derived_dir)
                        if not wide.exists():
                            raise FileNotFoundError(f"thiếu wide raw để regenerate tidy: {wide}")
                        uf = unit_factors.get((report_id, item), 1.0)
                        tidy_df = wide_csv_to_tidy(wide, uf)
                        src.parent.mkdir(parents=True, exist_ok=True)
                        write_tidy_csv(tidy_df, src)  # rỗng → ghi header-only (query → 0.0)
                        regen_count += 1
                        if tidy_df.empty:
                            print(f"   ⚠️ wide→tidy rỗng (header-only): {wide.name}")
                shutil.copyfile(src, dst)
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
    rewritten = 0
    filtered_count = 0
    for q in sorted(records.values(), key=lambda r: r["id"]):
        clean = {k: v for k, v in q.items() if k not in _INTERNAL_FIELDS}

        # Lọc evidence + relevant_tables: chỉ giữ bảng thực sự dùng trong pandas_query.
        # Giúp TABLES_PRECISION tăng mạnh (15→1-3 tables/question).
        # ⚠️ Chỉ lọc khi query THÀNH CÔNG (answer != 0) — khi thất bại, bảng used có thể
        # SAI → lọc sẽ mất recall + cắt nhầm evidence. Giữ nguyên toàn bộ.
        ev = clean.get("evidence") or []
        rt = clean.get("relevant_tables") or []
        pq = clean.get("pandas_query") or ""
        answer = clean.get("answer")
        if ev and rt and pq and answer not in (0.0, None) and pq.strip() != "result = 0.0":
            new_ev, new_rt, new_pq = _filter_evidence_and_tables(ev, rt, pq)
            if len(new_ev) < len(ev):
                filtered_count += 1
            clean["evidence"] = new_ev
            clean["relevant_tables"] = new_rt
            clean["pandas_query"] = new_pq

        if "pandas_query" in clean:
            clean["pandas_query"] = _with_vn_num(clean["pandas_query"])
        # relevant_tables: `rid|table_N` → `rid|<start_line>` (đặc tả BTC)
        rt = clean.get("relevant_tables") or []
        new_rt = [_table_ref_to_line(k, start_lines) for k in rt]
        if new_rt != rt:
            rewritten += 1
        clean["relevant_tables"] = new_rt
        out_list.append(clean)

    (out_dir / "submission.json").write_text(
        json.dumps(out_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "n": len(out_list),
        "materialized": len(materialized),
        "tidy_regen": regen_count,
        "relevant_tables_rewritten": rewritten,
        "evidence_filtered": filtered_count,
        "missing_ids": missing_ids,
    }