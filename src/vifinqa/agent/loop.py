"""loop.py — vòng giải 1 câu hỏi: retrieve → codegen → exec → retry → finalize.

`solve(question, qid, pipeline, facts_index, llm, cfg)` → record (dict) sẵn sàng
ghi vào results.jsonl. Retry ≤2 lần khi exec lỗi (feed error vào LLM). Fallback
`result = 0.0` nếu thất bại hết (giữ record hợp lệ, không void bài do thiếu câu).

Evidence = wide table CSV M1 (`data/derived/tables/{report_id}/{table_id}.csv`).
csv_path trong record = `data/{report_id}__{table_id}.csv` (flat, builder copy ở M4b).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from vifinqa.codegen import LLMClient, build_messages
from vifinqa.config import ROOT
from vifinqa.etl.tidy import report_year, wide_to_tidy, write_tidy_csv
from vifinqa.retrieval.entity import Entities
from vifinqa.retrieval.facts_index import FactsIndex
from vifinqa.retrieval.pipeline import RetrievalPipeline
from vifinqa.retrieval.search import SearchResult
from vifinqa.sandbox import check_code, run_pandas

_MAX_SAMPLE_ROWS = 8
_MAX_PREVIEW_CHARS = 1400
_MAX_FACT_HINTS = 25


def _tidy_evidence_path(report_id: str, table_id: str, derived_dir: Path) -> Path:
    return derived_dir / "evidence" / f"{report_id}__{table_id}.csv"


def _ensure_tidy(r: SearchResult, derived_dir: Path) -> Path | None:
    """Wide table → tidy CSV schema chuẩn [chi_tieu, Mãsố, ky, value] (cached)."""
    dst = _tidy_evidence_path(r.report_id, r.table_id, derived_dir)
    if dst.exists():
        return dst
    src = derived_dir / "tables" / r.report_id / f"{r.table_id}.csv"
    if not src.exists():
        return None
    try:
        df = pd.read_csv(src, dtype=str, index_col=0)
        tidy = wide_to_tidy(df, report_year(r.report_id), r.unit_factor)
        if tidy.empty:
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_tidy_csv(tidy, dst)
        return dst
    except Exception:
        return None


def _build_table_card(
    var: int,
    r: SearchResult,
    facts_index: FactsIndex | None,
    derived_dir: Path,
) -> dict[str, Any] | None:
    """Tidy evidence → card cho prompt (columns + sample rows + fact hints)."""
    tpath = _ensure_tidy(r, derived_dir)
    if tpath is None:
        return None
    columns = ["chi_tieu", "Mãsố", "ky", "value"]
    sample_rows = ""
    try:
        df = pd.read_csv(tpath, dtype=str, nrows=_MAX_SAMPLE_ROWS)
        lines = []
        for _, row in df.iterrows():
            cells = [str(row[c]) if pd.notna(row[c]) else "" for c in df.columns]
            lines.append(" | ".join(cells))
        sample_rows = "\n".join(lines)
        if len(sample_rows) > _MAX_PREVIEW_CHARS:
            sample_rows = sample_rows[:_MAX_PREVIEW_CHARS] + " ..."
    except Exception:
        sample_rows = ""

    fact_hints: list[tuple[str, str]] | None = None
    if facts_index is not None:
        try:
            ff = facts_index.facts_for_table(r.report_id, r.table_id)
            if not ff.empty:
                pairs = list(zip(ff["item_code"].astype(str), ff["item_label"].astype(str)))
                fact_hints = [(c, lab) for c, lab in pairs if c and lab][:_MAX_FACT_HINTS]
        except Exception:
            fact_hints = None

    return {
        "var": var,
        "report_id": r.report_id,
        "position": r.position,
        "statement": r.statement,
        "unit": r.unit,
        "unit_factor": r.unit_factor,
        "columns": columns,
        "fact_hints": fact_hints,
        "sample_rows": sample_rows,
    }


def _evidence_path(report_id: str, table_id: str) -> str:
    """csv_path nộp bài (flat, bắt đầu `data/`). Builder copy file gốc → đây."""
    return f"data/{report_id}__{table_id}.csv"


_DF_REF_RE = re.compile(r"\bdf(\d+)\b")


def _df_refs_over(code: str, n: int) -> int | None:
    """Số df lớn nhất query tham chiếu vượt n (df{n+1}...). None nếu ổn."""
    if not code:
        return None
    max_ref = 0
    for m in _DF_REF_RE.finditer(code):
        max_ref = max(max_ref, int(m.group(1)))
    return max_ref if max_ref > n else None


def solve(
    question: str,
    qid: int,
    pipeline: RetrievalPipeline,
    facts_index: FactsIndex | None,
    llm: LLMClient,
    cfg,
    max_retries: int = 1,
) -> dict:
    """Giải 1 câu → record. Fallback answer=0.0 nếu codegen/exec thất bại."""
    derived_dir = cfg.resolved_derived_dir()
    root = ROOT

    results, entities = pipeline.search(question)
    results = results or []

    usable: list[SearchResult] = []
    cards: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        card = _build_table_card(i + 1, r, facts_index, derived_dir)
        if card is not None:
            cards.append(card)
            usable.append(r)

    evidence = {
        f"df{i+1}": str(_tidy_evidence_path(r.report_id, r.table_id, derived_dir))
        for i, r in enumerate(usable)
    }

    messages = build_messages(question, entities, cards)
    pandas_query = llm.generate_query(messages)

    answer: float | None = None
    exec_error = ""
    for attempt in range(max_retries + 1):
        if not pandas_query:
            exec_error = "LLM không trả code hợp lệ"
            break
        over = _df_refs_over(pandas_query, len(evidence))
        if over is not None:
            exec_error = f'query tham chiếu dfs["df{over}"] nhưng chỉ có dfs["df1"]..dfs["df{len(evidence)}"]'
            if attempt < max_retries:
                pandas_query = _repair(llm, messages, pandas_query, exec_error)
            continue
        ok, err = check_code(pandas_query, cfg.sandbox.max_code_len, cfg.sandbox.max_ast_nodes)
        if not ok:
            exec_error = f"ast_check: {err}"
            # Thử sửa qua LLM
            if attempt < max_retries:
                pandas_query = _repair(llm, messages, pandas_query, exec_error)
            continue
        out = run_pandas(pandas_query, evidence, root, timeout=cfg.sandbox.timeout)
        if out.get("ok"):
            answer = float(out["result"])
            break
        exec_error = out.get("error") or "exec thất bại"
        if attempt < max_retries:
            pandas_query = _repair(llm, messages, pandas_query, exec_error)

    if answer is None:
        pandas_query = "result = 0.0"
        answer = 0.0

    return _make_record(qid, question, usable, entities, pandas_query, answer, exec_error)


def _repair(llm: LLMClient, base_messages: list[dict], prev_code: str, error: str) -> str:
    """Feed lỗi vào LLM để sinh code sửa."""
    followup = base_messages + [
        {"role": "assistant", "content": f"```python\n{prev_code}\n```"},
        {
            "role": "user",
            "content": (
                f"Code trên lỗi khi chạy:\n{error}\n"
                "Hãy sửa (giữ quy tắc: truy cập dfs[\"df1\"]..dfs[\"dfN\"] hoặc alias "
                "`df1 = dfs[\"df1\"]`), dùng vn_num nếu cần parse số, gán result. "
                "Trả code mới trong ```python ... ```."
            ),
        },
    ]
    return llm.generate_query(followup)


def _make_record(
    qid: int,
    question: str,
    results: list[SearchResult],
    entities: Entities,
    pandas_query: str,
    answer: float,
    exec_error: str,
) -> dict:
    relevant_docs: list[str] = []
    seen: set[str] = set()
    for r in results:
        if r.report_id and r.report_id not in seen:
            seen.add(r.report_id)
            relevant_docs.append(r.report_id)
    relevant_tables = [r.relevant_tables_key() for r in results]
    evidence = [
        {"variable": f"df{i+1}", "csv_path": _evidence_path(r.report_id, r.table_id)}
        for i, r in enumerate(results)
    ]
    return {
        "id": qid,
        "question": question,
        "answer": float(answer),
        "relevant_docs": relevant_docs,
        "relevant_tables": relevant_tables,
        "evidence": evidence,
        "pandas_query": pandas_query,
        "_ok": answer != 0.0 or not exec_error,
        "_error": exec_error,
    }