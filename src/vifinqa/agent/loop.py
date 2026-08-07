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
    r: SearchResult,
    facts_index: FactsIndex | None,
    derived_dir: Path,
    var_name: str = "df1",
) -> dict[str, Any] | None:
    """Tidy evidence → card cho prompt (variable name + columns + sample rows + fact hints)."""
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
        "var_name": var_name,
        "table_ref": r.relevant_tables_key(),
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


_DF_REF_RE = re.compile(r"\bdf\d+\b")  # bare df1/df2 → không hợp lệ ở grader contract
_DFS_KEY_RE = re.compile(r'''dfs\s*\[\s*["']([^"']+)["']\s*\]''')
_DFS_DF_N_RE = re.compile(r'''dfs\s*\[\s*["']df\d+["']\s*\]''')  # dfs["df1"] → sai


def _auto_fix_dfs_keys(code: str, table_refs: list[str]) -> str:
    """Auto-fix `dfs["df1"]` → `dfs["<table_ref>"]` dựa trên thứ tự table_refs.

    Nếu LLM hallucinate `dfs["df1"]`, `dfs["df2"]`, tự động replace bằng table_ref
    tương ứng (theo thứ tự). Chỉ áp dụng nếu số lượng `dfs["dfN"]` <= số table_refs.
    """
    if not code or not table_refs:
        return code
    # Tìm tất cả dfs["dfN"]
    matches = list(_DFS_DF_N_RE.finditer(code))
    if not matches:
        return code
    # Map df1 → table_refs[0], df2 → table_refs[1], ...
    # Trích xuất số N từ "dfN"
    def get_df_idx(match: re.Match) -> int:
        key = match.group(0)  # vd: dfs["df1"]
        m = re.search(r'df(\d+)', key)
        return int(m.group(1)) if m else -1
    df_indices = [(m, get_df_idx(m)) for m in matches]
    # Chỉ fix nếu indices hợp lệ (1, 2, 3, ...) và <= số table_refs
    valid_indices = [idx for _, idx in df_indices if 1 <= idx <= len(table_refs)]
    if len(valid_indices) != len(df_indices):
        return code  # có index ngoài range → không auto-fix
    # Replace từ cuối về đầu để không lệch vị trí
    fixed_code = code
    for match, idx in sorted(df_indices, key=lambda x: x[0].start(), reverse=True):
        table_ref = table_refs[idx - 1]  # df1 → index 0
        fixed_code = fixed_code[:match.start()] + f'dfs["{table_ref}"]' + fixed_code[match.end():]
    return fixed_code


def _bad_refs(code: str, table_refs: set[str], n_tables: int) -> str | None:
    """Kiểm tra query dùng đúng contract grader BTC. Trả error string nếu sai, None nếu ổn.

    Theo spec BTC: variable names (df1, df2, ...) dùng TRỰC TIẾP trong query.
    - Cho phép bare `df1`, `df2`, ... (variable names từ evidence).
    - Cấm `dfs["..."]` (dfs không tồn tại trong grader BTC).
    """
    if not code:
        return None
    # Cấm dfs["..."] — dfs không tồn tại trong grader BTC
    if _DFS_KEY_RE.search(code):
        return (
            "dùng `dfs[\"...\"]` (SAI — dfs KHÔNG TỒN TẠI trong grader BTC). "
            "PHẢI dùng variable names từ evidence: df1, df2, df3, ... "
            "VÍ DỤ ĐÚNG: `t = df1` hoặc `t = df2`. "
            "VÍ DỤ SAI: `t = dfs[\"...\"]` → KeyError/NameError."
        )
    return None


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
        var_name = f"df{i+1}"  # bare variable name: df1, df2, ...
        card = _build_table_card(r, facts_index, derived_dir, var_name=var_name)
        if card is not None:
            cards.append(card)
            usable.append(r)

    # Evidence dict keyed theo table_ref (khớp grader: dfs["{report_id}|table_N"]).
    table_refs: set[str] = set()
    evidence: dict[str, str] = {}
    for r in usable:
        key = r.relevant_tables_key()
        table_refs.add(key)
        evidence[key] = str(_tidy_evidence_path(r.report_id, r.table_id, derived_dir))

    # Short-circuit: không có bảng truy hồi → fallback 0.0 ngay, không gọi LLM (tiết
    # kiệm ~120s/câu vô ích — xem phân tích fail: 26/470 câu wallcap có evidence=0).
    if not usable:
        return _make_record(
            qid, question, [], entities, "result = 0.0", 0.0,
            f"retrieval: không có bảng truy hồi ({len(results)} kết quả, 0 khả dụng)",
        )

    messages = build_messages(question, entities, cards)
    pandas_query = llm.generate_query(messages)

    answer: float | None = None
    exec_error = ""
    for attempt in range(max_retries + 1):
        if not pandas_query:
            exec_error = "LLM không trả code hợp lệ"
            break
        # Auto-fix dfs["df1"] → dfs["<table_ref>"] (nếu LLM hallucinate)
        pandas_query = _auto_fix_dfs_keys(pandas_query, list(table_refs))
        bad = _bad_refs(pandas_query, table_refs, len(evidence))
        if bad is not None:
            exec_error = bad
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
    # Nếu lỗi do dfs["..."], thêm ví dụ cụ thể
    extra_hint = ""
    if "dfs[" in error:
        extra_hint = (
            "\n\n⚠️ LỖI CỤ THỂ: Bạn đang dùng `dfs[\"...\"]` — SAI vì dfs KHÔNG TỒN TẠI.\n"
            "BẠN PHẢI dùng variable names từ evidence: df1, df2, df3, ...\n"
            "VÍ DỤ ĐÚNG: `t = df1` hoặc `t = df2`."
        )
    followup = base_messages + [
        {"role": "assistant", "content": f"```python\n{prev_code}\n```"},
        {
            "role": "user",
            "content": (
                f"Code trên lỗi khi chạy:\n{error}\n"
                "Hãy sửa theo đúng contract BTC: dùng variable names từ evidence "
                "(df1, df2, df3, ...) TRỰC TIẾP trong query. "
                "TUYỆT ĐỐI CẤM: `dfs[\"...\"]` (dfs không tồn tại).\n"
                "Chỉ dùng pd + builtins + vn_num (không np/math/re). Gán "
                "`result = round(<float>, 2)`. Trả code mới trong ```python ... ```."
                f"{extra_hint}"
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