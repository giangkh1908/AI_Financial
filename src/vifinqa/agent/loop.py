"""loop.py — vòng giải 1 câu hỏi: retrieve → codegen → exec → retry → finalize.

`solve(question, qid, pipeline, facts_index, llm, cfg)` → record (dict) sẵn sàng
ghi vào results.jsonl. Retry ≤2 lần khi exec lỗi (feed error vào LLM). Fallback
`result = 0.0` nếu thất bại hết (giữ record hợp lệ, không void bài do thiếu câu).

Evidence = wide table CSV M1 (`data/derived/tables/{report_id}/{table_id}.csv`).
csv_path trong record = `data/{report_id}__{table_id}.csv` (flat, builder copy ở M4b).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from vifinqa.codegen import LLMClient, build_messages
from vifinqa.config import ROOT
from vifinqa.engine.deterministic import build_template_query, solve_deterministic
from vifinqa.etl.merged_evidence import STATEMENTS
from vifinqa.etl.tidy import load_layout_dict, wide_csv_to_tidy, write_tidy_csv
from vifinqa.retrieval.entity import Entities
from vifinqa.retrieval.facts_index import FactsIndex
from vifinqa.retrieval.pipeline import RetrievalPipeline
from vifinqa.retrieval.search import SearchResult
from vifinqa.sandbox import check_code, run_pandas

_MAX_SAMPLE_ROWS = 8
_MAX_PREVIEW_CHARS = 1400
_MAX_FACT_HINTS = 25
# Bảng gộp (toàn bộ statement) — cần preview đủ chỉ tiêu để LLM tìm label.
_MAX_PREVIEW_CHARS_MERGED = 20000
_MAX_FACT_HINTS_MERGED = 400


def _tidy_evidence_path(report_id: str, table_id: str, derived_dir: Path) -> Path:
    return derived_dir / "evidence" / f"{report_id}__{table_id}.csv"


def _merged_evidence_path(report_id: str, statement: str, derived_dir: Path) -> Path:
    return derived_dir / "evidence_merged" / f"{report_id}__{statement}.csv"


_stmt_meta_cache: dict[tuple[Path, str], dict[tuple[str, str], dict]] = {}


def _load_statement_meta(derived_dir: Path) -> dict[tuple[str, str], dict]:
    """statement_meta.csv → {(report_id, statement): {src_table_ids: [...]}} (cached)."""
    key = (derived_dir, "meta")
    if key in _stmt_meta_cache:
        return _stmt_meta_cache[key]
    out: dict[tuple[str, str], dict] = {}
    p = derived_dir / "statement_meta.csv"
    if p.exists():
        try:
            df = pd.read_csv(p, dtype=str)
            for _, row in df.iterrows():
                try:
                    sids = json.loads(row["src_table_ids"])
                except (json.JSONDecodeError, KeyError):
                    sids = []
                out[(row["report_id"], row["statement"])] = {"src_table_ids": sids}
        except Exception:
            out = {}
    _stmt_meta_cache[key] = out
    return out


def _table_to_statement(derived_dir: Path) -> dict[tuple[str, str], str]:
    """{(report_id, table_id): statement} — reverse map từ statement_meta."""
    out: dict[tuple[str, str], str] = {}
    for (rid, stmt), meta in _load_statement_meta(derived_dir).items():
        for tid in meta.get("src_table_ids", []):
            out[(rid, tid)] = stmt
    return out


def _ensure_tidy(r: SearchResult, derived_dir: Path) -> Path | None:
    """Wide table → tidy CSV schema chuẩn [chi_tieu, Mãsố, ky, value] (cached).

    Dùng layout từ `layouts/{rid}.json` (nguồn sự thật cấu trúc) — không tự đoán cột.
    """
    dst = _tidy_evidence_path(r.report_id, r.table_id, derived_dir)
    if dst.exists():
        return dst
    src = derived_dir / "tables" / r.report_id / f"{r.table_id}.csv"
    if not src.exists():
        return None
    try:
        layout = load_layout_dict(derived_dir, r.report_id, r.table_id)
        tidy = wide_csv_to_tidy(src, r.unit_factor, layout)
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
    evidence_path: Path | None = None,
    fact_table_ids: list[str] | None = None,
    merged: bool = False,
) -> dict[str, Any] | None:
    """Tidy evidence → card cho prompt (variable name + columns + sample rows + fact hints).

    `merged=True`: evidence_path là bảng gộp statement (`evidence_merged/`), value đã
    chuẩn hoá VND (unit_factor=1), preview/hints đầy đủ toàn bộ chỉ tiêu.
    """
    tpath = evidence_path if evidence_path is not None else _ensure_tidy(r, derived_dir)
    if tpath is None:
        return None
    columns = ["chi_tieu", "Mãsố", "ky", "value"]
    sample_rows = ""
    max_chars = _MAX_PREVIEW_CHARS_MERGED if merged else _MAX_PREVIEW_CHARS
    try:
        df = pd.read_csv(tpath, dtype=str, nrows=_MAX_SAMPLE_ROWS if not merged else None)
        lines = []
        for _, row in df.iterrows():
            cells = [str(row[c]) if pd.notna(row[c]) else "" for c in df.columns]
            lines.append(" | ".join(cells))
        sample_rows = "\n".join(lines)
        if len(sample_rows) > max_chars:
            sample_rows = sample_rows[:max_chars] + " ..."
    except Exception:
        sample_rows = ""

    fact_hints: list[tuple[str, str]] | None = None
    if facts_index is not None:
        try:
            pairs: list[tuple[str, str]] = []
            for tid in (fact_table_ids or [r.table_id]):
                ff = facts_index.facts_for_table(r.report_id, tid)
                if ff.empty:
                    continue
                for _, row in ff.iterrows():
                    c = str(row.get("item_code") or "")
                    lab = str(row.get("item_label") or "")
                    if c and lab:
                        pairs.append((c, lab))
            if pairs:
                # dedupe (code, label) nhưng giữ thứ tự
                seen_pairs: set[tuple[str, str]] = set()
                merged_pairs: list[tuple[str, str]] = []
                for p in pairs:
                    if p not in seen_pairs:
                        seen_pairs.add(p)
                        merged_pairs.append(p)
                fact_hints = merged_pairs[:_MAX_FACT_HINTS_MERGED if merged else _MAX_FACT_HINTS]
        except Exception:
            fact_hints = None

    unit = "VND" if merged else r.unit
    unit_factor = 1.0 if merged else r.unit_factor
    return {
        "var_name": var_name,
        "table_ref": r.relevant_tables_key(),
        "report_id": r.report_id,
        "position": r.position,
        "statement": r.statement,
        "unit": unit,
        "unit_factor": unit_factor,
        "columns": columns,
        "fact_hints": fact_hints,
        "sample_rows": sample_rows,
        "merged": merged,
    }


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


def _plan_evidence(
    results: list[SearchResult],
    facts_index: FactsIndex | None,
    derived_dir: Path,
) -> tuple[list[SearchResult], list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Biến top-k SearchResult → (usable, cards, evidence_paths, evidence_plan).

    - Bảng statement (có trong `statement_meta`) → dùng **bảng gộp** toàn bộ BCTC
      (evidence_merged), dedupe theo (report_id, statement) — fix fragment-split.
    - Bảng notes / không có merged → tidy per-table như cũ.
    - `evidence` dict (key → abs path) cùng thứ tự `evidence_plan` = thứ tự var
      (df1, df2, ...) mà runner inject. `evidence_plan` dùng để dựng record cuối.
    """
    table_to_stmt = _table_to_statement(derived_dir)
    stmt_meta = _load_statement_meta(derived_dir)

    usable: list[SearchResult] = []
    cards: list[dict[str, Any]] = []
    evidence: dict[str, str] = {}
    evidence_plan: list[dict[str, Any]] = []
    merged_used: set[tuple[str, str]] = set()
    for r in results:
        var_name = f"df{len(evidence_plan) + 1}"
        stmt = table_to_stmt.get((r.report_id, r.table_id))
        if stmt and (r.report_id, stmt) not in merged_used:
            mpath = _merged_evidence_path(r.report_id, stmt, derived_dir)
            if mpath.exists():
                meta = stmt_meta.get((r.report_id, stmt)) or {}
                frags = meta.get("src_table_ids") or [r.table_id]
                card = _build_table_card(
                    r, facts_index, derived_dir, var_name=var_name,
                    evidence_path=mpath, fact_table_ids=frags, merged=True,
                )
                if card is not None:
                    merged_used.add((r.report_id, stmt))
                    cards.append(card)
                    usable.append(r)
                    evidence[f"{r.report_id}|merged_{stmt}"] = str(mpath)
                    evidence_plan.append({
                        "variable": var_name,
                        "report_id": r.report_id,
                        "kind": "stmt",
                        "name": stmt,
                        "fragments": frags,
                    })
                    continue
        # Bảng notes / không có merged → bảng đơn lẻ như cũ.
        card = _build_table_card(r, facts_index, derived_dir, var_name=var_name)
        if card is not None:
            cards.append(card)
            usable.append(r)
            key = r.relevant_tables_key()
            evidence[key] = str(_tidy_evidence_path(r.report_id, r.table_id, derived_dir))
            evidence_plan.append({
                "variable": var_name,
                "report_id": r.report_id,
                "kind": "table",
                "name": r.table_id,
                "fragments": [r.table_id],
            })
    return usable, cards, evidence, evidence_plan


_STMT_HINT_VI = {
    "income": "kết quả kinh doanh / thu nhập",
    "balance_sheet": "bảng cân đối kế toán",
    "cash_flow": "lưu chuyển tiền tệ",
}


def _precheck_hint(
    entities: Entities,
    cards: list[dict[str, Any]],
    usable: list[SearchResult],
    derived_dir: Path,
) -> str | None:
    """Hint (advisory) nếu evidence thiếu statement câu hỏi cần — KHÔNG chặn.

    Nếu câu hỏi có hint statement nhưng evidence không có bảng thuộc statement đó →
    nhắc LLM tìm metric ở notes tables (không kết luận 0.0 vội).
    """
    hint = entities.statement
    if not hint:
        return None
    if any(card.get("statement") == hint for card in cards):
        return None
    stmt_vi = _STMT_HINT_VI.get(hint, hint)
    return (
        f"Câu hỏi có vẻ cần BCTC {stmt_vi} (statement={hint}) nhưng không bảng nào trong "
        f"evidence thuộc statement này. Metric có thể nằm ở bảng thuyết minh/notes — "
        f"hãy tìm kiếm kỹ ở TẤT CẢ các bảng trước khi kết luận result = 0.0."
    )


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

    usable, cards, evidence, evidence_plan = _plan_evidence(results, facts_index, derived_dir)

    # Short-circuit: không có bảng truy hồi → fallback 0.0 ngay, không gọi LLM (tiết
    # kiệm ~120s/câu vô ích — xem phân tích fail: 26/470 câu wallcap có evidence=0).
    if not usable:
        return _make_record(
            qid, question, [], entities, "result = 0.0", 0.0,
            f"retrieval: không có bảng truy hồi ({len(results)} kết quả, 0 khả dụng)",
            evidence_plan,
        )

    # Deterministic engine trước: lookup đơn giản (1 ticker + 1 chỉ tiêu) → không cần LLM.
    det = solve_deterministic(
        question, entities, facts_index, list(evidence.values()), evidence_plan
    )
    if det is not None:
        var = det["row"]["var"]
        pandas_query = build_template_query(entities, det["row"], var, det["year"])
        return _make_record(
            qid, question, usable, entities, pandas_query, det["answer"],
            f"deterministic({det['tier']})", evidence_plan,
        )

    # Pre-check (advisory): thêm hint vào prompt nếu evidence thiếu statement câu hỏi cần.
    # KHÔNG hard-block (đã thấy 487/1012 câu LLM trả lời được bị block nhầm).
    pre_hint = _precheck_hint(entities, cards, usable, derived_dir)

    messages = build_messages(question, entities, cards)
    if pre_hint:
        messages[-1]["content"] += f"\n\n⚠️ HINT: {pre_hint}"
    pandas_query = llm.generate_query(messages)

    answer: float | None = None
    exec_error = ""
    table_refs = set(evidence.keys())
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

    return _make_record(qid, question, usable, entities, pandas_query, answer, exec_error, evidence_plan)


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
    evidence_plan: list[dict[str, Any]] | None = None,
) -> dict:
    evidence_plan = evidence_plan or []
    relevant_docs: list[str] = []
    seen: set[str] = set()
    for r in results:
        if r.report_id and r.report_id not in seen:
            seen.add(r.report_id)
            relevant_docs.append(r.report_id)
    # relevant_tables = các table vật lý tham gia evidence (bảng gộp → mọi fragment).
    relevant_tables: list[str] = []
    seen_rt: set[str] = set()
    for ev in evidence_plan:
        for frag in ev.get("fragments") or []:
            key = f"{ev['report_id']}|{frag}"
            if key not in seen_rt:
                seen_rt.add(key)
                relevant_tables.append(key)
    # evidence = đúng thứ tự biến thực thi (df1, df2, ...) — csv_path flat.
    evidence = [
        {
            "variable": ev["variable"],
            "csv_path": f"data/{ev['report_id']}__{ev['name']}.csv",
        }
        for ev in evidence_plan
    ]
    return {
        "id": qid,
        "question": question,
        "answer": float(answer),
        "relevant_docs": relevant_docs,
        "relevant_tables": relevant_tables,
        "evidence": evidence,
        "pandas_query": pandas_query,
        "_ok": answer != 0.0 or not exec_error or exec_error.startswith("deterministic"),
        "_error": exec_error,
    }