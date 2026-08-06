"""validate.py — re-exec mỗi pandas_query trên CSV đã đóng gói, so answer.

Mỗi record: chạy sandbox `run_pandas(query, evidence)` với path resolve từ `data_dir`,
so `answer` (abs tol `cfg.answer_abs_tol`). Check mọi `csv_path` start `data/` + file
tồn tại. Ghi `validation.jsonl` + trả summary.

Tối ưu: (1) short-circuit query trivia `result = 0.0` (fallback) — không spawn subprocess;
(2) song song hoá subprocess (ThreadPoolExecutor) — subprocess release GIL.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vifinqa.sandbox import check_code, run_pandas

_TRIVIAL_QUERY = "result = 0.0"


def _table_ref_from_csv_path(csv_path: str) -> str:
    """`data/{report_id}__table_{N}.csv` → `{report_id}|table_N` (khớp grader BTC:
    dfs keyed theo table_ref, không phải variable)."""
    name = csv_path.split("/", 1)[-1]  # bỏ prefix "data/"
    stem = name[:-len(".csv")] if name.endswith(".csv") else name
    marker = "__table_"
    idx = stem.find(marker)
    if idx < 0:
        raise ValueError(f"không tách được table_id: {csv_path}")
    report_id = stem[:idx]
    table_id = "table_" + stem[idx + len(marker):]
    return f"{report_id}|{table_id}"


def _validate_one(rec: dict, root: Path, abs_tol: float, timeout: int, max_code_len: int, max_ast_nodes: int) -> tuple[dict, str]:
    """Trả (verdict, status). status ∈ {ok, crash, mismatch, bad_path}."""
    qid = rec["id"]
    ev_list = rec.get("evidence", [])
    verdict = {"id": qid, "ok": False, "answer": rec.get("answer"), "executed": None, "error": None, "no_evidence": not ev_list}

    # check path
    for ev in ev_list:
        cp = ev.get("csv_path", "")
        if not cp.startswith("data/"):
            verdict["error"] = f"csv_path không bắt đầu data/: {cp}"
            return verdict, "bad_path"
        if not (root / cp).exists():
            verdict["error"] = f"csv_path không tồn tại: {cp}"
            return verdict, "bad_path"

    query = (rec.get("pandas_query") or "").strip()
    # short-circuit trivial fallback (không spawn subprocess)
    if query == _TRIVIAL_QUERY:
        verdict["executed"] = 0.0
        ans = float(rec.get("answer") or 0.0)
        if abs(0.0 - ans) <= abs_tol:
            verdict["ok"] = True
            return verdict, "ok"
        verdict["error"] = f"mismatch: exec=0.0 vs answer={ans}"
        return verdict, "mismatch"

    cok, cerr = check_code(query, max_code_len=max_code_len, max_ast_nodes=max_ast_nodes)
    if not cok:
        verdict["error"] = f"ast: {cerr}"
        return verdict, "crash"
    # Evidence keyed theo table_ref (khớp grader BTC), KHÔNG phải variable.
    # dfs["{report_id}|table_N"] — mirror sandbox.py grader.
    evidence = {}
    for ev in ev_list:
        try:
            key = _table_ref_from_csv_path(ev["csv_path"])
        except ValueError as e:
            verdict["error"] = str(e)
            return verdict, "bad_path"
        evidence[key] = str(root / ev["csv_path"])
    out = run_pandas(query, evidence, root, timeout=timeout)
    if not out.get("ok"):
        verdict["error"] = out.get("error", "")
        return verdict, "crash"
    executed = float(out["result"])
    verdict["executed"] = executed
    ans = float(rec.get("answer") or 0.0)
    if abs(executed - ans) <= abs_tol:
        verdict["ok"] = True
        return verdict, "ok"
    verdict["error"] = f"mismatch: exec={executed} vs answer={ans}"
    return verdict, "mismatch"


def validate(
    submission_json: Path,
    data_dir: Path,
    out_path: Path,
    abs_tol: float = 0.01,
    timeout: int = 20,
    workers: int = 8,
    max_code_len: int = 6000,
    max_ast_nodes: int = 500,
) -> dict:
    """Validate submission. Trả summary {total, ok, crash, mismatch, no_evidence, bad_path}."""
    records = json.loads(submission_json.read_text(encoding="utf-8"))
    root = Path(data_dir).parent  # out_dir: csv_path = "data/..." → resolve qua data_dir

    summary = {"total": len(records), "ok": 0, "crash": 0, "mismatch": 0, "no_evidence": 0, "bad_path": 0}
    verdicts: list[dict] = [None] * len(records)  # type: ignore[list-item]

    def _do(i: int, rec: dict) -> tuple[int, dict, str]:
        v, status = _validate_one(rec, root, abs_tol, timeout, max_code_len, max_ast_nodes)
        return i, v, status

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_do, i, rec) for i, rec in enumerate(records)]
        for fut in as_completed(futs):
            i, v, status = fut.result()
            verdicts[i] = v
            summary[status] += 1
            if v.get("no_evidence"):
                summary["no_evidence"] += 1

    with open(out_path, "w", encoding="utf-8") as fout:
        for v in verdicts:
            fout.write(json.dumps(v, ensure_ascii=False) + "\n")
    return summary