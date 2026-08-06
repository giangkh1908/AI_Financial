"""executor.py — chạy pandas_query trong subprocess cô lập (gọi runner.py).

`run_pandas(code, evidence, root)` → spawn ``python -I runner.py``, truyền payload
JSON qua stdin (code + evidence {table_ref: abs_path}), thu stdout JSON. Timeout
theo config sandbox. Evidence paths resolve về tuyệt đối trước khi truyền (runner
chạy isolated, không nhận path tương đối).

API chính cho codegen loop (agent.loop) và validate (submission).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_RUNNER = Path(__file__).resolve().parent / "runner.py"


def run_pandas(
    code: str,
    evidence: dict[str, str | Path],
    root: Path,
    timeout: int = 20,
) -> dict:
    """Chạy `code` với evidence CSVs. Trả {ok, result, error, stdout}.

    - `evidence`: {var_name: csv_path}. Path resolve về abs: nếu relative → root/path.
    - `root`: thư mục gốc để resolve path tương đối (thường ROOT).
    - `timeout`: giây, kill process nếu vượt.
    """
    root = Path(root)
    ev_abs: dict[str, str] = {}
    for var, p in evidence.items():
        pp = Path(p)
        if not pp.is_absolute():
            pp = root / pp
        ev_abs[var] = str(pp)
    payload = json.dumps({"code": code, "evidence": ev_abs})

    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(_RUNNER)],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "result": None, "error": f"timeout sau {timeout}s", "stdout": ""}
    except Exception as e:
        return {"ok": False, "result": None, "error": f"spawn runner: {e}", "stdout": ""}

    out = proc.stdout.strip()
    if not out:
        return {"ok": False, "result": None, "error": (proc.stderr or "no stdout").strip()[:500], "stdout": ""}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"ok": False, "result": None, "error": f"runner output không phải JSON: {out[:300]}", "stdout": ""}