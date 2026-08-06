"""runner.py — subprocess thực thi pandas_query trong sandbox cô lập.

Đọc stdin JSON: ``{"code": str, "evidence": {var: abs_csv_path}}``.
- Nạp mỗi evidence CSV ``dtype=str, index_col=0`` → DataFrame gán vào biến ``var``.
- Inject globals: ``pd, np, math, re, json`` + ``vn_num`` (parse số VN) + safe builtins.
- exec code (code phải gán ``result``).
- In stdout JSON: ``{"ok": bool, "result": float|null, "error": str|null, "stdout": str}``.

Chạy qua ``python -I`` (isolated mode: bỏ PYTHONPATH, user site-packages) để
giảm rủi ro injec module. Đây là tuyến 2 sau ast_check.

Không import thêm gì ngoài stdlib + pandas/numpy (đã cài trong .venv) — code LLM
không được import (ast_check chặn). ``vn_num`` = parse_vn_number reuse từ ETL.
"""

from __future__ import annotations

import builtins
import json
import math
import re
import sys
import traceback

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# vn_num — inline copy của vifinqa.etl.numbers.parse_vn_number.
# Runner chạy `python -I` (isolated, bỏ PYTHONPATH) → không import được vifinqa;
# inline để runner chỉ phụ thuộc pandas/numpy (trong venv site-packages).
# Giữ đồng bộ với etl/numbers.py: . nghìn, , thập phân, (x)/-x âm, - = rỗng.
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"^\s*(?P<sign>-|–)?\s*\(?\s*(?P<num>[\d.,\s]+)\s*\)?\s*%?\s*$")


def vn_num(s):
    """Parse chuỗi số VN → float (None nếu không phải số). Inline = parse_vn_number."""
    if s is None:
        return None
    s = str(s).strip().replace(" ", "").replace(" ", "")
    if not s or s in {"-", "–", "--", "/", "x", "X", "n/a", "N/A"}:
        return None
    m = _NUMBER_RE.match(s)
    if not m:
        return None
    num_str = m.group("num")
    if not num_str:
        return None
    negative = bool(m.group("sign")) or (s.startswith("(") and s.endswith(")"))
    if "," in num_str:
        num_str = num_str.replace(".", "").replace(",", ".")
    else:
        num_str = num_str.replace(".", "")
    try:
        val = float(num_str)
    except ValueError:
        return None
    return -val if negative else val

# Builtins an toàn — bỏ các hàm IO/reflect nguy hiểm (tuyến 2 sau AST).
_UNSAFE_BUILTINS = {
    "open", "eval", "exec", "compile", "input", "globals", "locals", "vars",
    "breakpoint", "__import__", "getattr", "setattr", "delattr",
    "exit", "quit", "help", "license", "copyright", "credits",
}
_SAFE_BUILTINS = {k: v for k, v in vars(builtins).items() if k not in _UNSAFE_BUILTINS}


def _to_float(result) -> float | None:
    """Ép result về float. Chấp nhận scalar / numpy / Series 1 phần tử / 1-cell DataFrame."""
    if result is None:
        return None
    # pandas Series/DataFrame 1 ô
    if isinstance(result, (pd.Series, pd.DataFrame)):
        if result.size == 1:
            result = result.iloc[0]
        else:
            return None
    if isinstance(result, (bool, np.bool_)):
        return float(int(result))
    if isinstance(result, (int, np.integer)):
        return float(result)
    if isinstance(result, (float, np.floating)):
        return float(result)
    # chuỗi số (vd "1234.5") — cố gắng parse
    if isinstance(result, str):
        try:
            return float(result.replace(",", ""))
        except ValueError:
            return None
    return None


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
        code = payload["code"]
        evidence = payload.get("evidence", {})
    except (json.JSONDecodeError, KeyError) as e:
        print(json.dumps({"ok": False, "result": None, "error": f"bad payload: {e}", "stdout": ""}))
        return 0

    # Nạp evidence CSV → dict {var: DataFrame}
    # Evidence = tidy CSV schema cố định [chi_tieu, Mãsố, ky, value] — đọc default
    # (không index_col) để khớp grader BTC. Query dùng .astype(str) cho cột text,
    # float()/astype(float) cho value → robust với mọi dtype inference.
    frames: dict[str, pd.DataFrame] = {}
    try:
        for var, path in evidence.items():
            frames[var] = pd.read_csv(path, dtype=str)
    except Exception as e:
        print(json.dumps({"ok": False, "result": None, "error": f"load evidence: {e}", "stdout": ""}))
        return 0

    g: dict = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "math": math,
        "re": re,
        "json": json,
        "vn_num": vn_num,
    }
    g.update(frames)

    captured: list[str] = []
    _orig_print = print

    def _safe_print(*args, **kwargs):
        try:
            captured.append(" ".join(str(a) for a in args))
        except Exception:
            pass

    g["print"] = _safe_print

    try:
        exec(code, g)  # noqa: S102 — sandbox đã ast_check + isolated mode + safe builtins
    except Exception:
        tb = traceback.format_exc(limit=4)
        print(json.dumps({"ok": False, "result": None, "error": tb, "stdout": "\n".join(captured)}))
        return 0

    if "result" not in g:
        print(json.dumps({"ok": False, "result": None, "error": "code không gán `result`", "stdout": "\n".join(captured)}))
        return 0

    val = _to_float(g["result"])
    if val is None:
        print(json.dumps({"ok": False, "result": None, "error": f"`result` không phải số: {g['result']!r}", "stdout": "\n".join(captured)}))
        return 0

    print(json.dumps({"ok": True, "result": val, "error": None, "stdout": "\n".join(captured)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())