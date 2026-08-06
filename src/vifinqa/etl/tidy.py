"""tidy.py — transform wide table → tidy CSV schema chuẩn cho submission.

Wide table M1 (`data/derived/tables/{rid}/{tid}.csv`) có cột lộn xộn
(`Mã số`/`Mãsố`, ngày `31/12/2016VND`, `Năm 2015VND`, `.1` trùng...). Để
pandas_query chạy lại được ổn định trên mọi môi trường (grader BTC), ta "ép
kiểu" mỗi bảng thành tidy CSV với schema cố định:

    chi_tieu, Mãsố, ky, value

- `chi_tieu`: nhãn dòng (tên chỉ tiêu) — cột đầu / index.
- `Mãsố`: mã số chỉ tiêu (cột "Mã số"/"Mãsố" nếu có, else "").
- `ky`: năm kỳ (rút từ tên cột kỳ; fallback năm báo cáo). Bỏ cột "đầu kỳ".
- `value`: giá trị VND sạch (vn_num(cell) × unit_factor).

Query model sẽ dùng: boolean mask trên cột + `.values[0]` — không `.index`,
không `.iloc` vị trí, không vn_num. Loader-agnostic (khớp grader default read).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from vifinqa.etl.numbers import parse_vn_number

# Tên cột mã số (biến thể OCR) / thuyết minh / đầu kỳ.
_CODE_COLS = ("Mãsố", "Mã số", "Mã số chỉ tiêu", "Codes")
_SKIP_COLS = ("Thuyết minh", "Thuyết minh.", "TM", "Chỉ tiêu", "Mã thuyết minh")
# Tên cột kỳ chỉ "đầu kỳ" (số dư đầu) — bỏ để tránh nhập nhằng cùng năm.
_OPENING_HINTS = ("01/01", "1/1/", "đầu kỳ", "đầu năm", "opening", "số đầu")

_YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


def report_year(report_id: str) -> int:
    """Năm báo cáo từ report_id (vd `HPG_financial_statements_2023_consolidated` → 2023)."""
    m = _YEAR_RE.search(report_id)
    return int(m.group(0)) if m else 0


def _period_year(col: str, fallback_year: int) -> str | None:
    """Rút năm từ tên cột kỳ. None nếu không phải cột kỳ / là cột đầu kỳ."""
    if not isinstance(col, str):
        return None
    low = col.lower()
    if any(h in low for h in _OPENING_HINTS):
        return None
    m = _YEAR_RE.search(col)
    if m:
        return m.group(0)
    # cột kỳ không có số năm (vd "Trong năm", "Số cuối năm") → năm báo cáo
    if fallback_year:
        return str(fallback_year)
    return None


def _is_code_col(col: str) -> bool:
    return any(c in col for c in _CODE_COLS)


def _is_skip_col(col: str) -> bool:
    return any(c in col for c in _SKIP_COLS)


def wide_to_tidy(df: pd.DataFrame, report_year: int, unit_factor: float = 1.0) -> pd.DataFrame:
    """Wide table (index_col=0, dtype=str) → tidy DataFrame [chi_tieu, Mãsố, ky, value].

    - Bỏ dòng không có nhãn (section title trùng) và dòng không có giá trị hợp lệ.
    - Mãsố = cột mã nếu có; else "".
    - value = parse_vn_number(cell) × unit_factor → VND; bỏ None.
    """
    # Chuẩn hoá cột đầu → chi_tieu (df đầu vào đã index_col=0 → index là nhãn)
    if df.index.name:
        chi = df.index.name
    else:
        chi = "chi_tieu"
    out: list[dict] = []
    # cột mã (nếu có) — lấy cột đầu tiên khớp
    code_col = None
    period_cols: list[str] = []
    for c in df.columns:
        cs = str(c)
        if _is_code_col(cs):
            if code_col is None:
                code_col = c
        elif _is_skip_col(cs):
            continue
        else:
            period_cols.append(c)

    for i, (idx_label, row) in enumerate(df.iterrows()):
        label = str(idx_label).strip()
        if not label or not row.notna().any():
            continue
        code = ""
        if code_col is not None:
            v = row[code_col]
            code = str(v).strip() if pd.notna(v) else ""
        for pc in period_cols:
            ky = _period_year(str(pc), report_year)
            if ky is None:
                continue
            cell = row[pc]
            if pd.isna(cell):
                continue
            val = parse_vn_number(str(cell))
            if val is None:
                continue
            out.append({"chi_tieu": label, "Mãsố": code, "ky": ky, "value": round(val * unit_factor, 6)})

    return pd.DataFrame(out, columns=["chi_tieu", "Mãsố", "ky", "value"])


def wide_csv_to_tidy(path: Path | str, unit_factor: float = 1.0) -> pd.DataFrame:
    """Đọc wide table CSV (index_col=0, str) → tidy DataFrame."""
    df = pd.read_csv(path, dtype=str, index_col=0)
    ryear = report_year(str(path))
    return wide_to_tidy(df, ryear, unit_factor)


def write_tidy_csv(df: pd.DataFrame, dst: Path | str) -> None:
    """Ghi tidy CSV (cột cố định, value float)."""
    pd.DataFrame(df, columns=["chi_tieu", "Mãsố", "ky", "value"]).to_csv(dst, index=False, float_format="%.6f")
