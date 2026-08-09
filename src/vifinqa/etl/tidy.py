"""tidy.py — transform wide table → tidy CSV schema chuẩn cho submission.

Wide table M1 (`data/derived/tables/{rid}/{tid}.csv`) có cột lộn xộn
(`Mã số`/`Mãsố`, ngày `31/12/2016VND`, `Năm 2015VND`, `.1` trùng...). Để
pandas_query chạy lại được ổn định trên mọi môi trường (grader BTC), ta "ép
kiểu" mỗi bảng thành tidy CSV với schema cố định:

    chi_tieu, Mãsố, ky, value

- `chi_tieu`: nhãn dòng (tên chỉ tiêu) — lấy từ layout.label_col (fallback cột đầu).
- `Mãsố`: mã số chỉ tiêu — layout.code_col (fallback cột tên "Mã số").
- `ky`: năm kỳ (rút từ header cột kỳ; fallback năm báo cáo). Bỏ cột "đầu kỳ".
- `value`: giá trị VND sạch (vn_num(cell) × unit_factor).

Layout (từ `layouts/{rid}.json` / catalog) là NGUỒN SỰ THẬT — cấm tự đoán cột
khi đã có. `unit_factor` từ layout; nếu không có layout (bảng hiếm) → fallback
đoán tên cột như cũ.

Query model sẽ dùng: boolean mask trên cột + `.values[0]` — không `.index`,
không `.iloc` vị trí, không vn_num. Loader-agnostic (khớp grader default read).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from vifinqa.etl.numbers import normalize_label, parse_vn_number

# Tên cột mã số (biến thể OCR) / thuyết minh / đầu kỳ (FALLBACK khi không có layout).
_CODE_COLS = ("Mãsố", "Mã số", "Mã số chỉ tiêu", "Codes")
_SKIP_COLS = ("Thuyết minh", "Thuyết minh.", "TM", "Chỉ tiêu", "Mã thuyết minh")
# Tên cột kỳ chỉ "đầu kỳ" (số dư đầu) — bỏ để tránh nhập nhằng cùng năm.
_OPENING_HINTS = ("01/01", "1/1/", "đầu kỳ", "đầu năm", "opening", "số đầu")

_YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


def report_year(report_id: str) -> int:
    """Năm báo cáo từ report_id (vd `HPG_financial_statements_2023_consolidated` → 2023)."""
    m = _YEAR_RE.search(report_id)
    return int(m.group(0)) if m else 0


def load_layout_dict(derived_dir: Path, report_id: str, table_id: str) -> dict | None:
    """Layout 1 bảng từ `layouts/{rid}.json` → dict (None nếu thiếu)."""
    p = derived_dir / "layouts" / f"{report_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return data.get(table_id)


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


def wide_to_tidy(df: pd.DataFrame, report_year: int, unit_factor: float = 1.0,
                 layout: dict | None = None) -> pd.DataFrame:
    """Wide table (index_col=0, dtype=str) → tidy DataFrame [chi_tieu, Mãsố, ky, value].

    Dùng layout (code_col/label_col/period_cols/unit_factor) khi có — nguồn sự thật.
    Fallback: đoán cột theo tên (Mã số/Thuyết minh/đầu kỳ). Bỏ dòng nhãn rỗng.
    """
    # cột mã (nếu có) — layout.code_col hoặc tên cột khớp
    code_col = None
    period_cols: list[str] = []
    if layout and layout.get("period_cols"):
        # wide CSV: cột index là label (đã index_col=0); period_cols của layout
        # tính trên grid gốc (kể cả cột label) → phải trừ đi 1 cho frame này.
        for c in df.columns:
            cs = str(c)
            if _is_skip_col(cs):
                continue
            period_cols.append(c)
        if layout.get("code_col") is not None:
            code_col = df.columns[layout["code_col"] - 1] if layout["code_col"] - 1 < len(df.columns) else None
    else:
        for c in df.columns:
            cs = str(c)
            if _is_code_col(cs):
                if code_col is None:
                    code_col = c
            elif _is_skip_col(cs):
                continue
            else:
                period_cols.append(c)

    out: list[dict] = []
    for idx_label, row in df.iterrows():
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
            out.append({"chi_tieu": normalize_label(label), "Mãsố": code, "ky": ky, "value": round(val * unit_factor, 6)})

    return pd.DataFrame(out, columns=["chi_tieu", "Mãsố", "ky", "value"])


def grid_to_tidy(grid_rows: list[list[str]], report_year: int,
                 unit_factor: float = 1.0, layout: dict | None = None) -> pd.DataFrame:
    """Raw grid (wide CSV header=None) → tidy [chi_tieu, Mãsố, ky, value].

    Layout nguồn sự thật: label_col / code_col / period_cols tính trên grid gốc
    (kể cả header). Data rows = sau header_idx. Fallback: header_idx=0, label=col0,
    period = mọi cột khác không phải Mã số/Thuyết minh.
    """
    out: list[dict] = []
    if not grid_rows:
        return pd.DataFrame(out, columns=["chi_tieu", "Mãsố", "ky", "value"])
    n_cols = max(len(r) for r in grid_rows)
    header_idx = 0
    period_row_idx = 0
    code_col = None
    label_col = 0
    period_cols: list[int] = []
    value_col = None
    if layout:
        header_idx = int(layout.get("header_idx") or 0)
        period_row_idx = int(layout.get("period_row_idx") or 0)
        if layout.get("code_col") is not None:
            code_col = int(layout["code_col"])
        if layout.get("label_col") is not None:
            label_col = int(layout["label_col"])
        period_cols = [int(x) for x in layout.get("period_cols") or []]
        if layout.get("value_col") is not None:
            value_col = int(layout["value_col"])
        uf = layout.get("unit_factor")
        if uf:
            unit_factor = float(uf)
    else:
        for c in range(n_cols):
            header = grid_rows[header_idx][c] if c < len(grid_rows[header_idx]) else ""
            if c == label_col:
                continue
            if _is_code_col(str(header)):
                code_col = c
            elif _is_skip_col(str(header)):
                continue
            else:
                period_cols.append(c)

    period_row = grid_rows[period_row_idx] if period_row_idx < len(grid_rows) else grid_rows[header_idx]
    for row in grid_rows[header_idx + 1:]:
        label = row[label_col].strip() if label_col < len(row) else ""
        if not label or not any(cell.strip() for cell in row):
            continue
        code = ""
        if code_col is not None and code_col < len(row):
            code = row[code_col].strip()
        if not period_cols and value_col is not None:
            # bảng không có cột kỳ (bảng %/danh sách): emit 1 row, ky = năm báo cáo
            if value_col < len(row):
                cell = row[value_col]
                if cell.strip():
                    val = parse_vn_number(cell)
                    if val is not None:
                        out.append({
                            "chi_tieu": normalize_label(label), "Mãsố": code,
                            "ky": str(report_year), "value": round(val * unit_factor, 6),
                        })
            continue
        for pc in period_cols:
            if pc >= len(row):
                continue
            header_cell = period_row[pc] if pc < len(period_row) else ""
            ky = _period_year(str(header_cell), report_year)
            if ky is None:
                continue
            cell = row[pc]
            if not cell.strip():
                continue
            val = parse_vn_number(cell)
            if val is None:
                continue
            out.append({"chi_tieu": normalize_label(label), "Mãsố": code, "ky": ky, "value": round(val * unit_factor, 6)})

    return pd.DataFrame(out, columns=["chi_tieu", "Mãsố", "ky", "value"])


def wide_csv_to_tidy(path: Path | str, unit_factor: float = 1.0,
                     layout: dict | None = None) -> pd.DataFrame:
    """Đọc wide table CSV (raw grid, header=None) → tidy DataFrame."""
    df = pd.read_csv(path, dtype=str, header=None, keep_default_na=False)
    ryear = report_year(str(path))
    return grid_to_tidy(df.values.tolist(), ryear, unit_factor, layout)


def write_tidy_csv(df: pd.DataFrame, dst: Path | str) -> None:
    """Ghi tidy CSV (cột cố định, value float)."""
    pd.DataFrame(df, columns=["chi_tieu", "Mãsố", "ky", "value"]).to_csv(dst, index=False, float_format="%.6f")
