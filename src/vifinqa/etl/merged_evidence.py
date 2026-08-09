"""merged_evidence.py — gộp facts tier → evidence bảng statement (fix fragment-split).

BCTC lõi (BS/IN/CF) có thể bị ETL tách thành NHIỀU table (CF 2 trang → table_9 +
table_10). Facts tier (M2) đã gộp qua `group_statement_fragments`+`build_asset`+
`emit_facts` → mỗi (item, kỳ) 1 dòng duy nhất. Module này đóng gói lại thành
**bảng evidence gộp** schema chuẩn (khớp MEMORY §4/§8):

    chi_tieu, Mãsố, ky, value

- chi_tieu = item_label (label sạch ASCII, bỏ escape `\(`/`\)`).
- Mãsố = item_code (NaN → "").
- ky = năm kỳ (chuỗi): rút từ period_label nếu ∈ {ry, ry-1, ry-2}, else năm báo cáo.
  Bỏ period_key `year_start` (số dư đầu kỳ — trùng năm cuối kỳ, gây nhập nhằng).
- value = value_vnd (đã chuẩn hoá VND).

Bỏ dòng label rỗng (section title / OCR noise) và period_label chứa "Quyết định"
(noise OCR — item_code rỗng, năm bịa từ số quy định). Dedupe (Mãsố, ky) giữ dòng đầu.
Order: giữ nguyên thứ tự facts (fragment-major → toàn bộ chỉ tiêu của statement
trong 1 bảng, LLM search theo label là đủ).
"""

from __future__ import annotations

import re

import pandas as pd

# Tên statement → tên file evidence gộp (cũng là marker phân loại trong csv_path).
STATEMENTS = ("balance_sheet", "income", "cash_flow")
_YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
_NOISE_HINTS = ("quyết định", "quyet dinh", "qđ-btc", "qd-btc")


def _clean_label(s) -> str:
    """Label sạch: bỏ escape `\(`/`\)` (formula-label của emit_facts), strip."""
    if s is None:
        return ""
    return str(s).replace("\\(", "(").replace("\\)", ")").strip()


def merge_statement_facts(report_year: int, facts: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """facts (cột item_code/item_label/period_key/period_label/value_vnd/src_table_ids)
    → {statement: (DataFrame [chi_tieu, Mãsố, ky, value], src_table_ids)}.

    Trả dict rỗng nếu không có statement nào. src_table_ids = các table_N vật lý
    hợp thành statement (để dựng relevant_tables + reverse map table→statement).
    """
    allowed = {report_year, report_year - 1, report_year - 2}
    out: dict[str, tuple[pd.DataFrame, list[str]]] = {}

    for stmt in STATEMENTS:
        sub = facts[facts["statement"] == stmt]
        if sub.empty:
            continue
        rows: list[dict] = []
        seen: set[tuple[str, str]] = set()
        table_ids: list[str] = []
        for _, f in sub.iterrows():
            label = _clean_label(f.get("item_label"))
            if not label:
                continue
            pk = str(f.get("period_key") or "")
            pl = str(f.get("period_label") or "")
            if pk == "year_start":
                continue  # số dư đầu kỳ — bỏ (trùng năm, nhập nhằng)
            low = pl.lower()
            if any(h in low for h in _NOISE_HINTS):
                continue
            m = _YEAR_RE.search(pl)
            if m and int(m.group(0)) in allowed:
                ky = m.group(0)
            else:
                ky = str(report_year)
            code = str(f.get("item_code") or "").strip()
            key = (code, ky)
            if key in seen:
                continue
            seen.add(key)
            val = f.get("value_vnd")
            if val is None or pd.isna(val):
                continue
            rows.append({"chi_tieu": label, "Mãsố": code, "ky": ky, "value": round(float(val), 6)})
            tid = str(f.get("src_table_ids") or "")
            if tid and tid not in table_ids:
                table_ids.append(tid)
        if rows:
            out[stmt] = (
                pd.DataFrame(rows, columns=["chi_tieu", "Mãsố", "ky", "value"]),
                table_ids,
            )
    return out
