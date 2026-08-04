"""statements.py — phân loại bảng: 3 BCTC lõi (balance_sheet/income/cash_flow) hay notes.

Bản M1: **tiêu đề (anchor) + xác nhận cấu trúc bảng** — anchor đơn lẻ bị false-positive
vì notes hay NHẮC tên BCTC ("ghi nhận trong báo cáo kết quả hoạt động kinh doanh",
"ngoài bảng cân đối kế toán", bảng form "Mẫu B02/TCTD-HN", cam kết ngoài bảng...).

Quy trình:
1. Negative filter: bảng thuyết minh / off-balance ("ngoài bảng", "ngoài báo cáo") → None.
2. Positive: tiêu đề BCTC xuất hiện trong anchor (search bất kỳ đâu — header ngân hàng gộp
   tên Cty + địa chỉ + tiêu đề thành 1 dòng).
3. Structural: bảng có header "Mã số" / "STT·Chỉ tiêu" HOẶC cột mã số (VAS/bank) → xác nhận
   đây là bảng thật. Notes chỉ có nhãn + giá trị → None.

M2 sẽ nâng cấp: gộp fragment, trích facts, item_code đầy đủ hơn.
"""

from __future__ import annotations

import re

from vifinqa.etl.numbers import is_period_cell, normalize_label
from vifinqa.etl.parser import TableGrid, find_header_row

# Tiêu đề BCTC lõi (đã normalize). Ngân hàng dùng "Báo cáo tình hình tài chính" cho CĐKT,
# "Báo cáo kết quả hoạt động" (thiếu "kinh doanh") cho KQKD. BCTC tiếng Anh (FPT/DBC/VGC
# 2024-2025) dùng "Balance Sheet"/"Income Statement"/"Statement of Cash Flows".
_STMT_TITLE_RE: dict[str, re.Pattern] = {
    "balance_sheet": re.compile(
        r"bao\s*cao\s*tinh\s*hinh\s*tai\s*chinh|bang\s*can\s*doi\s*ke\s*toan|"
        r"balance\s*sheet|statement\s*of\s*financial\s*position"
    ),
    "income": re.compile(
        r"bao\s*cao\s*ket\s*qua\s*hoat\s*dong(?:\s*kinh\s*doanh)?|"
        r"income\s*statement|statement\s*of\s*comprehensive\s*income"
    ),
    "cash_flow": re.compile(
        r"bao\s*cao\s*luu\s*chuyen\s*tien\s*te|"
        r"cash\s*flow\s*statement|statement\s*of\s*cash\s*flows"
    ),
}

# Negative: bảng thuyết minh / off-balance có nhắc "ngoài bảng"/"ngoài báo cáo"
_NEGATIVE_RE = re.compile(
    r"ngoai\s*bao\s*cao\s*tinh\s*hinh\s*tai\s*chinh|ngoai\s*bang\s*can\s*doi\s*ke\s*toan"
)

# Mã số VAS: 100, 411a, 60 ; Bank STT: I, VI, A, 1, 2
_VAS_CODE_RE = re.compile(r"^\d{1,3}[a-z]?$")
_BANK_STT_RE = re.compile(r"^[IVXLC]+$|^[A-Z]$|^\d{1,2}$")


def _has_statement_structure(grid: TableGrid) -> bool:
    """Xác nhận bảng có cấu trúc BCTC thật.

    BẮT BUỘC có cột năm (header parse được year/date) — loại notes kiểu
    "Số năm khấu hao" (50/25/40) hay "Nhóm nợ" (1,2,3) vốn KHÔNG có cột năm.
    Sau đó: header "Mã số"/"STT·Chỉ tiêu" HOẶC cột mã số (VAS/bank) trong vùng dữ liệu.
    """
    header_idx = find_header_row(grid)
    if header_idx >= grid.n_rows:
        return False
    header_row = grid.rows[header_idx]
    # 1) Phải có ít nhất 1 cột kỳ thật (ngày/năm HOẶC "Số cuối năm"/"Năm nay")
    if not any(is_period_cell(c) for c in header_row):
        return False

    # 2) Header nhận diện cột mã số (VAS "Mã số"/"Mãsố", bank "STT·Chỉ tiêu", English "Codes")
    joined = normalize_label(" ".join(header_row))
    if re.search(r"ma\s*so|codes", joined):
        return True
    if re.search(r"\bstt\b", joined) and re.search(r"chi\s*tieu", joined):
        return True

    # 3) Fallback: tồn tại cột mà ≥50% ô dữ liệu (sau header) là mã VAS hoặc STT ngân hàng
    if grid.n_rows - (header_idx + 1) >= 3:
        for col in range(grid.n_cols):
            vals = [
                grid.rows[r][col].strip()
                for r in range(header_idx + 1, grid.n_rows)
                if col < len(grid.rows[r])
            ]
            vals = [v for v in vals if v and v != "-"]
            if not vals:
                continue
            ok = sum(1 for v in vals if _VAS_CODE_RE.match(v) or _BANK_STT_RE.match(v))
            if ok / len(vals) >= 0.5:
                return True
    return False


def classify_statement(grid: TableGrid, anchor_text: str = "") -> str | None:
    """Trả 'balance_sheet' | 'income' | 'cash_flow' | None (notes)."""
    texts: list[str] = [normalize_label(" ".join(r)) for r in grid.rows[:3]]
    if anchor_text:
        texts += [normalize_label(ln) for ln in anchor_text.splitlines()]
    for t in texts:
        if _NEGATIVE_RE.search(t):
            return None

    stmt: str | None = None
    for ln in (anchor_text or "").splitlines():
        norm = normalize_label(ln)
        if not norm:
            continue
        for s, pat in _STMT_TITLE_RE.items():
            if pat.search(norm):
                stmt = s
                break
        if stmt:
            break

    if stmt is None:
        return None
    return stmt if _has_statement_structure(grid) else None
