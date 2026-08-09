"""format_classify.py — PHÂN LOẠI FORMAT BCTC → 1 FORMAT CHUNG.

Mục tiêu: mọi bảng statement (DN VAS / ngân hàng TCTD / chứng khoán / bảo hiểm /
BCTC tiếng Anh) đều map về cùng một layout chuẩn:

    TableLayout(header_idx, period_row_idx, code_col, label_col,
                period_cols, thuyet_minh_cols, unit_factor, unit_label,
                number_format)

Đây là tầng QUYẾT ĐỊNH của toàn pipeline: catalog, facts, merged_evidence, tidy,
evidence, retrieval đều tiêu thụ layout này. Sai ở đây = sai mọi thứ hạ nguồn.

Hai bug lịch sử mà module này sửa (đã verify trên OCR thật 100 công ty):
1. parse_table_grid KHÔNG xử lý rowspan → header ngân hàng `rowspan=2 colspan=2`
   lệch cột → bank BS facts hoá label=number, value=index.
2. find_header_row pick nhầm dòng kỳ khi header 2 tầng (dòng "Mã số/Chỉ tiêu"
   + dòng năm "2021 VND") → period_cols chứa cột code, label_col = cột giá trị.

Cách tiếp cận (đọc từ dữ liệu thật, không đoán):
- Header thật = dòng có marker "Mã số"/"Chỉ tiêu"/"STT"/"Codes" (normalize bỏ dấu).
  Nếu không có → dòng đầu có cột kỳ.
- Period = cột header chứa năm/ngày/kỳ-label ("Số cuối năm"...). Với header 2 tầng
  (rowspan), nhãn kỳ có năm nằm ở DÒNG NGAY SAU header → period_row_idx = header_idx+1.
- Code col = cột có header "Mã số"/"Codes" HOẶC cột ≥70% cell khớp mã VAS/bank.
- Label col = cột có header "Chỉ tiêu"/"Tài sản"/"Nguồn vốn" HOẶC cột text TB dài nhất
  (không kỳ, không code, không thuyết minh).
- Thuyết minh col = cột header "Thuyết minh" (bỏ qua khi emit facts).
- Unit = nhúng trong header kỳ ("Triệu VND" → 1e6) fallback dòng "Đơn vị tính:".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vifinqa.etl.numbers import (
    detect_number_format,
    detect_unit,
    is_period_cell,
    normalize_label,
)
from vifinqa.etl.parser import TableGrid

# marker cột mã số / chỉ tiêu / thuyết minh (đã normalize)
_CODE_HEADER_RE = re.compile(r"ma\s*so|codes")
_ITEM_HEADER_RE = re.compile(r"chi\s*tieu|chi\s?ti\s?eu")
_ASSET_HEADER_RE = re.compile(r"tai\s*san|nguon\s*von")
_NOTES_HEADER_RE = re.compile(r"thuyet\s*minh|note|ref")
# mã VAS: 100, 411a, 60, 111.1 ; ngân hàng: A, I, II, III, 1, 2, a, b
_VAS_CODE_RE = re.compile(r"^\d{1,4}[a-z]?(\.\d+)?$")
_BANK_CODE_RE = re.compile(r"^[IVXLC]+$|^[A-Za-z]$|^\d{1,2}$")


@dataclass
class TableLayout:
    """Layout chuẩn hoá của 1 bảng statement. Mọi format map về đây."""
    header_idx: int = 0
    period_row_idx: int = 0            # dòng chứa nhãn kỳ có năm
    code_col: int | None = None
    label_col: int | None = None
    period_cols: list[int] = field(default_factory=list)
    thuyet_minh_cols: list[int] = field(default_factory=list)
    unit_factor: float = 1.0
    unit_label: str = "VND"
    number_format: str = "vi"
    value_col: int | None = None       # bảng KHÔNG có cột kỳ (bảng %/danh sách): cột giá trị chính

    def period_headers(self, grid: TableGrid) -> list[str]:
        r = grid.rows[self.period_row_idx] if self.period_row_idx < grid.n_rows else []
        return [r[c] for c in self.period_cols if c < len(r)]


def _is_code_token(s: str, bank: bool) -> bool:
    s = s.strip()
    if not s:
        return False
    if bank:
        return bool(_BANK_CODE_RE.match(s))
    return bool(_VAS_CODE_RE.match(s))


def _code_col(grid: TableGrid, header_idx: int, period_cols: set[int]) -> int | None:
    """Cột mã: theo header 'Mã số'/'Codes' nếu có; fallback cột ≥70% cell là mã."""
    header = grid.rows[header_idx] if header_idx < grid.n_rows else []
    for c, cell in enumerate(header):
        if _CODE_HEADER_RE.search(normalize_label(cell)):
            return c
    # fallback: cột có ≥70% cell khớp VAS hoặc bank (độc lập nhau)
    best: tuple[int, float] | None = None
    for c in range(grid.n_cols):
        vals = [grid.rows[r][c].strip() for r in range(header_idx + 1, grid.n_rows)
                if c < len(grid.rows[r])]
        vals = [v for v in vals if v and v != "-"]
        if not vals:
            continue
        for bank in (False, True):
            ok = sum(1 for v in vals if _is_code_token(v, bank))
            ratio = ok / len(vals)
            if ratio >= 0.7 and (best is None or ratio > best[1]):
                best = (c, ratio)
    return best[0] if best else None


def _label_col(grid: TableGrid, header_idx: int, period_cols: set[int],
               code_col: int | None) -> int:
    """Cột nhãn: ưu tiên cột có header 'Chỉ tiêu'/'Tài sản'/'Nguồn vốn'.

    Header label có thể colspan sang nhiều cột (ACV: "TÀI SẢN"×2) — cột đầu là
    số section ("A."/"I."), cột sau là label thật. → trong các cột header-match,
    chọn cột có text TB dài nhất. Fallback: cột không kỳ/không code/có text dài.
    """
    def _avg_len(col: int) -> float:
        lens = [len(grid.rows[r][col].strip()) for r in range(header_idx + 1, grid.n_rows)
                if col < len(grid.rows[r]) and grid.rows[r][col].strip()]
        return sum(lens) / len(lens) if lens else 0.0

    header = grid.rows[header_idx] if header_idx < grid.n_rows else []
    matches = [c for c, cell in enumerate(header)
               if c not in period_cols and c != code_col
               and (_ITEM_HEADER_RE.search(normalize_label(cell))
                    or _ASSET_HEADER_RE.search(normalize_label(cell)))]
    if matches:
        return max(matches, key=_avg_len)

    candidates = [c for c in range(grid.n_cols)
                  if c not in period_cols and c != code_col]
    if not candidates:
        return 0
    return max(candidates, key=_avg_len)


def _thuyet_minh_cols(grid: TableGrid, header_idx: int) -> list[int]:
    header = grid.rows[header_idx] if header_idx < grid.n_rows else []
    return [c for c, cell in enumerate(header)
            if _NOTES_HEADER_RE.search(normalize_label(cell))]


def _periods_on(grid: TableGrid, row_idx: int) -> list[int]:
    row = grid.rows[row_idx] if row_idx < grid.n_rows else []
    return [c for c in range(len(row)) if is_period_cell(row[c])]


def _value_col(grid: TableGrid, header_idx: int, label_col: int,
               code_col: int | None) -> int | None:
    """Cột giá trị chính khi bảng KHÔNG có cột kỳ (bảng %/danh sách công ty con).

    Chọn cột đầu tiên (sau label/code) có ≥50% cell parse được số, ưu tiên header
    chứa `%|ty le|so huu|gia tri|biểu quyết`. Trả None nếu không có cột số rõ ràng.
    """
    from vifinqa.etl.numbers import parse_vn_number

    def _num_ratio(col: int) -> float:
        vals = []
        for r in range(header_idx + 1, grid.n_rows):
            row = grid.rows[r]
            if col < len(row):
                c = row[col].strip()
                if c and c != "-":
                    vals.append(c)
        if not vals:
            return 0.0
        ok = sum(1 for v in vals if parse_vn_number(v) is not None)
        return ok / len(vals)

    def _header_has_value(col: int) -> bool:
        hdr = grid.rows[header_idx][col] if header_idx < grid.n_rows and col < len(grid.rows[header_idx]) else ""
        return bool(re.search(r"%|ty\s*le|ti\s*le|so\s*huu|gia\s*tri|bieu\s*quyet|loi\s*ich", normalize_label(hdr)))

    skip = {label_col}
    if code_col is not None:
        skip.add(code_col)
    # ưu tiên cột header có dấu hiệu giá trị (%/tỷ lệ/giá trị) + đủ số
    best: int | None = None
    best_ratio = 0.0
    for c in range(grid.n_cols):
        if c in skip:
            continue
        ratio = _num_ratio(c)
        if ratio < 0.5:
            continue
        if _header_has_value(c):
            return c
        if ratio > best_ratio:
            best, best_ratio = c, ratio
    return best


def _filter_period_cols_by_group(grid: TableGrid, period_row_idx: int,
                                 period_cols: list[int], report_type: str) -> list[int]:
    """Lọc cột kỳ theo nhóm (report_type) nếu bảng có group-row "Tập đoàn/Công ty".

    Format đặc biệt (MSR 2015-2018): cùng 1 bảng statement chứa CẢ cột "Tập đoàn"
    (hợp nhất) lẫn "Công ty" (mẹ). Report consolidated → chỉ lấy cột "Tập đoàn";
    separate → chỉ lấy cột "Công ty". Bảng bình thường (không có group-row) →
    giữ nguyên period_cols.
    """
    if period_row_idx <= 0:
        return period_cols
    up = grid.rows[period_row_idx - 1]
    labels = {c: normalize_label(up[c].strip()) for c in period_cols if c < len(up)}
    groups = set(labels.values())
    if len(groups) <= 1:
        return period_cols  # không có group-row
    # mỗi nhóm khác → kiểm tra có phải "Tập đoàn"/"Công ty"
    has_group_word = any("tap doan" in v or "cong ty" in v for v in groups)
    if not has_group_word:
        return period_cols
    want_consolidated = report_type in ("consolidated", "aggregated")
    keep = []
    for c in period_cols:
        v = labels.get(c, "")
        is_cons = "tap doan" in v
        is_sep = "cong ty" in v and "tap doan" not in v
        if want_consolidated and is_cons:
            keep.append(c)
        elif not want_consolidated and is_sep:
            keep.append(c)
    # nếu lọc không ra nhóm nào (label mờ) → giữ nguyên (fail-safe)
    return keep if keep else period_cols


def detect_layout(grid: TableGrid, anchor_text: str = "",
                  report_type: str = "") -> TableLayout | None:
    """Phân loại 1 bảng → TableLayout chuẩn hoá. Trả None nếu không có cột kỳ.

    Xử lý header 2 tầng: nếu header có marker mã/chỉ-tiêu nhưng KHÔNG có cột kỳ
    (chỉ "Tại ngày 31 tháng 12 năm" / "Năm 2023"), nhãn kỳ có năm nằm ở dòng
    ngay sau → period_row_idx = header_idx + 1, period_cols tính trên dòng đó.
    `report_type` để lọc cột "Tập đoàn"/"Công ty" (MSR 2015-2018).
    """
    if grid.n_rows == 0:
        return None

    # 1) header thật
    header_idx = 0
    for i, row in enumerate(grid.rows[:4]):
        joined = normalize_label(" ".join(row))
        if _CODE_HEADER_RE.search(joined) or _ITEM_HEADER_RE.search(joined):
            header_idx = i
            break
    else:
        for i, row in enumerate(grid.rows[:4]):
            if any(is_period_cell(c) for c in row):
                header_idx = i
                break

    # 2) period columns
    period_cols = _periods_on(grid, header_idx)
    period_row_idx = header_idx
    if not period_cols and header_idx + 1 < grid.n_rows:
        nxt = _periods_on(grid, header_idx + 1)
        if nxt:
            period_cols, period_row_idx = nxt, header_idx + 1

    # 2b) lọc cột theo group "Tập đoàn/Công ty" nếu có (chỉ khi có period cols)
    if period_cols:
        period_cols = _filter_period_cols_by_group(grid, period_row_idx, period_cols, report_type)

    # 3) code / label / thuyết minh (cần trước để loại trừ khỏi value_col)
    period_set = set(period_cols)
    code_col = _code_col(grid, header_idx, period_set)
    label_col = _label_col(grid, header_idx, period_set, code_col)
    notes_cols = _thuyet_minh_cols(grid, header_idx)

    # 4) bảng KHÔNG có cột kỳ → thử detect cột giá trị (bảng %/danh sách công ty)
    value_col = None
    if not period_cols:
        value_col = _value_col(grid, header_idx, label_col, code_col)
        if value_col is None:
            return None

    # 5) unit + number format
    if period_cols:
        period_row = grid.rows[period_row_idx] if period_row_idx < grid.n_rows else []
        header_cells = [period_row[c] for c in period_cols if c < len(period_row)]
        uf, ul = detect_unit(header_cells, anchor_text)
    else:
        period_row = grid.rows[header_idx] if header_idx < grid.n_rows else []
        uf, ul = detect_unit([], anchor_text)

    # 6) number format trên toàn ô giá trị
    value_cells: list[str] = []
    for r in range(period_row_idx + 1, grid.n_rows):
        row = grid.rows[r]
        if period_cols:
            for pc in period_cols:
                if pc < len(row):
                    value_cells.append(row[pc])
        elif value_col is not None and value_col < len(row):
            value_cells.append(row[value_col])
    num_fmt = detect_number_format(value_cells)

    return TableLayout(
        header_idx=header_idx,
        period_row_idx=period_row_idx,
        code_col=code_col,
        label_col=label_col,
        period_cols=period_cols,
        thuyet_minh_cols=notes_cols,
        unit_factor=uf,
        unit_label=ul,
        number_format=num_fmt,
        value_col=value_col,
    )


def classify_table(grid: TableGrid, anchor_text: str = "",
                   report_type: str = "") -> tuple[str | None, TableLayout | None]:
    """Phân loại đầy đủ: (statement_type, layout). statement None nếu không phải BCTC.

    Statement type dựa anchor text (tiêu đề báo cáo) — layout xác nhận có cột kỳ.
    """
    from vifinqa.etl.statements import classify_statement
    stmt = classify_statement(grid, anchor_text)
    if stmt is None:
        return None, None
    layout = detect_layout(grid, anchor_text, report_type)
    if layout is None:
        return None, None
    return stmt, layout
