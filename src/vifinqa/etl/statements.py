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
from dataclasses import dataclass, field

from vifinqa.etl.numbers import (
    detect_number_format,
    is_period_cell,
    normalize_label,
    parse_number,
    parse_period_header,
)
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

# Mã số VAS: 100, 411a, 60 ; Bank STT: I, VI, A, 1, 2, a, b, c, g (chữ thường)
_VAS_CODE_RE = re.compile(r"^\d{1,3}[a-z]?$")
_BANK_STT_RE = re.compile(r"^[IVXLC]+$|^[A-Za-z]$|^\d{1,2}$")


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


# ===========================================================================
# M2 — Facts tier: item_code, merge fragment, build asset
# ===========================================================================

# Công thức trong nhãn: "60 = 50 - 51 - 52", "( 100 = 110 + 120 + ... )"
_FORMULA_RE = re.compile(r"\d+\s*=\s*\d+(\s*[+\-]\s*\d+)+")
# Dòng "mang sang trang sau" / "mang từ trang trước sang" — dấu hiệu biên fragment
_CARRY_RE = re.compile(r"mang\s*(?:sang|t[uừ]\s*(?:trang|t[uừ]))", re.IGNORECASE)


def _is_code_token(s: str) -> bool:
    """True nếu ô là mã hợp lệ (VAS số/kèm chữ HOẶC bank La Mã/chữ/số ngắn)."""
    s = s.strip()
    if not s:
        return False
    return bool(_VAS_CODE_RE.match(s) or _BANK_STT_RE.match(s))


def find_item_code_col(grid: TableGrid, header_idx: int) -> int | None:
    """Cột mã số: ≥70% giá trị (non-empty, non-`-`) khớp mã VAS/bank. Trả cột dày nhất.

    Phân biệt cột giá trị (số lớn có dấu `.` nghìn) và cột thuyết minh (note ref
    thưa, có dấu `.`): chọn cột có tỉ lệ khớp cao nhất — cột mã thật gần 100%.
    """
    n_data = grid.n_rows - (header_idx + 1)
    if n_data < 3:
        return None
    best_col: int | None = None
    best_ratio = 0.0
    for col in range(grid.n_cols):
        vals = [
            grid.rows[r][col].strip()
            for r in range(header_idx + 1, grid.n_rows)
            if col < len(grid.rows[r])
        ]
        vals = [v for v in vals if v and v != "-"]
        if not vals:
            continue
        ok = sum(1 for v in vals if _is_code_token(v))
        ratio = ok / len(vals)
        if ratio >= 0.7 and ratio > best_ratio:
            best_col, best_ratio = col, ratio
    return best_col


def _period_columns(header_row: list[str]) -> list[tuple[int, str, int | None, str]]:
    """Trả [(col_idx, period_key, year, period_label)] cho các cột kỳ trong header."""
    out: list[tuple[int, str, int | None, str]] = []
    for col, cell in enumerate(header_row):
        if col >= len(header_row):
            continue
        parsed = parse_period_header(cell)
        if parsed is None:
            # nhãn kỳ không có năm ("Số cuối năm") — vẫn là cột giá trị
            if is_period_cell(cell):
                out.append((col, "period_label", None, cell.strip()))
            continue
        key, year = parsed
        out.append((col, key, year, cell.strip()))
    return out


def _label_column(grid: TableGrid, header_idx: int, period_cols: list[int],
                  code_col: int | None) -> int:
    """Cột nhãn chỉ tiêu = cột (không kỳ, không mã) có độ dài text TB cao nhất.

    Nhãn chỉ tiêu là text dài ("Tiền và các khoản tương đương tiền"); section
    marker ("A.","I.") và cột thuyết minh (note ref ngắn) ngắn hơn → chọn cột dài.
    """
    candidates = [
        c for c in range(grid.n_cols)
        if c not in period_cols and c != code_col
    ]
    if not candidates:
        return 0
    best_col = candidates[0]
    best_len = -1.0
    for col in candidates:
        lens = [
            len(grid.rows[r][col].strip())
            for r in range(header_idx + 1, grid.n_rows)
            if col < len(grid.rows[r]) and grid.rows[r][col].strip()
        ]
        avg = sum(lens) / len(lens) if lens else 0.0
        if avg > best_len:
            best_col, best_len = col, avg
    return best_col


def header_signature(grid: TableGrid) -> tuple:
    """Chữ ký header để gom fragment: (n_cols, period_keys, period_years)."""
    header_idx = find_header_row(grid)
    header_row = grid.rows[header_idx] if header_idx < grid.n_rows else []
    periods = _period_columns(header_row)
    return (
        grid.n_cols,
        tuple(p[1] for p in periods),
        tuple(p[2] for p in periods),
    )


@dataclass
class Fragment:
    grid: TableGrid
    table_id: str
    header_idx: int


@dataclass
class StatementAsset:
    """Một BCTC lõi (đã gom fragment) → sẵn sàng emit facts."""
    statement: str
    period_cols: list[int]
    period_keys: list[str]             # period_key mỗi cột giá trị
    period_labels: list[str]           # nhãn gốc header mỗi cột giá trị
    code_col: int | None
    label_col: int
    unit_factor: float
    unit_label: str
    number_format: str = "vi"          # "vi" (. nghìn) | "en" (, nghìn)
    fragments: list[Fragment] = field(default_factory=list)
    src_table_ids: list[str] = field(default_factory=list)


def _is_section_title_row(row: list[str]) -> bool:
    """Dòng section title: mọi ô non-empty giống hệt nhau (vd "TÀI SẢN"×N)."""
    cells = [c.strip() for c in row if c.strip()]
    return bool(cells) and len(set(cells)) == 1


def build_asset(stmt: str, fragments: list[Fragment], unit_factor: float,
                unit_label: str) -> StatementAsset:
    """Xây asset từ các fragment cùng (stmt, signature).

    Lấy header/period/code/label col từ fragment đầu; giả định các fragment còn
    lại cùng cấu trúc (đã kiểm tra bằng header_signature khi gom).
    """
    first = fragments[0]
    header_idx = first.header_idx
    header_row = first.grid.rows[header_idx] if header_idx < first.grid.n_rows else []
    periods = _period_columns(header_row)
    period_cols = [p[0] for p in periods]
    period_keys = [p[1] for p in periods]
    period_labels = [p[3] for p in periods]
    code_col = find_item_code_col(first.grid, header_idx)
    label_col = _label_column(first.grid, header_idx, period_cols, code_col)
    # detect format số trên toàn bộ ô giá trị của mọi fragment (en: FPT/DBC/VGC)
    value_cells: list[str] = []
    for frag in fragments:
        g = frag.grid
        for r in range(frag.header_idx + 1, g.n_rows):
            row = g.rows[r]
            for pc in period_cols:
                if pc < len(row):
                    value_cells.append(row[pc])
    num_fmt = detect_number_format(value_cells)
    return StatementAsset(
        statement=stmt,
        period_cols=period_cols,
        period_keys=period_keys,
        period_labels=period_labels,
        code_col=code_col,
        label_col=label_col,
        unit_factor=unit_factor,
        unit_label=unit_label,
        number_format=num_fmt,
        fragments=fragments,
        src_table_ids=[f.table_id for f in fragments],
    )


# --- merge: gom table liên tiếp cùng (stmt, signature) ---


def group_statement_fragments(
    tables: list[tuple[int, str, TableGrid, str, str | None, float, str]],
) -> list[tuple[str, list[Fragment], float, str]]:
    """Gom table liên tiếp cùng (statement, header_signature) → list asset groups.

    `tables`: danh sách theo thứ tự report, mỗi phần tử
        (table_idx, table_id, grid, anchor, statement|None, unit_factor, unit_label).
    Trả list (statement, [Fragment, ...], unit_factor, unit_label) cho các group
    có statement không None (≥1 fragment).
    """
    groups: list[tuple[str, list[Fragment], float, str]] = []
    cur_stmt: str | None = None
    cur_sig: tuple | None = None
    cur_frags: list[Fragment] = []
    cur_unit_factor = 1.0
    cur_unit_label = "VND"

    def _flush() -> None:
        nonlocal cur_stmt, cur_sig, cur_frags
        if cur_stmt is not None and cur_frags:
            groups.append((cur_stmt, list(cur_frags), cur_unit_factor, cur_unit_label))
        cur_stmt, cur_sig, cur_frags = None, None, []

    for table_idx, table_id, grid, anchor, stmt, uf, ul in tables:
        if grid.n_rows == 0:
            _flush()
            continue
        header_idx = find_header_row(grid)
        frag = Fragment(grid=grid, table_id=table_id, header_idx=header_idx)
        if stmt is None:
            _flush()
            continue
        sig = header_signature(grid)
        if stmt == cur_stmt and sig == cur_sig and cur_frags:
            cur_frags.append(frag)
        else:
            _flush()
            cur_stmt, cur_sig = stmt, sig
            cur_frags = [frag]
            cur_unit_factor, cur_unit_label = uf, ul
    _flush()
    return groups


# --- emit facts từ asset ---


def emit_facts(asset: StatementAsset, ticker: str, year: int,
               report_type: str) -> list[dict]:
    """Duyệt data rows của asset → list fact dict (schema facts CSV).

    Bỏ header + section title + dòng không có giá trị nào. Dedupe theo
    item_code (non-empty) ở biên fragment ("mang sang/trang trước"). Label rỗng
    → kế thừa label dòng trước. Mỗi cột kỳ có giá trị → 1 fact (value_vnd = raw*factor).
    """
    facts: list[dict] = []
    seen_codes: set[str] = set()
    last_label = ""
    thousands = "," if asset.number_format == "en" else "."
    decimal = "." if asset.number_format == "en" else ","
    for frag in asset.fragments:
        grid = frag.grid
        for r in range(frag.header_idx + 1, grid.n_rows):
            row = grid.rows[r]
            if _is_section_title_row(row):
                continue
            label_raw = (
                row[asset.label_col].strip()
                if asset.label_col < len(row) else ""
            )
            code_raw = ""
            if asset.code_col is not None and asset.code_col < len(row):
                code_raw = row[asset.code_col].strip()
            # chỉ giữ mã khớp regex; phần rác (vd "TỔNG THU NHẬP...") → ""
            item_code = code_raw if _is_code_token(code_raw) else ""

            values: list[float | None] = []
            has_value = False
            for pc in asset.period_cols:
                v = parse_number(row[pc], thousands, decimal) if pc < len(row) else None
                values.append(v)
                if v is not None:
                    has_value = True
            if not has_value:
                if label_raw:
                    last_label = label_raw
                continue

            if label_raw:
                last_label = label_raw
            else:
                label_raw = last_label

            if item_code:
                if item_code in seen_codes:
                    continue
                seen_codes.add(item_code)

            for pc, pk, pl, v in zip(
                asset.period_cols, asset.period_keys, asset.period_labels, values
            ):
                if v is None:
                    continue
                facts.append({
                    "ticker": ticker,
                    "year": str(year),
                    "report_type": report_type,
                    "statement": asset.statement,
                    "item_code": item_code,
                    "item_label": normalize_label(label_raw),
                    "item_label_raw": label_raw,
                    "period_key": pk if pk != "period_label" else "",
                    "period_label": pl,
                    "value_vnd": repr(float(v) * asset.unit_factor),
                    "src_table_ids": frag.table_id,
                })
    return facts


# --- cross-sum validator (log warning, giữ giá trị gốc) ---


def validate_asset(asset: StatementAsset, facts: list[dict]) -> list[str]:
    """Kiểm tra chất lượng facts → list warning (KHÔNG sửa giá trị).

    - Nhãn chứa công thức ("60 = 50 - 51 - 52") → warning.
    - CĐKT: tổng TÀI SẢN (mã 270) == tổng NGUỒN VỐN (mã 440) nếu cả 2 có mặt.
    """
    warnings: list[str] = []
    for f in facts:
        if _FORMULA_RE.search(f.get("item_label_raw", "")):
            warnings.append(
                f"formula-label {asset.statement} code={f['item_code']}: "
                f"{f['item_label_raw'][:60]} (giữ giá trị gốc)"
            )
    if asset.statement == "balance_sheet":
        by_code: dict[str, dict[str, float]] = {}
        for f in facts:
            if f["item_code"] and f["period_key"]:
                key = f["item_code"]
                pk = f["period_key"]
                by_code.setdefault(key, {})[pk] = float(f["value_vnd"])
        for pk in {pk for d in by_code.values() for pk in d}:
            a = by_code.get("270", {}).get(pk)
            b = by_code.get("440", {}).get(pk)
            if a is not None and b is not None and abs(a - b) > 1.0:
                warnings.append(
                    f"cross-sum BS {pk}: 270={a:.0f} != 440={b:.0f}"
                )
    return warnings
