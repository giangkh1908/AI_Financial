"""numbers.py — parse số, đơn vị, nhãn, kỳ báo cáo từ OCR BCTC.

Chuẩn hoá là nền tảng của toàn pipeline: sai đơn vị = sai đáp án.
Quy ước chung:
- Số OCR: `.` = phân cách nghìn, `,` = thập phân, `(x)` hoặc `-x` = âm, `-`/`–` (đơn) = rỗng/0.
- Đơn vị: ưu tiên nhúng trong header cột năm (vd `31/12/2022Triệu VND`, `2018 Triệu đồng`),
  fallback dòng `Đơn vị tính:` / `ĐVT:` / `Đơn vị tiền tệ:`.
- Mọi value chuẩn hoá về VND: `value_vnd = raw_number * factor`.
"""

from __future__ import annotations

import re
import unicodedata

# Đơn vị → hệ số VND (trùng constants.UNIT_FACTORS nhưng dùng trực tiếp để ETL self-contained)
UNIT_FACTORS: dict[str, float] = {
    "nghìn": 1e3,
    "triệu": 1e6,
    "tỷ": 1e9,
    "trăm": 1e2,
    "đồng": 1.0,
    "vnd": 1.0,
}

# Regex đơn vị nhúng trong header cột năm. Khảo sát: "31/12/2018 VND", "2018VND",
# "31/12/2022Triệu VND" (ngân hàng), "2018 Triệu đồng" (không có chữ VND), "1/1/2018VND".
_HEADER_UNIT_RE = re.compile(
    r"(?P<label>(?:Nghìn|Triệu|Tỷ|Trăm)?\s*(?:VND|đồng))",
    re.IGNORECASE,
)
# Regex dòng đơn vị riêng: "Đơn vị tính: Triệu đồng Việt Nam", "ĐVT: triệu đồng",
# "Đơn vị tiền tệ: Triệu đồng" (chuẩn BCTC).
_UNIT_LINE_RE = re.compile(
    r"(?:Đơn\s*vị\s*(?:tính|tiền\s*tệ)?\s*[::/]|ĐVT\s*[::/])\s*(?P<label>[^:\n]+)",
    re.IGNORECASE,
)
# Regex năm cột: 31/12/2018 | 1/1/2018 | 2018
_YEAR_DATE_RE = re.compile(r"(?P<d>\d{1,2})\s*/\s*(?P<m>\d{1,2})\s*/\s*(?P<y>\d{4})")
_YEAR_BARE_RE = re.compile(r"(?<!\d)(?P<y>20\d\d)(?!\d)")
_RESTATED_RE = re.compile(r"trình\s*bày\s*lại|restated|đã\s*điều\s*chỉnh", re.IGNORECASE)
# Nhãn kỳ không có số năm (BCTC dùng "Số cuối năm"/"Năm nay"/"Cuối kỳ" thay ngày tháng;
# BCTC tiếng Anh dùng "Closing balance"/"Opening balance"/"Current year").
# ⚠️ KHÔNG khớp "Số năm" (bảng khấu hao notes) — chỉ các cụm kỳ rõ ràng.
_PERIOD_LABEL_RE = re.compile(
    r"\b(cuoi nam|dau nam|cuoi ky|dau ky|nam nay|nam truoc|nam hien tai|"
    r"nam ke toan|nam bao cao|ky nay|ky truoc|so cuoi nam|so dau nam|"
    r"so cuoi ky|so dau ky|cuoi nam nay|dau nam nay|"
    r"closing balance|opening balance|current year|prior year|this year|"
    r"last year|current period|prior period|end of year|beginning of year)\b"
)


def is_period_cell(cell: str) -> bool:
    """True nếu cell là cột kỳ: có ngày/năm parse được HOẶC nhãn kỳ (Số cuối năm...)."""
    if parse_period_header(cell) is not None:
        return True
    return bool(_PERIOD_LABEL_RE.search(normalize_label(cell)))


# ---------------------------------------------------------------------------
# Số
# ---------------------------------------------------------------------------

# Số âm: dấu trừ đứng trước (-1.234) hoặc trong ngoặc ((1.234)); % tuỳ chọn.
_NUMBER_RE = re.compile(r"^\s*(?P<sign>-|–)?\s*\(?\s*(?P<num>[\d.,\s]+)\s*\)?\s*%?\s*$")


def parse_vn_number(s: str) -> float | None:
    """Parse 1 ô số OCR → float; trả None nếu không phải số.

    Xử lý: bỏ dấu '.' nghìn, ',' → '.', `(x)`/`-x` → âm, `-`/`–` → None.
    - `parse_vn_number("(11.078.921.256)") == -11078921256`
    - `parse_vn_number("-1.234") == -1234.0`
    - `parse_vn_number("-") is None`
    - `parse_vn_number("4.037") == 4037`   (EPS: nghìn separator)
    """
    if s is None:
        return None
    s = s.strip().replace(" ", "").replace(" ", "")
    if not s or s in {"-", "–", "--", "/", "x", "X", "n/a", "N/A"}:
        return None
    m = _NUMBER_RE.match(s)
    if not m:
        return None
    num_str = m.group("num")
    if not num_str:
        return None
    negative = bool(m.group("sign")) or (s.startswith("(") and s.endswith(")"))
    # Dấu phân cách: loại '.' (nghìn), ',' (thập phân) → '.'
    if "," in num_str:
        num_str = num_str.replace(".", "").replace(",", ".")
    else:
        num_str = num_str.replace(".", "")
    try:
        val = float(num_str)
    except ValueError:
        return None
    return -val if negative else val


def parse_number(s: str, thousands: str = ".", decimal: str = ",") -> float | None:
    """Parse ô số theo quy ước phân cách đã biết.

    - Việt Nam (mặc định): `.` = nghìn, `,` = thập phân.
    - English: `thousands=","`, `decimal="."` (vd "8,198,590,237.083" → 8198590237.083).
    `(x)`/`-x` → âm; `-`/`–` → None.
    """
    if s is None:
        return None
    s = s.strip().replace(" ", "").replace(" ", "")
    if not s or s in {"-", "–", "--", "/", "x", "X", "n/a", "N/A"}:
        return None
    m = _NUMBER_RE.match(s)
    if not m:
        return None
    num_str = m.group("num")
    if not num_str:
        return None
    negative = bool(m.group("sign")) or (s.startswith("(") and s.endswith(")"))
    if decimal in num_str:
        num_str = num_str.replace(thousands, "").replace(decimal, ".")
    else:
        num_str = num_str.replace(thousands, "")
    try:
        val = float(num_str)
    except ValueError:
        return None
    return -val if negative else val


# ---------------------------------------------------------------------------
# Detect format số: Việt Nam (. nghìn, , thập phân) vs English (, nghìn, . thập phân)
# Dùng cho BCTC tiếng Anh (FPT/DBC/VGC 2024-2025).
# ---------------------------------------------------------------------------

_EN_THOUSANDS_RE = re.compile(r"^[+-]?\(?\d{1,3}(,\d{3})+(\.\d+)?\)?%?$")
_VI_THOUSANDS_RE = re.compile(r"^[+-]?\(?\d{1,3}(\.\d{3})+(,\d+)?\)?%?$")


def _clean_num_token(s: str) -> str:
    """Lấy phần số sạch (bỏ dấu ngoặc, khoảng trắng) để kiểm tra format."""
    if s is None:
        return ""
    s = s.strip().replace(" ", "").replace(" ", "")
    s = s.strip("()").strip()
    s = s.lstrip("+-").strip()
    s = s.rstrip("%")
    return s


def detect_number_format(cells: list[str]) -> str:
    """Trả 'en' hoặc 'vi' theo đa số ô giá trị.

    BCTC tiếng Anh dùng `,` làm phân cách nghìn ("8,198,590,237,083"); tiếng Việt
    dùng `.` ("8.198.590.237.083"). Ô có cả 2 dấu thì không mơ hồ (vi: 1.234,56 /
    en: 1,234.56). Mặc định 'vi' khi không quyết định được.
    """
    en = vi = 0
    for c in cells:
        tok = _clean_num_token(c)
        if not tok:
            continue
        if _EN_THOUSANDS_RE.match(tok):
            en += 1
        elif _VI_THOUSANDS_RE.match(tok):
            vi += 1
    return "en" if en > vi else "vi"


# ---------------------------------------------------------------------------
# Đơn vị
# ---------------------------------------------------------------------------


def normalize_label(s: str) -> str:
    """Chuẩn hoá nhãn: bỏ dấu (NFD), Đ/đ→D/d, hạ thường, dồn khoảng trắng.

    OCR hay lỗi dấu ("Triệu Đông Việt Nam") → normalize giúp khớp substring.
    ⚠️ U+0110 (Đ/đ) không phân rã bởi NFD → phải thay thế thủ công.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("Đ", "D").replace("đ", "d")
    return re.sub(r"\s+", " ", s.lower()).strip()


def unit_factor_from_label(label: str) -> float:
    """Nhãn đơn vị → hệ số VND. Kiểm tra nghìn→triệu→tỷ trước (word boundary).

    Chấp nhận cả nhãn có dấu lẫn đã normalize (tự normalize trong hàm).
    """
    norm = normalize_label(label)
    if not norm:
        return 1.0
    if re.search(r"\bnghin\b", norm):
        return 1e3
    if re.search(r"\btrieu\b", norm):
        return 1e6
    if re.search(r"\bty\b", norm):
        return 1e9
    if re.search(r"\btram\b", norm):
        return 1e2
    return 1.0


def detect_unit_in_header(cells: list[str]) -> tuple[float, str] | None:
    """Tìm đơn vị trong các cell header cột năm.

    Vd `31/12/2022Triệu VND` → (1e6, "Triệu VND"); `2018 VND` → (1.0, "VND");
    `2018 Triệu đồng` → (1e6, "Triệu đồng"). Trả None nếu không tìm thấy.
    """
    for c in cells:
        m = _HEADER_UNIT_RE.search(c)
        if m:
            label = m.group("label").strip()
            return unit_factor_from_label(label), label
    return None


def detect_unit_in_text(context_text: str) -> tuple[float, str] | None:
    """Tìm dòng đơn vị riêng `Đơn vị tính:` / `ĐVT:` / `Đơn vị tiền tệ:` trong context."""
    for m in _UNIT_LINE_RE.finditer(context_text):
        label = m.group("label").strip()
        if not label:
            continue
        norm = normalize_label(label)
        # Loại bỏ các câu dài không phải đơn vị tiền tệ (vd "Đơn vị báo cáo ...")
        if "vnd" not in norm and "dong" not in norm:
            continue
        return unit_factor_from_label(norm), label
    return None


def detect_unit(header_cells: list[str], context_text: str = "") -> tuple[float, str]:
    """Detect đơn vị cho bảng: ưu tiên header cột, fallback dòng đơn vị trong context.

    `context_text` = text ngay trước bảng (anchor) — KHÔNG phải cả trang, tránh bảng
    thứ 2 trên cùng trang thừa hưởng unit của bảng thứ 1. Trả (factor, label);
    mặc định (1.0, "VND") nếu không tìm thấy.
    """
    hit = detect_unit_in_header(header_cells)
    if hit:
        return hit
    if context_text:
        hit = detect_unit_in_text(context_text)
        if hit:
            return hit
    return 1.0, "VND"


# ---------------------------------------------------------------------------
# Kỳ báo cáo (period)
# ---------------------------------------------------------------------------


def parse_period_header(cell: str) -> tuple[str, int | None] | None:
    """Parse header cột năm → (period_key, year) hoặc None nếu không phải cột kỳ.

    - `31/12/2018 VND`      → ("year_end", 2018)
    - `1/1/2018 VND`        → ("year_start", 2018)
    - `2018 VND`            → ("flow_year", 2018)
    - `31/12/2021Triệu VND(trình bày lại)` → ("restated_cur", 2021)  (theo contract plan)
    - `Thuyết minh`, `Mã số` → None
    """
    if cell is None:
        return None
    # Bỏ phần đơn vị để isolate ngày/năm
    body = _HEADER_UNIT_RE.sub("", cell)
    restated = bool(_RESTATED_RE.search(cell))

    m = _YEAR_DATE_RE.search(body)
    if m:
        day, month, year = int(m.group("d")), int(m.group("m")), int(m.group("y"))
        if day == 1 and month == 1:
            key = "year_start"
        else:
            key = "year_end"
        if restated:
            key = "restated_cur"
        return key, year

    m = _YEAR_BARE_RE.search(body)
    if m:
        key = "flow_year"
        if restated:
            key = "restated_cur"
        return key, int(m.group("y"))

    return None
