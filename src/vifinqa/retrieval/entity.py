r"""entity.py — trích xuất entity tài chính từ câu hỏi tiếng Việt.

Entity grounding thu hẹp không gian tìm kiếm trước khi đánh hạng nội dung bảng:
ticker(s), năm(s), loại báo cáo, gợi ý loại BCTC, đơn vị.

Chiến lược (đã chốt M3):
- Company-name match longest-first (span che bare-ticker bên trong → Q4/Q8 "CTCP Chứng khoán FPT" → FTS, không phải FPT).
- Alias lọc theo corpus membership (Techcombank→TCB bị bỏ vì TCB ∉ 100 ticker).
- Parenthesized ticker `(VJC)` luôn giữ; bare ticker `\b[a-z]{2,4}\d?\b` khớp với membership.
- `statement` là hint MỀM (KHÔNG bao giờ hard filter — VJC "Lãi tiền gửi" ở notes table_50, statement='').

Không có dep nặng (chỉ re/unicodedata/csv) → test nhanh.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vifinqa.etl.numbers import normalize_label
from vifinqa.loader import load_stocks

# ---------------------------------------------------------------------------
# Các prefix pháp lý bị strip khi sinh variant tên công ty (để khớp câu hỏi viết gọn).
# ---------------------------------------------------------------------------
_LEGAL_PREFIXES = (
    "ngan hang tmcp",
    "ngan hang thuong mai co phan",
    "cong ty co phan",
    "tong cong ty",
    "ngan hang",
    "tap doan",
    "cong ty",
    "ctcp",
    "tct",
    "cty",
)

# Các key quá chung chung — KHÔNG dùng làm variant (khớp nhầm mọi ngân hàng/công ty).
_GENERIC_KEYS = {
    "ngan hang", "cong ty", "tap doan", "tong cong ty", "ctcp",
    "cong ty co phan", "tct", "cty", "tmcp", "thuong mai co phan",
    "co phan", "viet nam",
}

# Alias thương hiệu → ticker (đã normalize). Lọc theo corpus membership lúc load.
COMPANY_ALIASES: dict[str, str] = {
    "bidv": "BID",
    "vietcombank": "VCB",
    "vietinbank": "CTG",
    "sacombank": "STB",
    "vinamilk": "VNM",
    "masan": "MSN",
    "vingroup": "VIC",
    "hoa sen": "HSG",
    "hoa phat": "HPG",
    "techcombank": "TCB",  # TCB ∉ corpus → bị bỏ khi load (chỉ là đối tác trong câu hỏi)
}

# Lexicon gợi ý loại BCTC (hint MỀM). Scan order income → cash_flow → balance_sheet.
STATEMENT_HINT_LEXICON: dict[str, list[str]] = {
    "income": [
        "loi nhuan", "doanh thu", "gia von", "lai thuan", "lai vay", "lai tien gui",
        "lai lo", "lai ty gia", "thu nhap", "chi phi", "bien loi nhuan", "eps",
        "co tuc", "loi nhuan sau thue", "loi nhuan gop", "loi nhuan truoc thue",
    ],
    "cash_flow": [
        "luu chuyen tien", "dong tien", "cfo", "tien thuan", "dong tien hoat dong",
    ],
    "balance_sheet": [
        "tai san", "no phai tra", "phai thu", "phai tra", "ton kho", "tien gui",
        "von chu so huu", "quy", "du phong", "phai sinh", "tai san co dinh",
        "dau tu", "tien mat", "vo von", "du no", "no ngan han", "no dai han",
    ],
}

# Đơn vị (order quan trọng: nghìn tỷ trước tỷ, ...).
_UNIT_PATTERNS: tuple[tuple[float, str], ...] = (
    (1e12, "nghin ty dong"),
    (1e9, "ty dong"),
    (1e6, "trieu dong"),
    (1e3, "nghin dong"),
    (1.0, "dong"),
)


@dataclass(frozen=True)
class Entities:
    """Entity trích xuất từ 1 câu hỏi."""

    tickers: frozenset[str]
    years: frozenset[int]                # set đã expand range (filter IN, không single)
    year_ranges: tuple[tuple[int, int], ...]
    report_type: str | None              # None | "consolidated" | "separate"
    statement: str | None                # hint mềm: balance_sheet | income | cash_flow
    unit_factor: float
    unit_label: str
    matched_names: dict[str, str]        # ticker → text đã khớp (diagnostics)

    def as_dict(self) -> dict:
        return {
            "tickers": sorted(self.tickers),
            "years": sorted(self.years),
            "report_type": self.report_type,
            "statement": self.statement,
            "unit_factor": self.unit_factor,
        }


@dataclass(frozen=True)
class CompanyMap:
    """Bản đồ công ty → ticker, precompute để khớp nhanh."""

    tickers: tuple[str, ...]                       # upper, vd ("AAA", "HPG", ...)
    keys_norm: list[tuple[str, str]]               # (normalized variant, ticker), dài→ngắn
    aliases_norm: dict[str, str]                   # alias normalized → ticker (đã lọc corpus)
    bare_re: re.Pattern                            # \b(?:vjc|ht1|...)\b từ tickers


def _compact(s: str) -> str:
    """normalize_label + thay mọi ký tự không phải [a-z0-9] bằng space (khớp substring ổn định)."""
    return re.sub(r"[^a-z0-9]+", " ", normalize_label(s)).strip()


def _name_variants(company_name: str) -> set[str]:
    """Sinh các variant tên công ty đã chuẩn hoá (bỏ prefix pháp lý, hậu tố CTCP/CTY)."""
    c = _compact(company_name)
    out = {c}
    # strip tiền tố pháp lý lặp (vd "ctcp tap doan hoa phat" → "tap doan hoa phat" → "hoa phat")
    s = c
    changed = True
    while changed and s:
        changed = False
        for p in sorted(_LEGAL_PREFIXES, key=len, reverse=True):
            if s == p:
                break
            if s.startswith(p + " "):
                s = s[len(p) + 1:].strip()
                out.add(s)
                changed = True
                break
    # bỏ hậu tố " ctcp" / " cty" (compact đã thay "-"/"(" bằng space → "tap doan vingroup ctcp")
    for tail in (" ctcp", " cty"):
        if c.endswith(tail):
            out.add(c[: -len(tail)].strip())
    # loại key chung chung / quá ngắn
    return {v for v in out if v and v not in _GENERIC_KEYS and len(v) >= 4}


def load_company_map(stocks_csv_path: Path) -> CompanyMap:
    """Đọc code_stock.csv → CompanyMap (variant dài→ngắn, alias lọc corpus, bare_re)."""
    stocks = load_stocks(stocks_csv_path)  # {TICKER: company_name}
    tickers = tuple(sorted(stocks.keys()))

    keys: set[tuple[str, str]] = set()
    for ticker, name in stocks.items():
        for variant in _name_variants(name):
            keys.add((variant, ticker))
    keys_norm = sorted(keys, key=lambda kv: len(kv[0]), reverse=True)

    aliases_norm = {normalize_label(k): t for k, t in COMPANY_ALIASES.items() if t in stocks}
    bare_re = re.compile(
        r"\b(?:{})\b".format("|".join(t.lower() for t in sorted(tickers, key=len, reverse=True)))
    )
    return CompanyMap(tickers=tickers, keys_norm=keys_norm, aliases_norm=aliases_norm, bare_re=bare_re)


def extract_tickers(question: str, cmap: CompanyMap) -> tuple[frozenset[str], dict[str, str]]:
    """Ticker(s) trong câu hỏi: company-name → alias → bare/paren. Trả (frozenset, matched_names)."""
    q = _compact(question)
    tickers: set[str] = set()
    matched: dict[str, str] = {}
    spans: list[tuple[int, int]] = []  # span của company-name/alias → che bare ticker bên trong

    # 1) company-name match (longest-first) — span đè bare ticker.
    #    Word-boundary (KHÔNG dùng q.find raw): `q.find` match xuyên biên từ —
    #    "tổng tài sản bình quân" → "san binh quan" chứa "an binh" => false-positive ABB.
    for key, ticker in cmap.keys_norm:
        if ticker in tickers:
            continue
        m = re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", q)
        if m:
            tickers.add(ticker)
            matched[ticker] = key
            spans.append(m.span())

    # 2) alias match (word-ish)
    for key, ticker in cmap.aliases_norm.items():
        if ticker in tickers:
            continue
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", q):
            tickers.add(ticker)
            matched[ticker] = key
            spans.append((m.start(), m.end()))
            break

    # 3) bare ticker (suppress span bên trong company-name/alias)
    for m in cmap.bare_re.finditer(q):
        if any(a <= m.start() and m.end() <= b for a, b in spans):
            continue
        ticker = m.group(0).upper()
        if ticker not in tickers:
            tickers.add(ticker)
            matched[ticker] = m.group(0)

    # 4) parenthesized ticker `(vjc)` — luôn giữ
    #    Chạy trên normalize_label(question) (giữ dấu ngoặc) — _compact đã strip '()'.
    t_raw = normalize_label(question)
    for m in re.finditer(r"\(\s*([a-z]{2,4}\d?)\s*\)", t_raw):
        ticker = m.group(1).upper()
        if ticker in cmap.tickers and ticker not in tickers:
            tickers.add(ticker)
            matched[ticker] = m.group(0)

    return frozenset(tickers), matched


def extract_years(question: str) -> tuple[frozenset[int], tuple[tuple[int, int], ...]]:
    """Năm(s) trong câu hỏi: range (`2018–2024`) expand → set; clamp [2015, 2025].

    Clamp jointly (a trước, b theo a) để không sinh tuple đảo khi range nằm ngoài [2015,2025].
    """
    t = normalize_label(question)
    years: set[int] = set()
    ranges: list[tuple[int, int]] = []
    covered: set[int] = set()
    for m in re.finditer(r"(?<!\d)(20\d\d)\s*[–-]\s*(20\d\d)", t):
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        a = max(2015, min(a, 2025))   # clamp a trước
        b = max(a, min(b, 2025))      # clamp b theo a → không đảo
        ranges.append((a, b))
        for y in range(a, b + 1):
            covered.add(y)
            years.add(y)
    for m in re.finditer(r"(?<!\d)(20\d\d)(?!\d)", t):
        y = int(m.group(1))
        if 2015 <= y <= 2025 and y not in covered:
            years.add(y)
    return frozenset(years), tuple(ranges)


def extract_report_type(question: str) -> str | None:
    """"Công ty mẹ"/"BCTC riêng" → separate; "hợp nhất" → consolidated; None = cả hai."""
    t = normalize_label(question)
    if re.search(r"\bcong ty me\b|\bbctc rieng\b|\bbao cao rieng\b", t):
        return "separate"
    if re.search(r"\bhop nhat\b", t):
        return "consolidated"
    return None


def extract_statement_hint(question: str) -> str | None:
    """Gợi ý loại BCTC (hint MỀM) — scan income → cash_flow → balance_sheet, single hint."""
    t = normalize_label(question)
    for stmt in ("income", "cash_flow", "balance_sheet"):
        for kw in STATEMENT_HINT_LEXICON[stmt]:
            if re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", t):
                return stmt
    return None


def extract_units(question: str) -> tuple[float, str]:
    """Đơn vị trong câu hỏi → (factor, nhãn). Order: nghìn tỷ → tỷ → triệu → nghìn → đồng."""
    t = normalize_label(question)
    for factor, pat in _UNIT_PATTERNS:
        if re.search(r"(?<![a-z0-9])" + re.escape(pat) + r"(?![a-z0-9])", t):
            return factor, pat
    return 1.0, ""


def extract_entities(question: str, cmap: CompanyMap) -> Entities:
    """Tổng hợp toàn bộ entity từ 1 câu hỏi."""
    tickers, matched = extract_tickers(question, cmap)
    years, ranges = extract_years(question)
    unit_factor, unit_label = extract_units(question)
    return Entities(
        tickers=tickers,
        years=years,
        year_ranges=ranges,
        report_type=extract_report_type(question),
        statement=extract_statement_hint(question),
        unit_factor=unit_factor,
        unit_label=unit_label,
        matched_names=matched,
    )
