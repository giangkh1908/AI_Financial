"""deterministic.py — deterministic answer engine (không LLM).

2 tầng:
1. **Facts tier** (BS/IN/CF — item_label ASCII sạch, có Mãsố, value_vnd chuẩn VND)
   → primary cho câu có statement hint. Match label + bonus Mãsố.
2. **Evidence tier** (tidy/merged — mọi bảng retrieval trả, gồm notes)
   → fallback khi facts không match (notes items như "Lãi tiền gửi", "ngành Thương mại").

Trả None khi không đủ tự tin → caller fallback LLM. Nhanh (~1s), deterministic.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from vifinqa.etl.numbers import normalize_label
from vifinqa.retrieval.entity import Entities
from vifinqa.retrieval.facts_index import FactsIndex

# Filler thật sự — keyword cấu trúc câu hỏi, KHÔNG phải token metric.
_FILLER = set("""
cua trong nam la bao nhieu con ty me hop nhat rieng le cuoi dau ngay thang gio
do cac ai cong ty doanh nghiep ngan hang tmcp tct cty tap doan sacom vj
tong so du ton cuoi ky dau ky vnd dong trieu ty nghin duoc va cho voi
noi bo ctcp ket qua hoat dong bieu hien tai san co dinh vo von cong ty me
""".split())

_UNIT_LABELS = ("nghin ty dong", "ty dong", "trieu dong", "nghin dong", "dong")

_YEAR_RE = re.compile(r"(?<!\d)(20\d\d)(?!\d)")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Keyword báo câu phức tạp (so sánh/tăng trưởng/tỷ lệ/argmax) → fallback LLM.
_COMPLEX_RE = re.compile(
    r"\btang\b|\bgiam\b|\bso voi\b|\bchenh lech\b|\bnhieu hon\b|\bit hon\b|"
    r"\bty le\b|\bphan tram\b|\bbien loi nhuan\b|\bper\b|\beps\b|\bcao nhat\b|\bthap nhat\b|"
    r"\bnhanh nhat\b|\bcham nhat\b|\btrung binh\b|\btong cong\b|\bquy do\b"
)

_SCORE_MIN = 0.5  # ngưỡng tự tin tối thiểu


def is_complex(question: str) -> bool:
    return bool(_COMPLEX_RE.search(normalize_label(question)))


def metric_tokens(question: str, entities: Entities) -> list[str]:
    """Token metric từ câu hỏi: bỏ entity words + năm + đơn vị + filler."""
    t = normalize_label(question)
    for tok_text in entities.matched_names.values():
        t = re.sub(r"(?<![a-z0-9])" + re.escape(tok_text) + r"(?![a-z0-9])", " ", t)
    t = re.sub(r"\(\s*[a-z]{2,4}\d?\s*\)", " ", t)  # (VJC)
    t = _YEAR_RE.sub(" ", t)
    for lab in _UNIT_LABELS:
        t = t.replace(" " + lab + " ", " ")
    toks = _TOKEN_RE.findall(t)
    return [w for w in toks if w not in _FILLER and len(w) >= 2]


def _score_label(label: str, metric: list[str], has_code: bool = False) -> float:
    """Token-overlap metric vs label. Thưởng coverage, bonus Mãsố, phạt label dài."""
    lab_toks = set(label.split())
    if not lab_toks or not metric:
        return 0.0
    hit = sum(1 for m in metric if m in lab_toks)
    coverage = hit / len(metric)
    brevity = len(lab_toks) / max(1, len(metric))
    sc = coverage * 0.8 - min(0.25, (brevity - 1.0) * 0.1)
    if has_code:
        sc += 0.15
    return sc


def _facts_match(
    facts_index: FactsIndex,
    entities: Entities,
    metric: list[str],
    year: int,
) -> dict | None:
    """Match trong facts tier (BS/IN/CF). Trả row + answer hoặc None."""
    if not entities.statement:
        return None
    sub = facts_index.get_facts(
        next(iter(entities.tickers)), set(entities.years), entities.report_type, entities.statement
    )
    if sub.empty:
        return None
    best: dict | None = None
    for _, f in sub.iterrows():
        lab = str(f.get("item_label") or "")
        if not lab:
            continue
        code = str(f.get("item_code") or "")
        sc = _score_label(lab, metric, has_code=bool(code))
        if best is None or sc > best["sc"]:
            best = {
                "sc": sc,
                "chi_tieu": lab,
                "Mãsố": code,
                "ky": str(year),
                "value": f["value_vnd"],
                "src_table_ids": str(f.get("src_table_ids") or ""),
            }
    if best is None or best["sc"] < _SCORE_MIN:
        return None
    return best


def _evidence_match(
    evidence_paths: list[Path],
    metric: list[str],
    year: str,
) -> dict | None:
    """Match trong evidence CSVs (mọi bảng retrieval trả, gồm notes)."""
    best: dict | None = None
    for i, path in enumerate(evidence_paths):
        if not Path(path).exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception:
            continue
        if "chi_tieu" not in df.columns:
            continue
        for _, row in df.iterrows():
            lab_raw = str(row.get("chi_tieu") or "")
            if not lab_raw or lab_raw == "nan":
                continue
            if str(row.get("ky") or "") != str(year):
                continue
            lab = normalize_label(lab_raw)
            # Evidence tier = fallback notes (mã thường rỗng) → KHÔNG bonus mã.
            # Bonus mã chỉ dùng ở facts tier (statement rows có mã ổn định).
            sc = _score_label(lab, metric)
            if best is None or sc > best["sc"]:
                best = {
                    "sc": sc,
                    "chi_tieu": lab_raw,
                    "Mãsố": str(row.get("Mãsố") or ""),
                    "ky": str(year),
                    "value": row.get("value"),
                    "var": f"df{i + 1}",
                    "csv_path": str(path),
                }
    if best is None or best["sc"] < _SCORE_MIN:
        return None
    return best


def solve_deterministic(
    question: str,
    entities: Entities,
    facts_index: FactsIndex,
    evidence_paths: list[Path],
    evidence_plan: list[dict] | None = None,
) -> dict | None:
    """Giải lookup đơn giản. Trả {answer, row, year, tier} hoặc None.

    - Facts tier primary (có statement hint), evidence tier fallback (notes).
    - Chỉ lookup đơn giản (1 ticker, không complex arithmetic).
    """
    if len(entities.tickers) != 1 or is_complex(question):
        return None
    if not entities.years:
        return None
    metric = metric_tokens(question, entities)
    if not metric:
        return None
    year = max(entities.years)

    row = _facts_match(facts_index, entities, metric, year)
    tier = "facts"
    if row is not None:
        row["var"] = _facts_var(entities, evidence_plan or [])
        if not row.get("var"):
            # Facts match nhưng merged statement KHÔNG có trong evidence plan
            # (retrieval không trả statement) → không package được → thử evidence tier.
            row = None
    if row is None:
        row = _evidence_match(evidence_paths or [], metric, str(year))
        tier = "evidence"
    if row is None or not row.get("var"):
        return None
    try:
        raw = float(row["value"])
    except (TypeError, ValueError):
        return None
    return {"answer": round(raw / entities.unit_factor, 2), "row": row, "year": str(year), "tier": tier}


def _facts_var(entities: Entities, evidence_plan: list[dict]) -> str:
    """Map facts row (statement) → var dfN trong evidence plan (bảng gộp statement).

    Tìm entry kind=stmt có statement khớp + report_id chứa ticker (+ report_type nếu có).
    """
    ticker = next(iter(entities.tickers))
    for ev in evidence_plan:
        if ev.get("kind") != "stmt":
            continue
        if ev.get("name") != entities.statement:
            continue
        if ticker not in ev.get("report_id", ""):
            continue
        if entities.report_type and entities.report_type not in ev.get("report_id", ""):
            continue
        return ev["variable"]
    return ""


def build_template_query(entities: Entities, row: dict, var_name: str, year: str) -> str:
    """Sinh pandas_query template từ matched row (chạy lại trên evidence CSV)."""
    unit = entities.unit_factor
    div = f" / {unit:g}" if unit != 1.0 else ""
    code = row.get("Mãsố") or ""
    if code:
        mask = f"{var_name}[\"Mãsố\"].astype(str) == \"{code}\""
    else:
        # chi_tieu trong CSV đã chuẩn hoá ASCII (tidy/merged) → snippet phải khớp.
        norm = normalize_label(str(row.get("chi_tieu") or ""))
        words = [w for w in re.sub(r"[()]", " ", norm).split() if len(w) >= 3][:6]
        snippet = " ".join(words) if words else norm
        mask = f"{var_name}[\"chi_tieu\"].astype(str).str.contains(\"{snippet}\", case=False, na=False)"
    return (
        f'sub = {var_name}[({mask}) & ( {var_name}["ky"].astype(str) == "{year}")]\n'
        f'result = round(float(sub["value"].iloc[0]){div}, 2) if len(sub) > 0 else 0.0'
    )