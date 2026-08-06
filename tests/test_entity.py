"""test_entity.py — entity extraction golden (câu hỏi thật, đã verify M3)."""

from pathlib import Path

import json

import pytest

from vifinqa.retrieval.entity import (
    load_company_map,
    extract_tickers,
    extract_years,
    extract_report_type,
    extract_statement_hint,
    extract_units,
    extract_entities,
)

ROOT = Path(__file__).resolve().parents[1]
STOCKS = ROOT / "data" / "code_stock.csv"
QUESTIONS = ROOT / "data" / "questions" / "questions.jsonl"


def _load_questions() -> dict[int, str]:
    qs = {}
    with open(QUESTIONS, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qs[int(obj["id"])] = obj["question"]
    return qs


def test_load_company_map():
    cmap = load_company_map(STOCKS)
    assert len(cmap.tickers) == 100
    assert "HPG" in cmap.tickers and "FTS" in cmap.tickers
    # alias Techcombank→TCB phải bị lọc vì TCB ∉ corpus
    assert "techcombank" not in cmap.aliases_norm
    assert "bidv" in cmap.aliases_norm and cmap.aliases_norm["bidv"] == "BID"
    # variant dài → ngắn (longest-first): "chung khoan fpt" xuất hiện trước ticker dạng từ
    keys = [k for k, _ in cmap.keys_norm]
    assert "hoa phat" in keys
    assert "ctcp chung khoan fpt" in keys


def test_extract_tickers_golden():
    cmap = load_company_map(STOCKS)
    qs = _load_questions()
    cases = {
        1: {"VJC"},
        4: {"FTS"},        # CTCP Chứng khoán FPT → FTS, KHÔNG phải FPT
        7: {"HT1"},
        11: {"BID"},
        30: {"BVH"},
        369: {"HPG", "HSG", "MSR", "NKG"},
        400: {"HPG"},
        689: {"BID"},
        790: {"VIB", "BID"},   # BIDV alias → BID
    }
    for qid, expected in cases.items():
        tickers, matched = extract_tickers(qs[qid], cmap)
        assert tickers == frozenset(expected), f"Q{qid}: {tickers} != {expected}"
    # Q4: matched FTS qua company-name, span che bare "fpt"
    tickers, matched = extract_tickers(qs[4], cmap)
    assert matched["FTS"].startswith("ctcp chung khoan")


def test_extract_tickers_no_company():
    cmap = load_company_map(STOCKS)
    tickers, matched = extract_tickers("Chi phí quản lý doanh nghiệp năm 2024?", cmap)
    assert tickers == frozenset()
    assert matched == {}


def test_extract_years_golden():
    qs = _load_questions()
    years, ranges = extract_years(qs[378])           # 2018–2024
    assert years == frozenset(range(2018, 2025))
    assert ranges == ((2018, 2024),)
    years, _ = extract_years(qs[400])                # 2020-2024
    assert years == frozenset(range(2020, 2025))
    years, _ = extract_years(qs[689])                # 2022 so với 2021
    assert years == frozenset({2021, 2022})
    years, _ = extract_years("câu hỏi không có năm")
    assert years == frozenset()


def test_extract_report_type_golden():
    qs = _load_questions()
    assert extract_report_type(qs[1]) == "separate"     # công ty mẹ
    assert extract_report_type(qs[400]) == "consolidated"  # báo cáo hợp nhất
    assert extract_report_type(qs[790]) == "separate"   # BCTC riêng
    assert extract_report_type("Doanh thu năm 2023 của HPG?") is None


def test_extract_statement_hint_golden():
    qs = _load_questions()
    assert extract_statement_hint(qs[1]) == "income"       # Lãi tiền gửi
    assert extract_statement_hint(qs[11]) == "balance_sheet"  # Số dư tiền gửi TCTD
    assert extract_statement_hint(qs[7]) == "balance_sheet"   # Quỹ khen thưởng
    assert extract_statement_hint("Lưu chuyển tiền thuần năm 2017?") == "cash_flow"


def test_extract_units_golden():
    qs = _load_questions()
    assert extract_units(qs[1]) == (1e6, "trieu dong")     # triệu đồng
    assert extract_units(qs[12]) == (1e12, "nghin ty dong")  # nghìn tỷ đồng
    assert extract_units(qs[4]) == (1e9, "ty dong")        # tỷ đồng
    assert extract_units("Câu hỏi không có đơn vị?") == (1.0, "")


def test_extract_entities_compose():
    cmap = load_company_map(STOCKS)
    qs = _load_questions()
    e = extract_entities(qs[1], cmap)
    assert e.tickers == frozenset({"VJC"})
    assert e.years == frozenset({2018})
    assert e.report_type == "separate"
    assert e.statement == "income"
    assert e.unit_factor == 1e6
    assert e.as_dict()["tickers"] == ["VJC"]


@pytest.mark.skipif(not QUESTIONS.exists(), reason="thiếu data/questions/questions.jsonl")
def test_questions_file_present():
    assert len(_load_questions()) >= 1010
