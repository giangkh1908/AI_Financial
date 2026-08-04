"""Test loader trên dữ liệu thật (nhanh, không cần deps nặng)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_load_stocks():
    from vifinqa.loader import load_stocks

    stocks = load_stocks(DATA / "code_stock.csv")
    assert len(stocks) == 100
    assert "HPG" in stocks
    assert "Hòa Phát" in stocks["HPG"]


def test_load_questions():
    from vifinqa.loader import load_questions

    questions = load_questions(DATA / "questions" / "questions.jsonl")
    assert len(questions) == 1012
    assert "id" in questions[0] and "question" in questions[0]


def test_iter_reports():
    from vifinqa.loader import iter_reports

    reports = iter_reports(DATA)
    assert len(reports) == 1973
    r0 = reports[0]
    assert r0.report_type == "consolidated"
    assert r0.path.exists()
    assert r0.ticker and r0.year


def test_infer_report_type():
    from vifinqa.loader import infer_report_type

    assert infer_report_type("AAA_financial_statements_2015_consolidated") == "consolidated"
    assert infer_report_type("AAA_financial_statements_2015_separate") == "separate"
    assert infer_report_type("AAA_financial_statements_2015_aggregated") == "aggregated"
    assert infer_report_type("XYZ_note_01") == "other"


def test_infer_report_type_split_suffix():
    # Report tách phần có hậu tố số: loại nằm ở token giữa, không phải token cuối
    from vifinqa.loader import infer_report_type

    assert infer_report_type("HDB_financial_statements_2022_separate_1") == "separate"
    assert infer_report_type("NAB_financial_statements_2022_consolidated_2") == "consolidated"
