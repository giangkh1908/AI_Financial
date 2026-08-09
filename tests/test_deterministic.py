"""test_deterministic.py — deterministic answer engine (facts + evidence tier)."""

from __future__ import annotations

import pandas as pd
import pytest

from vifinqa.engine.deterministic import (
    build_template_query,
    is_complex,
    metric_tokens,
    solve_deterministic,
)
from vifinqa.retrieval.entity import Entities
from vifinqa.retrieval.facts_index import FactsIndex


def _entities(tickers=("HPG",), years=(2023,), stmt="income", unit=1.0, label=""):
    return Entities(
        tickers=frozenset(tickers),
        years=frozenset(years),
        year_ranges=((years[0], years[0]),),
        report_type=None,
        statement=stmt,
        unit_factor=unit,
        unit_label=label,
        matched_names={},
    )


class _FactsStub:
    """Giả lập FactsIndex + get_facts trả DataFrame cố định."""

    def __init__(self, rows):
        self._rows = rows

    def get_facts(self, ticker, years, report_type, statement):
        df = pd.DataFrame(self._rows, columns=[
            "ticker", "year", "report_type", "statement", "item_code", "item_label",
            "period_label", "value_vnd", "src_table_ids",
        ])
        return df[df["statement"] == statement]


def _facts_rows():
    return [
        {"ticker": "HPG", "year": 2023, "report_type": "consolidated", "statement": "income",
         "item_code": "60", "item_label": "loi nhuan sau thue", "period_label": "2023 VND",
         "value_vnd": 10_000_000_000.0, "src_table_ids": "table_6"},
        {"ticker": "HPG", "year": 2023, "report_type": "consolidated", "statement": "income",
         "item_code": "10", "item_label": "doanh thu thuan", "period_label": "2023 VND",
         "value_vnd": 100_000_000_000.0, "src_table_ids": "table_6"},
        {"ticker": "HPG", "year": 2023, "report_type": "consolidated", "statement": "cash_flow",
         "item_code": "20", "item_label": "luu chuyen tien thuan tu hoat dong kinh doanh",
         "period_label": "2023 VND", "value_vnd": 5_000_000_000.0, "src_table_ids": "table_9"},
    ]


def _write_evidence(tmp_path, name, rows):
    p = tmp_path / name
    pd.DataFrame(rows, columns=["chi_tieu", "Mãsố", "ky", "value"]).to_csv(p, index=False)
    return p


def test_metric_tokens_keeps_metric():
    e = Entities(
        tickers=frozenset({"HPG"}), years=frozenset({2023}), year_ranges=((2023, 2023),),
        report_type=None, statement="income", unit_factor=1e9, unit_label="ty dong",
        matched_names={"HPG": "hoa phat"},
    )
    toks = metric_tokens("Lợi nhuận sau thuế 2023 của HPG là bao nhiêu tỷ đồng?", e)
    assert "loi" in toks and "nhuan" in toks and "thue" in toks


def test_solve_facts_tier(tmp_path):
    ev = _write_evidence(tmp_path, "income.csv", [
        {"chi_tieu": "loi nhuan sau thue", "Mãsố": "60", "ky": "2023", "value": "10000000000"},
    ])
    plan = [{"variable": "df1", "report_id": "HPG_financial_statements_2023_consolidated",
             "kind": "stmt", "name": "income", "fragments": ["table_6"]}]
    e = _entities(stmt="income", unit=1e9, label="ty dong")
    r = solve_deterministic("Lợi nhuận sau thuế 2023 của HPG là bao nhiêu tỷ đồng?",
                            e, _FactsStub(_facts_rows()), [ev], plan)
    assert r is not None
    assert r["tier"] == "facts"
    assert r["answer"] == 10.0
    assert r["row"]["var"] == "df1"


def test_solve_evidence_tier_fallback(tmp_path):
    """Facts không match (notes) → evidence tier (notes table)."""
    ev = _write_evidence(tmp_path, "notes.csv", [
        {"chi_tieu": "Lãi tiền gửi", "Mãsố": "", "ky": "2018", "value": "208253201298"},
    ])
    e = Entities(tickers=frozenset({"VJC"}), years=frozenset({2018}),
                 year_ranges=((2018, 2018),), report_type="separate",
                 statement="income", unit_factor=1e6, unit_label="trieu dong",
                 matched_names={"VJC": "ctcp hang khong vietjet"})
    r = solve_deterministic(
        "Lãi tiền gửi năm 2018 của công ty mẹ VJC là bao nhiêu triệu đồng?",
        e, _FactsStub([]), [ev], [],
    )
    assert r is not None
    assert r["tier"] == "evidence"
    assert r["answer"] == 208253.20


def test_solve_complex_returns_none(tmp_path):
    ev = _write_evidence(tmp_path, "income.csv", [
        {"chi_tieu": "loi nhuan sau thue", "Mãsố": "60", "ky": "2023", "value": "10000000000"},
    ])
    e = _entities(stmt="income", unit=1e9)
    r = solve_deterministic(
        "Lợi nhuận sau thuế 2023 của HPG tăng bao nhiêu % so với 2022?",
        e, _FactsStub(_facts_rows()), [ev], [],
    )
    assert r is None  # complex → fallback LLM


def test_is_complex():
    assert is_complex("Doanh thu 2023 của HPG tăng bao nhiêu % so với 2022?")
    assert is_complex("Tỷ lệ nợ/vốn 2023 của HPG là bao nhiêu?")
    assert not is_complex("Lợi nhuận sau thuế 2023 của HPG là bao nhiêu tỷ đồng?")


def test_build_template_query_by_code(tmp_path):
    e = _entities(stmt="income", unit=1e9)
    row = {"Mãsố": "60", "chi_tieu": "loi nhuan sau thue"}
    q = build_template_query(e, row, "df1", "2023")
    assert 'df1["Mãsố"]' in q
    assert "/ 1e+09" in q


def test_build_template_query_by_label(tmp_path):
    e = _entities(stmt="income", unit=1.0)
    row = {"Mãsố": "", "chi_tieu": "Lãi tiền gửi"}
    q = build_template_query(e, row, "df1", "2018")
    assert "chi_tieu" in q
    assert "lai" in q  # snippet đã normalize ASCII (khớp CSV chuẩn hoá)