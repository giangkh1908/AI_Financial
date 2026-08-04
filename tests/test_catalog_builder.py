"""Test etl/catalog_builder.py — header_and_labels, unit_factor, merge_catalog_parts."""

from pathlib import Path

from vifinqa.etl.catalog_builder import CATALOG_HEADER, header_and_labels, merge_catalog_parts
from vifinqa.etl.parser import parse_table_grid


def test_header_and_labels_skips_section_titles():
    html = (
        "<table>"
        '<tr><td></td><td>Mã số</td><td>31/12/2018 VND</td></tr>'
        '<tr><td colspan="3">TÀI SẢN</td></tr>'
        "<tr><td>Tiền và các khoản tương đương tiền</td><td>110</td><td>2.515.617</td></tr>"
        "<tr><td>Đầu tư tài chính ngắn hạn</td><td>120</td><td>3.724</td></tr>"
        "</table>"
    )
    grid = parse_table_grid(html)
    header_text, row_labels = header_and_labels(grid, 0)
    assert "TÀI SẢN" not in row_labels
    assert "Tiền và các khoản tương đương tiền" in row_labels
    assert "Đầu tư tài chính ngắn hạn" in row_labels


def test_merge_catalog_parts_concatenates(tmp_path: Path):
    from vifinqa.etl.catalog_builder import CatalogRow

    def row(ticker: str) -> CatalogRow:
        return CatalogRow(
            report_id=f"{ticker}_financial_statements_2018_consolidated",
            ticker=ticker,
            year=2018,
            report_type="consolidated",
            table_id="table_1",
            page_no=1,
            unit="VND",
            unit_factor=1.0,
            is_statement=True,
            statement="balance_sheet",
            header_text="Mã số",
            row_labels="Tiền",
            n_rows=2,
            n_cols=3,
            anchor_context="Bảng cân đối",
        )

    from vifinqa.etl.catalog_builder import write_catalog_csv

    parts = tmp_path / "parts"
    parts.mkdir()
    write_catalog_csv([row("AAA")], parts / "catalog_AAA.csv")
    write_catalog_csv([row("HPG"), row("HPG")], parts / "catalog_HPG.csv")

    out = tmp_path / "catalog_tables.csv"
    n = merge_catalog_parts(parts, out)
    assert n == 3
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == CATALOG_HEADER
    assert lines[1].startswith("AAA_financial_statements")
    assert sum(1 for ln in lines if ln.startswith("HPG_")) == 2


def test_merge_catalog_parts_empty_writes_header(tmp_path: Path):
    parts = tmp_path / "parts"
    parts.mkdir()
    out = tmp_path / "catalog_tables.csv"
    n = merge_catalog_parts(parts, out)
    assert n == 0
    assert out.read_text(encoding="utf-8").splitlines()[0].split(",") == CATALOG_HEADER
