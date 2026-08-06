"""facts_index.py — truy xuất facts (377K dòng) đã chuẩn hoá từ facts_all.csv.

Dùng để: (1) verify bảng truy hồi có facts không, (2) cung cấp schema/label sạch cho
codegen (M4). Facts KHÔNG có cột report_id — tái dựng từ (ticker, year, report_type).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# report_id chuẩn: {TICKER}_financial_statements_{YEAR}_{type}[_{suffix}]
_REPORT_ID_RE = re.compile(r"^([A-Z0-9]+)_financial_statements_(\d{4})_(consolidated|separate|aggregated|other)")


def parse_report_id(report_id: str) -> tuple[str, int, str] | None:
    """report_id → (ticker, year, report_type); None nếu không khớp chuẩn."""
    m = _REPORT_ID_RE.match(report_id)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


class FactsIndex:
    """Index trong RAM trên facts_all.csv (377K rows, ~55MB, load 1 lần)."""

    def __init__(self, facts_all_path: Path):
        self.df = pd.read_csv(facts_all_path, dtype={"year": int}, low_memory=False)
        # index nhanh theo (ticker, year)
        self.df["year"] = self.df["year"].astype(int)

    def get_facts(
        self,
        ticker: str,
        years: set[int],
        report_type: str | None = None,
        statement: str | None = None,
    ) -> pd.DataFrame:
        """Lọc facts theo ticker + năm (+ loại báo cáo / loại BCTC)."""
        df = self.df
        m = df["ticker"] == ticker
        if years:
            m &= df["year"].isin(sorted(years))
        if report_type:
            m &= df["report_type"] == report_type
        if statement:
            m &= df["statement"] == statement
        return df.loc[m]

    def facts_for_table(self, report_id: str, table_id: str) -> pd.DataFrame:
        """Facts thuộc 1 bảng (table_N) trong 1 báo cáo.

        ⚠️ facts_all.csv không có cột report_id → không phân biệt được report trùng
        (vd HDB_..._separate vs _separate_1). Cần thêm report_id ở ETL (M2) để fix triệt để.
        Hiện match exact src_table_ids (không substring — table_1 không khớp table_10).
        """
        parsed = parse_report_id(report_id)
        if parsed is None:
            return self.df.iloc[0:0]
        ticker, year, rt = parsed
        sub = self.get_facts(ticker, {year}, rt)
        if sub.empty:
            return sub
        return sub.loc[sub["src_table_ids"] == table_id]

    def table_fact_coverage(self, report_id: str, table_id: str) -> int:
        """Số fact row map được về bảng này."""
        return len(self.facts_for_table(report_id, table_id))

    def verify(self, results: list) -> dict:
        """% top-k kết quả có ≥1 fact row (chỉ đếm bảng statement; notes không có facts)."""
        if not results:
            return {"covered": 0.0, "n": 0, "avg_facts": 0.0}
        cov = [r for r in results if self.table_fact_coverage(r.report_id, r.table_id) > 0]
        total_facts = sum(self.table_fact_coverage(r.report_id, r.table_id) for r in results)
        return {
            "covered": len(cov) / len(results),
            "n": len(results),
            "avg_facts": total_facts / len(results),
        }
