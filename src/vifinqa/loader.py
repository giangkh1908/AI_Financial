"""Đọc dữ liệu gốc (read-only): code_stock.csv, questions.jsonl, financial_statements."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportMeta:
    report_id: str   # vd AAA_financial_statements_2015_consolidated
    ticker: str      # vd AAA
    year: int        # vd 2015
    report_type: str # consolidated | separate | aggregated | other
    path: Path       # đường dẫn file _extracted.txt


def load_stocks(csv_path: Path) -> dict[str, str]:
    """code_stock.csv → {ticker: tên công ty}.

    File dạng 'Mã CK,Tên công ty' (có header). Ticker viết hoa.
    """
    mapping: dict[str, str] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = (row.get("Mã CK") or "").strip()
            name = (row.get("Tên công ty") or "").strip()
            if ticker:
                mapping[ticker] = name
    return mapping


def load_questions(jsonl_path: Path) -> list[dict]:
    """questions.jsonl → list[dict] giữ nguyên 1:1 thứ tự dòng."""
    out: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def infer_report_type(report_id: str) -> str:
    """Loại báo cáo = token cuối cùng của report_id.

    vd 'AAA_financial_statements_2015_consolidated' → 'consolidated'.
    Không khớp các loại đã biết → 'other'.
    """
    tokens = report_id.split("_")
    if not tokens:
        return "other"
    last = tokens[-1]
    known = {"consolidated", "separate", "aggregated"}
    return last if last in known else "other"


def iter_reports(data_dir: Path) -> list[ReportMeta]:
    """Quét toàn bộ financial_statements → list ReportMeta.

    Cấu trúc: {data_dir}/financial_statements/{TICKER}/{YEAR}/{report_id}/{report_id}_extracted.txt
    """
    fs_root = data_dir / "financial_statements"
    out: list[ReportMeta] = []
    if not fs_root.exists():
        return out
    for ticker_dir in sorted(fs_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for year_dir in sorted(ticker_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            year = int(year_dir.name)
            for report_dir in year_dir.iterdir():
                if not report_dir.is_dir():
                    continue
                report_id = report_dir.name
                txt = report_dir / f"{report_id}_extracted.txt"
                if txt.exists():
                    out.append(
                        ReportMeta(
                            report_id=report_id,
                            ticker=ticker,
                            year=year,
                            report_type=infer_report_type(report_id),
                            path=txt,
                        )
                    )
    return out
