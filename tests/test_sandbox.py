"""test_sandbox.py — AST safety check + executor end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vifinqa.sandbox import check_code, run_pandas
from vifinqa.config import ROOT


# ---------------------------------------------------------------------------
# AST check — escape cases phải fail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = 1.0",
        "import socket",
        "from os import path\nresult = 1.0",
        "f = open('x.txt')\nresult = 1.0",
        "result = __import__('os').system('ls')",
        "result = obj.__class__.__bases__",
        "result = df1.__globals__",
        "result = pd.read_csv('data/x.csv')",
        "result = eval('1+1')",
        "result = getattr(obj, 'x')",
        "result = subprocess.run(['ls'])",  # subprocess là blocked module (Name)
    ],
)
def test_check_code_blocks_dangerous(code: str) -> None:
    ok, err = check_code(code)
    assert not ok, f"phải chặn: {code!r} (err={err})"


def test_check_code_blocks_huge() -> None:
    big = "result = 0.0\n" + "x = 1\n" * 500
    ok, err = check_code(big, max_ast_nodes=300)
    assert not ok
    assert "nodes" in err or "dài" in err


def test_check_code_allows_valid() -> None:
    ok, err = check_code(
        "sub = df1[df1['Mã số'] == '60']\n"
        "val = vn_num(sub['2023 VND'].iloc[0]) if len(sub) > 0 else 0.0\n"
        "result = val / 1e9\n"
    )
    assert ok, err


def test_check_code_syntax_error() -> None:
    ok, err = check_code("result = 1 +")
    assert not ok
    assert "syntax" in err.lower()


# ---------------------------------------------------------------------------
# Executor — end-to-end trên wide table tạm
# ---------------------------------------------------------------------------

def _write_wide_table(path: Path) -> None:
    df = pd.DataFrame(
        {
            "Mã số": ["60", "61", "62"],
            "Thuyết minh": ["", "", ""],
            "2023 VND": ["6.800.388.315.081", "6.835.064.334.356", "(34.676.019.275)"],
            "2022 VND": ["8.444.429.054.516", "8.483.510.554.031", "(39.081.499.515)"],
        },
        index=[
            "Lợi nhuận sau thuế TNDN(60 = 50 - 51 - 52)",
            "Cổ đông của công ty mẹ",
            "Cổ đông không kiểm soát",
        ],
    )
    df.index.name = None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8")


def test_run_pandas_filter_by_code(tmp_path: Path) -> None:
    csv = tmp_path / "table_6.csv"
    _write_wide_table(csv)
    code = (
        "df1 = dfs['df1']\n"
        "sub = df1[df1['Mã số'] == '60']\n"
        "val = vn_num(sub['2023 VND'].iloc[0]) if len(sub) > 0 else 0.0\n"
        "result = val / 1e9\n"
    )
    out = run_pandas(code, {"df1": csv}, tmp_path, timeout=15)
    assert out["ok"], out.get("error")
    assert abs(out["result"] - 6800.388315081) < 1e-3


def test_run_pandas_negative_parens(tmp_path: Path) -> None:
    csv = tmp_path / "table_6.csv"
    _write_wide_table(csv)
    code = (
        "df1 = dfs['df1']\n"
        "sub = df1[df1['Mã số'] == '62']\n"
        "val = vn_num(sub['2023 VND'].iloc[0]) if len(sub) > 0 else 0.0\n"
        "result = val / 1e6\n"
    )
    out = run_pandas(code, {"df1": csv}, tmp_path, timeout=15)
    assert out["ok"], out.get("error")
    assert out["result"] < 0
    assert abs(out["result"] - (-34676.019275)) < 1e-3


def test_run_pandas_no_result_assigned(tmp_path: Path) -> None:
    csv = tmp_path / "table_6.csv"
    _write_wide_table(csv)
    out = run_pandas("x = 5", {"df1": csv}, tmp_path, timeout=15)
    assert not out["ok"]
    assert "result" in out["error"].lower()


def test_run_pandas_empty_filter_safe(tmp_path: Path) -> None:
    csv = tmp_path / "table_6.csv"
    _write_wide_table(csv)
    code = (
        "df1 = dfs['df1']\n"
        "sub = df1[df1['Mã số'] == '999']\n"
        "val = vn_num(sub['2023 VND'].iloc[0]) if len(sub) > 0 else 0.0\n"
        "result = val\n"
    )
    out = run_pandas(code, {"df1": csv}, tmp_path, timeout=15)
    assert out["ok"]
    assert out["result"] == 0.0


def test_run_pandas_single_df_alias(tmp_path: Path) -> None:
    """1 CSV → grader inject alias `df` + `dfs` dict; cả 2 đều truy cập được."""
    csv = tmp_path / "table_6.csv"
    _write_wide_table(csv)
    code = (
        "sub = df[df['Mã số'] == '60']\n"
        "val = vn_num(sub['2023 VND'].iloc[0]) if len(sub) > 0 else 0.0\n"
        "result = val / 1e9\n"
    )
    out = run_pandas(code, {"df1": csv}, tmp_path, timeout=15)
    assert out["ok"], out.get("error")
    assert abs(out["result"] - 6800.388315081) < 1e-3


def test_run_pandas_bare_df_nameerror(tmp_path: Path) -> None:
    """Bare `df1` (không qua dfs) → NameError ở grader contract → fail (bảo vệ)."""
    csv = tmp_path / "table_6.csv"
    _write_wide_table(csv)
    code = (
        "sub = df1[df1['Mã số'] == '60']\n"
        "result = 1.0\n"
    )
    out = run_pandas(code, {"df1": csv}, tmp_path, timeout=15)
    assert not out["ok"]
    assert "df1" in out["error"] or "name" in out["error"].lower()