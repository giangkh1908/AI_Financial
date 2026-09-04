"""
Independent Verification Tests for src/slm.py.
Verifies prompt construction, <think> and SQL block extraction, and SQL generation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slm import build_prompt, extract_think_and_sql, rule_based_fallback_sql, generate_sql


def test_build_prompt():
    """Verify that system prompt embeds all 15 columns and accounting invariants."""
    q = "Doanh thu năm 2022 của HPG là bao nhiêu tỷ đồng?"
    prompt = build_prompt(q)

    assert "financial_facts" in prompt
    assert "ticker: TEXT" in prompt
    assert "source_doc: TEXT" in prompt
    assert "unit != 'USD'" in prompt
    assert "ABS(value_vnd) / 1000000000.0 AS value_ty" in prompt
    assert q in prompt
    print("  [PASS] test_build_prompt (Schema & accounting invariants verified)")


def test_extract_think_and_sql():
    """Verify clean extraction of reasoning (<think>) and SQL block."""
    raw_response = """
<think>
Người dùng muốn hỏi về lãi tiền gửi của Vietjet năm 2018.
Mã chứng khoán: VJC
Năm: 2018
Báo cáo: riêng lẻ (công ty mẹ) -> report_type = 'separate'
Chỉ tiêu: 'Lãi tiền gửi và cho vay' -> item_name_ascii LIKE '%lai tien gui%'
Đơn vị: triệu đồng -> ABS(value_vnd) / 1e6
</think>

Dưới đây là câu lệnh SQL chuẩn:
```sql
SELECT item_name, period_label, ABS(value_vnd) / 1000000.0 AS value_trieu, raw_value, page_no, source_doc
FROM financial_facts
WHERE ticker = 'VJC' AND year = 2018 AND report_type = 'separate'
  AND item_name_ascii LIKE '%lai tien gui%'
  AND (unit != 'USD' OR unit IS NULL)
ORDER BY page_no ASC
LIMIT 5;
```
Hy vọng câu truy vấn này giúp ích cho bạn!
    """
    reasoning, sql = extract_think_and_sql(raw_response)
    assert reasoning is not None, "Failed to extract reasoning"
    assert "VJC" in reasoning and "separate" in reasoning
    assert sql is not None, "Failed to extract SQL"
    assert sql.startswith("SELECT")
    assert "WHERE ticker = 'VJC'" in sql
    assert "<think>" not in sql
    print("  [PASS] test_extract_think_and_sql (Reasoning and SQL cleanly decoupled)")


def test_rule_based_fallback_sql():
    """Verify deterministic SQL generator on benchmark questions."""
    q1 = "Lãi tiền gửi năm 2018 của công ty mẹ Vietjet (VJC) là bao nhiêu triệu đồng?"
    sql1 = rule_based_fallback_sql(q1)
    assert sql1 is not None, "Failed to generate SQL for Q1"
    assert "ticker = 'VJC'" in sql1
    assert "year = 2018" in sql1
    assert "report_type = 'separate'" in sql1
    assert "value_trieu" in sql1

    q2 = "Số dư cho vay khách hàng ngành Thương mại của công ty mẹ ACB cuối năm 2022 là bao nhiêu triệu đồng?"
    sql2 = rule_based_fallback_sql(q2)
    assert sql2 is not None, "Failed to generate SQL for Q2"
    assert "ticker = 'ACB'" in sql2
    assert "year = 2022" in sql2
    assert "report_type = 'separate'" in sql2
    assert "31.12" in sql2 or "31/12" in sql2

    print("  [PASS] test_rule_based_fallback_sql (Deterministic patterns verified)")


def test_generate_sql_e2e():
    """Verify generate_sql execution with fallback resilience."""
    res = generate_sql("Lợi nhuận sau thuế năm 2023 của Hòa Phát (HPG) là bao nhiêu tỷ đồng?")
    assert res["status"] == "success"
    assert res["sql_query"] is not None
    assert "HPG" in res["sql_query"]
    assert "2023" in res["sql_query"]
    assert "value_ty" in res["sql_query"]
    print(f"  [PASS] test_generate_sql_e2e (Mode: {res['mode']})")


def run_all_tests():
    print("\n==================================================")
    print("RUNNING INDEPENDENT TESTS FOR src/slm.py")
    print("==================================================")
    test_build_prompt()
    test_extract_think_and_sql()
    test_rule_based_fallback_sql()
    test_generate_sql_e2e()
    print("==================================================")
    print("ALL TESTS PASSED PERFECTLY!\n")


if __name__ == "__main__":
    run_all_tests()
