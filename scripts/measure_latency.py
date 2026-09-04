"""
Latency Benchmark Script for End-to-End Query Execution.
Measures:
1. SLM Inference Latency (Ollama qwen3.5-4b-sql)
2. SQLite Execution Latency (financial.db B-Tree Index)
3. Total End-to-End Latency
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.slm import generate_sql
from src.db import execute_query, extract_provenance


def measure_single_query(question: str):
    print(f"\n==================================================")
    print(f"BENCHMARK CÂU HỎI: \"{question}\"")
    print(f"==================================================")

    t0 = time.perf_counter()

    # Step 1: SLM Text-to-SQL
    t_slm_start = time.perf_counter()
    slm_res = generate_sql(question, allow_fallback=False)
    t_slm_end = time.perf_counter()
    slm_latency_ms = round((t_slm_end - t_slm_start) * 1000, 2)

    print(f"1. SLM Text-to-SQL:")
    print(f"   • Trạng thái: {slm_res['status']}")
    print(f"   • Model: {slm_res.get('model')}")
    print(f"   • Độ trễ SLM: {slm_latency_ms:,.2f} ms ({slm_latency_ms/1000:.2f}s)")
    print(f"   • SQL sinh ra:\n     {slm_res.get('sql_query')}")

    sql_query = slm_res.get("sql_query")
    if not sql_query:
        print("   [!] Lỗi: Model không sinh được SQL.")
        return

    # Step 2: DB Execution
    t_db_start = time.perf_counter()
    db_res = execute_query(sql_query)
    t_db_end = time.perf_counter()
    db_latency_ms = round((t_db_end - t_db_start) * 1000, 2)

    print(f"\n2. SQLite Execution (data/financial.db):")
    print(f"   • Trạng thái: {db_res['status']}")
    print(f"   • Số dòng trả về: {db_res['row_count']}")
    print(f"   • Độ trễ CSDL: {db_latency_ms:,.2f} ms")

    # Step 3: Provenance extraction
    provenance = extract_provenance(db_res.get("rows", []))

    total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    print(f"\n3. Kết quả Dữ liệu:")
    for idx, r in enumerate(db_res.get("rows", [])[:3], 1):
        print(f"   [{idx}] {r}")

    print(f"\n4. Dẫn chứng Kiểm toán (Provenance):")
    for p in provenance[:2]:
        print(f"   • {p['ticker']} | {p['company_name']} | Trang {p['page_no']} | File: {p['source_doc']} | Raw: {p['raw_value']}")

    print(f"\n--------------------------------------------------")
    print(f"TỔNG THỜI GIAN END-TO-END: {total_latency_ms:,.2f} ms ({total_latency_ms/1000:.2f}s)")
    print(f"  - SLM Reasoning & Gen:  {slm_latency_ms:,.2f} ms ({slm_latency_ms/total_latency_ms*100:.1f}%)")
    print(f"  - SQLite B-Tree Query:  {db_latency_ms:,.2f} ms ({db_latency_ms/total_latency_ms*100:.1f}%)")
    print(f"==================================================\n")


if __name__ == "__main__":
    q = "Lãi tiền gửi năm 2018 của công ty mẹ Vietjet (VJC) là bao nhiêu triệu đồng?"
    measure_single_query(q)
